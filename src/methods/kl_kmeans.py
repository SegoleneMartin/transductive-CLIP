from src.utils import get_one_hot, get_one_hot_full, Logger
from tqdm import tqdm
import torch
import time
from copy import deepcopy
import numpy as np
from .. import utils


class BASE(object):

    def __init__(self, model, device, log_file, args):
        self.device = device
        self.iter = args.iter
        self.model = model
        self.log_file = log_file
        self.logger = Logger(__name__, self.log_file)
        self.init_info_lists()
        self.args = args
        self.eps = 1e-15


    def init_info_lists(self):
        self.timestamps = []
        self.criterions = []
        self.test_acc = []


    def record_convergence(self, new_time, criterions):
        """
        inputs:
            new_time : scalar
            criterions : torch.Tensor of shape [n_task]
        """
        self.criterions.append(criterions)
        self.timestamps.append(new_time)


    def compute_acc(self, y_q):
        """
        inputs:
            y_q : torch.Tensor of shape [n_task, n_query] :
        """

        preds_q = self.u.argmax(2)
        accuracy = (preds_q == y_q).float().mean(1, keepdim=True)
        self.test_acc.append(accuracy)


    def compute_acc_clustering(self, query, y_q, support, y_s_one_hot):
        n_task = query.shape[0]
        preds_q = self.u.argmax(2)
        preds_q_one_hot = get_one_hot_full(preds_q, self.args.n_ways)

        prototypes = ((preds_q_one_hot.unsqueeze(-1) * query.unsqueeze(2)).sum(1)) / (preds_q_one_hot.sum(1).clamp(min=self.eps).unsqueeze(-1))
        cluster_sizes = preds_q_one_hot.sum(1).unsqueeze(-1) # N x K
        nonzero_clusters = cluster_sizes > self.eps
        prototypes = prototypes * nonzero_clusters 
        
        if self.args.graph_matching == True:
            new_preds_q = utils.compute_graph_matching(preds_q, prototypes, self.args)
                
        else:
            new_preds_q = utils.compute_basic_matching(preds_q, prototypes, self.args)

        accuracy = (new_preds_q == y_q).float().mean(1, keepdim=True)
        self.test_acc.append(accuracy)


    def get_logs(self):
        self.criterions = torch.stack(self.criterions, dim=0).cpu().numpy()
        self.test_acc = torch.cat(self.test_acc, dim=1).cpu().numpy()
        return {'timestamps': self.timestamps, 'criterions':self.criterions,
                'acc': self.test_acc}


    def run_task(self, task_dic, shot=10):
        """
        inputs:
            task_dic : dictionnary with n_tasks few-shot tasks
            shot : scalar, number of shots
        """

        # Extract support and query
        y_s = task_dic['y_s']               # [n_task, shot]
        y_q = task_dic['y_q']               # [n_task, n_query]
        support = task_dic['x_s']           # [n_task, shot, feature_dim]
        query = task_dic['x_q']             # [n_task, n_query, feature_dim]

        # Transfer tensors to GPU if needed
        support = support.to(self.device).double()
        query = query.to(self.device).double()
        y_s = y_s.long().squeeze(2).to(self.device)
        y_q = y_q.long().squeeze(2).to(self.device)
        del task_dic
           
        # Run adaptation
        self.run_method(support=support, query=query, y_s=y_s, y_q=y_q)

        # Extract adaptation logs
        logs = self.get_logs()
        return logs


class KL_KMEANS(BASE):

    def __init__(self, model, device, log_file, args):
        super().__init__(model=model, device=device, log_file=log_file, args=args)


    def __del__(self):
        self.logger.del_logger()
        

    def kl_divergence(self, P, Q):
        P = P + self.eps
        Q = Q + self.eps
        KLdiv = torch.sum(P * torch.log(P / Q), dim=-1)
        return KLdiv
     

    def run_method(self, support, query, y_s, y_q):
        """
        Corresponds to the KL_KMEANS inference
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]
            y_q : torch.Tensor of shape [n_task, n_query]

        updates :from copy import deepcopy
            self.u : torch.Tensor of shape [n_task, n_query, num_class]         (hard labels)
            self.w : torch.Tensor of shape [n_task, num_class, feature_dim]     (centroids)
        """

        self.logger.info(" ==> Executing KL KMEANS with T = {}".format(self.args.T))
        
        y_s_one_hot = get_one_hot(y_s)
        n_task, n_support, n_ways = y_s_one_hot.shape
        
        self.u = deepcopy(query)
        u_old = deepcopy(self.u)

        pbar = tqdm(range(self.iter))
        for i in pbar:
            t0 = time.time()
            
            # Update centroids by averaging the assigned samples
            cluster_sizes = self.u.sum(1).unsqueeze(-1)
            nonzero_clusters = cluster_sizes > 0
            self.w = (self.u.transpose(1, 2) @ query) / cluster_sizes.clamp(min=1)
            self.w *= nonzero_clusters.float()  # set zero-sized cluster'
            
            # Assign samples to the cluster with minimum KL divergence
            divs = self.kl_divergence(query.unsqueeze(2), self.w.unsqueeze(1))
            labels = torch.argmin(divs, dim=-1)
            self.u.zero_()
            self.u.scatter_(2, labels.unsqueeze(-1), 1.0)
    
            criterions = (u_old - self.u).norm(dim=(1,2)).mean(0) 
            t1 = time.time()
            self.record_convergence(new_time=t1-t0, criterions=criterions)
            u_old = deepcopy(self.u)
            
            if i in [0, 1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                pbar.set_description(f"Criterion: {criterions}")
                self.record_convergence(new_time=(t1-t0) / n_task, criterions=criterions)
                t1 = time.time()

        t1 = time.time()
        self.record_convergence(new_time=(t1-t0) / n_task, criterions=criterions)
        if self.args.acc_clustering == True:
            self.compute_acc_clustering(query, y_q, support, y_s_one_hot)
        else:
            self.compute_acc(y_q=y_q)

