from src.utils import get_one_hot, get_one_hot_full, Logger
from tqdm import tqdm
import torch
import time
from copy import deepcopy
import numpy as np
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.sparse import csr_matrix
from scipy.optimize import linear_sum_assignment

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
        self.lambd = args.lambd


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
        preds_q_one_hot = get_one_hot_full(preds_q, self.args.num_classes_test)
        new_preds_q = torch.zeros_like(y_q)

        if self.args.shots == 0:
            prototypes = ((preds_q_one_hot.unsqueeze(-1) * query.unsqueeze(2)).sum(1)) / (preds_q_one_hot.sum(1).clamp(min=self.eps).unsqueeze(-1))
            cluster_sizes = preds_q_one_hot.sum(1).unsqueeze(-1) # N x K
            nonzero_clusters = cluster_sizes > self.eps
            prototypes = prototypes * nonzero_clusters 
        else:
            prototypes = ((preds_q_one_hot.unsqueeze(-1) * query.unsqueeze(2)).sum(1) + (y_s_one_hot.unsqueeze(-1) * support.unsqueeze(2)).sum(1)) / (preds_q_one_hot.sum(1) + y_s_one_hot.sum(1)).unsqueeze(-1)

        list_clusters = []
        list_A = []
        for task in range(n_task):
            clusters = []
            num_clusters = len(torch.unique(preds_q[task]))
            A = np.zeros((num_clusters, int(self.args.n_ways)))
            for i, cluster in enumerate(preds_q[task]):
                if cluster.item() not in clusters:
                    A[len(clusters), :] = - prototypes[task, cluster].cpu().numpy()
                    clusters.append(cluster.item())
            list_A.append(A)
            list_clusters.append(clusters)
        
        for task in range(n_task):
            A = list_A[task]
            clusters = list_clusters[task]
            #__, matching_classes = min_weight_full_bipartite_matching(csr_matrix(A), maximize=False)
            __, matching_classes = linear_sum_assignment(A, maximize=False)
            for i, cluster in enumerate(preds_q[task]):
                new_preds_q[task, i] = matching_classes[clusters.index(cluster)]
     
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

        scaler = MinMaxScaler(feature_range=(0, 1))
        support, query, scale, ratio = scaler(support, query)
           
        # Run adaptation
        self.run_method(support=support, query=query, y_s=y_s, y_q=y_q)

        # Extract adaptation logs
        logs = self.get_logs()
        return logs


class PADDLE(BASE):

    def __init__(self, model, device, log_file, args):
        super().__init__(model=model, device=device, log_file=log_file, args=args)


    def __del__(self):
        self.logger.del_logger()
     
    def get_logits(self, samples):
        """
        inputs:
            samples : torch.Tensor of shape [n_task, shot, feature_dim]
        returns :
            logits : torch.Tensor of shape [n_task, shot, num_class]
        """
        diff = self.w.unsqueeze(1) - samples.unsqueeze(2)  # N x n x K x C
        logits = (diff.square_()).sum(dim=-1)
        return - 1 / 2 * logits  # N x n x K

    def A(self, p):
        """
        inputs:

            p : torch.tensor of shape [n_tasks, q_shot, num_class]
                where p[i,j,k] = probability of point j in task i belonging to class k
                (according to our L2 classifier)
        returns:
            v : torch.Tensor of shape [n_task, q_shot, num_class]
        """
        n_samples = p.size(1)
        v = p.sum(1) / n_samples
        return v

    def A_adj(self, v, q_shot):
        """
        inputs:
            V : torch.tensor of shape [n_tasks, num_class]
            q_shot : int
        returns:
            p : torch.Tensor of shape [n_task, q_shot, num_class]
        """
        p = v.unsqueeze(1).repeat(1, q_shot, 1) / q_shot
        return p
    
    def u_update(self, query):
        """
        inputs:
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
         
        updates:
            self.u : torch.Tensor of shape [n_task, n_query, num_class]
        """
        feature_size, n_query = query.size(-1), query.size(1)
        logits = self.get_logits(query)
        self.u = (logits + self.lambd * self.A_adj(self.v, n_query)).softmax(2)

    def v_update(self):
        """
        inputs:
        """
        p = self.u
        self.v = torch.log(self.A(p) + self.eps) + 1
                
    def init_w(self, support, y_s_one_hot):
        """
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]

        updates :
            self.w : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
        counts = (y_s_one_hot.sum(1)).unsqueeze(-1)
        weights = (y_s_one_hot.unsqueeze(-1) * (support.unsqueeze(2))).sum(1)
        self.w = weights.div_(counts)          

    def w_update(self, support, query, y_s_one_hot):
        """
        Corresponds to w_k updates
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s_one_hot : torch.Tensor of shape [n_task, s_shot, n_ways]


        updates :
            self.w : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
         
        num = (query.unsqueeze(2) * self.u.unsqueeze(3)).sum(1)
        den  = self.u.sum(1)
        if self.args.shots != 0:
            num.add_((support.unsqueeze(2) * y_s_one_hot.unsqueeze(3)).sum(1))
            den.add_(y_s_one_hot.sum(1))
        self.w = num.div_(den.unsqueeze(2)) 

    def run_method(self, support, query, y_s, y_q):
        """
        Corresponds to the PADDLE inference
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]
            y_q : torch.Tensor of shape [n_task, n_query]

        updates :from copy import deepcopy
            self.u : torch.Tensor of shape [n_task, n_query, num_class]         (labels)
            self.w : torch.Tensor of shape [n_task, num_class, feature_dim]     (centroids)
        """

        self.logger.info(" ==> Executing PADDLE with LAMBDA = {} and T = {}".format(self.lambd, self.args.T))
        
        y_s_one_hot = get_one_hot(y_s)
        n_task, n_support, n_ways = y_s_one_hot.shape
        
        self.v = torch.zeros(n_task, n_ways).to(self.device)
        self.init_w(support, y_s_one_hot)

        pbar = tqdm(range(self.iter))
        for i in pbar:
            t0 = time.time()

            # Update assignments
            self.u_update(query)

            # update on dual variable v
            self.v_update()

            # Update centroids by averaging the assigned samples
            self.w_update(support, query, y_s_one_hot)
            
            t1 = time.time()
            u_old = deepcopy(self.u)

            if i in [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                criterions = (u_old - self.u).norm(dim=(1,2)).mean(0) 
                pbar.set_description(f"Criterion: {criterions}")
                self.record_convergence(new_time=(t1-t0) / n_task, criterions=criterions)
                t1 = time.time()

        t1 = time.time()
        self.record_convergence(new_time=(t1-t0) / n_task, criterions=criterions)
        if self.args.acc_clustering == True:
            self.compute_acc_clustering(query, y_q, support, y_s_one_hot)
        else:
            self.compute_acc(y_q=y_q)


class MinMaxScaler(object):
    """MinMax Scaler

    Transforms each channel to the range [a, b].

    Parameters
    ----------
    feature_range : tuple
        Desired range of transformed data.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __call__(self, query, support):
        features = torch.cat([query, support], dim=1)
        dist = (features.max(dim=1, keepdim=True)[0] - features.min(dim=1, keepdim=True)[0])
        dist[dist==0.] = 1.
        scale = 1.0 /  dist
        ratio = features.min(dim=1, keepdim=True)[0]
        query.mul_(scale).sub_(ratio)
        support.mul_(scale).sub_(ratio)
        return query, support, scale, ratio