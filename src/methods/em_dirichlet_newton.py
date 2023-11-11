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
        self.lambd = int(args.num_classes_test / 5) * args.n_query  #int(args.num_classes_test / args.k_eff) * args.n_query * args.fact
        self.model = model
        self.log_file = log_file
        self.logger = Logger(__name__, self.log_file)
        self.init_info_lists()
        self.args = args
        self.eps = 1e-15
        self.iter_mm = args.iter_mm


    def init_info_lists(self):
        self.timestamps = []
        self.criterions = []
        self.test_acc = []


    def get_logits(self, samples):
        """
        inputs:
            samples : torch.Tensor of shape [n_task, shot, feature_dim]
        returns :
            logits : torch.Tensor of shape [n_task, shot, num_class]
        """
        l1 = torch.lgamma(self.alpha.sum(-1)).unsqueeze(1)
        l2 = - torch.lgamma(self.alpha).sum(-1).unsqueeze(1)
        if self.args.shots == 0:
            l3 = ((self.alpha.unsqueeze(1) - 1) * torch.log(samples + self.eps).unsqueeze(2)).sum(-1)
        else:
            l3 = ((self.alpha.unsqueeze(1) - 1) * samples.unsqueeze(2)).sum(-1)
        logits = l1 + l2 + l3
        return - logits  # N x n x K
    

    def record_convergence(self, new_time, criterions):
        """
        inputs:
            new_time : scalar
            criterions : torch.Tensor of shape [n_task]
        """
        self.criterions.append(criterions)
        self.timestamps.append(new_time)

    def newton(self, f, df, alpha_0):
        """
        inputs:
            f : function operating on vectors of shape [n_task, num_class, feature_size], returns [n_task, num_class, feature_size]
            df : function operating on vectors of shape [n_task, num_class, feature_size], returns [n_task, num_class, feature_size]
            alpha_0 : init, [n_task, num_class, feature_size]
        returns :
            alpha : zero of f, torch.Tensor of shape [n_task, shot, num_class]
        """
        alpha = deepcopy(alpha_0)

        for l in range(1000):
            alpha_new = alpha - f(alpha) / df(alpha)
            #criterion = torch.norm(alpha_new - alpha)**2 / (alpha.shape[1] * alpha.shape[2]) 
            criterion = torch.norm(alpha_new - alpha)**2 / torch.norm(alpha)**2
            alpha = deepcopy(alpha_new)
            #print("crierion", criterion)
            if criterion < 1e-13:
                return alpha
        #print('DID NOT CONVERGE')
        return alpha
    
    def init_newton(self, y):
        gamma = - torch.polygamma(0, torch.Tensor([1.0]).to('cuda'))
        init = torch.where(y > -2.22, torch.exp(y) + 1/2, -1 / (y + gamma)) 
        return init

    def define_f(self, y):
        """
        inputs:
            support : [n_task, n_supp, feature_size]
            query : [n_task, n_query, feature_size]
            y_s_one_hot : [n_task, n_supp, num_class]
        returns :
            f : function that maps [n_task, num_class, feature_size] to [n_task, num_class, feature_size]
        """
        def f(alpha):
            return torch.polygamma(0, alpha) - y
        return f

    def define_df(self):
        """
        inputs:
            y_s_one_hot : [n_task, n_supp, num_class]
        returns :
            df_inv : linear function that maps [n_task, num_class, feature_size] to [n_task, num_class, feature_size]
        """
        def df(alpha):
            return torch.polygamma(1, alpha)
        return df
        
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
        support = support.to(self.device).float()
        query = query.to(self.device).float()
        y_s = y_s.long().squeeze(2).to(self.device)
        y_q = y_q.long().squeeze(2).to(self.device)
        del task_dic
                   
        # Run adaptation
        self.run_method(support=support, query=query, y_s=y_s, y_q=y_q)

        # Extract adaptation logs
        logs = self.get_logs()
        return logs


class EM_DIRICHLET(BASE):

    def __init__(self, model, device, log_file, args):
        super().__init__(model=model, device=device, log_file=log_file, args=args)


    def __del__(self):
        self.logger.del_logger() 
        

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
    
    
    def u_update(self, query, n_support):
        """
        inputs:
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
         
        updates:
            self.u : torch.Tensor of shape [n_task, n_query, num_class]
        """
        __, n_query = query.size(-1), query.size(1)
        logits = self.get_logits(query)
        self.u = (- logits + self.lambd * self.A_adj(self.v, n_query)).softmax(2)
        

    def v_update(self):
        """
        inputs:
        """
        p = self.u
        self.v = torch.log(self.A(p) + self.eps) + 1

        
    def update_alpha(self, sum_alpha_0, y_cst):
        n_task = y_cst.shape[0]
        sum_alpha = deepcopy(sum_alpha_0)
        for l in range(self.iter_mm):
            
            y = y_cst + torch.polygamma(0, sum_alpha).unsqueeze(-1)
            alpha_init = self.init_newton(y)
            f = self.define_f(y)
            df = self.define_df()
            alpha_new = self.newton(f, df, alpha_init)

            if l > 0:
                criterion = torch.norm(alpha_new - alpha)**2 / torch.norm(alpha)**2
                if l % 200 == 0 or (l <500 and l%1==0):
                    alpha_diff = ((alpha - alpha_new).norm(dim=(1,2)) / alpha.norm(dim=(1,2))).mean(0) 
                    print('iter', l, 'criterion', criterion)
                    t1 = time.time()
                    self.record_convergence(new_time=(t1-self.t0) / n_task, criterions=alpha_diff)
                if criterion < 1e-9:
                    break
            alpha = deepcopy(alpha_new)
            sum_alpha = alpha.sum(-1)
            
        self.alpha = deepcopy(alpha_new)
    

    def run_method(self, support, query, y_s, y_q):
        """
        Corresponds to the EM DIRICHLET inference
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]
            y_q : torch.Tensor of shape [n_task, n_query]

        updates :from copy import deepcopy
            self.u : torch.Tensor of shape [n_task, n_query, num_class]         (soft labels)
            self.v : torch.Tensor of shape [n_task, num_class]                  (dual variable)
            self.alpha : torch.Tensor of shape [n_task, num_class, feature_dim]     (dirichlet parameters)
        """

        self.logger.info(" ==> Executing EM-DIRICHLET with LAMBDA = {} and T = {}".format(self.lambd, self.args.T))
        
        y_s_one_hot = get_one_hot(y_s)
        n_task, n_support, n_ways = y_s_one_hot.shape
        self.zero_value = torch.polygamma(1, torch.Tensor([1]).to(self.device)).float() #.double()
        self.log_gamma_1 = torch.lgamma(torch.Tensor([1]).to(self.device)).float() 

        # Initialization
        self.u = deepcopy(query) # initialize u to the probabilities given by CLIP
        u_old = deepcopy(self.u)
        self.v = torch.zeros(n_task, n_ways).to(self.device)
        self.alpha = torch.ones((n_task, n_ways, n_ways)).to(self.device)
        alpha_old = deepcopy(self.alpha)
        self.t0 = time.time()
        
        if self.args.shots != 0:  # inplace operations to save memory
            support.add_(self.eps)
            query.add_(self.eps)
            support.log_()
            query.log_()
        
        #pbar = tqdm(range(self.iter))
        pbar = tqdm(range(2))
        for i in pbar:

            # update of dirichlet parameter alpha
            if self.args.shots == 0: # in this case, avoid errors du to possibly empty clusters by estimating alpha only for non empty clusters
                cluster_sizes = self.u.sum(dim=1).unsqueeze(-1).float()#.double() # N x K  
                nonzero_clusters = cluster_sizes > self.eps
                y_cst = ((self.u.unsqueeze(-1) * torch.log(query + self.eps).unsqueeze(2)).sum(1)) / (self.u.sum(1).clamp(min=self.eps).unsqueeze(-1))
                y_cst = y_cst * nonzero_clusters + (1 - 1*nonzero_clusters) * torch.ones_like(y_cst) * (-10)
                #self.update_alpha(self.alpha.sum(-1), y_cst)
                self.update_alpha(torch.ones((n_task, n_ways, n_ways)).to(self.device).sum(-1), y_cst)
                alpha_new = self.alpha * nonzero_clusters + alpha_old * (1 - 1*nonzero_clusters)
                self.alpha = alpha_new
                del alpha_new
                
            else:
                y_s_sum = y_s_one_hot.sum(dim=1)  # Shape [n_task, num_class]
                u_sum = self.u.sum(dim=1) 
                y_cst = (1 / (y_s_sum + u_sum)).unsqueeze(-1)
                y_cst = y_cst * ((y_s_one_hot.unsqueeze(-1) * support.unsqueeze(2)).sum(dim=1) + (self.u.unsqueeze(-1) * query.unsqueeze(2)).sum(dim=1))
                #y_cst = 1 / (y_s_one_hot.sum(1) + self.u.sum(1)).unsqueeze(-1) * ((y_s_one_hot.unsqueeze(-1) * torch.log(support +self.eps).unsqueeze(2)).sum(1) + (self.u.unsqueeze(-1) * torch.log(query + self.eps).unsqueeze(2)).sum(1))
                self.update_alpha(self.alpha.sum(-1), y_cst)
            
            # update on dual variable v
            self.v_update()

            # update on assignment variable u
            self.u_update(query, n_support)

            u_old = deepcopy(self.u)
            alpha_diff = ((alpha_old - self.alpha).norm(dim=(1,2)) / alpha_old.norm(dim=(1,2))).mean(0) 
            criterions = alpha_diff
            alpha_old = deepcopy(self.alpha)
            t1 = time.time()
            
            if i in [0, 1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                pbar.set_description(f"Criterion: {criterions}")
                #self.record_convergence(new_time=(t1-t0) / n_task, criterions=criterions)
                t1 = time.time()
            
            #print('u', self.u)
        t1 = time.time()
        self.record_convergence(new_time=(t1-self.t0) / n_task, criterions=criterions)
        if self.args.acc_clustering == True:
            self.compute_acc_clustering(query, y_q, support, y_s_one_hot)
        else:
            self.compute_acc(y_q=y_q)


