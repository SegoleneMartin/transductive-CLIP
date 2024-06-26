from src.utils import get_one_hot,  Logger, clip_weights, compute_graph_matching, compute_basic_matching
from tqdm import tqdm
import torch
import time
from copy import deepcopy
import numpy as np


class BASE(object):

    def __init__(self, model, device, log_file, args):
        self.device = device
        self.iter = args.iter
        self.lambd = int(args.num_classes_test / 5) * args.n_query
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
        l3 = ((self.alpha.unsqueeze(1) - 1) *
              torch.log(samples + self.eps).unsqueeze(2)).sum(-1)
        logits = l1 + l2 + l3
        return logits  # N x n x K

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

    def compute_acc_clustering(self, query, y_q):
        n_task = query.shape[0]
        preds_q = self.u.argmax(2)
        preds_q_one_hot = get_one_hot(preds_q, self.args.n_class)

        prototypes = ((preds_q_one_hot.unsqueeze(-1) * query.unsqueeze(2)).sum(1)
                      ) / (preds_q_one_hot.sum(1).clamp(min=self.eps).unsqueeze(-1))
        cluster_sizes = preds_q_one_hot.sum(1).unsqueeze(-1)  # N x K
        nonzero_clusters = cluster_sizes > self.eps
        prototypes = prototypes * nonzero_clusters

        if self.args.use_softmax_feature:
            probs = prototypes
        else:
            text_features = clip_weights(
                self.model, self.args.classnames, self.args.template, self.device).float()
            probs = torch.zeros(n_task, self.args.n_class,
                                self.args.n_class).to(self.device)
            for task in range(n_task):
                image_features = prototypes[task] / \
                    prototypes[task].norm(dim=-1, keepdim=True)
                probs[task] = (self.args.T * image_features @
                               text_features.T).softmax(dim=-1)  # K

        if self.args.graph_matching == True:
            new_preds_q = compute_graph_matching(preds_q, probs, self.args)

        else:
            new_preds_q = compute_basic_matching(preds_q, probs, self.args)

        accuracy = (new_preds_q == y_q).float().mean(1, keepdim=True)
        self.test_acc.append(accuracy)

    def get_logs(self):
        self.criterions = torch.stack(self.criterions, dim=0).cpu().numpy()
        self.test_acc = torch.cat(self.test_acc, dim=1).cpu().numpy()
        return {'timestamps': np.array(self.timestamps).mean(), 'criterions': self.criterions,
                'acc': self.test_acc}

    def run_task(self, task_dic):
        """
        inputs:
            task_dic : dictionnary with n_task few-shot tasks
            shot : scalar, number of shots
        """

        # Extract support and query
        y_q = task_dic['y_q']               # [n_task, n_query]
        query = task_dic['x_q']             # [n_task, n_query, feature_dim]

        # Transfer tensors to GPU if needed
        query = query.to(self.device).float()
        y_q = y_q.long().squeeze(2).to(self.device)
        del task_dic

        # Run adaptation
        self.run_method(query=query, y_q=y_q)

        # Extract adaptation logs
        logs = self.get_logs()
        return logs


class HARD_EM_DIRICHLET(BASE):

    def __init__(self, model, device, log_file, args):
        super().__init__(model=model, device=device, log_file=log_file, args=args)

    def __del__(self):
        self.logger.del_logger()

    def u_update(self, query):
        """
        inputs:
            query : torch.Tensor of shape [n_task, n_query, feature_dim]

        updates:
            self.u : torch.Tensor of shape [n_task, n_query, num_class]
        """
        __, n_query = query.size(-1), query.size(1)
        logits = self.get_logits(query)
        self.u = (logits + self.lambd *
                  self.v.unsqueeze(1) / n_query).softmax(2)

    def v_update(self):
        """
        updates:
            self.v : torch.Tensor of shape [n_task, num_class]
            --> corresponds to the log of the class proportions
        """
        self.v = torch.log(self.u.sum(1) / self.u.size(1) + self.eps) + 1

    def curvature(self, alpha):
        digam = torch.polygamma(0, alpha + 1)
        return torch.where(alpha > 1e-11, abs(2 * (self.log_gamma_1 - torch.lgamma(alpha + 1) + digam * alpha) / alpha**2), self.zero_value), digam

    def update_alpha(self, alpha_0, y_cst):
        alpha = deepcopy(alpha_0)

        for l in range(self.iter_mm):
            curv, digam = self.curvature(alpha)
            b = digam - \
                torch.polygamma(0, alpha.sum(-1)).unsqueeze(-1) - curv * alpha
            b = b - y_cst
            a = curv
            delta = b**2 + 4 * a
            alpha_new = (- b + torch.sqrt(delta)) / (2 * a)

            if l > 0 and l % 50 == 0:
                criterion = torch.norm(
                    alpha_new - alpha)**2 / torch.norm(alpha)**2
                if l % 1000 == 0:
                    print('iter', l, 'criterion', criterion)
                if criterion < 1e-11:
                    break
            alpha = deepcopy(alpha_new)
        self.alpha = deepcopy(alpha_new)

    def objective_function(self, support, query, y_s_one_hot):
        l1 = torch.lgamma(self.alpha.sum(-1)).unsqueeze(1)
        l2 = - torch.lgamma(self.alpha).sum(-1).unsqueeze(1)
        l3 = ((self.alpha.unsqueeze(1) - 1) *
              torch.log(query + self.eps).unsqueeze(2)).sum(-1)
        datafit_query = -(self.u * (l1 + l2 + l3)).sum(-1).sum(1)
        l1 = torch.lgamma(self.alpha.sum(-1)).unsqueeze(1)
        l2 = - torch.lgamma(self.alpha).sum(-1).unsqueeze(1)
        l3 = ((self.alpha.unsqueeze(1) - 1) *
              torch.log(support + self.eps).unsqueeze(2)).sum(-1)
        datafit_support = -(y_s_one_hot * (l1 + l2 + l3)).sum(-1).sum(1)
        datafit = 1 / 2 * (datafit_query + datafit_support)

        reg_ent = (self.u * torch.log(self.u + self.eps)).sum(-1).sum(1)

        props = self.u.mean(1)
        part_complexity = - self.lambd * \
            (props * torch.log(props + self.eps)).sum(-1)

        return datafit + reg_ent + part_complexity

    def run_method(self, query, y_q):
        """
        Corresponds to the Hard EM DIRICHLET inference
        inputs:
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_q : torch.Tensor of shape [n_task, n_query]

        updates :from copy import deepcopy
            self.u : torch.Tensor of shape [n_task, n_query, num_class]         (soft labels)
            self.v : torch.Tensor of shape [n_task, num_class]                  (dual variable)
            self.w : torch.Tensor of shape [n_task, num_class, feature_dim]     (centroids)
        """
        self.logger.info(
            " ==> Executing HARD EM-DIRICHLET with LAMBDA = {} and T = {}".format(self.lambd, self.args.T))

        self.zero_value = torch.polygamma(1, torch.Tensor(
            [1]).to(self.device)).float()  # .double()
        self.log_gamma_1 = torch.lgamma(
            torch.Tensor([1]).to(self.device)).float()

        n_task, n_class = query.shape[0], self.args.num_classes_test

        # Initialization
        self.v = torch.zeros(n_task, n_class).to(
            self.device)        # dual variable set to zero
        if self.args.use_softmax_feature:
            self.u = deepcopy(query)
        else:
            raise ValueError(
                "The selected method is unable to handle query features that are not in the unit simplex")
        self.alpha = torch.ones((n_task, n_class, n_class)).to(self.device)
        alpha_old = deepcopy(self.alpha)
        t0 = time.time()

        pbar = tqdm(range(self.iter))
        for i in pbar:

            # update of dirichlet parameter alpha
            cluster_sizes = self.u.sum(
                dim=1).unsqueeze(-1).float()  # .double() # N x K
            nonzero_clusters = cluster_sizes > self.eps
            y_cst = ((self.u.unsqueeze(-1) * torch.log(query + self.eps).unsqueeze(2)
                      ).sum(1)) / (self.u.sum(1).clamp(min=self.eps).unsqueeze(-1))
            y_cst = y_cst * nonzero_clusters + \
                (1 - 1 * nonzero_clusters) * torch.ones_like(y_cst) * (-10)
            self.update_alpha(self.alpha, y_cst)
            alpha_new = self.alpha * nonzero_clusters + \
                alpha_old * (1 - 1 * nonzero_clusters)
            self.alpha = alpha_new
            del alpha_new

            # update on dual variable v (= log of class proportions)
            self.v_update()

            # update hard assignment variable u
            self.u_update(query)
            labels = torch.argmax(self.u, dim=-1)
            self.u.zero_()
            self.u.scatter_(2, labels.unsqueeze(-1), 1.0)

            alpha_diff = ((alpha_old - self.alpha).norm(dim=(1, 2)
                                                        ) / alpha_old.norm(dim=(1, 2))).mean(0)
            criterions = alpha_diff
            alpha_old = deepcopy(self.alpha)
            t1 = time.time()

            pbar.set_description(f"Criterion: {criterions}")
            t1 = time.time()
            self.record_convergence(
                new_time=(t1-t0) / n_task, criterions=criterions)

        self.compute_acc_clustering(query, y_q)
