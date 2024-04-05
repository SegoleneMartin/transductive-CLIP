#######################################################################
# This file contains the implementation of the PADDLE method,
# adapted from the NeurIPS 2022 paper entitled "Towards practical few-shot query sets: Transductive minimum description length inference":
# https://github.com/SegoleneMartin/PADDLE
#######################################################################

from src.utils import get_one_hot,  Logger, clip_weights
from tqdm import tqdm
import torch
import time
from copy import deepcopy
import numpy as np


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

    def get_logs(self):
        self.criterions = torch.stack(self.criterions, dim=0).cpu().numpy()
        self.test_acc = torch.cat(self.test_acc, dim=1).cpu().numpy()
        return {'timestamps': np.array(self.timestamps).mean(), 'criterions': self.criterions,
                'acc': self.test_acc}

    def run_task(self, task_dic, shot):
        """
        inputs:
            task_dic : dictionnary with n_task few-shot tasks
            shot : scalar, number of shots
        """

        # Extract support and query
        y_s = task_dic['y_s']               # [n_task, shot]
        y_q = task_dic['y_q']               # [n_task, n_query]
        support = task_dic['x_s']           # [n_task, shot, feature_dim]
        query = task_dic['x_q']             # [n_task, n_query, feature_dim]

        # Transfer tensors to GPU if needed
        support = support.to(self.device)
        query = query.to(self.device)
        y_s = y_s.long().squeeze(2).to(self.device)
        y_q = y_q.long().squeeze(2).to(self.device)

        # Run method
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

    def init_w(self, support, query, y_s):
        """
        Init prototypes
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]
        """
        # Compute initial prototypes by averaging support features per class
        y_s_one_hot = get_one_hot(y_s, self.n_class)
        counts = (y_s_one_hot.sum(1)).unsqueeze(-1)
        weights = (y_s_one_hot.unsqueeze(-1) * (support.unsqueeze(2))).sum(1)
        init_prototypes = weights.div_(counts)

        return init_prototypes

    def w_update(self, support, query, y_s_one_hot):
        """
        Corresponds to w_k updates
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s_one_hot : torch.Tensor of shape [n_task, s_shot, n_class]


        updates :
            self.w : torch.Tensor of shape [n_task, num_class, feature_dim]
        """

        num = (query.unsqueeze(2) * self.u.unsqueeze(3)).sum(1)
        den = self.u.sum(1)
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

        self.logger.info(" ==> Executing PADDLE with LAMBDA = {} and T = {}".format(
            self.lambd, self.args.T))

        n_task, self.n_class = query.shape[0], self.args.num_classes_test

        # Initialization
        self.v = torch.zeros(n_task, self.n_class).to(
            self.device)        # dual variable set to zero
        if self.args.use_softmax_feature:
            self.u = deepcopy(query)
        else:
            self.u = torch.zeros(
                (n_task, query.shape[1], self.n_class)).to(self.device)
            text_features = clip_weights(
                self.model, self.args.classnames, self.args.template, self.device)
            for task in range(n_task):
                image_features = query[task] / \
                    query[task].norm(dim=-1, keepdim=True)
                sim = (self.args.T * (image_features @ text_features.T)
                       ).softmax(dim=-1)  # N* K
                self.u[task] = sim
        self.w = self.init_w(support=support, query=query, y_s=y_s)
        y_s_one_hot = get_one_hot(y_s, self.n_class)
        t0 = time.time()

        pbar = tqdm(range(self.iter))
        for i in pbar:

            # Update assignments
            self.u_update(query)

            # update on dual variable v
            self.v_update()

            # Update centroids by averaging the assigned samples
            self.w_update(support, query, y_s_one_hot)

            u_old = deepcopy(self.u)
            criterions = (u_old - self.u).norm(dim=(1, 2)).mean(0)
            pbar.set_description(f"Criterion: {criterions}")
            t1 = time.time()
            self.record_convergence(
                new_time=(t1-t0) / n_task, criterions=criterions)

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
        dist = (features.max(dim=1, keepdim=True)[
                0] - features.min(dim=1, keepdim=True)[0])
        dist[dist == 0.] = 1.
        scale = 1.0 / dist
        ratio = features.min(dim=1, keepdim=True)[0]
        query.mul_(scale).sub_(ratio)
        support.mul_(scale).sub_(ratio)
        return query, support, scale, ratio
