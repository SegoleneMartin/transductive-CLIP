from src.utils import get_one_hot,  Logger, clip_weights, compute_graph_matching, compute_basic_matching
from tqdm import tqdm
import torch
import time
from copy import deepcopy
import numpy as np


class BASE(object):

    def __init__(self, model, device, log_file, args):
        self.device = device
        self.model = model
        self.log_file = log_file
        self.logger = Logger(__name__, self.log_file)
        self.init_info_lists()
        self.args = args

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
        diff = self.w.unsqueeze(1) - samples.unsqueeze(2)  # N x n x K x C
        logits = (diff.square_()).sum(dim=-1)
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

       # Extract query
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


class CLIP(BASE):

    def __init__(self, model, device, log_file, args):
        super().__init__(model=model, device=device, log_file=log_file, args=args)

    def __del__(self):
        self.logger.del_logger()

    def run_method(self, query, y_q):
        """
        Corresponds to the ZERO SHOT CLIP inference
        inputs:
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_q : torch.Tensor of shape [n_task, n_query]

        updates :from copy import deepcopy
            self.u : torch.Tensor of shape [n_task, n_query, num_class]         (soft labels)
            self.v : torch.Tensor of shape [n_task, num_class]                  (dual variable)
            self.w : torch.Tensor of shape [n_task, num_class, feature_dim]     (centroids)
        """

        self.logger.info(" ==> Executing CLIP")
        n_task = query.shape[0]

        n_task, n_class = query.shape[0], self.args.num_classes_test

        # Initialization
        if self.args.use_softmax_feature:
            self.u = deepcopy(query)
        else:
            self.u = torch.zeros(
                (n_task, query.shape[1], n_class)).to(self.device)
            text_features = clip_weights(
                self.model, self.args.classnames, self.args.template, self.device)
            for task in range(n_task):
                image_features = query[task] / \
                    query[task].norm(dim=-1, keepdim=True)
                sim = (self.args.T * (image_features @ text_features.T)
                       ).softmax(dim=-1)  # N* K
                self.u[task] = sim

        u_old = deepcopy(self.u)
        self.record_convergence(new_time=0, criterions=(
            u_old - self.u).norm(dim=(1, 2)).mean(0))

        self.compute_acc(y_q)
