#######################################################################
# This file contains the implementation of the TIM-GD method,
# adapted from the NeurIPS 2020 paper entitled "TIM: Transductive Information Maximization",
# and of the ALPHA_TIM method, adapted from the NeurIPS 2021 paper entitled
# "Realistic evaluation of transductive few-shot learning".
# https://github.com/mboudiaf/TIM and
# https://github.com/oveilleux/Realistic_Transductive_Few_Shot
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
        self.loss_weights = args.loss_weights.copy()
        self.temp = args.temp

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

    def compute_acc(self, y_q, logits_q):
        """
        inputs:
            y_q : torch.Tensor of shape [n_task, n_query] :
        """
        preds_q = logits_q.argmax(2)
        accuracy = (preds_q == y_q).float().mean(1, keepdim=True)
        self.test_acc.append(accuracy)

    def get_logs(self):
        self.criterions = torch.stack(
            self.criterions, dim=0).detach().cpu().numpy()
        self.test_acc = torch.cat(self.test_acc, dim=1).detach().cpu().numpy()
        return {'timestamps': np.array(self.timestamps).mean(), 'criterions': self.criterions,
                'acc': self.test_acc}

    def run_task(self, task_dic, shot=10):
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
        support = support.to(self.device).float()  # .double()
        query = query.to(self.device).float()  # .double()
        y_s = y_s.long().squeeze(2).to(self.device)
        y_q = y_q.long().squeeze(2).to(self.device)
        del task_dic

        # Run adaptation
        self.run_method(support=support, query=query, y_s=y_s, y_q=y_q)

        # Extract adaptation logs
        logs = self.get_logs()
        return logs


class TIM_GD(BASE):

    def __init__(self, model, device, log_file, args):
        super().__init__(model=model, device=device, log_file=log_file, args=args)
        self.lr = float(args.lr_tim)

    def __del__(self):
        self.logger.del_logger()

    def get_logits(self, samples):
        """
        inputs:
            samples : torch.Tensor of shape [n_task, shot, feature_dim]

        returns :
            logits : torch.Tensor of shape [n_task, shot, num_class]
        """

        n_task = samples.size(0)
        logits = self.temp * (samples.matmul(self.weights.transpose(1, 2))
                              - 1 / 2 * (self.weights **
                                         2).sum(2).view(n_task, 1, -1)
                              - 1 / 2 * (samples**2).sum(2).view(n_task, -1, 1))  #
        return logits

    def init_weights(self, support, query, y_s):
        """
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]

        updates :
            self.weights : torch.Tensor of shape [n_task, num_class, feature_dim]
        """

        self.model.eval()
        n_task, n_class = query.shape[0], self.args.num_classes_test
        one_hot = get_one_hot(y_s, n_class).to(self.device)
        counts = one_hot.sum(1).view(n_task, -1, 1)
        weights = one_hot.transpose(1, 2).matmul(support)
        self.weights = weights / counts
        logits_q = self.get_logits(query).detach()
        q_probs = logits_q.softmax(2)
        return q_probs

    def run_method(self, support, query, y_s, y_q):
        """
        Corresponds to the TIM-GD inference
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]
            y_q : torch.Tensor of shape [n_task, n_query]

        updates :
            self.weights : torch.Tensor of shape [n_task, num_class, feature_dim]
        """

        self.logger.info(" ==> Executing TIM with T = {}".format(self.args.T))
        self.init_weights(support=support, y_s=y_s, query=query)

        n_task = support.shape[0]
        self.weights.requires_grad_()
        optimizer = torch.optim.Adam([self.weights], lr=self.lr)
        n_task, n_class = query.shape[0], self.args.num_classes_test
        y_s_one_hot = get_one_hot(y_s, n_class)
        self.model.train()

        t0 = time.time()
        pbar = tqdm(range(self.iter))
        for i in pbar:
            weights_old = deepcopy(self.weights.detach())
            logits_s = self.get_logits(support)
            logits_q = self.get_logits(query)

            ce = - (y_s_one_hot * torch.log(logits_s.softmax(2) + 1e-12)
                    ).sum(2).mean(1).sum(0)
            q_probs = logits_q.softmax(2)
            q_cond_ent = - (q_probs * torch.log(q_probs + 1e-12)
                            ).sum(2).mean(1).sum(0)
            q_ent = - (q_probs.mean(1) *
                       torch.log(q_probs.mean(1) + 1e-12)).sum(1).sum(0)
            loss = self.loss_weights[0] * ce - \
                (self.loss_weights[1] * q_ent -
                 self.loss_weights[2] * q_cond_ent)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            weight_diff = (weights_old - self.weights).norm(dim=-1).mean(-1)
            criterions = weight_diff
            pbar.set_description(f"Criterion: {criterions}")
            t1 = time.time()
            self.record_convergence(
                new_time=(t1-t0) / n_task, criterions=criterions)

        self.model.eval()
        self.compute_acc(y_q=y_q, logits_q=logits_q)


class ALPHA_TIM(BASE):

    def __init__(self, model, device, log_file, args):
        super().__init__(model=model, device=device, log_file=log_file, args=args)
        self.lr = float(args.lr_alpha_tim)
        self.entropies = args.entropies.copy()
        self.alpha_value = args.alpha_value

    def __del__(self):
        self.logger.del_logger()

    def get_logits(self, samples):
        """
        inputs:
            samples : torch.Tensor of shape [n_task, shot, feature_dim]

        returns :
            logits : torch.Tensor of shape [n_task, shot, num_class]
        """

        n_task = samples.size(0)
        logits = self.temp * (samples.matmul(self.weights.transpose(1, 2))
                              - 1 / 2 * (self.weights **
                                         2).sum(2).view(n_task, 1, -1)
                              - 1 / 2 * (samples**2).sum(2).view(n_task, -1, 1))  #
        return logits

    def init_weights(self, support, query, y_s):
        """
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]

        updates :
            self.weights : torch.Tensor of shape [n_task, num_class, feature_dim]
        """

        self.model.eval()
        n_task, n_class = query.shape[0], self.args.num_classes_test
        one_hot = get_one_hot(y_s, n_class).to(self.device)
        counts = one_hot.sum(1).view(n_task, -1, 1)
        weights = one_hot.transpose(1, 2).matmul(support)
        self.weights = weights / counts
        logits_q = self.get_logits(query).detach()
        q_probs = logits_q.softmax(2)
        return q_probs

    def run_method(self, support, query, y_s, y_q):
        """
        Corresponds to the TIM-GD inference
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]
            y_q : torch.Tensor of shape [n_task, n_query]

        updates :
            self.weights : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
        self.logger.info(" ==> Executing ALPHA_TIM with ALPHA = {} and T = {}".format(
            self.alpha_value, self.args.T))
        self.init_weights(support=support, y_s=y_s, query=query)

        self.weights.requires_grad_()
        n_task, n_class = query.shape[0], self.args.num_classes_test
        optimizer = torch.optim.Adam([self.weights], lr=self.lr)
        y_s_one_hot = get_one_hot(y_s, n_class)
        self.model.train()
        t0 = time.time()

        pbar = tqdm(range(self.iter))
        for i in pbar:
            weights_old = deepcopy(self.weights.detach())
            logits_s = self.get_logits(support)
            logits_q = self.get_logits(query)
            q_probs = logits_q.softmax(2)

            # Cross entropy type
            if self.entropies[0] == 'Shannon':
                ce = - (y_s_one_hot * torch.log(logits_s.softmax(2) +
                        1e-12)).sum(2).mean(1).sum(0)
            elif self.entropies[0] == 'Alpha':
                ce = torch.pow(y_s_one_hot, self.alpha_value) * \
                    torch.pow(logits_s.softmax(2) +
                              1e-12, 1 - self.alpha_value)
                ce = ((1 - ce.sum(2))/(self.alpha_value - 1)).mean(1).sum(0)
            else:
                raise ValueError("Entropies must be in ['Shannon', 'Alpha']")

            # Marginal entropy type
            if self.entropies[1] == 'Shannon':
                q_ent = - (q_probs.mean(1) *
                           torch.log(q_probs.mean(1))).sum(1).sum(0)
            elif self.entropies[1] == 'Alpha':
                q_ent = ((1 - (torch.pow(q_probs.mean(1), self.alpha_value)
                               ).sum(1)) / (self.alpha_value - 1)).sum(0)
            else:
                raise ValueError("Entropies must be in ['Shannon', 'Alpha']")

            # Conditional entropy type
            if self.entropies[2] == 'Shannon':
                q_cond_ent = - (q_probs * torch.log(q_probs +
                                1e-12)).sum(2).mean(1).sum(0)
            elif self.entropies[2] == 'Alpha':
                q_cond_ent = ((1 - (torch.pow(q_probs + 1e-12, self.alpha_value)
                                    ).sum(2)) / (self.alpha_value - 1)).mean(1).sum(0)
            else:
                raise ValueError("Entropies must be in ['Shannon', 'Alpha']")

            # Loss
            loss = self.loss_weights[0] * ce - \
                (self.loss_weights[1] * q_ent -
                 self.loss_weights[2] * q_cond_ent)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # t1 = time.time()

            weight_diff = (weights_old - self.weights.detach()
                           ).norm(dim=-1).mean()
            criterions = weight_diff
            pbar.set_description(f"Criterion: {criterions}")
            t1 = time.time()
            self.record_convergence(
                new_time=(t1-t0) / n_task, criterions=criterions)

        self.model.eval()
        self.compute_acc(y_q=y_q, logits_q=logits_q)
