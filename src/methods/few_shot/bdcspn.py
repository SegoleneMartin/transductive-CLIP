#######################################################################
# This file contains the implementation of the BD-CSPN method,
# adapted from the ECCV 2020 paper entitled "Prototype rectification for few-shot learning."
#######################################################################

import torch.nn.functional as F
from tqdm import tqdm
import torch
from src.utils import Logger, get_one_hot
import time
import numpy as np


class BDCSPN(object):
    def __init__(self, model, device, log_file, args):
        self.device = device
        self.norm_type = args.norm_type
        self.temp = args.temp
        self.model = model
        self.log_file = log_file
        self.n_class = args.n_class
        self.logger = Logger(__name__, self.log_file)
        self.init_info_lists()

    def __del__(self):
        self.logger.del_logger()

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

    def get_logits(self, w, samples):
        """
        inputs:
            samples : torch.Tensor of shape [n_task, shot, feature_dim]
            w : torch.Tensor of shape [n_task, num_class, feature_dim]
        returns :
            logits : torch.Tensor of shape [n_task, shot, num_class]
        """
        w = w / w.norm(p=2, dim=-1, keepdim=True)
        samples = samples / samples.norm(p=2, dim=-1, keepdim=True)
        if len(w.shape) == 3:
            diff = w.unsqueeze(1) - samples.unsqueeze(2)  # N x n x K x C
        else:
            diff = w.unsqueeze(0) - samples.unsqueeze(1)  # n x K x C
        logits = (diff.square_()).sum(dim=-1)
        return - 1 / 2 * logits  # N x n x K

    def compute_acc(self, y_q, preds_q):
        """
        inputs:
            y_q : torch.Tensor of shape [n_task, n_query] :
        """
        accuracy = (preds_q == y_q).float().mean(1, keepdim=True)
        self.test_acc.append(accuracy)

    def get_logs(self):
        self.criterions = torch.stack(
            self.criterions, dim=0).detach().cpu().numpy()
        self.test_acc = torch.cat(self.test_acc, dim=1).detach().cpu().numpy()
        return {'timestamps': np.array(self.timestamps).mean(), 'criterions': self.criterions,
                'acc': self.test_acc}

    def normalization(self, z_s, z_q, train_mean):
        """
            inputs:
                z_s : torch.Tensor of shape [n_task, shot, feature_dim]
                z_q : torch.Tensor of shape [n_task, n_query, feature_dim]
                train_mean: torch.Tensor of shape [n_task, 1, feature_dim]
        """
        # Ensure train_mean is unsqueezed to match the dimensions of z_s and z_q
        train_mean = train_mean.unsqueeze(1)

        # CL2N Normalization
        if self.norm_type == 'CL2N':
            z_s = z_s - train_mean
            z_s = z_s / z_s.norm(p=2, dim=2, keepdim=True)
            z_q = z_q - train_mean
            z_q = z_q / z_q.norm(p=2, dim=2, keepdim=True)

        # L2 Normalization
        elif self.norm_type == 'L2N':
            z_s = z_s / z_s.norm(p=2, dim=2, keepdim=True)
            z_q = z_q / z_q.norm(p=2, dim=2, keepdim=True)

        return z_s, z_q

    def proto_rectification(self, support, query, y_s, shot):
        """
        Perform prototype rectification for each task.
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]
            y_q : torch.Tensor of shape [n_task, n_query]
            shot : scalar indicating the number of shots
        """
        self.logger.info(
            f" ==> Executing proto_rectification on {shot} shot tasks ...")

        n_task, n_query, feature_dim = query.shape
        prototypes = torch.zeros(n_task, self.n_class,
                                 feature_dim, device=query.device)

        # Compute initial prototypes by averaging support features per class
        y_s_one_hot = get_one_hot(y_s, self.n_class)
        counts = (y_s_one_hot.sum(1)).unsqueeze(-1)
        weights = (y_s_one_hot.unsqueeze(-1) * (support.unsqueeze(2))).sum(1)
        init_prototypes = weights.div_(counts)

        # Iterate over each task
        for j in tqdm(range(n_task)):
            # Compute mean of support set to shift the query set for rectification
            eta = support[j].mean(0) - query[j].mean(0)
            query_shifted = query[j] + eta

            # Combine the support and shifted query sets
            query_aug = torch.cat(
                (support[j], query_shifted), dim=0)  # n x dim_features

            # Compute weights of query_aug using the temperature-scaled cosine similarity
            cos_sim = self.get_logits(init_prototypes[j], query_aug)
            u = (self.temp * cos_sim).softmax(-1)  # n x num_class

            # Rectify prototypes based on the nearest neighbors (=1) and their weights
            query_aug = query_aug / query_aug.norm(p=2, dim=-1, keepdim=True)
            counts = (u.sum(0)).unsqueeze(-1)
            weights = (u.unsqueeze(-1) * (query_aug.unsqueeze(1))).sum(0)
            prototypes[j] = weights.div_(counts)

        return prototypes

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
        train_mean = support.mean(1)

        support, query = self.normalization(support, query, train_mean)

        # Run method
        self.run_method(support=support, query=query,
                        y_s=y_s, y_q=y_q, shot=shot)

        # Extract adaptation logs
        logs = self.get_logs()

        return logs

    def run_method(self, support, query, y_s, y_q, shot):
        """
        Corresponds to the BD-CSPN inference
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]
            y_q : torch.Tensor of shape [n_task, n_query]
        """
        self.logger.info(f" ==> Executing BD-CSPN")

        t0 = time.time()
        # Prototype rectification
        prototypes = self.proto_rectification(
            support=support, query=query, y_s=y_s, shot=shot)

        # Calculate distances between all support and query features
        cos_sim = self.get_logits(prototypes, query)
        u = (self.temp * cos_sim).softmax(-1)  # n x num_class
        preds_q = u.argmax(2)

        t1 = time.time()
        # Record the information
        self.record_convergence(
            new_time=t1-t0, criterions=torch.zeros(1).to(self.device))

        self.compute_acc(y_q, preds_q)
