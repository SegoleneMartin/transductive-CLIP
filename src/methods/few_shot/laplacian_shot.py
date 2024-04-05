#######################################################################
# This file contains the implementation of the LaplacianShot method,
# adapted from the ICML 2020 paper entitled "LaplacianShot: Laplacian Regularized Few Shot Learning":
# https://github.com/imtiazziko/LaplacianShot
#######################################################################

from src.utils import Logger, get_one_hot
from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F
from tqdm import tqdm
import torch
import time
from numpy import linalg as LA
import numpy as np
import math
from scipy import sparse
import matplotlib
matplotlib.use('Agg')


class LAPLACIAN_SHOT(object):
    def __init__(self, model, device, log_file, args):
        self.device = device
        self.knn = args.knn
        self.norm_type = args.norm_type
        self.iter = args.iter
        self.number_tasks = args.batch_size
        self.model = model
        self.log_file = log_file
        self.logger = Logger(__name__, self.log_file)
        self.shots = args.shots
        self.lmd = args.lmd
        self.temp = args.temp
        self.timestamps = []
        self.criterions = []
        self.ent_energy = []
        self.test_acc = []
        self.args = args

    def __del__(self):
        self.logger.del_logger()

    def record_info(self, acc, ent_energy, new_time, criterions):
        """
        inputs:
            acc_list : torch.Tensor of shape [iter]
            ent_energy : torch.Tensor of shape [iter]
            new_time: torch.Tensor of shape [iter]
        """
        self.test_acc.append(acc)
        self.criterions.append(criterions)
        self.ent_energy.append(ent_energy)
        self.timestamps.append(new_time)

    def get_logs(self):
        self.test_acc = torch.stack(
            self.test_acc, dim=0).squeeze(2).cpu().numpy()
        self.ent_energy = np.array(self.ent_energy)
        self.timestamps = np.array(self.timestamps).mean()
        return {'timestamps': np.array(self.timestamps).mean(),
                'acc': self.test_acc,
                'ent_energy': self.ent_energy,
                'criterions': self.criterions}

    def normalization(self, z_s, z_q, train_mean):
        """
            inputs:
                z_s : np.Array of shape [n_task, s_shot, feature_dim]
                z_q : np.Array of shape [n_task, q_shot, feature_dim]
                train_mean: np.Array of shape [feature_dim]
        """
        z_s = z_s.cpu()
        z_q = z_q.cpu()

        # CL2N Normalization
        if self.norm_type == 'CL2N':
            z_s = z_s - train_mean
            z_s = z_s / LA.norm(z_s, 2, 2)[:, :, None]
            z_q = z_q - train_mean
            z_q = z_q / LA.norm(z_q, 2, 2)[:, :, None]

        # L2 Normalization
        elif self.norm_type == 'L2N':
            z_s = z_s / LA.norm(z_s, 2, 2)[:, :, None]
            z_q = z_q / LA.norm(z_q, 2, 2)[:, :, None]
        return z_s, z_q

    def create_affinity(self, X):
        N, D = X.shape

        nbrs = NearestNeighbors(n_neighbors=self.knn).fit(X)
        dist, knnind = nbrs.kneighbors(X)

        row = np.repeat(range(N), self.knn - 1)
        col = knnind[:, 1:].flatten()
        data = np.ones(X.shape[0] * (self.knn - 1))
        W = sparse.csc_matrix((data, (row, col)), shape=(N, N), dtype=np.float)
        return W

    def normalize(self, Y_in):
        maxcol = np.max(Y_in, axis=1)
        Y_in = Y_in - maxcol[:, np.newaxis]
        N = Y_in.shape[0]
        size_limit = 150000
        if N > size_limit:
            batch_size = 1280
            Y_out = []
            num_batch = int(math.ceil(1.0 * N / batch_size))
            for batch_idx in range(num_batch):
                start = batch_idx * batch_size
                end = min((batch_idx + 1) * batch_size, N)
                tmp = np.exp(Y_in[start:end, :])
                tmp = tmp / (np.sum(tmp, axis=1)[:, None])
                Y_out.append(tmp)
            del Y_in
            Y_out = np.vstack(Y_out)
        else:
            Y_out = np.exp(Y_in)
            Y_out = Y_out / (np.sum(Y_out, axis=1)[:, None])

        return Y_out

    def entropy_energy(self, Y, unary, kernel, bound_lambda, batch=False):
        tot_size = Y.shape[0]
        pairwise = kernel.dot(Y)
        if batch == False:
            temp = (unary * Y) + (-bound_lambda * pairwise * Y)
            E = (Y * np.log(np.maximum(Y, 1e-20)) + temp).sum()
        else:
            batch_size = 1024
            num_batch = int(math.ceil(1.0 * tot_size / batch_size))
            E = 0
            for batch_idx in range(num_batch):
                start = batch_idx * batch_size
                end = min((batch_idx + 1) * batch_size, tot_size)
                temp = (unary[start:end] * Y[start:end]) + \
                    (-bound_lambda * pairwise[start:end] * Y[start:end])
                E = E + (Y[start:end] *
                         np.log(np.maximum(Y[start:end], 1e-20)) + temp).sum()

        return E

    def bound_update(self, unary, kernel, bound_lambda, y_s, y_q, task_i, bound_iteration=20, batch=False):
        oldE = float('inf')
        Y = self.normalize(-unary)
        E_list = []
        out_list = []
        acc_list = []
        timestamps = []
        t0 = time.time()
        for i in range(bound_iteration):
            additive = -unary
            mul_kernel = kernel.dot(Y)
            Y = -bound_lambda * mul_kernel
            additive = additive - Y
            Y = self.normalize(additive)
            E = self.entropy_energy(Y, unary, kernel, bound_lambda, batch)
            E_list.append(E)
            # print('entropy_energy is ' +repr(E) + ' at iteration ',i)
            l = np.argmax(Y, axis=1)
            # out = np.take(y_s, l)
            out = l
            timestamps.append(time.time()-t0)

            if (i > 1 and (abs(E - oldE) <= 1e-6 * abs(oldE))):
                # print('Converged')
                out_list.append(torch.from_numpy(out))
                acc_list.append(
                    (torch.from_numpy(y_q[task_i]) == torch.from_numpy(out)).float())
                for j in range(bound_iteration-i-1):
                    out_list.append(out_list[i].detach().clone())
                    acc_list.append(acc_list[i].detach().clone())
                    E_list.append(E_list[i])
                    timestamps.append(0)
                break

            else:
                oldE = E.copy()

                out_list.append(torch.from_numpy(out))
                acc_list.append(
                    (torch.from_numpy(y_q[task_i]) == torch.from_numpy(out)).float())
            t0 = time.time()

        out_list = torch.stack(out_list, dim=0)
        acc_list = torch.stack(acc_list, dim=0).mean(dim=1, keepdim=True)

        return out, acc_list, E_list, timestamps

    def run_task(self, task_dic, shot):
        # Extract support and query
        y_s, y_q = task_dic['y_s'], task_dic['y_q']
        x_s, x_q = task_dic['x_s'], task_dic['x_q']

        # Extract features
        support, query = self.normalization(z_s=x_s, z_q=x_q, train_mean=None)
        y_s = y_s.squeeze(2).to(self.device)
        y_q = y_q.squeeze(2).to(self.device)
        support = support.to(self.device)
        query = query.to(self.device)

        n_class = self.args.num_classes_test
        one_hot = get_one_hot(y_s, n_class)
        counts = one_hot.sum(1).view(support.size()[0], -1, 1)
        weights = one_hot.transpose(1, 2).matmul(support)
        support = weights / counts
        support = support.cpu().numpy()

        query = query.cpu().numpy()
        y_s = y_s.cpu().numpy()
        y_q = y_q.cpu().numpy()

        # Run adaptation
        self.run_prediction(support=support, query=query,
                            y_s=y_s, y_q=y_q, shot=shot)

        # Extract adaptation logs
        logs = self.get_logs()

        return logs

    def run_prediction(self, support, query, y_s, y_q, shot):
        """
        Corresponds to the LaplacianShot inference
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]
            y_q : torch.Tensor of shape [n_task, q_shot]
        records :
            accuracy
            ent_energy
            inference time
        """
        self.logger.info(
            " ==> Executing LAPLACIAN SHOT with lmd = {}".format(self.lmd))
        t0 = time.time()
        for i in tqdm(range(self.number_tasks)):

            substract = support[i][:, None, :] - query[i]
            distance = LA.norm(substract, 2, axis=-1)
            unary = distance.transpose() ** 2
            W = self.create_affinity(query[i])
            preds, acc_list, ent_energy, times = self.bound_update(unary=unary, kernel=W, bound_lambda=self.lmd, y_s=y_s, y_q=y_q, task_i=i,
                                                                   bound_iteration=self.iter)

            t1 = time.time()
            self.record_info(acc=acc_list, ent_energy=ent_energy,
                             new_time=t1 - t0, criterions=[0])
