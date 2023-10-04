import torch.nn.functional as F
from src.utils import get_one_hot, Logger
from tqdm import tqdm
import torch
import time
import numpy as np
from copy import deepcopy
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import datasets
import clip
import csv
from sklearn.linear_model import LogisticRegression


class CLIP_LINEAR_PROBE(object):

    def __init__(self, model, device, log_file, args):
        self.device = device
        self.log_file = log_file
        self.logger = Logger(__name__, self.log_file)
        self.init_info_lists()
        self.model = model


    def init_info_lists(self):
        self.timestamps = []
        self.criterions = []
        self.test_acc = []

    
    
    def record_info(self, y_q, preds_q):
        """
        inputs:
            y_q : torch.Tensor of shape [n_task, n_query] :
        """
        accuracy = (preds_q == y_q).float().mean(1, keepdim=True)
        print('acc', accuracy)
        self.test_acc.append(accuracy)

    def get_logs(self):

        #self.criterions = torch.stack(self.criterions, dim=0).cpu().numpy()
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
        y_q = task_dic['y_q']               # [n_task, n_query]
        y_s = task_dic['y_s']               # [n_task, n_query]
        support = task_dic['x_s']             # [n_task, n_query, feature_dim]
        query = task_dic['x_q']             # [n_task, n_query, feature_dim]

        # Transfer tensors to GPU if needed
        support = support.float().cpu().numpy()
        query = query.float().cpu().numpy()
        y_q = y_q.long().squeeze(2).cpu().numpy()
        y_s = y_s.long().squeeze(2).cpu().numpy()
        
        # Run adaptation
        self.run_method(support=support, query=query, y_s=y_s, y_q=y_q)

        # Extract adaptation logs
        logs = self.get_logs()

        return logs


    def run_method(self, support, query, y_s, y_q):

        #self.logger.info(" ==> Executing CLIP")
        
        preds_q = np.zeros_like(y_q)
        n_task = y_q.shape[0]
        n_query = y_q.shape[1]
        
        for i in range(n_task):
            classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
            classifier.fit(support[i], y_s[i])

            # Evaluate using the logistic regression classifier
            preds_q[i] = classifier.predict(query[i])
        
        preds_q = torch.from_numpy(preds_q)
        y_q = torch.from_numpy(y_q)
        self.record_info(y_q=y_q, preds_q=preds_q)


