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

class CLIP(object):

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
        y_s_original = task_dic['y_s_original']               # [n_task, n_query]
        y_q_original = task_dic['y_q_original']               # [n_task, n_query]
        query = task_dic['x_q']             # [n_task, n_query, feature_dim]

        # Transfer tensors to GPU if needed
        query = query.to(self.device).float()
        y_q_original = y_q_original.long().squeeze(2).to(self.device)
        y_q = y_q.long().squeeze(2).to(self.device)
        y_s_original = y_s_original.long().squeeze(2).to(self.device)
        
        # Run adaptation
        self.run_method(query=query, y_s_original=y_s_original, y_q=y_q, y_q_original=y_q_original)

        # Extract adaptation logs
        logs = self.get_logs()

        return logs


    def run_method(self, query, y_s_original, y_q, y_q_original):

        #self.logger.info(" ==> Executing CLIP")
        
        data = datasets.ImageFolder('./data/imagenet/')
        preds_q = torch.zeros_like(y_q)
        n_task = y_q.shape[0]
        n_query = y_q.shape[1]
        
        for i in range(n_task):
            unique_labels = torch.flip(torch.unique(y_s_original, sorted=False), dims=(0,))
            file = open('data/idx_class_name.csv', 'r') 
            reader = list(csv.reader(file, delimiter=","))
            file.close()

            # Prepare the inputs
            text_inputs = torch.cat([clip.tokenize("an image of a {}".format(reader[i][2])) for i in unique_labels]).to(self.device)
            with torch.no_grad():
                text_features = self.model.encode_text(text_inputs).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            for n in range(n_query):

                # Calculate feature
                image_features = query[i, n].unsqueeze(0)
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                values, indices = similarity[0].topk(1)
                #image_features = query[i, n]
                #values, indices = image_features.topk(1)
                preds_q[i, n] = unique_labels[indices[0].item()]
    
        self.record_info(y_q=y_q_original, preds_q=preds_q)


