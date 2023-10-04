import torch
from src.utils import Logger,load_pickle, save_pickle
import os
from itertools import cycle
from torchvision import datasets
import csv
import clip


class Tasks_Generator:
    def __init__(self, k_eff, shot, n_query, n_ways, loader_support, loader_query, model):
        self.k_eff = k_eff
        self.shot = shot
        self.n_query = n_query
        self.loader_support = loader_support
        self.loader_query = loader_query
        self.n_ways = n_ways
        self.model = model
        
    def get_task(self, data_support, data_query, labels_support, labels_query):
        """
        inputs:
            data_support : torch.tensor of shape [shot * k_eff, channels, H, W]
            data_query : torch.tensor of shape [n_query, channels, H, W]
            labels_support :  torch.tensor of shape [shot * k_eff + n_query]
            labels_query :  torch.tensor of shape [n_query]
        returns :
            task : Dictionnary : x_support : torch.tensor of shape [k_eff * shot, channels, H, W]
                                 x_query : torch.tensor of shape [n_query, channels, H, W]
                                 y_support : torch.tensor of shape [k_eff * shot]
                                 y_query : torch.tensor of shape [n_query]
        """

        unique_labels = torch.flip(torch.unique(labels_support, sorted=False), dims=(0,))
        #print('unique', unique_labels)
        new_labels_support = torch.zeros_like(labels_support)
        new_labels_query = torch.zeros_like(labels_query)
        #precisions_support = torch.zeros(self.n_ways, 1024, 1024)
        for j, y in enumerate(unique_labels):
            new_labels_support[labels_support == y] = j #y # j
            new_labels_query[labels_query == y] = j #y # j
                
        #text_inputs = torch.cat([clip.tokenize("an image of a {}".format(reader[i][2])) for i in unique_labels]).to('cuda')
        #with torch.no_grad():
        #    text_embs = self.model.encode_text(text_inputs).float().to('cpu').float()
        #text_embs.div_(text_embs.norm(dim=-1, keepdim=True))
   
        #print('unique_labels', unique_labels)
        new_data_query = data_query[:, unique_labels]
        new_data_support = data_support[:, unique_labels]
        torch.cuda.empty_cache()

        #task = {'x_s': data_support, 'y_s': new_labels_support.long(),
        #        'x_q': data_query, 'y_q': new_labels_query.long(), 'y_q_original': labels_query.long(), 'y_s_original': labels_support.long(), 'text_embs': text_embs,
        #       'S_s': precisions_support}
        

        task = {'x_s': new_data_support, 'y_s': new_labels_support.long(),
                'x_q': new_data_query, 'y_q': new_labels_query.long(), 'y_q_original': labels_query.long(), 'y_s_original': labels_support.long()}
        #task = {'y_s': new_labels_support.long(),
        #        'x_q': data_query, 'y_q': new_labels_query.long(), 'y_q_original': labels_query.long(), 'y_s_original': labels_support.long(), 'text_embs': text_embs}
        
        #task = {'x_s': data_support, 'y_s': new_labels_support.long(),
        #        'x_q': data_query, 'y_q': new_labels_query.long(), 'y_q_original': labels_query.long(), 'y_s_original': labels_support.long(), 'text_embs': text_embs}
        return task

    def generate_tasks(self):
        """
        returns :
            merged_task : { x_support : torch.tensor of shape [batch_size, k_eff * shot, channels, H, W]
                            x_query : torch.tensor of shape [batch_size, k_eff * query_shot, channels, H, W]
                            y_support : torch.tensor of shape [batch_size, k_eff * shot]
                            y_query : torch.tensor of shape [batch_size, k_eff * query_shot]
                            train_mean: torch.tensor of shape [feature_dim]}
        """
        tasks_dics = []

        for support, query in zip(self.loader_support, self.loader_query):
            (data_support, labels_support) = support
            (data_query, labels_query) = query
            task = self.get_task(data_support, data_query, labels_support, labels_query)
            tasks_dics.append(task)

        feature_size = data_support.size()[-1]
        #feature_size = 768
        #feature_size = 1000
        
        # Now merging all tasks into 1 single dictionnary
        merged_tasks = {}
        n_tasks = len(tasks_dics)
        for key in tasks_dics[0].keys():
            n_samples = tasks_dics[0][key].size(0)
            if key == 'x_s' or key == 'x_q':
                merged_tasks[key] = torch.cat([tasks_dics[i][key] for i in range(n_tasks)], dim=0).view(n_tasks,
                                                                                                        n_samples, feature_size)
            elif key == 'text_embs':
                merged_tasks[key] = torch.cat([tasks_dics[i][key] for i in range(n_tasks)], dim=0)
                
            elif key == 'y_s' or key == 'y_q' or key == 'y_s_original' or key == 'y_q_original':
                merged_tasks[key] = torch.cat([tasks_dics[i][key] for i in range(n_tasks)], dim=0).view(n_tasks,
                                                                                                        n_samples, -1)
            elif key == 'alpha':
                merged_tasks[key] = torch.cat([tasks_dics[i][key] for i in range(n_tasks)], dim=0).view(n_tasks,
                                                                                                        feature_size, feature_size)
            else:
                merged_tasks[key] = torch.cat([tasks_dics[i][key] for i in range(n_tasks)], dim=0).view(n_tasks,
                                                                                     self.n_ways, feature_size, feature_size, -1)
            

        return merged_tasks

