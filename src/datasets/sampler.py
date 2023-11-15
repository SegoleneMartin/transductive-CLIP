import torch
import numpy as np
import math
import random

class CategoriesSampler():
    """
            CategorySampler
            inputs:
                label : All labels of dataset
                n_batch : Number of batches to load
                k_eff : Number of classification ways (k_eff)
                s_shot : Support shot
                n_query : Size of query set
                sampling : 'balanced': Balanced query class distribution: Standard class sampling Few-Shot setting
                           'dirichlet': Dirichlet's distribution over query data: For class imbalanced Few-Shot setting
                           'uniform' : Uniform distribution over query data: Realistic distribution of the data given the dataset
                alpha : Dirichlet's concentration parameter
            returns :
                sampler : CategoriesSampler object that will yield batch when iterated
                When iterated returns : batch
                        data : torch.tensor [n_support + n_query, channel, H, W]
                               [support_data, query_data]
                        labels : torch.tensor [n_support + n_query]
                               [support_labels, query_labels]
    """

    def __init__(self, label_support, label_query, n_batch, k_eff, n_ways, s_shot, n_query, sampling, force_query_size=False):
        self.n_batch = n_batch                      # the number of iterations in the dataloader
        self.k_eff = k_eff
        self.s_shot = s_shot
        self.n_query = n_query
        self.sampling = sampling
        self.n_ways = n_ways
        self.force_query_size = force_query_size
        
    def create_list_classes(self, label_support, label_query):

        label_support = np.array(label_support)     # all data label
        self.m_ind_support = []                     # the data index of each class
        for i in range(max(label_support) + 1):
            ind = np.argwhere(label_support == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind_support.append(ind)

        label_query = np.array(label_query)         # all data label
        self.m_ind_query = []                       # the data index of each class
        for i in range(max(label_support) + 1):
            ind = np.argwhere(label_query == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind_query.append(ind)

        self.list_classes = []
        for i_batch in range(self.n_batch):
            classes = torch.randperm(len(self.m_ind_support))[:self.n_ways]
            self.list_classes.append(classes)  # random sample num_class indexs
            


class SamplerSupport:
    def __init__(self, cat_samp):
        self.name = "SamplerSupport"
        self.list_classes = cat_samp.list_classes
        self.n_batch = cat_samp.n_batch
        self.s_shot = cat_samp.s_shot
        self.m_ind_support = cat_samp.m_ind_support

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            support = []
            classes = self.list_classes[i_batch]
            for c in classes:
                l = self.m_ind_support[c]                       # all data indexs of this class
                pos = torch.randperm(len(l))[:self.s_shot]                 # select all data
                support.append(l[pos])
            support = torch.cat(support)
           
            yield support


class SamplerQuery:
    def __init__(self, cat_samp):
        self.name = "SamplerQuery"
        self.list_classes = cat_samp.list_classes
        self.n_batch = cat_samp.n_batch
        self.k_eff = cat_samp.k_eff
        self.m_ind_query = cat_samp.m_ind_query
        self.n_query = cat_samp.n_query
        self.sampling = cat_samp.sampling
        self.force_query_size = cat_samp.force_query_size

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            self.k_eff = random.randint(3, 10)
            query_size = 0
            n_trials = 0
            while query_size < self.n_query and n_trials < 1:
                classes = self.list_classes[i_batch][torch.randperm(len(self.list_classes[i_batch]))[:self.k_eff]]
                query = []

                assert self.sampling in ['balanced', 'uniform']

                if self.sampling == 'balanced':
                    query_samples = np.repeat(self.n_query // self.k_eff, self.k_eff)
                    for c, nb_shot in zip(classes, query_samples):
                        l = self.m_ind_query[c]                         # all data indexs of this class
                        pos = torch.randperm(len(l))[:nb_shot]          # sample n_per data index of this class
                        query.append(l[pos])
                    query = torch.cat(query)

                elif self.sampling == "uniform":
                    complete_possible_samples = self.m_ind_query[classes[0]]
                    for c in classes[1:]:
                        complete_possible_samples = torch.cat((complete_possible_samples, self.m_ind_query[c]))
                    pos = torch.randperm(len(complete_possible_samples))[:self.n_query]
                    query = complete_possible_samples[pos]
                    
                if self.force_query_size == False:
                    n_trials += 1
                query_size = len(query)
                #print("query_size", query_size)
            yield query

