import torch
import numpy as np
import random

class CategoriesSampler_zero_shot():
    """
    CategorySampler
    inputs:
        n_batch : Number of batches to load
        k_eff : Number of classification effecive classes represented in the query set
        n_query : Size of query set
    returns :
        sampler : CategoriesSampler object that will yield batch when iterated
        When iterated returns : batch
            data : torch.tensor [n_support + n_query, channel, H, W]
                    [query_data]
            labels : torch.tensor [n_support + n_query]
                    [query_labels]
    """

    def __init__(self, n_batch, k_eff, n_ways, n_query, force_query_size=False):
        self.n_batch = n_batch
        self.k_eff = k_eff
        self.n_query = n_query
        self.n_ways = n_ways
        self.force_query_size = force_query_size
        self.m_ind_query = []
        self.list_classes = [i for i in range(n_ways)]
        
    def create_list_classes(self, label_query):

        label_query = np.array(label_query)
        for i in range(self.n_ways):
            ind = np.argwhere(label_query == i).reshape(-1)
            self.m_ind_query.append(torch.from_numpy(ind))

class SamplerQuery_zero_shot:
    def __init__(self, cat_samp):
        self.name = "SamplerQuery"
        self.list_classes = cat_samp.list_classes
        self.n_batch = cat_samp.n_batch
        self.k_eff = cat_samp.k_eff
        self.m_ind_query = cat_samp.m_ind_query
        self.n_query = cat_samp.n_query
        self.force_query_size = cat_samp.force_query_size

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            self.k_eff = random.randint(3, 10)
            query_size = 0
            n_trials = 0
            while query_size < self.n_query and n_trials < 1:
                classes = [self.list_classes[i] for i in torch.randperm(len(self.list_classes))[:self.k_eff].tolist()]
                query = []
                complete_possible_samples = self.m_ind_query[classes[0]]
                for c in classes[1:]:
                    complete_possible_samples = torch.cat((complete_possible_samples, self.m_ind_query[c]))
                pos = torch.randperm(len(complete_possible_samples))[:self.n_query]
                query = complete_possible_samples[pos]
                
                if self.force_query_size == False:
                    n_trials += 1
                query_size = len(query)
            yield query

   