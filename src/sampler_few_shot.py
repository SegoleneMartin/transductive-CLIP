import torch
import numpy as np


class CategoriesSampler_few_shot():
    """
            CategorySampler
            inputs:
                label : All labels of dataset
                n_batch : Number of batches to load
                k_eff : Number of classification ways (k_eff)
                s_shot : Support shot
                n_query : Size of query set
                alpha : Dirichlet's concentration parameter
            returns :
                sampler : CategoriesSampler object that will yield batch when iterated
                When iterated returns : batch
                        data : torch.tensor [n_support + n_query, channel, H, W]
                               [support_data, query_data]
                        labels : torch.tensor [n_support + n_query]
                               [support_labels, query_labels]
    """

    def __init__(self, n_batch, k_eff, n_class, s_shot, n_query, force_query_size=False):
        # the number of iterations in the dataloader
        self.n_batch = n_batch
        self.k_eff = k_eff
        self.s_shot = s_shot
        self.n_query = n_query
        self.n_class = n_class
        self.force_query_size = force_query_size
        self.list_classes = [i for i in range(n_class)]

    def create_list_classes(self, label_support, label_query):

        label_support = np.array(label_support)     # all data label
        self.m_ind_support = []                     # the data index of each class
        for i in range(max(label_support) + 1):
            # all data index of this class
            ind = np.argwhere(label_support == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind_support.append(ind)

        label_query = np.array(label_query)         # all data label
        self.m_ind_query = []                       # the data index of each class
        for i in range(max(label_support) + 1):
            # all data index of this class
            ind = np.argwhere(label_query == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind_query.append(ind)


class SamplerSupport_few_shot:
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
            classes = self.list_classes
            for c in classes:
                # all data indexs of this class
                l = self.m_ind_support[c]
                # select all data
                pos = torch.randperm(len(l))[:self.s_shot]
                support.append(l[pos])
            support = torch.cat(support)

            yield support


class SamplerQuery_few_shot:
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
            query_size = 0
            n_trials = 0
            while query_size < self.n_query and n_trials < 1:
                classes = [self.list_classes[i] for i in torch.randperm(
                    len(self.list_classes))[:self.k_eff].tolist()]
                query = []

                complete_possible_samples = self.m_ind_query[classes[0]]
                for c in classes[1:]:
                    complete_possible_samples = torch.cat(
                        (complete_possible_samples, self.m_ind_query[c]))
                pos = torch.randperm(len(complete_possible_samples))[
                    :self.n_query]
                query = complete_possible_samples[pos]

                if self.force_query_size == False:
                    n_trials += 1
                query_size = len(query)
            yield query
