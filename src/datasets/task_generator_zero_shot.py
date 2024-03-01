import torch

class Tasks_Generator:
    def __init__(self, k_eff, n_query, n_ways, loader_query, model, args):
        """
        Initialize the Tasks_Generator.
        :param k_eff: Effective number of samples per class in support set.
        :param n_query: Number of samples in the query set.
        :param n_ways: Number of classes (typically the total number of classes in the test set).
        :param loader_query: Data loader for the query set.
        :param model: Model used for extracting features.
        :param args: Additional arguments.
        """

        self.k_eff = k_eff
        self.n_query = n_query
        self.loader_query = loader_query
        self.n_ways = n_ways
        self.model = model
        self.args = args
        

    def get_task(self, data_query, labels_query):
        """
        inputs:
            data_query : torch.tensor of shape [n_query, channels, H, W]
            labels_query :  torch.tensor of shape [n_query]
        returns :
            task : Dictionnary : x_query : torch.tensor of shape [n_query, channels, H, W]
                                 y_query : torch.tensor of shape [n_query]
        """

        task = {'x_q': data_query, 'y_q': labels_query.long()}
        return task


    def generate_tasks(self):
        """
        returns :
            merged_task : { x_query : torch.tensor of shape [batch_size, k_eff * query_shot, channels, H, W]
                            y_query : torch.tensor of shape [batch_size, k_eff * query_shot]}
        """
        tasks_dics = []

        for query in self.loader_query:
            (data_query, labels_query) = query
            task = self.get_task(data_query, labels_query)
            tasks_dics.append(task)

        feature_size = data_query.size()[-1]

        # Now merging all tasks into 1 single dictionnary
        merged_tasks = {}
        n_tasks = len(tasks_dics)
        for key in tasks_dics[0].keys():
            n_samples = tasks_dics[0][key].size(0)
            if key == 'x_q':
                merged_tasks[key] = torch.cat([tasks_dics[i][key] for i in range(n_tasks)], dim=0).view(n_tasks,
                                                                                                        n_samples, feature_size)
            elif key == 'y_q':
                merged_tasks[key] = torch.cat([tasks_dics[i][key] for i in range(n_tasks)], dim=0).view(n_tasks,
                                                                                                        n_samples, -1)
            else:
                raise Exception("Wrong dict key")

        return merged_tasks

