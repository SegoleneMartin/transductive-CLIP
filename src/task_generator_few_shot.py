import torch


class Tasks_Generator_few_shot:
    def __init__(self, k_eff, shot, n_query, n_class, loader_support, loader_query, model, args):
        """
        Initialize the Tasks_Generator.
        :param k_eff: Effective number of samples per class in support set.
        :param shot: Number of samples per class in the support set.
        :param n_query: Number of samples in the query set.
        :param n_class: Number of classes (typically the total number of classes in the test set).
        :param loader_support: Data loader for the support set.
        :param loader_query: Data loader for the query set.
        :param model: Model used for extracting features.
        :param args: Additional arguments.
        """

        self.k_eff = k_eff
        self.shot = shot
        self.n_query = n_query
        self.loader_support = loader_support
        self.loader_query = loader_query
        self.n_class = n_class
        self.model = model
        self.args = args

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

        unique_labels = torch.flip(torch.unique(
            labels_support, sorted=False), dims=(0,))
        new_labels_support = torch.zeros_like(labels_support)
        new_labels_query = torch.zeros_like(labels_query)

        for j, y in enumerate(unique_labels):
            new_labels_support[labels_support == y] = j
            new_labels_query[labels_query == y] = j

        if self.args.use_softmax_feature == True:
            new_data_query = data_query[:, unique_labels]
            new_data_support = data_support[:, unique_labels]
        else:
            new_data_query = data_query
            new_data_support = data_support
            new_labels_support = labels_support
            new_labels_query = labels_query

        torch.cuda.empty_cache()

        task = {'x_s': new_data_support, 'y_s': new_labels_support.long(),
                'x_q': new_data_query, 'y_q': new_labels_query.long()}
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
            task = self.get_task(data_support, data_query,
                                 labels_support, labels_query)
            tasks_dics.append(task)

        feature_size = data_support.size()[-1]

        # Now merging all tasks into 1 single dictionnary
        merged_tasks = {}
        n_task = len(tasks_dics)
        for key in tasks_dics[0].keys():
            n_samples = tasks_dics[0][key].size(0)
            if key == 'x_s' or key == 'x_q':
                merged_tasks[key] = torch.cat([tasks_dics[i][key] for i in range(n_task)], dim=0).view(n_task,
                                                                                                       n_samples, feature_size)
            elif key == 'y_s' or key == 'y_q':
                merged_tasks[key] = torch.cat([tasks_dics[i][key] for i in range(n_task)], dim=0).view(n_task,
                                                                                                       n_samples, -1)
            else:
                raise Exception("Wrong dict key")

        return merged_tasks
