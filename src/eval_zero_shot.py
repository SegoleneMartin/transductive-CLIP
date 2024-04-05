import os
import numpy as np

# Import utilities and datasets
from src.utils import Logger, load_pickle, extract_features_softmax, extract_features_visual, compute_confidence_interval
from src.datasets import OxfordPets, EuroSAT, UCF101, SUN397, Caltech101, DescribableTextures, FGVCAircraft, Food101, Flowers102, StanfordCars, ImageNet
from src.datasets import build_data_loader
from src.task_generator_zero_shot import Tasks_Generator_zero_shot
from src.sampler_zero_shot import CategoriesSampler_zero_shot, SamplerQuery_zero_shot

# Import methods for zero-shot learning
from src.methods.zero_shot.inductive_clip import CLIP  # i.e. zero-shot CLIP
from src.methods.zero_shot.kl_kmeans import KL_KMEANS
from src.methods.zero_shot.em_dirichlet import EM_DIRICHLET
from src.methods.zero_shot.hard_em_dirichlet import HARD_EM_DIRICHLET
from src.methods.zero_shot.em_gaussian import EM_GAUSSIAN
from src.methods.zero_shot.em_gaussian_cov import EM_GAUSSIAN_COV
from src.methods.zero_shot.soft_kmeans import SOFT_KMEANS
from src.methods.zero_shot.hard_kmeans import HARD_KMEANS

# Dataset list for few-shot learning tasks
dataset_list = {
    "oxfordpets": OxfordPets,
    "eurosat": EuroSAT,
    "ucf101": UCF101,
    "sun397": SUN397,
    "caltech101": Caltech101,
    "dtd": DescribableTextures,
    "fgvcaircraft": FGVCAircraft,
    "food101": Food101,
    "flowers102": Flowers102,
    "stanfordcars": StanfordCars,
    "imagenet": ImageNet
}


class Evaluator_zero_shot:
    def __init__(self, device, args, log_file):
        self.device = device
        self.args = args
        self.log_file = log_file
        self.logger = Logger(__name__, self.log_file)

    def run_full_evaluation(self, model, preprocess):
        """
        Run the full evaluation process over all tasks.
        :param model: The model to be evaluated.
        :param preprocess: Preprocessing function for data.
        :return: Mean accuracies of the evaluation.
        """
        model.eval()

        # Initialize dataset and data loaders
        dataset = dataset_list[self.args.dataset](self.args.dataset_path)
        self.args.classnames = dataset.classnames
        self.args.template = dataset.template
        # data_loaders = self.initialize_data_loaders(dataset, preprocess)
        data_loaders = None

        # Extract and load features
        extracted_features_dic_query = self.extract_and_load_features(
            model, dataset, data_loaders)
        all_features_query = extracted_features_dic_query['concat_features'].to(
            'cpu')
        all_labels_query = extracted_features_dic_query['concat_labels'].long().to(
            'cpu')

        # Run evaluation for each task and collect results
        mean_accuracies, mean_times = self.evaluate_tasks(
            model, all_features_query, all_labels_query)

        # Log final results
        self.report_results(mean_accuracies, mean_times)

    def initialize_data_loaders(self, dataset, preprocess):
        """
        Initialize data loaders for training, validation, and testing.
        :param dataset: The dataset object.
        :param preprocess: Preprocessing function for data.
        :return: Dictionary of data loaders for test set (train and val sets not required in the zero-shot setting).
        """
        batch_size = 1024
        data_loaders = {
            # 'train': build_data_loader(data_source=dataset.train_x, batch_size=batch_size, is_train=False, shuffle=False, tfm=preprocess),
            # 'val': build_data_loader(data_source=dataset.val, batch_size=batch_size, is_train=False, shuffle=False, tfm=preprocess),
            'test': build_data_loader(data_source=dataset.test, batch_size=batch_size, is_train=False, shuffle=False, tfm=preprocess)
        }
        return data_loaders

    def extract_and_load_features(self, model, dataset, data_loaders):
        """
        Extract and load features for the evaluation.
        :param model: The model to be evaluated.
        :param dataset: The dataset object.
        :param data_loaders: Data loaders for train, val, and test.
        """

        # Load the features: either the softmax features, either the visual embeddings
        if self.args.use_softmax_feature == True:
            # extract_features_softmax(model, dataset, data_loaders['test'], 'test', self.args, self.device, list_T=[self.args.T])
            filepath_query = 'data/{}/saved_features/{}_softmax_{}_T{}.plk'.format(
                self.args.dataset, self.args.used_test_set, self.args.backbone, self.args.T)
        else:
            # extract_features_visual(model, dataset, data_loaders['test'], 'test', self.args, self.device, list_T=[self.args.T])
            # filepath_query = 'data/{}/saved_features/{}_visual_{}.plk'.format(self.args.dataset, self.args.used_test_set, self.args.backbone)
            filepath_query = 'data/{}/saved_features/{}_{}.plk'.format(
                self.args.dataset, self.args.used_test_set, self.args.backbone)

        extracted_features_dic_query = load_pickle(filepath_query)

        return extracted_features_dic_query

    def get_method_builder(self, model, device, args, log_file):
        # Initialize method classifier builder
        method_info = {'model': model, 'device': device,
                       'log_file': log_file, 'args': args}

        # zero-shot methods
        if args.name_method == 'KL_KMEANS':
            method_builder = KL_KMEANS(**method_info)
        elif args.name_method == 'EM_DIRICHLET':
            method_builder = EM_DIRICHLET(**method_info)
        elif args.name_method == 'HARD_EM_DIRICHLET':
            method_builder = HARD_EM_DIRICHLET(**method_info)
        elif args.name_method == 'EM_GAUSSIAN':
            method_builder = EM_GAUSSIAN(**method_info)
        elif args.name_method == 'EM_GAUSSIAN_COV':
            method_builder = EM_GAUSSIAN_COV(**method_info)
        elif args.name_method == 'SOFT_KMEANS':
            method_builder = SOFT_KMEANS(**method_info)
        elif args.name_method == 'HARD_KMEANS':
            method_builder = HARD_KMEANS(**method_info)
        elif args.name_method == 'CLIP':
            method_builder = CLIP(**method_info)
        else:
            raise ValueError(
                "The method your entered does not exist or is not a zero-shot method. Please check the spelling")
        return method_builder

    def evaluate_tasks(self, model, all_features_query, all_labels_query):

        self.logger.info("=> Runnning evaluation with method {} on {} dataset".format(
            self.args.name_method, self.args.used_test_set))

        results = []
        results_time = []
        results_task = []
        results_task_time = []

        # Start the evaluation over the tasks
        for i in range(int(self.args.number_tasks/self.args.batch_size)):

            # Create sampler for transductive zero-shot tasks
            sampler = CategoriesSampler_zero_shot(self.args.batch_size,
                                                  self.args.k_eff, self.args.n_class, self.args.n_query, force_query_size=True)
            sampler.create_list_classes(all_labels_query)
            sampler_query = SamplerQuery_zero_shot(sampler)

            # Get the query samples at the indexes given by the samplers
            test_loader_query = []
            for indices in sampler_query:
                test_loader_query.append(
                    (all_features_query[indices, :], all_labels_query[indices]))

            # Prepare the tasks
            task_generator = Tasks_Generator_zero_shot(
                k_eff=self.args.k_eff, n_class=self.args.n_class, n_query=self.args.n_query, loader_query=test_loader_query, model=model, args=self.args)
            tasks = task_generator.generate_tasks()

            # Load the method (e.g. EM_DIRICHLET)
            method = self.get_method_builder(
                model=model, device=self.device, args=self.args, log_file=self.log_file)

            # Run task
            logs = method.run_task(task_dic=tasks)
            acc_mean, acc_conf = compute_confidence_interval(
                logs['acc'][:, -1])
            timestamps, criterions = logs['timestamps'], logs['criterions']
            results_task.append(acc_mean)
            results_task_time.append(timestamps)

            del tasks

        results.append(results_task)
        results_time.append(results_task_time)
        mean_accuracies = np.asarray(results).mean(1)
        mean_times = np.asarray(results_time).mean(1)

        return mean_accuracies, mean_times

    def report_results(self, mean_accuracies, mean_times):
        self.logger.info('----- Final results -----')

        if self.args.use_softmax_feature == True:
            word = '_softmax'
        else:
            word = '_visual'
        path = 'results_zero_shot/{}/{}'.format(
            self.args.used_test_set, self.args.dataset)

        # if save results mode
        if self.args.save_results == True:
            var = str(self.args.shots) + '\t' + str(self.args.n_query) + \
                '\t' + str(self.args.number_tasks)
            var_names = 'shots' + '\t' + 'n_query' + '\t' + 'n_task' + '\t' + 'acc' + '\n'

            path = 'results_zero_shot/{}/{}'.format(
                self.args.used_test_set, self.args.dataset)
            name_file = path + \
                '/{}_{}shot.txt'.format(self.args.name_method +
                                        word, self.args.shots)

            if not os.path.exists(path):
                os.makedirs(path)
            if os.path.isfile(name_file) == True:
                f = open(name_file, 'a')
            else:
                f = open(name_file, 'w')
                f.write(var_names + '\t' + '\n')

            self.logger.info('{}-shot mean test accuracy over {} tasks: {}'.format(
                self.args.shots, self.args.number_tasks, mean_accuracies[0]))
            self.logger.info('{}-shot mean time over {} tasks: {}'.format(
                self.args.shots, self.args.number_tasks, mean_times[0][0]))
            f.write(str(var)+'\t')
            f.write(str(round(100 * mean_accuracies[0], 1)) + '\t')
            f.write('\n')
            f.close()

        else:
            self.logger.info('{}-shot mean test accuracy over {} tasks: {}'.format(
                self.args.shots, self.args.number_tasks, mean_accuracies[0]))
            self.logger.info('{}-shot mean time over {} tasks: {}'.format(
                self.args.shots, self.args.number_tasks, mean_times[0][0]))
