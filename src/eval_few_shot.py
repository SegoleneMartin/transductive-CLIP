import os
import numpy as np

# Import utilities and datasets
from src.utils import Logger, load_pickle, extract_features_softmax, extract_features_visual, compute_confidence_interval
from src.datasets import OxfordPets, EuroSAT, UCF101, SUN397, Caltech101, DescribableTextures, FGVCAircraft, Food101, Flowers102, StanfordCars, ImageNet
from src.datasets import build_data_loader
from src.task_generator_few_shot import Tasks_Generator_few_shot
from src.sampler_few_shot import CategoriesSampler_few_shot, SamplerQuery_few_shot, SamplerSupport_few_shot

# Import methods for few-shot learning
from src.methods.few_shot.em_dirichlet import EM_DIRICHLET
from src.methods.few_shot.hard_em_dirichlet import HARD_EM_DIRICHLET
from src.methods.few_shot.paddle import PADDLE
from src.methods.few_shot.bdcspn import BDCSPN
from src.methods.few_shot.tim import ALPHA_TIM
from src.methods.few_shot.laplacian_shot import LAPLACIAN_SHOT

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


class Evaluator_few_shot:
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
        data_loaders = self.initialize_data_loaders(dataset, preprocess)

        # Extract and load features
        extracted_features_dic_support, extracted_features_dic_query = self.extract_and_load_features(
            model, dataset, data_loaders)
        all_features_support = extracted_features_dic_support['concat_features'].to(
            'cpu')
        all_labels_support = extracted_features_dic_support['concat_labels'].long().to(
            'cpu')
        all_features_query = extracted_features_dic_query['concat_features'].to(
            'cpu')
        all_labels_query = extracted_features_dic_query['concat_labels'].long().to(
            'cpu')

        # Run evaluation for each task and collect results
        mean_accuracies, mean_times = self.evaluate_tasks(
            model, all_features_support, all_labels_support, all_features_query, all_labels_query)

        # Log final results
        self.report_results(mean_accuracies, mean_times)

    def initialize_data_loaders(self, dataset, preprocess):
        """
        Initialize data loaders for training, validation, and testing.
        :param dataset: The dataset object.
        :param preprocess: Preprocessing function for data.
        :return: Dictionary of data loaders for train, val, and test.
        """
        batch_size = 1024
        data_loaders = {
            'train': build_data_loader(data_source=dataset.train_x, batch_size=batch_size, is_train=False, shuffle=False, tfm=preprocess),
            'val': build_data_loader(data_source=dataset.val, batch_size=batch_size, is_train=False, shuffle=False, tfm=preprocess),
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
            extract_features_softmax(model, dataset, data_loaders['test'], 'test', self.args, self.device, list_T=[self.args.T])
            extract_features_softmax(model, dataset, data_loaders['val'], 'val', self.args, self.device, list_T=[self.args.T])
            extract_features_softmax(model, dataset, data_loaders['train'], 'train', self.args, self.device, list_T=[self.args.T])

            filepath_support = 'data/{}/saved_features/train_softmax_{}_T{}.plk'.format(
                self.args.dataset, self.args.backbone, self.args.T)
            filepath_query = 'data/{}/saved_features/{}_softmax_{}_T{}.plk'.format(
                self.args.dataset, self.args.used_test_set, self.args.backbone, self.args.T)
        else:
            extract_features_visual(model, dataset, data_loaders['test'], 'test', self.args, self.device)
            extract_features_visual(model, dataset, data_loaders['val'], 'val', self.args, self.device)
            extract_features_visual(model, dataset, data_loaders['train'], 'train', self.args, self.device)

            filepath_support = 'data/{}/saved_features/train_visual_{}.plk'.format(self.args.dataset, self.args.backbone)
            filepath_query = 'data/{}/saved_features/{}_visual_{}.plk'.format(self.args.dataset, self.args.used_test_set, self.args.backbone)

        extracted_features_dic_support = load_pickle(filepath_support)
        extracted_features_dic_query = load_pickle(filepath_query)

        return extracted_features_dic_support, extracted_features_dic_query

    def get_method_val_param(self):
        # fixes for each method the name of the parameter on which validation is performed
        if self.args.name_method == 'LAPLACIAN_SHOT':
            self.val_param = self.args.lmd
        elif self.args.name_method == 'ALPHA_TIM':
            self.val_param = self.args.alpha_value
        elif self.args.name_method == 'PADDLE':
            self.val_param = self.args.lambd
        elif self.args.name_method == 'BDCSPN':
            self.val_param = self.args.temp

    def set_value_opt_param(self, opt_param):
        # fixes for each method the name of the parameter on which validation is performed
        if self.args.name_method == 'LAPLACIAN_SHOT':
            self.args.lmd = opt_param
        elif self.args.name_method == 'ALPHA_TIM':
            self.args.alpha_value = opt_param
        elif self.args.name_method == 'PADDLE':
            self.args.lambd = opt_param
        elif self.args.name_method == 'BDCSPN':
            self.args.temp = opt_param

    def set_method_opt_param(self):
        if self.args.use_softmax_feature == True:
            word = '_softmax'
        else:
            word = '_visual'
        path = 'results_few_shot/val/{}'.format(self.args.dataset)
        name_file = path + \
            '/{}_s{}.txt'.format(self.args.name_method + word, self.args.shots)

        # for imagenet we use the val set of the caltech101 dataset to tune the parameter
        if self.args.dataset == 'imagenet':
            path = 'results_few_shot/val/{}'.format('caltech101')
            name_file = path + \
                '/{}_s{}.txt'.format(self.args.name_method +
                                     word, self.args.shots)

        try:
            f = open(name_file, 'r')
            list_param, list_acc = [], []
            for i, line in enumerate(f):
                if i < 2:
                    continue
                line = line.split('\t')
                list_param.append(float(line[0]))
                list_acc.append(float(line[1]))
            list_acc = np.array(list_acc)
            index = np.argwhere(list_acc == np.amax(list_acc))[-1][0]
            opt_param = list_param[index]
            print('opt param', opt_param)
            self.set_value_opt_param(opt_param)
            f.close()

        except:

            raise ValueError(
                "The optimal parameter was not found. Please make sure you have performed the tuning of the parameter on the validation set.")

    def get_method_builder(self, model, device, args, log_file):
        # Initialize method classifier builder
        method_info = {'model': model, 'device': device,
                       'log_file': log_file, 'args': args}

        # few-shot methods
        if args.name_method == 'EM_DIRICHLET':
            method_builder = EM_DIRICHLET(**method_info)
        elif args.name_method == 'HARD_EM_DIRICHLET':
            method_builder = HARD_EM_DIRICHLET(**method_info)
        elif args.name_method == 'PADDLE':
            method_builder = PADDLE(**method_info)
        elif args.name_method == 'BDCSPN':
            method_builder = BDCSPN(**method_info)
        elif args.name_method == 'LAPLACIAN_SHOT':
            method_builder = LAPLACIAN_SHOT(**method_info)
        elif args.name_method == 'ALPHA_TIM':
            method_builder = ALPHA_TIM(**method_info)

        else:
            raise ValueError(
                "The method your entered does not exist or is not a few-shot method. Please check the spelling")
        return method_builder

    def evaluate_tasks(self, model, all_features_support, all_labels_support, all_features_query, all_labels_query):

        self.logger.info("=> Runnning evaluation with method {} on {} dataset".format(
            self.args.name_method, self.args.used_test_set))

        results = []
        results_time = []
        results_task = []
        results_task_time = []

        # Start the evaluation over the tasks
        for i in range(int(self.args.number_tasks/self.args.batch_size)):

            # Create sampler for transductive few-shot tasks
            sampler = CategoriesSampler_few_shot(self.args.batch_size,
                                                 self.args.k_eff, self.args.n_class, self.args.shots, self.args.n_query, force_query_size=True)
            sampler.create_list_classes(all_labels_support, all_labels_query)
            sampler_support = SamplerSupport_few_shot(sampler)
            sampler_query = SamplerQuery_few_shot(sampler)

            # Get the query and support samples at the indexes given by the samplers
            test_loader_query = []
            for indices in sampler_query:
                test_loader_query.append(
                    (all_features_query[indices, :], all_labels_query[indices]))

            test_loader_support = []
            for indices in sampler_support:
                test_loader_support.append(
                    (all_features_support[indices, :], all_labels_support[indices]))

            # Prepare the tasks
            task_generator = Tasks_Generator_few_shot(k_eff=self.args.k_eff, shot=self.args.shots, n_query=self.args.n_query,
                                                      n_class=self.args.n_class, loader_support=test_loader_support, loader_query=test_loader_query, model=model, args=self.args)
            tasks = task_generator.generate_tasks()

            # Load the method (e.g. EM_DIRICHLET)
            method = self.get_method_builder(
                model=model, device=self.device, args=self.args, log_file=self.log_file)
            # set the optimal parameter for the method if the test set is used
            if self.args.used_test_set == 'test' and self.args.tunable:
                self.set_method_opt_param()

            # Run task
            logs = method.run_task(task_dic=tasks, shot=self.args.shots)
            acc_mean, acc_conf = compute_confidence_interval(
                logs['acc'][:, -1])
            timestamps, criterions = logs['timestamps'], logs['criterions']
            results_task.append(acc_mean)
            results_task_time.append(timestamps)

            del method
            del tasks

        mean_accuracies = np.asarray(results_task).mean()
        mean_times = np.asarray(results_task_time).mean()

        return mean_accuracies, mean_times

    def report_results(self, mean_accuracies, mean_times):
        self.logger.info('----- Final results -----')

        if self.args.use_softmax_feature == True:
            word = '_softmax'
        else:
            word = '_visual'
        path = 'results_few_shot/{}/{}'.format(
            self.args.used_test_set, self.args.dataset)

        # If validation mode
        if self.args.used_test_set == 'val':
            self.get_method_val_param()
            name_file = path + \
                '/{}_s{}.txt'.format(self.args.name_method +
                                     word, self.args.shots)

            if not os.path.exists(path):
                os.makedirs(path)
            if os.path.isfile(name_file) == True:
                f = open(name_file, 'a')
            else:
                f = open(name_file, 'w')
                f.write('val_param' + '\t' + 'acc' + '\n')

            self.logger.info('{}-shot mean test accuracy over {} tasks: {}'.format(
                self.args.shots, self.args.number_tasks, mean_accuracies))

            f.write(str(self.val_param) + '\t')
            f.write(str(round(100 * mean_accuracies, 2)) + '\t')
            f.write('\n')
            f.close()

        # if test mode
        elif self.args.used_test_set == 'test' and self.args.save_results == True:
            var = str(self.args.shots) + '\t' + \
                str(self.args.n_query) + '\t' + str(self.args.k_eff)
            var_names = 'shots' + '\t' + 'n_query' + '\t' + 'k_eff' + '\t' + 'acc' + '\n'

            path = 'results_few_shot/{}/{}'.format(
                self.args.used_test_set, self.args.dataset)
            name_file = path + \
                '/{}_s{}.txt'.format(self.args.name_method +
                                     word, self.args.shots)

            if not os.path.exists(path):
                os.makedirs(path)
            if os.path.isfile(name_file) == True:
                f = open(name_file, 'a')
            else:
                f = open(name_file, 'w')
                f.write(var_names + '\t' + '\n')

            self.logger.info('{}-shot mean test accuracy over {} tasks: {}'.format(
                self.args.shots, self.args.number_tasks, mean_accuracies))
            self.logger.info('{}-shot mean time over {} tasks: {}'.format(
                self.args.shots, self.args.number_tasks, mean_times))
            f.write(str(var)+'\t')
            f.write(str(round(100 * mean_accuracies, 1)) + '\t')
            f.write('\n')
            f.close()

        else:
            self.logger.info('{}-shot mean test accuracy over {} tasks: {}'.format(
                self.args.shots, self.args.number_tasks, mean_accuracies))
            self.logger.info('{}-shot mean time over {} tasks: {}'.format(
                self.args.shots, self.args.number_tasks, mean_times))
