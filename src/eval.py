# -*- coding: utf-8 -*-
import numpy as np
from src.utils import compute_confidence_interval, Logger
from src.methods.fuzzy_kmeans import FUZZY_KMEANS
from src.methods.kl_kmeans import KL_KMEANS
from src.methods.em_dirichlet import EM_DIRICHLET
from src.methods.paddle import PADDLE
from src.methods.tim import ALPHA_TIM, TIM_GD
from src.methods.hard_em_dirichlet import HARD_EM_DIRICHLET
from src.methods.clip_inductive import CLIP
from src.methods.clip_linear_probe import CLIP_LINEAR_PROBE
from src.datasets import Tasks_Generator, SamplerSupport, SamplerQuery, CategoriesSampler, build_data_loader
from src.datasets import OxfordPets, EuroSAT, UCF101, Caltech101, DescribableTextures, FGVCAircraft, Food101, Flowers102, StanfordCars, ImageNet, SUN397
import os
from src.utils import load_pickle, extract_features

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


class Evaluator:
    def __init__(self, device, args, log_file):
        self.device = device
        self.args = args
        self.log_file = log_file
        self.logger = Logger(__name__, self.log_file)
                     

    def run_full_evaluation(self, model, preprocess):
        """
        Run the evaluation over all the tasks
        inputs:
            model : The loaded model containing the feature extractor
            args : All parameters

        returns :
            results : List of the mean accuracy for each number of support shots
        """
        model.eval()
        
        # Define the data loaders
        #dataset = dataset_list[self.args.dataset](self.args.dataset_path)
        #train_loader = build_data_loader(data_source=dataset.train_x, batch_size=1024, is_train=False, shuffle=False, tfm=preprocess)
        #val_loader = build_data_loader(data_source=dataset.val, batch_size=1024, is_train=False, shuffle=False, tfm=preprocess)
        #test_loader = build_data_loader(data_source=dataset.test, batch_size=1024, is_train=False, shuffle=False, tfm=preprocess)

        # Extract features of query, support and val for all the temperatures (if they do not already exist)
        #extract_features(model, dataset, test_loader, 'test', self.args, self.device)
        #extract_features(model, dataset, val_loader, 'val', self.args, self.device)
        #extract_features(model, dataset, train_loader, 'train', self.args, self.device)

        # Load the features for the given temperature
        if self.args.used_test_set == 'test':  # if the inference is on the test set, set the temperature to the optimal one
            self.args.T = self.args.T_opts[self.args.dataset]

        if self.args.method == 'clip_linear_probe': # use image embeddings as feature vectors
            filepath_support = 'data/{}/saved_features/train_{}.plk'.format(self.args.dataset, self.args.backbone)
            filepath_query = 'data/{}/saved_features/{}_{}.plk'.format(self.args.dataset, self.args.used_test_set, self.args.backbone)
        else: # use softmaxs as feature vectors
            filepath_support = 'data/{}/saved_features/train_softmax_{}_T{}.plk'.format(self.args.dataset, self.args.backbone, self.args.T)
            filepath_query = 'data/{}/saved_features/{}_softmax_{}_T{}.plk'.format(self.args.dataset, self.args.used_test_set, self.args.backbone, self.args.T)

        extracted_features_dic_support = load_pickle(filepath_support)
        extracted_features_dic_query = load_pickle(filepath_query)

        all_features_support = extracted_features_dic_support['concat_features'].to('cpu')
        all_labels_support = extracted_features_dic_support['concat_labels'].long().to('cpu')
        all_features_query = extracted_features_dic_query['concat_features'].to('cpu')
        all_labels_query = extracted_features_dic_query['concat_labels'].long().to('cpu')
    
        self.logger.info("=> Runnning full evaluation with method: {}".format(self.args.name_method))

        results = []
  
        if self.args.shots == 0:
            shot = 1  # ugly solution for now because the code was not really adapted to 0 shot yet
        else:
            shot = self.args.shots
            
        results_task = []
        for i in range(int(self.args.number_tasks/self.args.batch_size)):
            if (self.args.dataset == 'stanfordcars' or self.args.dataset == 'sun397') and self.args.used_test_set == 'val': # the validation set of stanford cars does not contain enough samples
                force_query_size = True
            else:
                force_query_size = False

            # create sampler for transductive few-shot tasks
            sampler = CategoriesSampler(all_labels_support, all_labels_query, self.args.batch_size,
                                    self.args.k_eff, self.args.n_ways, shot, self.args.n_query, 
                                    self.args.sampling, force_query_size)
            sampler.create_list_classes(all_labels_support, all_labels_query)
            sampler_support = SamplerSupport(sampler)
            sampler_query = SamplerQuery(sampler)

            # get the query and support samples at the indexes given by the samplers
            test_loader_query = []
            for indices in sampler_query :
                test_loader_query.append((all_features_query[indices,:], all_labels_query[indices]))

            test_loader_support = []
            for indices in sampler_support :
                test_loader_support.append((all_features_support[indices,:], all_labels_support[indices]))
            
            # Prepare the tasks
            task_generator = Tasks_Generator(k_eff=self.args.k_eff, n_ways=self.args.n_ways, shot=shot, n_query=self.args.n_query, loader_support=test_loader_support, loader_query=test_loader_query, model=model)
            tasks = task_generator.generate_tasks()

            # Load the method (e.g. EM_DIRICHLET)
            # select the optimal lambda (will probably be removed soon as finally there is no need for tuning lambda)
            if self.args.method == 'em_dirichlet' or self.args.method == 'hard_em_dirichlet':
                if self.args.lambd_opt == False:
                    self.args.lambd = int(self.args.num_classes_test / self.args.k_eff) * self.args.n_query * self.args.fact # grid search on args.fact
                else:
                    self.args.lambd = int(self.args.num_classes_test /  self.args.k_eff) *  self.args.n_query *  self.args.fact_opts[self.args.dataset] # use the optimal fact given in the config file of the method
            method = self.get_method_builder(model=model)

            # Run task
            logs = method.run_task(task_dic=tasks, shot=shot)
            acc_mean, acc_conf = compute_confidence_interval(logs['acc'][:, -1])
            timestamps, criterions = logs['timestamps'], logs['criterions']
            # print(timestamps, criterions)
            results_task.append(acc_mean)
            del method
            del tasks
        results.append(results_task)

        mean_accuracies = np.asarray(results).mean(1)
    
        # Report results in .txt files
        if self.args.name_method in ['FUZZY_KMEANS', 'KL_KMEANS']:
                param = str(self.args.shots) + '\t' + str(self.args.n_query) + '\t' + str(self.args.k_eff) + '\t' + str(self.args.T)
                param_names = 'shots' + '\t' + 'n_query' + '\t' + 'k_eff' + '\t' + 'T'
        if self.args.name_method in ['EM_DIRICHLET', 'HARD_EM_DIRICHLET']:
                param = str(self.args.shots) + '\t' + str(self.args.n_query) + '\t' + str(self.args.k_eff) + '\t' + str(self.args.iter_mm) + '\t' + str(self.args.lambd) + '\t' + str(self.args.T)
                param_names ='shots' + '\t' + 'n_query' + '\t' + 'k_eff' + '\t' + 'iter_mm' + '\t' + 'lambd' + '\t' + 'T'
        if self.args.name_method == 'CLIP_LINEAR_PROBE':
                param = str(self.args.shots)+  '\t' + str(self.args.k_eff)  + '\t' + str(self.args.n_query)
                param_names = 'shots' + '\t' + 'k_eff' + '\t' + 'n_query'       
            
        self.logger.info('----- Final test results -----')
        
        if self.args.save_results == True:
           
            path = 'results/{}/{}'.format(self.args.used_test_set, self.args.dataset)
            name_file = path + '/{}.txt'.format(self.args.name_method)

            if not os.path.exists(path):
                os.makedirs(path)
            if os.path.isfile(name_file) == True:
                f = open(name_file, 'a')
                #print('Adding to already existing .txt file to avoid overwritting')
            else:
                f = open(name_file, 'w')
                f.write(param_names +'\t' + '\n')
                
            self.logger.info('{}-shot mean test accuracy over {} tasks: {}'.format(self.args.shots, self.args.number_tasks,
                                                                                    mean_accuracies[0]))
            f.write(str(param)+'\t')
            f.write(str(round(100 * mean_accuracies[0], 1)) +'\t' )
            f.write('\n')
            f.close()
            
        else:
            self.logger.info('{}-shot mean test accuracy over {} tasks: {}'.format(self.args.shots, self.args.number_tasks, mean_accuracies[0]))
            

        return mean_accuracies

    def get_method_builder(self, model):
        # Initialize method classifier builder
        method_info = {'model': model, 'device': self.device, 'log_file': self.log_file, 'args': self.args}
        if self.args.name_method == 'FUZZY_KMEANS':
            method_builder = FUZZY_KMEANS(**method_info)
        elif self.args.name_method == 'KL_KMEANS':
            method_builder = KL_KMEANS(**method_info)
        elif self.args.name_method == 'EM_DIRICHLET':
            method_builder = EM_DIRICHLET(**method_info)
        elif self.args.name_method == 'HARD_EM_DIRICHLET':
            method_builder = HARD_EM_DIRICHLET(**method_info)
        elif self.args.name_method == 'PADDLE':
            method_builder = PADDLE(**method_info)
        elif self.args.name_method == 'ALPHA_TIM':
            method_builder = ALPHA_TIM(**method_info)
        elif self.args.name_method == 'CLIP_LINEAR_PROBE':
            method_builder = CLIP_LINEAR_PROBE(**method_info)
        else:
            self.logger.exception("Method must be in ['FUZZY_KMEANS', 'KL_KMEANS', 'EM_DIRICHLET', 'HARD_EM_DIRICHLET', 'CLIP_LINEAR_PROBE']")
            raise ValueError("Method must be in ['FUZZY_KMEANS', 'KL_KMEANS', 'EM_DIRICHLET', 'HARD_EM_DIRICHLET', 'CLIP_LINEAR_PROBE']")
        return method_builder
