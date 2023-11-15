# -*- coding: utf-8 -*-
import numpy as np
from src.utils import compute_confidence_interval, Logger
from src.methods.fuzzy_kmeans import FUZZY_KMEANS
from src.methods.kl_kmeans import KL_KMEANS
#from src.methods.em_dirichlet_newton import EM_DIRICHLET
from src.methods.em_dirichlet import EM_DIRICHLET
from src.methods.em_gaussian import EM_GAUSSIAN
from src.methods.em_gaussian_cov import EM_GAUSSIAN_COV
from src.methods.paddle import PADDLE
from src.methods.bdcspn import BDCSPN
from src.methods.soft_kmeans import SOFT_KMEANS
from src.methods.hard_kmeans import HARD_KMEANS
from src.methods.tim import ALPHA_TIM, TIM_GD
from src.methods.hard_em_dirichlet import HARD_EM_DIRICHLET
from src.methods.inductive_clip import CLIP
from src.methods.laplacian_shot import LAPLACIAN_SHOT
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
        dataset = dataset_list[self.args.dataset](self.args.dataset_path)
        self.args.classnames = dataset.classnames
        self.args.template = dataset.template
        train_loader = build_data_loader(data_source=dataset.train_x, batch_size=1024, is_train=False, shuffle=False, tfm=preprocess)
        val_loader = build_data_loader(data_source=dataset.val, batch_size=1024, is_train=False, shuffle=False, tfm=preprocess)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=1024, is_train=False, shuffle=False, tfm=preprocess)

        # Extract features of query, support and val for all the temperatures (if they do not already exist)
        extract_features(model, dataset, test_loader, 'test', self.args, self.device, list_T=[self.args.T])
        extract_features(model, dataset, val_loader, 'val', self.args, self.device, list_T=[self.args.T])
        extract_features(model, dataset, train_loader, 'train', self.args, self.device, list_T=[self.args.T])

        # Load the features for the given temperature
        if self.args.used_test_set == 'test' and self.args.shots > 0 and self.args.tunable == True:  # if the inference is on the test set, set the temperature to the optimal one found during validation
            if self.args.use_softmax_feature == True:
                word = ''
            else:
                word = '_visual'
            path = 'results_T_fixed_fewshot/val/{}'.format(self.args.dataset)
            name_file = path + '/{}_s{}.txt'.format(self.args.name_method + word, self.args.shots)
            
            if self.args.dataset == 'imagenet':
                path = 'results_T_fixed_fewshot/val/{}'.format('caltech101')
                name_file = path + '/{}_s{}.txt'.format(self.args.name_method + word, self.args.shots)
                
            print(" path", path)
            try:
                f =  open(name_file, 'r')
                list_param, list_acc = [], []
                for i, line in enumerate(f):
                    if i < 2 :
                        continue
                    line = line.split('\t')
                    list_param.append(float(line[0]))
                    list_acc.append(float(line[1]))
                list_acc = np.array(list_acc)
                index = np.argwhere(list_acc == np.amax(list_acc))[-1][0]
                opt_param = list_param[index]
                self.set_method_opt_param(opt_param)
                
            except:
                
                raise ValueError("The optimal parameter was not found. Please make sure you have performed the tuning of the parameter on the validation set.")
                   
        if self.args.used_test_set == 'val':
            self.args.used_train_set = 'val'
        else:
            self.args.used_train_set = 'train'

        if self.args.use_softmax_feature == True:
            filepath_support = 'data/{}/saved_features/{}_softmax_{}_T{}.plk'.format(self.args.dataset, self.args.used_train_set, self.args.backbone, self.args.T)
            filepath_query = 'data/{}/saved_features/{}_softmax_{}_T{}.plk'.format(self.args.dataset, self.args.used_test_set, self.args.backbone, self.args.T)
        else:
            filepath_support = 'data/{}/saved_features/{}_{}.plk'.format(self.args.dataset, self.args.used_train_set, self.args.backbone)
            filepath_query = 'data/{}/saved_features/{}_{}.plk'.format(self.args.dataset, self.args.used_test_set, self.args.backbone)


        extracted_features_dic_support = load_pickle(filepath_support)
        extracted_features_dic_query = load_pickle(filepath_query)

        all_features_support = extracted_features_dic_support['concat_features'].to('cpu')
        all_labels_support = extracted_features_dic_support['concat_labels'].long().to('cpu')
        all_features_query = extracted_features_dic_query['concat_features'].to('cpu')
        all_labels_query = extracted_features_dic_query['concat_labels'].long().to('cpu')
    
    
        self.logger.info("=> Runnning full evaluation with method: {}".format(self.args.name_method))

        results = []
        results_time = []
  
        if self.args.shots == 0:
            shot = 1  # ugly solution for now because the code was not really adapted to 0 shot yet
        else:
            shot = self.args.shots
            
        results_task = []
        results_task_time = []
        for i in range(int(self.args.number_tasks/self.args.batch_size)):

            # create sampler for transductive few-shot tasks
            sampler = CategoriesSampler(all_labels_support, all_labels_query, self.args.batch_size,
                                    self.args.k_eff, self.args.n_ways, shot, self.args.n_query, 
                                    self.args.sampling, force_query_size=True)
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
            task_generator = Tasks_Generator(k_eff=self.args.k_eff, n_ways=self.args.n_ways, shot=shot, n_query=self.args.n_query, loader_support=test_loader_support, loader_query=test_loader_query, model=model, args=self.args)
            tasks = task_generator.generate_tasks()

            # Load the method (e.g. EM_DIRICHLET)
            method = self.get_method_builder(model=model)

            # Run task
            logs = method.run_task(task_dic=tasks, shot=shot)
            acc_mean, acc_conf = compute_confidence_interval(logs['acc'][:, -1])
            timestamps, criterions = logs['timestamps'], logs['criterions']
            results_task.append(acc_mean)
            results_task_time.append(timestamps)
            del method
            del tasks
        results.append(results_task)
        results_time.append(results_task_time)

        mean_accuracies = np.asarray(results).mean(1)
        mean_times = np.asarray(results_time).mean(1)
    
        self.logger.info('----- Final results -----')
        
        
        ## If validation mode, report results
        if self.args.used_test_set == 'val': 

            self.get_method_val_param()
                        
            if self.args.use_softmax_feature == True:
                word = ''
            else:
                word = '_visual'
            path = 'results_T_fixed_fewshot/{}/{}'.format(self.args.used_test_set, self.args.dataset)
            name_file = path + '/{}_s{}.txt'.format(self.args.name_method  + word, self.args.shots)

            if not os.path.exists(path):
                os.makedirs(path)
            if os.path.isfile(name_file) == True:
                f = open(name_file, 'a')
                #print('Adding to already existing .txt file to avoid overwritting')
            else:
                f = open(name_file, 'w')
                f.write('val_param' + '\t' + 'acc' + '\n')
                
            self.logger.info('{}-shot mean test accuracy over {} tasks: {}'.format(self.args.shots, self.args.number_tasks, mean_accuracies[0]))
            
            f.write(str(self.val_param) + '\t')
            f.write(str(round(100 * mean_accuracies[0], 2)) + '\t' )
            f.write('\n')
            f.close()
    
    
        elif  self.args.used_test_set == 'test' and self.args.save_results == True:
            
            # Report results in .txt files
            var = str(self.args.shots) + '\t' + str(self.args.n_query) + '\t' + str(self.args.k_eff) 
            var_names = 'shots' + '\t' + 'n_query' + '\t' + 'k_eff' + '\t' + 'acc' + '\n'
           
            #if self.args.graph_matching == True:
            #    word = 'graph'
            #else:
            #    word = 'basic'
            if self.args.use_softmax_feature == True:
                word = ''
            else:
                word = '_visual'
            #path = 'results_T_fixed_fewshot/{}/{}'.format(self.args.used_test_set, self.args.dataset)
            path = 'results_query/{}/{}'.format(self.args.used_test_set, self.args.dataset)
            name_file = path + '/{}_s{}.txt'.format(self.args.name_method + word, self.args.shots)

            if not os.path.exists(path):
                os.makedirs(path)
            if os.path.isfile(name_file) == True:
                f = open(name_file, 'a')
                #print('Adding to already existing .txt file to avoid overwritting')
            else:
                f = open(name_file, 'w')
                f.write(var_names +'\t' + '\n')
                
            self.logger.info('{}-shot mean test accuracy over {} tasks: {}'.format(self.args.shots, self.args.number_tasks, mean_accuracies[0]))
            self.logger.info('{}-shot mean time over {} tasks: {}'.format(self.args.shots, self.args.number_tasks, mean_times[0][0]))
            f.write(str(var)+'\t')
            f.write(str(round(100 * mean_accuracies[0], 1)) +'\t' )
            f.write('\n')
            f.close()
            
        else:
            self.logger.info('{}-shot mean test accuracy over {} tasks: {}'.format(self.args.shots, self.args.number_tasks, mean_accuracies[0]))
            self.logger.info('{}-shot mean time over {} tasks: {}'.format(self.args.shots, self.args.number_tasks, mean_times[0][0]))
            
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
        elif self.args.name_method == 'BDCSPN':
            method_builder = BDCSPN(**method_info)
        elif self.args.name_method == 'LAPLACIAN_SHOT':
            method_builder = LAPLACIAN_SHOT(**method_info)
        elif self.args.name_method == 'EM_GAUSSIAN':
            method_builder = EM_GAUSSIAN(**method_info)
        elif self.args.name_method == 'EM_GAUSSIAN_COV':
            method_builder = EM_GAUSSIAN_COV(**method_info)
        elif self.args.name_method == 'SOFT_KMEANS':
            method_builder = SOFT_KMEANS(**method_info)
        elif self.args.name_method == 'HARD_KMEANS':
            method_builder = HARD_KMEANS(**method_info)
        elif self.args.name_method == 'ALPHA_TIM':
            method_builder = ALPHA_TIM(**method_info)
        elif self.args.name_method == 'CLIP_LINEAR_PROBE':
            method_builder = CLIP_LINEAR_PROBE(**method_info)
        elif self.args.name_method == 'CLIP':
            method_builder = CLIP(**method_info)
        else:
            self.logger.exception("The method your entered does not exist. Please check the spelling.")
            raise ValueError("The method your entered does not exist. Please check the spelling")
        return method_builder
    
    
    def get_method_val_param(self):
        # fixes for each method the name of the parameter on which validation is performed
        if self.args.name_method ==  'LAPLACIAN_SHOT':
            self.val_param = self.args.lmd
        elif self.args.name_method == 'ALPHA_TIM':
            self.val_param = self.args.alpha_value
        elif self.args.name_method == 'PADDLE':
            self.val_param = self.args.lambd
        elif self.args.name_method == 'BDCSPN':
            self.val_param = self.args.temp
            
            
    def set_method_opt_param(self, opt_param):
        # fixes for each method the name of the parameter on which validation is performed
        if self.args.name_method  ==  'LAPLACIAN_SHOT':
            self.args.lmd = opt_param
        elif self.args.name_method == 'ALPHA_TIM':
            self.args.alpha_value = opt_param
        elif self.args.name_method == 'PADDLE':
            self.args.lambd = opt_param
        elif self.args.name_method == 'BDCSPN':
            self.args.temp = opt_param
            
