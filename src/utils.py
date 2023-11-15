import torch
import time
import numpy as np
import shutil
from tqdm import tqdm
import logging
import os
import pickle
import torch.nn.functional as F
from typing import List
import yaml
from ast import literal_eval
import logging
import copy
import requests
import torch.nn as nn
from IPython.display import Image, display
from PIL import Image as pimage
from scipy.optimize import linear_sum_assignment
import clip  # pylint: disable=import-outside-toplevel

def get_one_hot(y_s):
    n_ways = torch.unique(y_s).size(0)
    eye = torch.eye(n_ways).to(y_s.device)
    one_hot = []
    for y_task in y_s:
        one_hot.append(eye[y_task].unsqueeze(0))
    one_hot = torch.cat(one_hot, 0)
    return one_hot

def get_one_hot_full(y_s, K):
    n_ways = K
    eye = torch.eye(n_ways).to(y_s.device)
    one_hot = []
    for y_task in y_s:
        one_hot.append(eye[y_task].unsqueeze(0))
    one_hot = torch.cat(one_hot, 0)
    return one_hot

def get_logs_path(model_path, method, shot):
    exp_path = '_'.join(model_path.split('/')[1:])
    file_path = os.path.join('tmp', exp_path, method)
    os.makedirs(file_path, exist_ok=True)
    return os.path.join(file_path, f'{shot}.txt')

def setup_logger(filepath):
    file_formatter = logging.Formatter(
        "[%(asctime)s %(filename)s:%(lineno)s] %(levelname)-8s %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger = logging.getLogger('example')
    # handler = logging.StreamHandler()
    # handler.setFormatter(file_formatter)
    # logger.addHandler(handler)

    file_handle_name = "file"
    if file_handle_name in [h.name for h in logger.handlers]:
        return
    if os.path.dirname(filepath) != '':
        if not os.path.isdir(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
    file_handle = logging.FileHandler(filename=filepath, mode="a")
    file_handle.set_name(file_handle_name)
    file_handle.setFormatter(file_formatter)
    logger.addHandler(file_handle)
    logger.setLevel(logging.DEBUG)
    return logger


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', folder='result/default'):
    os.makedirs(folder, exist_ok=True)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(folder + '/' + filename, folder + '/model_best.pth.tar')


def load_checkpoint(model, model_path, type='best'):
    if type == 'best':
        checkpoint = torch.load('{}/model_best.pth.tar'.format(model_path),
                                )
    elif type == 'last':
        checkpoint = torch.load('{}/checkpoint.pth.tar'.format(model_path),
                                )
    else:
        assert False, 'type should be in [best, or last], but got {}'.format(type)
    state_dict = checkpoint['state_dict']
    names = []
    for k, v in state_dict.items():
        names.append(k)
    model.load_state_dict(state_dict)


def compute_confidence_interval(data, axis=0):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a, axis=axis)
    std = np.std(a, axis=axis)
    pm = 1.96 * (std / np.sqrt(a.shape[axis]))
    return m, pm

class CfgNode(dict):
    """
    CfgNode represents an internal node in the configuration tree. It's a simple
    dict-like container that allows for attribute-based access to keys.
    """

    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        # Recursively convert nested dictionaries in init_dict into CfgNodes
        init_dict = {} if init_dict is None else init_dict
        key_list = [] if key_list is None else key_list
        for k, v in init_dict.items():
            if type(v) is dict:
                # Convert dict to CfgNode
                init_dict[k] = CfgNode(v, key_list=key_list + [k])
        super(CfgNode, self).__init__(init_dict)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __str__(self):
        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        r = ""
        s = []
        for k, v in sorted(self.items()):
            seperator = "\n" if isinstance(v, CfgNode) else " "
            attr_str = "{}:{}{}".format(str(k), seperator, str(v))
            attr_str = _indent(attr_str, 2)
            s.append(attr_str)
        r += "\n".join(s)
        return r

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, super(CfgNode, self).__repr__())


def _decode_cfg_value(v):
    if not isinstance(v, str):
        return v
    try:
        v = literal_eval(v)
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(replacement, original, key, full_key):
    original_type = type(original)

    replacement_type = type(replacement)

    # The types must match (with some exceptions)
    if replacement_type == original_type:
        return replacement

    def conditional_cast(from_type, to_type):
        if replacement_type == from_type and original_type == to_type:
            return True, to_type(replacement)
        else:
            return False, None

    casts = [(tuple, list), (list, tuple)]
    try:
        casts.append((str, unicode))  # noqa: F821
    except Exception:
        pass

    for (from_type, to_type) in casts:
        converted, converted_value = conditional_cast(from_type, to_type)
        if converted:
            return converted_value

    raise ValueError(
        "Type mismatch ({} vs. {}) with values ({} vs. {}) for config "
        "key: {}".format(
            original_type, replacement_type, original, replacement, full_key
        )
    )


def load_cfg_from_cfg_file(file: str):
    cfg = {}
    assert os.path.isfile(file) and file.endswith('.yaml'), \
        '{} is not a yaml file'.format(file)

    with open(file, 'r') as f:
        cfg_from_file = yaml.safe_load(f)

    for key in cfg_from_file:
        for k, v in cfg_from_file[key].items():
            cfg[k] = v

    cfg = CfgNode(cfg)
    return cfg


def merge_cfg_from_list(cfg: CfgNode,
                        cfg_list: List[str]):
    new_cfg = copy.deepcopy(cfg)
    assert len(cfg_list) % 2 == 0, cfg_list
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        subkey = full_key.split('.')[-1]
        #assert subkey in cfg, 'Non-existent key: {}'.format(full_key)
        if subkey in cfg:
            value = _decode_cfg_value(v)
            value = _check_and_coerce_cfg_value_type(
                value, cfg[subkey], subkey, full_key
            )
            setattr(new_cfg, subkey, value)
        else:
            value = _decode_cfg_value(v)
            setattr(new_cfg, subkey, value)
    return new_cfg

class Logger:
    def __init__(self, module_name, filename):
        self.module_name = module_name
        self.filename = filename
        self.formatter = self.get_formatter()
        self.file_handler = self.get_file_handler()
        self.stream_handler = self.get_stream_handler()
        self.logger = self.get_logger()

    def get_formatter(self):
        log_format = '[%(name)s]: [%(levelname)s]: %(message)s'
        formatter = logging.Formatter(log_format)
        return formatter

    def get_file_handler(self):
        file_handler = logging.FileHandler(self.filename)
        file_handler.setFormatter(self.formatter)
        return file_handler

    def get_stream_handler(self):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(self.formatter)
        return stream_handler

    def get_logger(self):
        logger = logging.getLogger(self.module_name)
        logger.setLevel(logging.INFO)
        logger.addHandler(self.file_handler)
        logger.addHandler(self.stream_handler)
        return logger

    def del_logger(self):
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)

    def info(self, msg):
        self.logger.info(msg)

    def debug(self, msg):
        self.logger.debug(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def critical(self, msg):
        self.logger.critical(msg)

    def exception(self, msg):
        self.logger.exception(msg)

def make_log_dir(log_path, dataset, method):
    log_dir = os.path.join(log_path, dataset, method)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    return log_dir

def get_log_file(log_path, dataset, method):
    log_dir = make_log_dir(log_path=log_path, dataset=dataset, method=method)
    i = 0
    filename = os.path.join(log_dir, '{}_run_{}.log'.format(method, i))
    while os.path.exists(os.path.join(log_dir, '{}_run_%s.log'.format(method)) % i):
        i += 1
        filename = os.path.join(log_dir, '{}_run_{}.log'.format(method, i))
    return filename
   

def download_image(url):
    urllib_request = urllib.request.Request(
        url,
        data=None,
        headers={"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"},
    )
    with urllib.request.urlopen(urllib_request, timeout=10) as r:
        img_stream = io.BytesIO(r.read())
    return img_stream


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)
    
def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)
        

def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)
    
    
def extract_features(model, dataset, loader, set_name, args, 
                          device, list_T=[10, 20, 30, 40, 50]):
    """
        inputs:
            model : The loaded model containing the feature extractor
            train_loader : Train data loader
            args : arguments
            device : GPU device

        returns :
            Saves the features in data/args.dataset/saved_features/ for T in list_T under the name 
            '{}_softmax_{}_T{}.plk'.format(set_name, args.backbone, T)
    """
    for T in list_T:
        # Check if features are already saved
        features_save_path = 'data/{}/saved_features/{}_softmax_{}_T{}.plk'.format(args.dataset, set_name, args.backbone, T)
        #features_save_path = 'data/{}/saved_features/{}_{}.plk'.format(args.dataset, set_name, args.backbone)
        if os.path.exists(features_save_path):
            print('Features already saved for split {} and T = {}, skipping'.format(set_name, T))
            continue
        else:
            print('Extracting features on {} for T = {}'.format(args.dataset,T))
    
        # Create text embeddings for all classes in the dataset
        text_features = clip_weights(model, dataset.classnames, dataset.template, device).float()
    
        # Extract features and labels
        for i, (images, labels) in enumerate(tqdm(loader)):
        
        #for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                image_features = model.encode_image(images).float()
                image_features /= image_features.norm(dim=-1, keepdim=True)
                similarity = (T * image_features @ text_features.T).softmax(dim=-1)
                if i == 0:
                    all_features = similarity.cpu()
                    #all_features = image_features
                    all_labels = labels.cpu()
                else:
                    all_features = torch.cat((all_features, similarity.cpu()), dim=0)
                    #all_features = torch.cat((all_features, image_features), dim=0)
                    all_labels = torch.cat((all_labels, labels.cpu()), dim=0)

        # Save features
        extracted_features_dic = {'concat_features': all_features, 'concat_labels':all_labels}
        try :
            os.mkdir('data/{}/saved_features/'.format(args.dataset))
        except:
            pass
        save_pickle(features_save_path, extracted_features_dic)


def clip_weights(model, classnames, template, device):
    
    new_classnames = []
    for classname in classnames:
        classname = classname.replace('_', ' ')
        new_classnames.append(classname)
    classnames = new_classnames
    
    text_inputs = torch.cat([clip.tokenize([template.format(classname) for classname in classnames])]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features


def compute_graph_matching(preds_q, probs, args):
        
        new_preds_q = torch.zeros_like(preds_q)
        n_task = preds_q.shape[0]
        list_clusters = []
        list_A = []
        
        for task in range(n_task):
            clusters = []
            num_clusters = len(torch.unique(preds_q[task]))
            A = np.zeros((num_clusters, int(args.n_ways)))
            for i, cluster in enumerate(preds_q[task]):
                if cluster.item() not in clusters:
                    A[len(clusters), :] = - probs[task, cluster].cpu().numpy()
                    clusters.append(cluster.item())
            list_A.append(A)
            list_clusters.append(clusters)

        for task in range(n_task):
            A = list_A[task]
            clusters = list_clusters[task]
            __, matching_classes = linear_sum_assignment(A, maximize=False)
            for i, cluster in enumerate(preds_q[task]):
                new_preds_q[task, i] = matching_classes[clusters.index(cluster)]
        
        return new_preds_q
        

def compute_basic_matching(preds_q, probs, args):
    
    new_preds_q = torch.zeros_like(preds_q)
    n_task = preds_q.shape[0]
    
    for task in range(n_task):
        matching_classes = probs[task].argmax(dim=-1) # K
        new_preds_q[task] = matching_classes[preds_q[task]]
        
    return new_preds_q
