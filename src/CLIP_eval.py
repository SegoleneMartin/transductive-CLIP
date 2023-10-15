from IPython.display import Image, display
from clip_retrieval.clip_client import ClipClient, Modality
from PIL import Image as pimage
import torch
import clip  # pylint: disable=import-outside-toplevel
import torch
import urllib
import numpy as np
from utils import get_image_emb, save_pickle, load_pickle
import os
import torch.nn.functional as F
from torchvision.datasets import CIFAR100
from torchvision import datasets
import pandas as pd
from tqdm import tqdm

device = "cuda"
dataset = 'fgvcaircraft'
path_features = 'data/{}/saved_features/query_softmax_RN50_T20.plk'.format(dataset)

saved_features = load_pickle(path_features)
preds_q = saved_features['concat_features'].argmax(-1).to(device)
y_q = saved_features['concat_labels'].to(device)
print(preds_q.shape, y_q.shape)
acc = (preds_q == y_q).float().mean()
print(acc)
