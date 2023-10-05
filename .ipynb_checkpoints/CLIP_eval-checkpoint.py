from IPython.display import Image, display
from PIL import Image as pimage
import torch
import clip  # pylint: disable=import-outside-toplevel
import torch
import urllib
import numpy as np
from src.utils import save_pickle, load_pickle
import os
import torch.nn.functional as F
from torchvision import datasets
import pandas as pd
from tqdm import tqdm

device = "cuda"
dataset = 'food101'
path_features = 'data/{}/saved_features/test_softmax_RN50_T100.plk'.format(dataset)

saved_features = load_pickle(path_features)
preds_q = saved_features['concat_features'].argmax(-1).to(device)
y_q = saved_features['concat_labels'].to(device)
acc = (preds_q == y_q).float().mean()
print(acc)