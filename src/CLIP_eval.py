import numpy as np
from utils import load_pickle


device = "cuda"
#dataset = 'fgvcaircraft'
for dataset in ['fgvcaircraft', 'oxfordpets', 'eurosat', 'ucf101', 'caltech101', 'dtd', 'fgvcaircraft', 'food101', 'flowers102', 'stanfordcars', 'imagenet', 'sun397']:
    print('DATASET', dataset)
    path_features = 'data/{}/saved_features/test_softmax_RN50_T10.plk'.format(dataset)

    saved_features = load_pickle(path_features)
    preds_q = saved_features['concat_features'].argmax(-1).to(device)
    y_q = saved_features['concat_labels'].to(device)
    #print(preds_q.shape, y_q.shape)
    acc = (preds_q == y_q).float().mean()
    print(acc)
