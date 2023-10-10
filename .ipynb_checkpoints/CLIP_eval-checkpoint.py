# just a simple way to evaluate CLIP zero shot from the saved features

from src.utils import save_pickle, load_pickle

device = "cuda"
dataset = 'food101'
path_features = 'data/{}/saved_features/test_softmax_RN50_T100.plk'.format(dataset)

saved_features = load_pickle(path_features)
preds_q = saved_features['concat_features'].argmax(-1).to(device)
y_q = saved_features['concat_labels'].to(device)
acc = (preds_q == y_q).float().mean()
print(acc)