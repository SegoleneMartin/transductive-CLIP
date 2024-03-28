use_softmax_features=True
method='hard_em_dirichlet'
######
for dataset in sun397 caltech101 food101 ucf101 flowers102 dtd eurosat fgvcaircraft oxfordpets sun397 stanfordcars imagenet
do
python main.py --opts batch_size 1 number_tasks 1 dataset ${dataset} shots 0 use_softmax_features ${use_softmax_features} method ${method} use_features True
done