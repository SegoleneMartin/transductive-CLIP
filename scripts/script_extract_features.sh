n_query=75
k_eff=5
used_test_set='val'
s=0
acc_clustering=True
device=0
######
method='hard_em_dirichlet'
for dataset in sun397 caltech101 food101 ucf101 flowers102 dtd eurosat fgvcaircraft oxfordpets sun397 stanfordcars imagenet
do
python main.py --opts batch_size 1 number_tasks 1 dataset ${dataset} shots ${s} device ${device}
done