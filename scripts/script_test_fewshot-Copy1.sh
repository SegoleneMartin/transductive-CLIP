n_query=75
k_eff=5
used_test_set='test'
acc_clustering=False
device=0
s=16
######
for method in hard_em_dirichlet em_dirichlet #kl_kmeans em_gaussian em_gaussian_cov hard_kmeans soft_kmeans fuzzy_kmeans em_gaussian_cov
do
for dataset in imagenet sun397
do
python main.py --opts batch_size 1 number_tasks 500 dataset ${dataset} shots ${s} n_query ${n_query} k_eff ${k_eff} save_results True device ${device} seed 0 method ${method} used_test_set ${used_test_set} acc_clustering ${acc_clustering} use_softmax_feature True
done
for dataset in fgvcaircraft caltech101 food101 ucf101 flowers102 dtd eurosat oxfordpets stanfordcars
do
python main.py --opts batch_size 100 number_tasks 500 dataset ${dataset} shots ${s} n_query ${n_query} k_eff ${k_eff} save_results True device ${device} seed 0 method ${method} used_test_set ${used_test_set} acc_clustering ${acc_clustering} use_softmax_feature True
done
done
