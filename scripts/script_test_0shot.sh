n_query=75
k_eff=5
used_test_set='test'
s=0
acc_clustering=True
device=0
######
for method in hard_em_dirichlet em_dirichlet kl_kmeans em_gaussian em_gaussian_cov hard_kmeans soft_kmeans fuzzy_kmeans em_gaussian_cov
do
for dataset in caltech101 food101
do
python main.py --opts batch_size 100 number_tasks 1000 dataset ${dataset} shots ${s} n_query ${n_query} k_eff ${k_eff} save_results True device ${device} seed 0 method ${method} used_test_set ${used_test_set} acc_clustering ${acc_clustering} use_softmax_feature True
done
done