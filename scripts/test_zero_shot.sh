n_query=75
used_test_set='test'
s=0
device=0
######
for method in hard_em_dirichlet em_dirichlet soft_kmeans em_gaussian_cov kl_kmeans em_gaussian hard_kmeans clip_inductive
do
for dataset in food101
do
python main.py --opts batch_size 100 number_tasks 1000 dataset ${dataset} shots ${s} n_query ${n_query} save_results True device ${device} seed 0 method ${method} used_test_set ${used_test_set} use_softmax_feature True
done
done
###
for method in hard_em_dirichlet em_dirichlet soft_kmeans em_gaussian_cov kl_kmeans em_gaussian hard_kmeans
do
for dataset in food101
do
python main.py --opts batch_size 100 number_tasks 1000 dataset ${dataset} shots ${s} n_query ${n_query} save_results True device ${device} seed 0 method ${method} used_test_set ${used_test_set} use_softmax_feature False
done
done