n_query=75
k_eff=5
used_test_set='test'
acc_clustering=False
device=0
######
method='hard_em_dirichlet'
for dataset in caltech101 food101
do
for s in 1 2 4 8 16
do
python main.py --opts batch_size 100 number_tasks 1000 dataset ${dataset} shots ${s} n_query ${n_query} k_eff ${k_eff} save_results True device ${device} seed 0 method ${method} used_test_set ${used_test_set} acc_clustering ${acc_clustering} use_softmax_feature True
done
done
######
method='em_dirichlet'
for dataset in caltech101 food101
do
for s in 1 2 4 8 16
do
python main.py --opts batch_size 100 number_tasks 1000 dataset ${dataset} shots ${s} n_query ${n_query} k_eff ${k_eff} save_results True device ${device} seed 0 method ${method} used_test_set ${used_test_set} acc_clustering ${acc_clustering} use_softmax_feature True
done
done
######
method='alpha_tim'
for dataset in caltech101 food101
do
for s in 1 2 4 8 16
do
python main.py --opts batch_size 100 number_tasks 1000 dataset ${dataset} shots ${s} n_query ${n_query} k_eff ${k_eff} save_results True device ${device} seed 0 method ${method} used_test_set ${used_test_set} acc_clustering ${acc_clustering} use_softmax_feature True
done
done
######
method='bdcspn'
for dataset in caltech101 food101
do
for s in 1 2 4 8 16
do
python main.py --opts batch_size 100 number_tasks 1000 dataset ${dataset} shots ${s} n_query ${n_query} k_eff ${k_eff} save_results True device ${device} seed 0 method ${method} used_test_set ${used_test_set} acc_clustering ${acc_clustering} use_softmax_feature True
done
done
######
method='paddle'
for dataset in caltech101 food101
do
for s in 1 2 4 8 16
do
python main.py --opts batch_size 100 number_tasks 1000 dataset ${dataset} shots ${s} n_query ${n_query} k_eff ${k_eff} save_results True device ${device} seed 0 method ${method} used_test_set ${used_test_set} acc_clustering ${acc_clustering} use_softmax_feature True
done
done
######
method='laplacian_shot'
for dataset in caltech101 food101
do
for s in 1 2 4 8 16
do
python main.py --opts batch_size 100 number_tasks 1000 dataset ${dataset} shots ${s} n_query ${n_query} k_eff ${k_eff} save_results True device ${device} seed 0 method ${method} used_test_set ${used_test_set} acc_clustering ${acc_clustering} use_softmax_feature True
done
done