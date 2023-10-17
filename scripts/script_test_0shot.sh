n_query=75
k_eff=5
used_test_set='test'
s=0
acc_clustering=True
device=0
######
method='hard_em_dirichlet'
for dataset in sun397
do
python main.py --opts batch_size 100 number_tasks 10000 dataset ${dataset} shots ${s} n_query ${n_query} k_eff ${k_eff} save_results True device ${device} seed 0 method ${method} used_test_set ${used_test_set} acc_clustering ${acc_clustering}
done
######
method='em_dirichlet'
for dataset in sun397
do
python main.py --opts batch_size 100 number_tasks 10000 dataset ${dataset} shots ${s} n_query ${n_query} k_eff ${k_eff} save_results True device ${device} seed 0 method ${method} used_test_set ${used_test_set} acc_clustering ${acc_clustering}
done
######
method='kl_kmeans'
for dataset in sun397
do
python main.py --opts batch_size 100 number_tasks 10000 dataset ${dataset} shots ${s} n_query ${n_query} k_eff ${k_eff} save_results True device ${device} seed 0 method ${method} used_test_set ${used_test_set} acc_clustering ${acc_clustering}
done
######
method='fuzzy_kmeans'
for dataset in sun397
do
python main.py --opts batch_size 100 number_tasks 10000 dataset ${dataset} shots ${s} n_query ${n_query} k_eff ${k_eff} save_results True device ${device} seed 0 method ${method} used_test_set ${used_test_set} acc_clustering ${acc_clustering}
done
