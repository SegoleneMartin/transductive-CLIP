n_query=75
k_eff=5
used_test_set='val'
s=0
acc_clustering=True
device=0
######
method='hard_em_dirichlet'
for dataset in sun397
do
for T in 1 2 3 4 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100
do
python main.py --opts batch_size 100 number_tasks 100 dataset ${dataset} T ${T} shots ${s} n_query ${n_query} k_eff ${k_eff} save_results True device ${device} seed 0 method ${method} used_test_set ${used_test_set} acc_clustering ${acc_clustering}
done
done
######
method='em_dirichlet'
for dataset in sun397
do
for T in 1 2 3 4 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100
do
python main.py --opts batch_size 100 number_tasks 100 dataset ${dataset} T ${T} shots ${s} n_query ${n_query} k_eff ${k_eff} save_results True device ${device} seed 0 method ${method} used_test_set ${used_test_set} acc_clustering ${acc_clustering}
done
done
######
method='kl_kmeans'
for dataset in sun397
do
for T in 1 2 3 4 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100
do
python main.py --opts batch_size 100 number_tasks 100 dataset ${dataset} T ${T} shots ${s} n_query ${n_query} k_eff ${k_eff} save_results True device ${device} seed 0 method ${method} used_test_set ${used_test_set} acc_clustering ${acc_clustering}
done
done
######
method='fuzzy_kmeans'
for dataset in sun397
do
for T in 1 2 3 4 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100
do
python main.py --opts batch_size 100 number_tasks 100 dataset ${dataset} T ${T} shots ${s} n_query ${n_query} k_eff ${k_eff} save_results True device ${device} seed 0 method ${method} used_test_set ${used_test_set} acc_clustering ${acc_clustering}
done
done
