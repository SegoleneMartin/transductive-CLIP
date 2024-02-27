n_query=1200
k_eff=80
used_test_set='test'
s=0
acc_clustering=True
device=0
######
for method in hard_em_dirichlet 
do
for dataset in imagenet
do
python main.py --opts batch_size 1 number_tasks 10 dataset ${dataset} shots ${s} n_query ${n_query} k_eff ${k_eff} save_results True device ${device} seed 0 method ${method} used_test_set ${used_test_set} acc_clustering ${acc_clustering} use_softmax_feature True
done
done
###
n_query=150
k_eff=10
used_test_set='test'
s=0
acc_clustering=True
device=0
######
for method in hard_em_dirichlet 
do
for dataset in imagenet
do
python main.py --opts batch_size 25 number_tasks 1000 dataset ${dataset} shots ${s} n_query ${n_query} k_eff ${k_eff} save_results True device ${device} seed 0 method ${method} used_test_set ${used_test_set} acc_clustering ${acc_clustering} use_softmax_feature True
done
done
###
n_query=300
k_eff=20
used_test_set='test'
s=0
acc_clustering=True
device=0
######
for method in hard_em_dirichlet 
do
for dataset in imagenet
do
python main.py --opts batch_size 25 number_tasks 1000 dataset ${dataset} shots ${s} n_query ${n_query} k_eff ${k_eff} save_results True device ${device} seed 0 method ${method} used_test_set ${used_test_set} acc_clustering ${acc_clustering} use_softmax_feature True
done
done
###
n_query=600
k_eff=40
used_test_set='test'
s=0
acc_clustering=True
device=0
######
for method in hard_em_dirichlet 
do
for dataset in imagenet
do
python main.py --opts batch_size 10 number_tasks 1000 dataset ${dataset} shots ${s} n_query ${n_query} k_eff ${k_eff} save_results True device ${device} seed 0 method ${method} used_test_set ${used_test_set} acc_clustering ${acc_clustering} use_softmax_feature True
done
done
###
n_query=1200
k_eff=80
used_test_set='test'
s=0
acc_clustering=True
device=0
######
for method in hard_em_dirichlet 
do
for dataset in imagenet
do
python main.py --opts batch_size 1 number_tasks 1000 dataset ${dataset} shots ${s} n_query ${n_query} k_eff ${k_eff} save_results True device ${device} seed 0 method ${method} used_test_set ${used_test_set} acc_clustering ${acc_clustering} use_softmax_feature True
done
done
###