n_query=35
k_eff=5
used_test_set='val'
acc_clustering=False
device=0
######
method='alpha_tim'
for dataset in flowers102 ucf101 fgvcaircraft
do
for s in 1 2 4 8 16
do
for alpha_value in 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 1.0 1.1
do
python main.py --opts batch_size 5 number_tasks 5 dataset ${dataset} alpha_value ${alpha_value} shots ${s} n_query ${n_query} k_eff ${k_eff} save_results True device ${device} seed 0 method ${method} used_test_set ${used_test_set} acc_clustering ${acc_clustering} use_softmax_feature True
python main.py --opts batch_size 5 number_tasks 5 dataset ${dataset} alpha_value ${alpha_value} shots ${s} n_query ${n_query} k_eff ${k_eff} save_results True device ${device} seed 0 method ${method} used_test_set ${used_test_set} acc_clustering ${acc_clustering} use_softmax_feature False
done
done
done
######
method='bdcspn'
for dataset in flowers102 ucf101 fgvcaircraft
do
for s in 1 2 4 8 16
do
for temp in 1.0 3.0 5.0 10.0 20.0 30.0 40.0 50.0 60.0
do
python main.py --opts batch_size 5 number_tasks 5 dataset ${dataset} temp ${temp} shots ${s} n_query ${n_query} k_eff ${k_eff} save_results True device ${device} seed 0 method ${method} used_test_set ${used_test_set} acc_clustering ${acc_clustering} use_softmax_feature True
python main.py --opts batch_size 5 number_tasks 5 dataset ${dataset} temp ${temp} shots ${s} n_query ${n_query} k_eff ${k_eff} save_results True device ${device} seed 0 method ${method} used_test_set ${used_test_set} acc_clustering ${acc_clustering} use_softmax_feature False
done
done
done
######
method='paddle'
for dataset in flowers102 ucf101 fgvcaircraft
do
for s in 1 2 4 8 16
do
for lambd in 0.0 1.0 2.0 5.0 10.0 20.0 35.0 50.0 100.0
do
python main.py --opts batch_size 5 number_tasks 5 dataset ${dataset} lambd ${lambd} shots ${s} n_query ${n_query} k_eff ${k_eff} save_results True device ${device} seed 0 method ${method} used_test_set ${used_test_set} acc_clustering ${acc_clustering} use_softmax_feature True
python main.py --opts batch_size 5 number_tasks 5 dataset ${dataset} lambd ${lambd} shots ${s} n_query ${n_query} k_eff ${k_eff} save_results True device ${device} seed 0 method ${method} used_test_set ${used_test_set} acc_clustering ${acc_clustering} use_softmax_feature False
done
done
done
######
method='laplacian_shot'
for dataset in flowers102 ucf101 fgvcaircraft
do
for s in 1 2 4 8 16
do
for lmd in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
python main.py --opts batch_size 5 number_tasks 5 dataset ${dataset} lmd ${lmd} shots ${s} n_query ${n_query} k_eff ${k_eff} save_results True device ${device} seed 0 method ${method} used_test_set ${used_test_set} acc_clustering ${acc_clustering} use_softmax_feature True
python main.py --opts batch_size 5 number_tasks 5 dataset ${dataset} lmd ${lmd} shots ${s} n_query ${n_query} k_eff ${k_eff} save_results True device ${device} seed 0 method ${method} used_test_set ${used_test_set} acc_clustering ${acc_clustering} use_softmax_feature False
done
done
done