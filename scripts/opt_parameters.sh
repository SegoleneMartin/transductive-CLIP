n_query=35
number_tasks=5
batch_size=5
k_eff=5
used_test_set='val'
use_softmax_features=True
######
for dataset in food101 sun397 caltech101 ucf101 flowers102 dtd eurosat fgvcaircraft oxfordpets sun397 stanfordcars
do
for s in 1 2 4 8 16
do
## alpha_tim
for alpha_value in 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5 7.0
do
python main.py --opts method alpha_tim batch_size ${batch_size} number_tasks ${number_tasks} dataset ${dataset} alpha_value ${alpha_value} shots ${s} n_query ${n_query} k_eff ${k_eff} save_results True seed 0 used_test_set ${used_test_set} use_softmax_feature ${use_softmax_features}
done
## bdcspn
for temp in 1.0 3.0 5.0 10.0 20.0 30.0 40.0 50.0 60.0 #0.0001 0.001 0.01 0.1 1.0 10.0 100.0 1000.0
do
python main.py --opts method bdcspn batch_size ${batch_size} number_tasks ${number_tasks} dataset ${dataset} temp ${temp} shots ${s} n_query ${n_query} k_eff ${k_eff} save_results True seed 0 used_test_set ${used_test_set} use_softmax_feature ${use_softmax_features}
done
## paddle
for lambd in 0.0 1.0 2.0 5.0 10.0 20.0 35.0 50.0 100.0
do
python main.py --opts method paddle batch_size ${batch_size} number_tasks ${number_tasks} dataset ${dataset} lambd ${lambd} shots ${s} n_query ${n_query} k_eff ${k_eff} save_results True seed 0 used_test_set ${used_test_set} use_softmax_feature ${use_softmax_features}
done
## laplacian_shot
for lmd in 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0
do
python main.py --opts method laplacian_shot batch_size ${batch_size} number_tasks ${number_tasks} dataset ${dataset} lmd ${lmd} shots ${s} n_query ${n_query} k_eff ${k_eff} save_results True seed 0 used_test_set ${used_test_set} use_softmax_feature ${use_softmax_features}
done
done
done
