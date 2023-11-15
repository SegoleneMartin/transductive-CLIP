# Extending few-shot classification knowledge to CLIP


## 1. Getting started

### 1.1. Requirements

- numpy
- clip
- torch
- matplotlib
- pandas
- sklearn
- scipy


### 1.2 Download datasets and splits
We follow the links given in the Github repository of TIP-Adapter and we use CoOp-style train/val/test splits for all datasets except ImageNet where the validation set is used as test set.

The downloaded datasets should be placed in the folder data/ the following way:

    .
    ├── ...
    ├── data                    
    │   ├── eurosat       
    │   ├── flowers102        
    │   └── caltech101               
    └── ...

### 1.3 Extracting and saving the features
For a fixed temperature ($T=30$ recommended), we extract and save the features defined as
```math 
z_n = \text{softmax}(T \cos(f_{\text{im}}(x_n), f_{\text{text}}(t_k) )) 
```

For instance, for the dataset eurosat, the temperature T=30 and the backbone RN50, the features will be saved under

    eurosat
    ├── saved_features                    
    │   ├── test_softmax_RN50_T10.plk
    │   ├── val_softmax_RN50_T10.plk           
    │   ├── train_softmax_RN50_T10.plk           
    └── ...

The feature extraction process can take a while but once it is done the method runs pretty fast.

### 1.4 Setting the optimal parameter with the validation set (for alpha_tim, paddle, bdcspn and laplacian_shot only)
Given a fixed dataset and a fixed method, we run the method on 1000 transductive tasks constituting of images of the validation set. More specifically, these tasks are constituted with
- $|\mathbb{Q}|=35$
- $k_{\text{eff}}=5$
- 0 shots

The accuracy is then evaluated on these tasks and the best temperature can be determined this way.
To run the temperature tuning process, run
```bash scripts/script_val_fewshot.sh```
The results will be saved in results/val/method.

Note: as no validation set is provided for ImageNet, we use the same optimal temperature as for Caltech101.

### 1.5 Evaluation in a 0-shot setting
To evaluate the methods in a 0-shot setting, run
```bash scripts/script_test_0shot.sh```
The results will be saved in results/test/method.

### 1.6 Evaluation in a s-shot setting, with s >0
To evaluate the methods in a s-shot setting, run
```bash scripts/script_test_fewshot.sh```
The results will be saved in results/test/method.






