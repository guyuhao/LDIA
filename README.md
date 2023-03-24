# LDIA
This repository contains code for our paper: "LDIA: Label distribution inference attack against federated learning in edge computing".
***
## Code usage: 
### Prepare dataset
1. Please create a **"data"** folder in the same directory of the code to save the raw dataset.
2.  Please create a **"checkpoints"** folder in the same directory of the code to save the model parameters during federated learning. 
    For each dataset in ["ag_news", "cifar10", "covertype", "imdb", "mnist", "purchase"], please create a corresponding folder in **"checkpoints"**, such as **"checkpoints/mnist"**.
3. For MNIST and CIFAR-10, torchvision is used to automatically download the raw dataset to **"data"** when network is available. In addition, torchtext is used for AG's News and IMDB.
4. For Covertype, please download from https://archive.ics.uci.edu/ml/datasets/covertype in advance and unzip it to **"data"**.
5. As for Purchase, please download the raw dataset derived by <a href="https://ieeexplore.ieee.org/abstract/document/7958568/">Shokri et al.</a> in advance and save them to **"data"**.
   We provide preprocessing scripts for both datasets in the dictionary **"data_preprocess"**. Please execute the script as follows to generate Purchase dataset suitable for LDIA.
```
python3 preprocess_purchase.py
```
### Prepare configuration file
We provide a configuration file template **"config_template.yaml"** in the directory **"tests"** for running our code. You can refer to it to generate your own configuration file as needed. 

We also provide the configuration files of experiments in our paper in the **"final/final_config"** directory for reference.
### Run scripts
All experimental scripts are located in the dictionary **"tests"**. You can execute the scripts by providing the parameter **--config** which specified the configuration file path as follows.
1. To evaluate LDIA
```angular2html
python3 attack_compare.py --config ../final/final_config/compare/mnist.yaml
```
2. To evaluate the impact of selected participants in fractional aggregation scenario
```angular2html
python3 select_client_number_impact.py --config ../final/final_config/other/cifar10.yaml
```
3. To evaluate the impact of number of observed rounds in LDIA
```angular2html
python3 rounds_impact.py --config ../final/final_config/other/cifar10.yaml
```
4. To evaluate the LDIA performance in various observed round ranges
```angular2html
python3 start_round_impact.py --config ../final/final_config/other/cifar10.yaml
```
5. To evaluate the impact of auxiliary data size in LDIA
```angular2html
python3 shadow_size_impact.py --config ../final/final_config/other/cifar10.yaml
```
6. To evaluate the LDIA performance with an unbalance auxiliary dataset
```angular2html
python3 shadow_non_iid_impact.py --config ../final/final_config/other/cifar10.yaml
```
7. To evaluate the impact of number of participants
```angular2html
python3 client_number_impact.py --config ../final/final_config/other/cifar10.yaml
```
8. To evaluate the impact of number of classes
```angular2html
python3 class_number_impact.py --config ../final/final_config/other/purchase.yaml
```
9. To evaluate the impact of hyper-parameters for target models
```angular2html
python3 learning_rate_impact.py --config ../final/final_config/other/cifar10.yaml
python3 local_epochs_impact.py --config ../final/final_config/other/cifar10.yaml
python3 batch_size_impact.py --config ../final/final_config/other/cifar10.yaml
```

To evaluate the effectiveness of LDIA under defense.
1. To evaluate LDIA under dropout defense
```angular2html
python3 defense_dropout.py --config ../final/final_config/defense/dropout_purchase.yaml
```
2. To evaluate LDIA under differential privacy defense
```angular2html
python3 defense_dp.py --config ../final/final_config/defense/dp_mnist.yaml
```
3. To evaluate LDIA under gradient compression
```angular2html
python3 defense_gc.py --config ../final/final_config/defense/gc_mnist.yaml
```

## Code architecture
```angular2html
.
├── attack               # implementation of LDIA, Random, LLG+, and Updates-Leak attack
├── common               # implementation of loading dataset, training and testing single model
├── dataset
├── defense              # implementation of two defenses: differential privacy and gradient compression
├── federated            # implementation of federated learning
├── final
│   └── final_config     # to save configuration files of all experiments
├── metric               # implementation of computing output layer updates for LDIA
├── model                # architecture of all target models
└── tests                # experimental scripts
    ├── config_template.yaml
    └── result           # to save the log output by experimental scripts
```

***
## Citation
If you use this code, please cite the following paper: 
### <a href="https://www.sciencedirect.com/science/article/abs/pii/S2214212623000595">LDIA</a>
```
@article{GU2023103475,
title = {LDIA: Label distribution inference attack against federated learning in edge computing},
journal = {Journal of Information Security and Applications},
volume = {74},
pages = {103475},
year = {2023},
issn = {2214-2126},
doi = {https://doi.org/10.1016/j.jisa.2023.103475},
url = {https://www.sciencedirect.com/science/article/pii/S2214212623000595},
author = {Yuhao Gu and Yuebin Bai},
keywords = {Federated learning, Edge computing, Privacy leakage, Label distribution inference, User-level privacy}
```