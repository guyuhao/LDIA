# LDIA

***
## Code usage: 
### Prepare dataset
1. Please create a **"data"** folder in the same directory of the code to save the raw dataset.
2.  Please create a **"checkpoints"** folder in the same directory of the code to save the model parameters during federated learning. 
    For each dataset in ["ag_news", "cifar10", "covertype", "imdb", "mnist", "purchase"], please create a corresponding folder in **"checkpoints"**, such as **"checkpoints/mnist"**.
3. For MNIST and CIFAR-10, torchvision is used to automatically download the raw dataset to **"data"** when network is available. In addition, torchtext is used for AG's News, IMDB.
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
Experimental scripts are located in the dictionary **"tests"**. You can execute the scripts by providing the parameter **--config** which specified the configuration file path as follows.
1. To evaluate Random, LLG+ and Updates-Leak
```angular2html
python3 attack_compare.py --config ../final/final_config/compare/mnist.yaml
```
## Code architecture
```angular2html
.
├── attack               # implementation of Random, LLG+, and Updates-Leak attack
├── common               # implementation of loading dataset, training and testing single model
├── dataset
├── defense              # implementation of two defenses: differential privacy and gradient compression
├── federated            # implementation of federated learning
├── final
│   └── final_config     # to save configuration files of all experiments
├── model                # architecture of all target models
└── tests                # experimental scripts
    ├── config_template.yaml
    └── result           # to save the log output by experimental scripts
```