import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.utils import data

from dataset.utils import generate_loaders, client_split, auxiliary_test_partition

all_test_dataset = None


def load_data(name, args, probability_list=None, quantity=None):
    """
    load train and test data of Covertype from local file

    :param name: 'train' or 'test'
    :param args: configuration
    :param probability_list: preset Dirichlet distribution
    :param quantity: the value of n for #C=n
    :return: tuple contains:
        (1) client_datasets: users' dataset
        (2) test_dataset: test dataset
    """
    if probability_list is None:
        probability_list = []
    dataset_path = '../../data'
    file_data = pd.read_csv(dataset_path + '/covertype/covtype.data', header=None, skipinitialspace=True)

    x = file_data.iloc[:, :54]
    y = file_data.iloc[:, 54]
    x = np.array(x)
    y = np.array(y)
    y = y - 1
    # split dataset into train and test set by 7:3
    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.3)

    indices = np.random.choice(np.arange(len(x_train)), args['target_train_size'], replace=False)
    x_train = x_train[indices]
    y_train = y_train[indices]
    # use Min-Max to normalize features of training data
    scaler = preprocessing.MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    # use Min-Max to normalize features of testing data
    scaler = preprocessing.MinMaxScaler()
    x_test = scaler.fit_transform(x_test)

    client_datasets = client_split(
            dataset=None, x=x_train, y=y_train, name=name, args=args,
            probability_list=probability_list, quantity=quantity)
    test_dataset = data.TensorDataset(torch.tensor(x_test), torch.tensor(y_test))
    return client_datasets, test_dataset


def get_dataloader(args, shadow_non_iid=None, quantity=None):
    """
    generate loaders of Covertype dataset

    :param args: configuration
    :param shadow_non_iid: the unbalance level of auxiliary dataset, None means balanced
    :param quantity: the value of n for #C=n data distribution setting
    :return: tuple contains:
        (1) train_loaders: loaders of users' datasets;
        (2) test_loader: loader of test dataset;
        (3) auxiliary_loader: loader of auxiliary dataset
        (4) probe_loader: loader of probe dataset, used by Updates-Leak
    """
    train_loaders = get_train_loaders(args, quantity=quantity)
    test_loader, auxiliary_loader, probe_loader = \
        get_test_auxiliary_loaders(args, shadow_non_iid=shadow_non_iid)
    return train_loaders, test_loader, auxiliary_loader, probe_loader


def get_train_loaders(args, quantity=None):
    """
    generate loaders of users' datasets

    :param args: configuration
    :param quantity: the value of n for #C=n data distribution setting
    :return: loaders of users' datasets
    """
    global all_test_dataset
    batch_size = args['target_batch_size']
    (train_datasets), all_test_dataset = load_data('train', args, quantity=quantity)
    if isinstance(train_datasets, tuple):
        train_datasets, probability_list = train_datasets
    train_loaders = generate_loaders(train_datasets, batch_size)
    return train_loaders


def get_test_auxiliary_loaders(args, shadow_non_iid=None):
    """
    generate loaders of test dataset, auxiliary dataset and probe dataset from the original test data

    :param args: configuration
    :param shadow_non_iid: the unbalance level of auxiliary dataset, None means balanced
    :return: tuple contains:
        (1) test_loader: loader of test dataset;
        (2) auxiliary_loader: loader of auxiliary dataset
        (3) probe_loader: loader of probe dataset, used by Updates-Leak
    """
    global all_test_dataset
    batch_size = args['target_batch_size']
    auxiliary_dataset, probe_dataset, test_dataset = \
        auxiliary_test_partition(all_test_dataset, args, shadow_non_iid=shadow_non_iid)
    auxiliary_loader = generate_loaders([auxiliary_dataset], batch_size)[0]
    probe_loader = generate_loaders([probe_dataset], batch_size)[0]
    test_loader = generate_loaders([test_dataset], batch_size)[0]
    return test_loader, auxiliary_loader, probe_loader
