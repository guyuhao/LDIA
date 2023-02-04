import numpy as np
import pandas as pd
import torch
from torch.utils import data

from dataset.utils import generate_loaders, client_split, auxiliary_test_partition

exclude_classes = [25, 29, 65, 93]  # the label whose number of samples is below 100, excluded in experiments

target_classes = None


def load_data(name, args, target_classes=None,
              partition=False, probability_list=None, quantity=None):
    """
    load train or test data of Purchase from local file

    :param name: 'train' or 'test'
    :param args: configuration
    :param target_classes: the target classes to build dataset in current experiment
    :param partition: whether to partition dataset, True for train dataset
    :param probability_list: preset Dirichlet distribution
    :param quantity: the value of n for #C=n
    :return: dataset
    """
    if probability_list is None:
        probability_list = []
    dataset_path = '../../data'
    csv_data = pd.read_csv(dataset_path + '/purchase/dataset_purchase_' + name,
                           header=None, skipinitialspace=True)
    x = csv_data.drop(columns=[0]).values.astype('float32')
    y = np.squeeze(csv_data.drop(columns=list(range(1, 601))).values.astype('int')) - 1

    target_indices = np.isin(y, target_classes)
    x = x[target_indices]
    temp_y = y[target_indices]
    y = np.array([np.where(target_classes == temp)[0][0] for temp in temp_y])
    if name == 'train':
        args['target_train_size'] = len(y)
        args['client_train_size'] = int(args['target_train_size']/args['n_client'])

    if partition:
        return client_split(
            dataset=None, x=x, y=y, name=name, args=args,
            probability_list=probability_list, quantity=quantity)
    else:
        dataset = data.TensorDataset(torch.tensor(x), torch.tensor(y))
        return dataset


def get_dataloader(args, shadow_non_iid=None, quantity=None):
    """
    generate loaders of Purchase dataset

    :param args: configuration
    :param shadow_non_iid: the unbalance level of auxiliary dataset, None means balanced
    :param quantity: the value of n for #C=n data distribution setting
    :return: tuple contains:
        (1) train_loaders: loaders of users' datasets;
        (2) test_loader: loader of test dataset;
        (3) auxiliary_loader: loader of auxiliary dataset
        (4) probe_loader: loader of probe dataset, used by Updates-Leak
    """
    # randomly pick target classes in current experiment, excluding the labels whose number of samples is too small
    global target_classes
    if target_classes is None or len(target_classes) != args['num_classes']:
        all_classes = np.setdiff1d(np.arange(100), np.array(exclude_classes))
        target_classes = np.random.choice(all_classes, args['num_classes'], replace=False)

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
    global target_classes
    batch_size = args['target_batch_size']
    train_datasets = load_data('train', args, target_classes=target_classes,
                               partition=True, quantity=quantity)
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
    global target_classes
    batch_size = args['target_batch_size']
    test_dataset = load_data('test', args, target_classes=target_classes, partition=False)
    auxiliary_dataset, probe_dataset, test_dataset = \
        auxiliary_test_partition(test_dataset, args, shadow_non_iid=shadow_non_iid)
    auxiliary_loader = generate_loaders([auxiliary_dataset], batch_size)[0]
    probe_loader = generate_loaders([probe_dataset], batch_size)[0]
    test_loader = generate_loaders([test_dataset], batch_size)[0]
    return test_loader, auxiliary_loader, probe_loader
