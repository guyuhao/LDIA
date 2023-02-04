# -*- coding: utf-8 -*-

from dataset import purchase_dataset, covertype_dataset, cifar_dataset, mnist_dataset

from dataset import ag_news_dataset, imdb_dataset

dataset_class = {
    'purchase': purchase_dataset,
    'covertype': covertype_dataset,
    'cifar10': cifar_dataset,
    'mnist': mnist_dataset,
    'ag_news': ag_news_dataset,
    'imdb': imdb_dataset,
}


def get_dataloader(args, shadow_non_iid=None, quantity=None):
    """
    generate all loaders according to dataset name

    :param args: configuration
    :param shadow_non_iid: the unbalance level of auxiliary dataset, None means balanced
    :param quantity: the value of n for #C=n data distribution setting
    :return: loaders
    """
    return dataset_class.get(args['dataset']).get_dataloader(
        args, shadow_non_iid=shadow_non_iid, quantity=quantity)


def get_train_loaders(args, quantity=None):
    """
    generate loaders of users' dataset according to dataset name

    :param args: configuration
    :param quantity: the value of n for #C=n data distribution setting
    :return: loaders of users' dataset
    """
    return dataset_class.get(args['dataset']).get_train_loaders(
        args, quantity=quantity)


def get_test_auxiliary_loaders(args, shadow_non_iid=None):
    """
    generate loaders of test dataset, auxiliary dataset and probe dataset according to the dataset name

    :param args: configuration
    :param shadow_non_iid: the unbalance level of auxiliary dataset, None means balanced
    :return: tuple contains:
        (1) test_loader: loader of test dataset;
        (2) auxiliary_loader: loader of auxiliary dataset
        (3) probe_loader: loader of probe dataset, used by Updates-Leak
    """
    return dataset_class.get(args['dataset']).get_test_auxiliary_loaders(
        args,
        shadow_non_iid=shadow_non_iid)


def get_num_classes(dataset):
    """
    get classes number of the target dataset

    :param str dataset: target dataset name
    :return: classes number of the target dataset
    """
    data_dict = {
        'covertype': 7,
        'cifar10': 10,
        'mnist': 10,
        'ag_news': 4,
        'imdb': 2
    }
    if dataset not in data_dict.keys():
        return 0
    return data_dict[dataset]


