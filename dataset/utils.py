#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：gyh 
@File    ：utils.py
@Author  ：Gu Yuhao
@Date    ：2022/4/20 下午4:05 

Implementation of general functions in data processing
"""
import datetime
import logging
import random
from collections import defaultdict
from random import shuffle

import numpy as np
import torch
from torch.utils import data
from torch.utils.data import BatchSampler

from dataset.image_dataset import ImageDataset


def get_dataset_labels(target_dataset, name):
    """
    get labels of dataset

    :param target_dataset: dataset
    :param name: dataset name
    :return: labels
    """
    indices = None
    if isinstance(target_dataset, data.Subset):
        parent_dataset = target_dataset.dataset
        indices = target_dataset.indices
    else:
        parent_dataset = target_dataset
    result = None
    if isinstance(parent_dataset, data.TensorDataset):
        result = np.array(parent_dataset.tensors[1])
    elif isinstance(parent_dataset, ImageDataset):
        result = np.array(parent_dataset.targets)
    elif 'cifar' in name or name in ['mnist']:
        result = np.array(parent_dataset.targets)
    elif name in ['ag_news']:
        result = np.array([int(label)-1 for label, _ in parent_dataset])
    elif name in ['imdb']:
        result = np.array([1 if label == 'pos' else 0 for label, _ in parent_dataset])
    if indices is not None:
        result = result[indices]
    return result


def iid_partition(indices, per_size, args):
    """
    IID data partition for FL participants

    :param indices: indices of train dataset to divide
    :param per_size: the data size of each participant
    :param args: configuration
    :return: list of train dataset indices for each FL participant
    """
    client_indices = []
    shuffle(indices)
    for i in range(args['n_client']):
        client_indices.append(indices[i * per_size: (i + 1) * per_size])
    return client_indices


def distribution_based_non_iid_partition(indices, labels, args, probability_list=None):
    """
    distribution-based label imbalance partition for FL participants using Dirichlet distribution, refers to pn~Dir(a)

    :param indices: indices of train dataset to divide
    :param labels: labels of train dataset, corresponds to indices
    :param args: configuration
    :param probability_list: preset Dirichlet distribution
    :return: tuple containing:
        (1) client_indices: indices of train dataset for each user
        (2) probability_list: the Dirichlet distribution used in current partition
    """
    if probability_list is None:
        probability_list = []
    data_classes = {}
    # save indices of train dataset for each class
    for ind in indices:
        label = int(labels[ind])
        if label in data_classes:
            data_classes[label].append(ind)
        else:
            data_classes[label] = [ind]
    no_classes = len(data_classes.keys())
    per_participant_list = defaultdict(list)

    # for each class, use Dirichlet distribution to split dataset for all clients
    while True:
        client_indices = []
        for n in range(no_classes):
            class_size = len(data_classes[n])
            shuffle(data_classes[n])
            if n == len(probability_list):
                probability = np.random.dirichlet(np.array(args['n_client'] * [args['non_iid_alpha']]))
                probability_list.append(probability)
            else:
                probability = np.array(probability_list[n])
            sampled_probabilities = class_size * probability
            for user in range(args['n_client']):
                data_size = int(round(sampled_probabilities[user]))
                sampled_list = data_classes[n][:min(len(data_classes[n]), data_size)]
                per_participant_list[user].extend(sampled_list)
                data_classes[n] = data_classes[n][min(len(data_classes[n]), data_size):]
        for user in range(args['n_client']):
            client_indices.append(per_participant_list[user])
        # ensure at least one sample in each dataset, otherwise re-generate
        re_gen = False
        for temp in client_indices:
            if len(temp) == 0:
                print("re gen")
                re_gen = True
                seed = np.random.randint(0, 100000)
                np.random.seed(seed)
                random.seed(seed)
                break
        if not re_gen:
            break
    return client_indices, probability_list


def quantity_based_non_iid_partition(indices, labels, args, quantity):
    """
    quantity-based label imbalance partition for FL participants, refers to #C=n

    :param indices: indices of train dataset to divide
    :param labels: labels of train dataset, corresponds to indices
    :param args: configuration
    :param quantity: the number of labels whose samples are non-zero
    :return: indices of train dataset for each user
    """
    data_labels = {}
    # save indices of train dataset for each class
    for ind in indices:
        label = int(labels[ind])
        if label in data_labels:
            data_labels[label].append(ind)
        else:
            data_labels[label] = [ind]

    # random assign n classes for each user
    client_labels_list = []
    label_clients = [[] for i in range(args['num_classes'])]
    for client in range(args['n_client']):
        client_labels = np.sort(np.random.choice(args['num_classes'], quantity, replace=False))
        client_labels_list.append(client_labels)
        for label in client_labels:
            label_clients[label].append(client)
    # for each label, divide the corresponding samples equally among users who own the label
    client_indices = [[] for i in range(args['n_client'])]
    for n in range(args['num_classes']):
        subset_count = len(label_clients[n])
        if subset_count == 0:
            continue
        subset_size = len(data_labels[n])//subset_count
        indices = data_labels[n]
        shuffle(indices)
        for i in range(subset_count):
            client_indices[label_clients[n][i]] += indices[subset_size*i: subset_size*(i+1)]
    return client_indices


def generate_loaders(datasets, batch_size, shuffle=True, collate_fn=None,
                     samplers=None):
    """
    generate loaders from datasets

    :param datasets: datasets containing X and Y
    :param int batch_size: batch of loader
    :param bool shuffle: whether to shuffle loader
    :param collate_fn: definition of processing batch samples for text datasets
    :param samplers: definition of sampling functions for text datasets
    :return: loaders
    """
    loaders = []
    for i, dataset in enumerate(datasets):
        if dataset is None or len(dataset) <= 0:
            loaders.append(None)
        else:
            loader = data.DataLoader(dataset=dataset,
                                     batch_size=batch_size,
                                     shuffle=shuffle,
                                     drop_last=False,
                                     collate_fn=collate_fn,
                                     batch_sampler=samplers[i] if samplers is not None else None)
            loaders.append(loader)
    return loaders


def auxiliary_test_partition(dataset, args, shadow_non_iid=None):
    """
    divide the dataset into test dataset for FL, auxiliary dataset and probe dataset

    :param dataset: the divided dataset
    :param args: configuration
    :param shadow_non_iid: the unbalance level of auxiliary dataset, None means balanced
    :return: tuple containing auxiliary dataset, probe dataset and test dataset for FL
    """
    auxiliary_dataset, probe_dataset, test_dataset = None, None, None
    labels = get_dataset_labels(dataset, args['dataset'])
    auxiliary_indices, probe_indices, test_indices = None, None, None
    # divide the dataset if auxiliary_size and probe_size larger than 0 else return the test dataset directly
    if args['auxiliary_size'] > 0 and args['probe_size'] > 0:
        # calculate the data size of each label for auxiliary, probe and test dataset
        per_auxiliary_size, per_probe_size, per_test_size = \
            int(args['auxiliary_size']/args['num_classes']), \
            int(args['probe_size'] / args['num_classes']), \
            int(args['target_test_size']/args['num_classes'])
        # use Dirichlet distribution to generate auxiliary dataset
        if shadow_non_iid is None:
            auxiliary_size_list = [per_auxiliary_size] * args['num_classes']
        else:
            probability_list = np.random.dirichlet(np.array(args['num_classes'] * [shadow_non_iid]))
            auxiliary_size_list = args['auxiliary_size'] * probability_list
            for i, temp in enumerate(auxiliary_size_list):
                auxiliary_size_list[i] = max(int(temp), 1)
            auxiliary_size_list = np.int_(auxiliary_size_list)
        # divide the dataset into auxiliary, probe and test dataset
        for i in range(args['num_classes']):
            temp_indices = np.where(labels == i)[0]
            if len(temp_indices) < auxiliary_size_list[i] + per_probe_size:
                logging.warning('few class: {}, size: {}'.format(i, len(temp_indices)))
            shuffle(temp_indices)
            temp_probe_indices, temp_auxiliary_indices, temp_test_indices = \
                temp_indices[: per_probe_size], \
                temp_indices[per_probe_size: auxiliary_size_list[i] + per_probe_size], \
                temp_indices[auxiliary_size_list[i] + per_probe_size: auxiliary_size_list[i] + per_probe_size + per_test_size]
            auxiliary_indices = np.concatenate((auxiliary_indices, temp_auxiliary_indices)) \
                if auxiliary_indices is not None else temp_auxiliary_indices
            probe_indices = np.concatenate((probe_indices, temp_probe_indices)) \
                if probe_indices is not None else temp_probe_indices
            test_indices = np.concatenate((test_indices, temp_test_indices)) \
                if test_indices is not None else temp_test_indices
        if isinstance(dataset, list):
            auxiliary_dataset = list([dataset[index] for index in auxiliary_indices])
            probe_dataset = list([dataset[index] for index in probe_indices])
        else:
            auxiliary_dataset = torch.utils.data.Subset(dataset, auxiliary_indices)
            probe_dataset = torch.utils.data.Subset(dataset, probe_indices)
    else:
        test_indices = (np.random.choice(np.arange(len(labels)), args['target_test_size'], replace=False)).tolist()
    if isinstance(dataset, list):
        test_dataset = list([dataset[index] for index in test_indices])
    else:
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
    return auxiliary_dataset, probe_dataset, test_dataset


def client_split(dataset, x, y, name, args,
                 probability_list=None, quantity=None):
    """
    partition the dataset for all users

    :param dataset: the divided dataset, conflict to x
    :param x: the divided features, conflict to dataset
    :param y: the labels of the divide dataset
    :param name: dataset name
    :param args: configuration
    :param probability_list: preset Dirichlet distribution, only for distribution-based label imbalance setting
    :param quantity: the number of labels whose samples are non-zero, only for quantity-based label imbalance setting
    :return: tuple containing:
        (1) datasets: users' datasets
        (2) probability_list: Dirichlet distribution used in current partition
    """
    if probability_list is None:
        probability_list = []
    datasets = []
    all_indices = np.arange(len(y))
    # randomly sample according to the target_train_size
    size = min(args['target_train_size'], len(y)) if name == 'train' \
        else min(args['target_test_size'], len(y))
    temp_indices = np.random.choice(all_indices, size, replace=False)
    all_indices = temp_indices
    # get the indices of users' datasets using different partition
    per_size = args['client_train_size'] if name == 'train' else args['client_test_size']
    if args['non_iid'] == 0:
        client_indices = iid_partition(all_indices, per_size, args)
    else:
        if quantity is None:
            client_indices, probability_list = distribution_based_non_iid_partition(
                all_indices, y, args, probability_list)
        else:
            client_indices = quantity_based_non_iid_partition(all_indices, y, args, quantity)

    if dataset is not None:
        for indices in client_indices:
            if isinstance(dataset, list):
                sub_dataset = list([dataset[index] for index in indices])
                datasets.append(sub_dataset)
            else:
                sub_dataset = torch.utils.data.Subset(dataset, indices)
                datasets.append(sub_dataset)
    elif x is not None:
        for indices in client_indices:
            if args['dataset'] in ['purchase', 'covertype', 'ag_news', 'imdb']:
                dataset = torch.utils.data.TensorDataset(
                    torch.tensor(x[indices]),
                    torch.tensor(y[indices]))
            datasets.append(dataset)
    return datasets, probability_list


# Implementation of bucket sampler for text datasets
class BucketSampler(BatchSampler):
    def __init__(self, dataset, batch_size, tokenizer, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_dataset = len(dataset)
        self.drop_last = drop_last
        self.tokenizer = tokenizer

        indices = [(i, len(tokenizer(s[1]))) for i, s in enumerate(self.dataset)]
        random.shuffle(indices)
        pooled_indices = []
        # create pool of indices with similar lengths
        for i in range(0, len(indices), self.batch_size * 100):
            pooled_indices.extend(sorted(indices[i:i + self.batch_size * 100], key=lambda x: x[1]))

        self.pooled_indices = [x[0] for x in pooled_indices]

    def __iter__(self):
        # yield indices for current batch
        for i in range(0, len(self.pooled_indices), self.batch_size):
            yield self.pooled_indices[i:i + self.batch_size]

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore
