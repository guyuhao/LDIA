#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：gyh 
@File    ：data_augmentation.py
@Author  ：Gu Yuhao
@Date    ：2022/7/1 上午11:18 

"""
import copy

import Augmentor
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms
from dataset.image_dataset import ImageDataset
from dataset.utils import get_dataset_labels, generate_loaders, BucketSampler


def data_augmentation(loader, args):
    """
    enlarge auxiliary set by data augmentation

    :param loader: loader of auxiliary set
    :param args: configuration
    :return: loader of enlarged auxiliary set
    """
    target_size = args['client_train_size']

    all_dataset = copy.deepcopy(loader.dataset)
    all_dataset.dataset.transform = transforms.Compose([transforms.ToTensor()])

    class_images = []
    for i in range(args['num_classes']):
        class_images.append([])

    for image, label in all_dataset:
        temp_x = (image.permute(1, 2, 0)).numpy()
        if temp_x.shape[-1] == 1:
            temp_x = np.squeeze(temp_x, axis=len(temp_x.shape) - 1)
        class_images[label].append(np.expand_dims(temp_x, axis=0))

    target_images, target_labels = None, []
    for i in range(args['num_classes']):
        p = Augmentor.DataPipeline(class_images[i], [i] * len(class_images[i]))
        p.rotate(probability=0.8, max_left_rotation=15, max_right_rotation=20)  # 旋转
        p.shear(probability=0.8, max_shear_left=10, max_shear_right=15)  # 错切变换
        p.zoom(probability=0.8, min_factor=0.5, max_factor=1.5)  # 放大缩小

        images, labels = p.sample(target_size)
        images = np.squeeze(images, axis=1)
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=-1)
        target_images = np.concatenate((target_images, images), axis=0) if target_images is not None else images
        target_labels += labels

    target_dataset = ImageDataset(target_images,
                                  torch.tensor(target_labels),
                                  transform=loader.dataset.dataset.transform)

    x, y = [], []
    for image, label in target_dataset:
        x.append(image.unsqueeze(0))
        y.append(label)
    x = torch.cat(x, dim=0)
    y = torch.tensor(y)

    result_dataset = torch.utils.data.TensorDataset(x, y)
    result_loader = data.DataLoader(dataset=result_dataset, batch_size=args['target_batch_size'], shuffle=True)
    return result_loader


def data_duplicate(loader, args):
    """
    enlarge auxiliary set by duplication

    :param loader: loader of auxiliary set
    :param args: configuration
    :return: loader: loader of enlarged auxiliary set
    """
    target_size = args['client_train_size']

    # decide whether it needs to enlarge or not
    duplicate = False
    labels = get_dataset_labels(loader.dataset, args['dataset'])
    for i in range(args['num_classes']):
        if len(np.where(labels == i)[0]) < target_size:
            duplicate = True
            break
    if not duplicate:
        return loader

    # enlarge image dataset
    if args['dataset'] in ['mnist', 'cifar10', 'fashion_mnist']:
        all_dataset = copy.deepcopy(loader.dataset)
        if isinstance(all_dataset, data.Subset):
            all_dataset.dataset.transform = transforms.Compose([transforms.ToTensor()])
        else:
            all_dataset.transform = transforms.Compose([transforms.ToTensor()])

        class_images = []
        for i in range(args['num_classes']):
            class_images.append([])

        for image, label in all_dataset:
            temp_x = (image.permute(1, 2, 0)).numpy()
            class_images[label].append(temp_x)

        target_images, target_labels = None, []
        for i in range(args['num_classes']):
            times = target_size // len(class_images[i])
            remain = target_size % len(class_images[i])
            temp_images = []
            for j in range(times):
                temp_images += class_images[i]
            if remain > 0:
                temp_indices = np.random.choice(np.arange(len(class_images[i])), remain, replace=False)
                for index in temp_indices:
                    temp_images.append(class_images[i][index])
            target_images = np.concatenate((target_images, np.array(temp_images)), axis=0) if target_images is not None else np.array(temp_images)
            target_labels += ([i]*target_size)

        target_dataset = ImageDataset(target_images,
                                      torch.tensor(target_labels),
                                      transform=loader.dataset.dataset.transform)

        x, y = [], []
        for image, label in target_dataset:
            x.append(image.unsqueeze(0))
            y.append(label)
        x = torch.cat(x, dim=0)
        y = torch.tensor(y)
    # enlarge tabular datasets
    elif args['dataset'] in ['purchase', 'covertype', 'adult']:
        all_dataset = copy.deepcopy(loader.dataset)
        class_data = []
        for i in range(args['num_classes']):
            class_data.append([])
        for feature, label in all_dataset:
            class_data[label].append(torch.unsqueeze(feature, 0))
        target_features, target_labels = None, []
        for i in range(args['num_classes']):
            times = target_size // len(class_data[i])
            remain = target_size % len(class_data[i])
            temp_features = []
            for j in range(times):
                temp_features += class_data[i]
            if remain > 0:
                temp_indices = np.random.choice(np.arange(len(class_data[i])), remain, replace=False)
                for index in temp_indices:
                    temp_features.append(class_data[i][index])
            class_features = torch.cat(temp_features, dim=0)
            target_features = torch.cat((target_features, class_features), dim=0) \
                if target_features is not None else class_features
            target_labels += ([i] * target_size)

        x = target_features
        y = torch.tensor(target_labels)
    # enlarge text datasets
    elif args['dataset'] in ['ag_news', 'imdb']:
        all_dataset = copy.deepcopy(loader.dataset)
        class_data = []
        for i in range(args['num_classes']):
            class_data.append([])
        if args['dataset'] == 'ag_news':
            for label, text in all_dataset:
                class_data[label-1].append(text)
        elif args['dataset'] == 'imdb':
            for label, text in all_dataset:
                class_data[1 if label == 'pos' else 0].append(text)
        target_features, target_labels = [], []
        for i in range(args['num_classes']):
            times = target_size // len(class_data[i])
            remain = target_size % len(class_data[i])
            class_features = []
            for j in range(times):
                class_features += class_data[i]
            if remain > 0:
                temp_indices = np.random.choice(np.arange(len(class_data[i])), remain, replace=False)
                for index in temp_indices:
                    class_features.append(class_data[i][index])
            target_features = target_features + class_features
            if args['dataset'] == 'ag_news':
                target_labels += ([i+1] * target_size)
            elif args['dataset'] == 'imdb':
                target_labels += (['pos' if i == 1 else 'neg'] * target_size)

        x = target_features
        y = target_labels

    # build enlarged loader for image and tabular datasets
    if isinstance(y, torch.Tensor):
        result_dataset = torch.utils.data.TensorDataset(x, y)
        result_loader = data.DataLoader(dataset=result_dataset, batch_size=args['target_batch_size'], shuffle=True)
    # build enlarged loader for text datasets
    else:
        result_dataset = list(zip(y, x))
        result_loader = generate_loaders(
            [result_dataset], 1, shuffle=False, collate_fn=loader.collate_fn,
            samplers=[BucketSampler(
                dataset=result_dataset,
                batch_size=args['target_batch_size'],
                tokenizer=loader.batch_sampler.tokenizer)]
        )[0]
    return result_loader

