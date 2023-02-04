#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：gyh 
@File    ：utils.py
@Author  ：Gu Yuhao
@Date    ：2022/5/9 下午3:12 

Implementation of general functions in attack
"""
import logging
import random

import numpy as np
from torch import nn, Tensor

from common import kl_divergence
from common.utils import chebyshev_distance, cosine_similarity


def get_attack_kld_label(size_list, all_size, decimal=2):
    """
    calculate label distribution

    :param size_list: number of samples per label
    :param all_size: total data size
    :param decimal: limited digits
    :return: label distribution
    """
    result = []
    for i, size in enumerate(size_list):
        if i != len(size_list)-1:
            result.append(round(size/all_size, decimal))
        else:
            break
    result.append(max(1-np.sum(result), 0))
    return result


def get_quantity_label_distribution(all_size, num_classes, quantity):
    """
    generate random label distribution while fixing some labels zero

    :param all_size: total data size
    :param num_classes: number of classes
    :param quantity: the number of labels whose samples are non-zero
    :return: number of samples per label
    """
    result = [0] * num_classes
    # randomly pick 'quantity' classes as the labels whose samples are non-zero
    target_class = np.random.choice(list(range(num_classes)), quantity, replace=False)
    target_class = sorted(target_class)

    # generate random label distribution while keeping some labels zero
    temp_list = []
    for i in range(quantity-1):
        temp_list.append(random.randint(1, all_size))
    if len(temp_list) > 0:
        temp_list = sorted(temp_list)
        result[target_class[0]] = temp_list[0]-0
        for i in range(1, quantity-1):
            result[target_class[i]] = temp_list[i]-temp_list[i-1]
        result[target_class[-1]] = (all_size-temp_list[-1])
    else:
        result[target_class[-1]] = all_size
    return result


def get_random_label_distribution(all_size, num_classes):
    """
    generate random label distribution

    :param all_size: total data size
    :param num_classes: number of classes
    :return: number of samples per label
    """
    result = []
    temp_list = []
    for i in range(num_classes-1):
        temp_list.append(random.randint(0, all_size))
    temp_list = sorted(temp_list)
    result.append(temp_list[0]-0)
    for i in range(1, num_classes-1):
        result.append(temp_list[i]-temp_list[i-1])
    result.append(all_size-temp_list[-1])
    return result


def cal_loss(targets: Tensor, predicts: Tensor):
    """
    calculate attack performance metrics between real and predicted label distributions

    :param targets: real label distributions
    :param predicts: predicted label distributions
    :return: tuple containing KL-div, Cheb and Cosine
    """
    if targets is None or predicts is None:
        return None
    count = targets.shape[0]
    kld_result, chebyshev_result, cos_result = np.zeros(count), np.zeros(count), np.zeros(count)
    for i in range(count):
        temp = kl_divergence(targets[i], predicts[i])
        logging.warning('index {}, target: {}'.format(i, targets[i]))
        logging.warning('index {}, output: {}'.format(i, predicts[i]))
        kld_result[i] = temp.item()
        chebyshev_result[i] = chebyshev_distance(targets[i], predicts[i]).item()
        cos_result[i] = cosine_similarity(targets[i], predicts[i])
    kld_result = round(np.mean(kld_result), 5)
    chebyshev_result = round(np.mean(chebyshev_result), 5)
    cos_result = round(np.mean(cos_result), 5)
    return kld_result, chebyshev_result, cos_result
