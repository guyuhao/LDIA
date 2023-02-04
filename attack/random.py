# -*- coding: utf-8 -*-
"""
Implementation of random distribution attack
"""

import numpy as np
import torch.nn

from attack.utils import get_attack_kld_label, get_random_label_distribution, cal_loss
from common.model_process import *
from dataset.utils import get_dataset_labels


def random_attack(client_loaders, args):
    targets, predicts = [], []
    for client_loader in client_loaders:
        size = len(client_loader.dataset)
        temp = get_random_label_distribution(size, args['num_classes'])
        predict = get_attack_kld_label(temp, size, decimal=3)
        predicts.append(predict)

        labels = get_dataset_labels(client_loader.dataset, args['dataset'])
        all_size = len(labels)
        size_list = [len(np.where(labels == label)[0]) for label in range(args['num_classes'])]
        target = get_attack_kld_label(size_list, all_size)
        targets.append(target)
    return cal_loss(torch.tensor(targets), torch.tensor(predicts))
