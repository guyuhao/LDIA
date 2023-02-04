# -*- coding: utf-8 -*-
"""
Implementation of DP defense, refers to "User-Level Privacy-Preserving Federated Learning: Analysis and Performance Optimization"

refers to the implementation on https://github.com/AdamWei-boop/Federated-Learning-with-Local-Differential-Privacy
"""
import copy

import numpy as np
import torch


def noise_add(noise_scale, w):
    w_noise = copy.deepcopy(w)
    if isinstance(w[0], np.ndarray):
        noise = np.random.normal(0, noise_scale, w.size())
        w_noise = w_noise + noise
    else:
        for k in range(len(w)):
            for i in w[k].keys():
                noise = np.random.normal(0, noise_scale, w[k][i].size())
                noise = torch.from_numpy(noise).float()
                w_noise[k][i] = w_noise[k][i] + noise
    return w_noise


def get_1_norm(params_a):
    sum = 0
    if isinstance(params_a, np.ndarray):
        sum += pow(np.linalg.norm(params_a, ord=2), 2)
    else:
        for i in params_a.keys():
            if len(params_a[i]) == 1:
                sum += pow(np.linalg.norm(params_a[i].cpu().numpy(), ord=2), 2)
            else:
                a = copy.deepcopy(params_a[i].cpu().numpy())
                for j in a:
                    x = copy.deepcopy(j.flatten())
                    sum += pow(np.linalg.norm(x, ord=2), 2)
    norm = np.sqrt(sum)
    return norm


def client_differential_privacy(model, args, noise_scale):
    """
    perform LDP introduced in paper

    :param model: client model
    :param args: configuration
    :param noise_scale: the standard deviation of additive noises
    """
    w = model.state_dict()
    w_noise = noise_add(noise_scale, [w])
    model.load_state_dict(w_noise[0])


