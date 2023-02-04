#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：gyh 
@File    ：utils.py
@Author  ：Gu Yuhao
@Date    ：2022/5/9 下午3:12 

Implementation of general functions in the whole project
"""
import time

import torch


def gjm_distance(p, q):
    """
    calculate GJM distance between two distributions, not used
    """
    alpha = 0.5
    temp = q * torch.pow(torch.abs(1 - torch.pow(p / (q+1e-14), alpha)), 1 / alpha)
    result = torch.sum(temp)
    return result


def kl_divergence(p, q):
    """
    calculate KL-div between two distributions
    """
    result = None
    for i in range(len(p)):
        if p[i] == 0:
            temp = p[i]
        elif q[i] == 0:
            temp = (p[i] * torch.log2(p[i] / 1e-7))
        else:
            temp = (p[i] * torch.log2(p[i] / q[i]))
        result = temp if result is None else result+temp
    return result


def chebyshev_distance(p, q):
    """
    calculate Cheb between two distributions
    """
    return torch.max(torch.abs(p-q))


def cosine_similarity(p, q):
    """
    calculate Cosine between two distributions
    """
    cos = torch.nn.CosineSimilarity(dim=0)
    return cos(p, q)


def canberra_distance(p, q):
    """
    calculate Canberra distance between two distributions, not used
    """
    result = torch.div(torch.abs(p-q), torch.abs(p)+torch.abs(q))
    result[result != result] = 0
    return torch.sum(result)


def js_divergence(p, q):
    """
    calculate JS divergence between two distributions, not used
    """
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def var_loss_function(targets, outputs):
    """
    calculate the variance loss between targets and outputs
    """
    result = 0.0
    for i in range(len(targets)):
        result += torch.abs(torch.var(targets[i])-torch.var(outputs[i]))
    return result


def cal_conv2d_output_dim(input_dim, kernel_size, stride=1, padding=0, dilation=1):
    """
    calculate the output dimension of the conv layer
    """
    H_in, W_in = input_dim[0], input_dim[1]
    if isinstance(kernel_size, tuple):
        H_out = (H_in+2*padding-dilation*(kernel_size[0]-1)-1)//stride+1
        W_out = (W_in+2*padding-dilation*(kernel_size[1]-1)-1)//stride+1
    else:
        H_out = (H_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        W_out = (W_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    return H_out, W_out


def cal_maxpool2d_output_dim(input_dim, kernel_size, stride=None, padding=0, dilation=1):
    """
    calculate the output dimension of the max pooling layer
    """
    if stride is None:
        stride = kernel_size
    H_in, W_in = input_dim[0], input_dim[1]
    H_out = (H_in+2*padding-dilation*(kernel_size-1)-1)//stride+1
    W_out = (W_in+2*padding-dilation*(kernel_size-1)-1)//stride+1
    return H_out, W_out


def print_running_time(title, start_time):
    """
    print the executing period

    :param title: title of the execution
    :param start_time: the start time before executing
    :return: executing period
    """
    if start_time is not None:
        end_time = time.time()
        print('{}, time: {}s'.format(title, end_time-start_time))
    start_time = time.time()
    return start_time
