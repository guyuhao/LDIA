#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：gyh 
@File    ：utils.py
@Author  ：Gu Yuhao
@Date    ：2022/7/13 下午5:38 

Implementation of general functions in federated learning
"""
import copy

import numpy as np
import torch


def fed_aggregation(global_model, client_models, args, dp=False):
    """
    perform aggregation in FL

    :param global_model: the model architecture
    :param client_models: parameters uploaded by users
    :param args: configuration
    :param bool dp: whether to use dp defense
    :return: aggregated global model
    """
    global_model = fed_avg(client_models, global_model, args, dp=dp)
    return global_model


def fed_avg(client_models, global_model, args, dp=False):
    """
    perform FedAvg aggregation

    :param client_models: parameters uploaded by users
    :param global_model: the model architecture
    :param args: configuration
    :param dp: whether to use dp defense
    :return: aggregated global model
    """
    avg_state_dict = copy.deepcopy(client_models[0]['model'])

    local_state_dicts = list()
    data_size_list = list()
    for model in client_models:
        local_state_dicts.append(model['model'])
        data_size_list.append(int(model['data_size']))

    all_data_size = np.sum(data_size_list)
    client_num = len(client_models)
    for layer in avg_state_dict.keys():
        avg_state_dict[layer] *= 0
        for client_index in range(client_num):
            temp = (data_size_list[client_index] / all_data_size) * local_state_dicts[client_index][layer]
            if avg_state_dict[layer].dtype == torch.int64:
                avg_state_dict[layer] += (temp.type(torch.int64))
            else:
                avg_state_dict[layer] += temp
    global_model.load_state_dict(avg_state_dict, strict=False)
    return global_model

