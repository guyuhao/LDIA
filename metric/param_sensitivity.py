#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Implementation of calculating output layer updates
"""

import torch


def param_sensitivity(model, loader, class_list, args, **param_dict):
    global_model = param_dict['global_model']

    new_w_params = list((model.parameters()))[-2].data
    old_w_params = list((global_model.parameters()))[-2].data
    result_w = old_w_params - new_w_params
    final_result = torch.flatten(result_w).cpu().numpy().tolist()

    return final_result
