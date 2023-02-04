#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Implementation of gradient compression defense, refers to "Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training"
"""
import torch


class GradientCompression:
    def __init__(self, gc_percent):
        self.thresh_hold = 0.
        self.gc_percent = gc_percent

    def update_thresh_hold(self, tensor):
        tensor_copy = tensor.clone().detach()
        tensor_copy = torch.abs(tensor_copy)
        reshape_tensor = tensor_copy.reshape(1, -1)
        survivial_values = torch.topk(reshape_tensor,
                                      round(reshape_tensor.shape[1] * (1-self.gc_percent)))
        if len(survivial_values[0][0]) == 0:
            self.thresh_hold = None
        else:
            self.thresh_hold = survivial_values[0][0][-1]

    def prune_tensor(self, tensor):
        background_tensor = torch.zeros(tensor.shape).to(torch.float)
        if 'cuda' in str(tensor.device):
            background_tensor = background_tensor.cuda()
        if self.thresh_hold is None:
            tensor = background_tensor
        else:
            tensor = torch.where(abs(tensor) >= self.thresh_hold, tensor, background_tensor)
        return tensor

    def compression(self, parameters):
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        for p in parameters:
            gradients = p.grad.detach()
            # calculate the threshold based on the compression ratio
            self.update_thresh_hold(gradients)
            # prune gradients based on the threshold
            result = self.prune_tensor(gradients)
            p.grad = result
        return
