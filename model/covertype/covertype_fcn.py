# -*- coding: utf-8 -*-
"""
FCN model architecture for Covertype dataset
"""
import torch.nn as nn
from torch.nn import init

from model.base_model import BaseModel


class CovertypeFcn(BaseModel):

    def __init__(self, param_dict):
        super(CovertypeFcn, self).__init__(
            dataset=param_dict['dataset'],
            type='fcn',
            param_dict=param_dict
        )
        self.loss_function = nn.CrossEntropyLoss()

        self.classifier = nn.Sequential(
            nn.Linear(54, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 7)
        )

        self.init_optim(param_dict)

    def set_cuda(self):
        self.classifier.cuda()


def covertype_fcn(**kwargs):
    model = CovertypeFcn(**kwargs)
    return model
