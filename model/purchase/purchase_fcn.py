# -*- coding: utf-8 -*-
"""
FCN model architecture for Purchase dataset
"""

import torch.nn as nn
from torch.nn import init

from model.base_model import BaseModel


class PurchaseFcn(BaseModel):

    def __init__(self, param_dict, dropout_p=None):
        super(PurchaseFcn, self).__init__(
            dataset=param_dict['dataset'],
            type='fcn',
            param_dict=param_dict
        )
        self.loss_function = nn.CrossEntropyLoss()

        if dropout_p is None:
            self.classifier = nn.Sequential(
                nn.Linear(600, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.ReLU(),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(600, 1024),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(dropout_p)
            )
        self.last_layer = nn.Linear(128, param_dict['num_classes'])

        self.init_optim(param_dict)

    def set_cuda(self):
        self.classifier.cuda()
        self.last_layer.cuda()

    def set_cpu(self):
        self.classifier.cpu()
        self.last_layer.cpu()

    def forward(self, x):
        x = self.classifier(x)
        x = self.last_layer(x)
        return x

    def get_last_layer_input(self, x):
        x = self.classifier(x)
        return x

    def init_weight(self):
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):  # 判断是否是线性层
                init.xavier_uniform_(layer.weight, gain=1)
        return


def purchase_fcn(**kwargs):
    model = PurchaseFcn(**kwargs)
    return model
