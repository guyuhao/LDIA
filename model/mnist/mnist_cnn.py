# -*- coding: utf-8 -*-
"""
CNN model architecture for MNIST dataset
"""

import torch.nn as nn

from model.base_model import BaseModel


class MnistCnn(BaseModel):
    def __init__(self, param_dict):
        super(MnistCnn, self).__init__(
            dataset=param_dict['dataset'],
            type='cnn',
            param_dict=param_dict
        )
        self.loss_function = nn.CrossEntropyLoss()

        input_size = 1
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=input_size, out_channels=32, kernel_size=(5, 5)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.fc_features = 4 * 4 * 32
        self.last_layer = nn.Linear(in_features=self.fc_features, out_features=param_dict['num_classes'])

        self.init_optim(param_dict)

    def set_cuda(self):
        self.features.cuda()
        self.last_layer.cuda()

    def set_cpu(self):
        self.features.cpu()
        self.last_layer.cpu()

    def forward(self, x):
        x = self.features(x)
        x = self.last_layer(x)
        return x

    def get_last_layer_input(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def mnist_cnn(**kwargs):
    model = MnistCnn(**kwargs)
    return model

