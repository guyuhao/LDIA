# -*- coding: utf-8 -*-
"""
CNN model architecture for LDIA attack model
"""
import torch
import torch.nn as nn
from torch import optim
from torch.autograd.function import InplaceFunction

from torch.nn import init

from common.utils import cal_conv2d_output_dim, cal_maxpool2d_output_dim
from model.base_model import BaseModel
import torch.nn.functional as F

n_digits = 3


class round_func(InplaceFunction):
    """
    Limit the digits of the output of the attack model
    """
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return torch.round(input * 10 ** n_digits) / (10 ** n_digits)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class AttackModel(BaseModel):
    def __init__(self, param_dict):
        super(AttackModel, self).__init__(
            dataset='',
            type='fcn',
            param_dict=param_dict
        )
        # the indices of the attack model input to support multiple sensitivities, but only the output layer updates in paper
        self.split_indices = param_dict['split_indices']

        self.loss_function = nn.KLDivLoss()

        input_dim = param_dict['input_dim']

        self.features_list, self.classifier_list = [], []
        # build a CNN sub-model for each sensitivity, but only a CNN for the output layer updates in paper
        for i in range(input_dim[0]):
            # calculate the ratio between the kernel width and length in the first conv layer
            ratio = (self.split_indices[i + 1] - self.split_indices[i]) // input_dim[1]
            if ratio >= 100:
                ratio = ratio // 100 * 100
            elif ratio >= 10:
                ratio = ratio // 10 * 10
            else:
                ratio = 1

            conv_output_channel_1, conv_kernel_size_1, conv_stride_1, conv_padding_1 = 64, (5, 5 * ratio), 4, 5
            pool_kernel_size_1, pool_stride_1 = 2, 2
            conv_output_channel_2, conv_kernel_size_2, conv_padding_2 = 192, 5, 2
            pool_kernel_size_2, pool_stride_2 = 2, 2

            conv_output_1 = cal_conv2d_output_dim(
                input_dim=(input_dim[1], self.split_indices[i + 1] - self.split_indices[i]),
                kernel_size=conv_kernel_size_1,
                stride=conv_stride_1,
                padding=conv_padding_1)
            pool_output_1 = cal_maxpool2d_output_dim(input_dim=conv_output_1,
                                                     kernel_size=pool_kernel_size_1,
                                                     stride=pool_stride_1)
            conv_output_2 = cal_conv2d_output_dim(input_dim=pool_output_1,
                                                  kernel_size=conv_kernel_size_2,
                                                  padding=conv_padding_2)
            pool_output_2 = cal_maxpool2d_output_dim(input_dim=conv_output_2,
                                                     kernel_size=pool_kernel_size_2,
                                                     stride=pool_stride_2)

            modules = []
            # use two conv layers if the output dimension of the first conv layer doesn't contain 0
            if pool_output_1[0] > 0 and pool_output_1[1] > 0:
                modules = [
                    nn.Conv2d(1, conv_output_channel_1, kernel_size=conv_kernel_size_1, stride=conv_stride_1,
                              padding=conv_padding_1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=pool_kernel_size_1, stride=pool_stride_1)
                ]
                if pool_output_2[0] > 0 and pool_output_2[1] > 0:
                    fc_features = conv_output_channel_2 * pool_output_2[0] * pool_output_2[1]
                    modules.append(nn.Conv2d(conv_output_channel_1, conv_output_channel_2,
                                             kernel_size=conv_kernel_size_2, padding=conv_padding_2))
                    modules.append(nn.ReLU(inplace=True))
                    modules.append(nn.MaxPool2d(kernel_size=pool_kernel_size_2, stride=pool_stride_2))
                else:
                    fc_features = conv_output_channel_1 * pool_output_1[0] * pool_output_1[1]
            # use only one conv layer if the output dimension of the first conv layer doesn't contain 0
            # only used in the fractional aggregation scenario
            else:
                fc_features = input_dim[1] * (self.split_indices[i + 1] - self.split_indices[i])

            modules.append(nn.Flatten())
            self.features_list.append(nn.Sequential(*modules))

            self.classifier_list.append(nn.Sequential(
                nn.Linear(in_features=fc_features, out_features=512),
                nn.ReLU(),
                nn.Linear(in_features=512, out_features=64),
                nn.ReLU(),
                nn.Linear(in_features=64, out_features=param_dict['output_dim']),
            ))

        if self.cuda:
            for features in self.features_list:
                features.cuda()
            for classifier in self.classifier_list:
                classifier.cuda()

        self.output_dim = param_dict['output_dim']
        self.round_f = round_func.apply

        self.init_optim(param_dict)
        self.init_weight()

    def init_optim(self, param_dict):
        """
        initialize optimizer

        :param dict param_dict: parameters of optimizer
        """
        parameters = []

        for i in range(len(self.features_list)):
            parameters.append({'params': self.features_list[i].parameters()})
        for i in range(len(self.classifier_list)):
            parameters.append({'params': self.classifier_list[i].parameters()})

        # default use SGD except that param_dict requires using Adam
        if 'optim' in param_dict and param_dict['optim'] == 'adam':
            self.optimizer = optim.Adam(parameters, lr=param_dict['lr'])
        else:
            self.optimizer = optim.SGD(parameters,
                                       momentum=param_dict['momentum'],
                                       weight_decay=param_dict['wd'],
                                       lr=param_dict['lr'])

        # use MultiStepLR scheduler only param_dict contains stone
        if 'stone' in param_dict:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                  milestones=param_dict['stone'],
                                                                  gamma=param_dict['gamma'])

    def init_weight(self):
        for classifier in self.classifier_list:
            for layer in classifier:
                if isinstance(layer, nn.Linear):
                    init.xavier_uniform_(layer.weight, gain=1)

    def forward(self, x):
        result = None
        x = x.float()
        # support combining the predictions on multiple sensitivities, but only use the output layer updates in paper
        for i in range(len(self.features_list)):
            features = self.features_list[i]
            classifier = self.classifier_list[i]
            temp_x = torch.unsqueeze(x[:, :, self.split_indices[i]: self.split_indices[i + 1]], dim=1)
            temp_x = features(temp_x)
            temp_x = classifier(temp_x)
            result = result + temp_x if result is not None else temp_x

        result = result / len(self.features_list)
        result = F.softmax(result, dim=1)
        # limit the digits of the predicted label proportions
        result = self.round_f(result)
        return result


def attack_model(**kwargs):
    model = AttackModel(**kwargs)
    return model
