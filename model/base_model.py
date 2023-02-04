"""
Base class of all model architecture
"""
import os
from abc import abstractmethod

import torch
from torch import nn, optim

from common.constants import CHECKPOINT_PATH


class BaseModel(nn.Module):
    def __init__(self, dataset, type, param_dict):
        nn.Module.__init__(self)
        self.dataset = dataset  # dataset name
        self.type = type  # model type, support bert, fcn, and resnet
        self.cuda = param_dict['cuda']  # whether to use cuda, 1 means use and 0 otherwise
        self.learning_rate = param_dict['lr']  # learning rate of the model

        self.optimizer = None
        self.scheduler = None
        self.loss_function = None

    def init_optim(self, param_dict):
        """
        initialize optimizer

        :param dict param_dict: parameters of optimizer
        """
        parameters = self.parameters()

        # default use SGD except that param_dict requires using Adam
        if 'optim' in param_dict and param_dict['optim'] == 'adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, parameters), lr=param_dict['lr'])
        else:
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, parameters),
                                       momentum=param_dict['momentum'],
                                       weight_decay=param_dict['wd'],
                                       lr=param_dict['lr'])

        # use MultiStepLR scheduler only param_dict contains stone
        if 'stone' in param_dict:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                  milestones=param_dict['stone'],
                                                                  gamma=param_dict['gamma'])

    def save(self, name=None, id=None):
        """
        save model to local file located in CHECKPOINT_PATH/"dataset"

        :param str name: name of local file, use default name if not provided
        :param int id: user id
        """
        path = '{}/{}'.format(CHECKPOINT_PATH, self.dataset)
        if name is None:
            if id is None:
                filepath = '{}/{}_{}'.format(path, self.role, self.type)  # for active party
            else:
                filepath = '{}/{}_{}_{}'.format(path, self.role, self.type, id)  # for passive party
        else:
            filepath = '{}/{}'.format(path, name)
        temp_dict = self.state_dict()
        # ignore parameters of the embedding layer for text datasets to save memory
        if self.dataset in ['ag_news', 'imdb']:
            del temp_dict['embed.weight']

        torch.save(temp_dict, filepath)

    def load(self, name=None, id=None, delete=False):
        """
        load model from local file located in CHECKPOINT_PATH/"dataset"

        :param name: name of local file, use default name if not provided
        :param id: id of passive party
        :param bool delete: whether delete local file after loading
        :return: bool, whether to load successfully or not
        """
        path = '{}/{}'.format(CHECKPOINT_PATH, self.dataset)
        if name is None:
            if id is None:
                filepath = '{}/{}_{}'.format(path, self.role, self.type)  # for active party
            else:
                filepath = '{}/{}_{}_{}'.format(path, self.role, self.type, id)  # for passive party
        else:
            filepath = '{}/{}'.format(path, name)
        if os.path.isfile(filepath):
            checkpoint = torch.load(filepath)
            self.load_state_dict(checkpoint, strict=False)
            if delete:
                os.remove(filepath)
            return True
        else:
            print('error load {}'.format(name))
            return False

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        x = x.float()
        if self.cuda:
            x = x.cuda()
        return self.classifier(x)

    def predict(self, x):
        return self.forward(x)

    def backward(self):
        """
        backward using gradients of optimizer
        """
        self.optimizer.step()
        self.optimizer.zero_grad()

    def step(self):
        if self.scheduler is not None:
            self.scheduler.step()

    @abstractmethod
    def set_cuda(self):
        pass

    @abstractmethod
    def set_cpu(self):
        pass
