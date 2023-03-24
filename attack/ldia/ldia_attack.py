# -*- coding: utf-8 -*-
"""
Implementation of attack model training and inferring step in LDIA
"""

import numpy as np
import torch.nn
import copy
import os
from torch.utils import data

from attack.ldia.attack_model import attack_model
from attack.utils import get_attack_kld_label, cal_loss
from common import DATA_PATH
from common.model_process import *
from dataset.utils import get_dataset_labels
from metric.param_sensitivity import param_sensitivity


def get_last_layer_length(model):
    """
    calculate the dimension of the flattened output layer parameters

    :param model: the target model
    :return: the dimension of the flattened output layer parameters
    """
    temp = list(model.parameters())[-2].data
    length_w = (torch.flatten(temp)).shape[0]
    return length_w


class LdiaAttack:
    def __init__(self, auxiliary_loader, args, model_arch, metrics=None, select_client=False):
        """
        initialize LDIA attack

        :param auxiliary_loader: loader of auxiliary set
        :param args: configuration
        :param model_arch: target model architecture
        :param metrics: sensitivities for attack model input, only output layer updates in paper
        :param select_client: whether is fractional aggregation scenario
        """
        super(LdiaAttack, self).__init__()
        # only use the output layer updates by default
        self.metrics = ['param'] if metrics is None else metrics
        self.auxiliary_loader = auxiliary_loader
        self.args = args
        self.train_dataset = None
        self.test_dataset = None
        self.model = None
        self.select_client = select_client

        self.model_arch = copy.deepcopy(model_arch)
        # save the length of attack model input for each sensitivity
        self.metric_length_dict = {
            'param': get_last_layer_length(self.model_arch)
        }

        # save indices of attack model input for each sensitivity, only use ouptut layer updates in paper
        self.metric_split_indices = [0]
        current_index = 0
        for metric in self.metrics:
            current_index += self.metric_length_dict[metric]
            self.metric_split_indices.append(current_index)

    def get_train_dataset(self, loaders=None, x=None, save=False):
        """
        get train dataset for attack model

        :param loaders: loaders of shadow datasets, used to generate labels of train dataset
        :param x: features of train dataset
        :param save: whether to save train dataset
        """
        file = DATA_PATH + 'ldia_train_data_{}.npz'.format(self.args['dataset'])

        # load train dataset from local file directly
        if self.args['load_train_data'] and os.path.isfile(file):
            temp = np.load(file)
            x = torch.from_numpy(temp['attack_x'])
            y = torch.from_numpy(temp['attack_y'])
            if self.args['cuda']:
                x = x.cuda()
                y = y.cuda()
        # generate labels of train dataset for attack model from shadow datasets
        else:
            y = None
            for temp_loader in loaders:
                temp_y = self.get_label(temp_loader)
                y = torch.cat((y, temp_y), dim=0) if y is not None else temp_y

            if save:
                np.savez(DATA_PATH + 'ldia_train_data_{}.npz'.format(self.args['dataset']),
                         attack_x=x.detach(),
                         attack_y=y.detach())
        # use Min-Max to normalize features, not used for output layer updates
        x = self.normalize(x)

        self.train_dataset = torch.utils.data.TensorDataset(x, y)

    def normalize(self, x):
        """
        use Min-Max to normalize attack model inputs, not used for output layer updates

        :param x: attack model inputs
        :return: normalized inputs
        """
        dim0 = x.shape[0]
        for i in range(dim0):
            for j in range(len(self.metric_split_indices)-1):
                if self.metrics[j] == 'param':
                    continue
                temp = x[i, :, self.metric_split_indices[j]: self.metric_split_indices[j+1]]
                max = torch.max(temp)
                min = torch.min(temp)
                temp -= min
                temp /= max - min
        return x

    def get_feature_per_round(self, index, round, train=False):
        """
        calculate sensitivities (output layer updates in paper) of each model in each round

        :param index: id of target/shadow model
        :param round: the target communication round
        :param train: whether to calculate for shadow model, True or False
        :return: sensitivities (only output layer updates in paper) of each model in each round
        """
        x = None

        # only use output layer updates (parma) in paper
        cal_functions = []
        for metric in self.metrics:
            if metric == 'param':
                cal_functions.append(param_sensitivity)

        init = copy.deepcopy(self.model_arch)
        target_model = copy.deepcopy(init)
        global_model = copy.deepcopy(init)
        # load parameters of shadow global model and shadow local model for further calculation
        if not train:
            target_model.load(name='target_{}_{}_{}client_{}_{}'.format(
                'select' if self.select_client else 'all',
                self.args['target_model'], self.args['n_client'], index, round))
            global_model_name = 'target_{}_{}_{}client_global_{}'.format(
                'select' if self.select_client else 'all',
                self.args['target_model'], self.args['n_client'], round)
            global_model.load(name=global_model_name)
        # load parameters of global model and local model for further calculation
        else:
            target_model.load(name='shadow_{}_{}_{}client_local_{}_{}'.format(
                'select' if self.select_client else 'all',
                self.args['target_model'], self.args['n_client'], index, round), delete=True)
            global_model.load(name='shadow_{}_{}_{}client_global_{}_{}'.format(
                'select' if self.select_client else 'all',
                self.args['target_model'], self.args['n_client'], index, round), delete=True)
        target_model.set_cuda()
        global_model.set_cuda()

        # calculate sensitivities (only output layer updates in paper)
        for cal_function in cal_functions:
            temp_list = cal_function(model=target_model,
                                     loader=self.auxiliary_loader,
                                     class_list=list(range(self.args['num_classes'])),
                                     args=self.args,
                                     global_model=global_model)
            temp_x = torch.tensor(temp_list)
            x = torch.cat((x, temp_x), dim=-1) if x is not None else temp_x
        return x

    def get_label(self, loader):
        """
        calculate label distribution of a dataset

        :param loader: loader of dataset
        :return: label distribution
        """
        labels = get_dataset_labels(loader.dataset, self.args['dataset'])
        all_size = len(labels)
        size_list = [len(np.where(labels == label)[0]) for label in range(self.args['num_classes'])]
        y = get_attack_kld_label(size_list, all_size)
        y = torch.unsqueeze(torch.tensor(y, dtype=torch.float32), 0)
        return y

    def get_dataset(self, ids, loader, rounds, train=False):
        """
        generate dataset for attack model, whose feature is output layer updates, label is label distribution

        :param ids: id list of target/shadow models
        :param loader: loader of target/shadow datasets, used to generate labels
        :param rounds: observed rounds
        :param train: whether to generate train dataset, True or False
        :return: tuple containing features and labels
        """
        x = None
        for round in rounds:
            temp_x = self.get_feature_per_round(ids, round, train=train)
            temp_x = torch.unsqueeze(temp_x, dim=0)
            x = torch.cat((x, temp_x), dim=0) if x is not None else temp_x
        x = torch.unsqueeze(x, dim=0)

        y = self.get_label(loader)
        return x, y

    def get_test_dataset(self, client_ids, loaders, rounds):
        """
        get test dataset for attack model and save in test_dataset

        :param client_ids: id list of target users, used to generate test features
        :param loaders: loaders of target datasets, used to generate test labels
        :param rounds: observed rounds
        """
        file = DATA_PATH + 'ldia_test_data_{}.npz'.format(self.args['dataset'])
        # load test dataset from local file directly
        if self.args['load_test_data'] and os.path.isfile(file):
            temp = np.load(file)
            x = torch.from_numpy(temp['attack_x'])
            y = torch.from_numpy(temp['attack_y'])
            if self.args['cuda']:
                x = x.cuda()
                y = y.cuda()
        # generate test dataset
        else:
            x, y = None, None

            for i, models in enumerate(client_ids):
                temp_x, temp_y = self.get_dataset(client_ids[i], loaders[i], rounds)
                x = torch.cat((x, temp_x), dim=0) if x is not None else temp_x
                y = torch.cat((y, temp_y), dim=0) if y is not None else temp_y

            if self.args['save_data']:
                np.savez(DATA_PATH + 'ldia_test_data_{}.npz'.format(self.args['dataset']),
                         attack_x=x.detach(),
                         attack_y=y.detach())
        # normalize attack inputs, not used for output layer updates
        x = self.normalize(x)

        self.test_dataset = torch.utils.data.TensorDataset(x, y)

    def init_attack_model(self):
        """
        initialize attack model

        :return: attack model
        """
        param_dict = {
            'input_dim': (len(self.metrics), self.train_dataset.tensors[0].shape[1]),
            'output_dim': self.args['num_classes'],
            'cuda': self.args['cuda'],
            'lr': self.args['ldia_learning_rate'],
            'momentum': self.args['ldia_momentum'],
            'wd': self.args['ldia_wd']
        }
        if self.args['ldia_stone'] is not None:
            param_dict['stone'] = self.args['ldia_stone']
            param_dict['gamma'] = self.args['ldia_gamma']

        param_dict['optim'] = 'adam'

        param_dict['split_indices'] = self.metric_split_indices

        return attack_model(param_dict=param_dict)

    def train_and_test(self, train=True):
        """
        train attack model of LDIA and get predictions on test dataset

        :param train: whether to train the attack model, True of False
        :return: tuple containing predicted label distributions and real distributions
        """
        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.args['ldia_batch_size'],
                                                  shuffle=False)

        if train:
            train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.args['ldia_batch_size'],
                                                       shuffle=True)
            # initialize attack model
            self.model = self.init_attack_model()
            # train attack model
            train_model(
                model=self.model,
                model_type='ldia',
                train_loader=train_loader,
                args=self.args,
                debug=self.args['debug'],
                load=self.args['load_attack']
            )

        self.model.eval()
        # get predicted label distributions and real distributions
        result_predicts, result_targets = None, None
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                if self.args['cuda']:
                    inputs, targets = inputs.cuda(), targets.cuda()
                predicts = self.model.forward(inputs)
                result_predicts = torch.cat((result_predicts, predicts), dim=0) if result_predicts is not None \
                    else predicts
                result_targets = torch.cat((result_targets, targets), dim=0) if result_targets is not None \
                    else targets
        return result_predicts, result_targets

    def get_client_clusters(self, client_rounds_dict):
        """
        cluster target users based on their selected rounds in fractional aggregation scenario, an attack model will be trained for each cluster

        :param client_rounds_dict: the selected rounds of each user
        :return: dict, key is selected rounds, value is users with the same select rounds
        """
        client_clusters = dict()
        clustered_clients = []
        for i in range(self.args['n_client']):
            if i in clustered_clients:
                continue
            target_rounds = tuple(client_rounds_dict[i]['rounds'])
            client_clusters[target_rounds] = [i]
            clustered_clients.append(i)
            for j in range(i+1, self.args['n_client']):
                if j in clustered_clients:
                    continue
                temp_rounds = tuple(client_rounds_dict[j]['rounds'])
                if temp_rounds == target_rounds:
                    client_clusters[target_rounds].append(j)
                    clustered_clients.append(j)
        return client_clusters

    def attack(self, shadow_loaders=None, all_train_x=None,
               client_rounds_dict=None, client_loaders=None,
               target_rounds=None):
        """
        conduct LDIA to get attack performance

        :param shadow_loaders: loaders of shadow datasets
        :param all_train_x: features of the training data of the attack model
        :param client_rounds_dict: the selected rounds of each user
        :param client_loaders: loaders of users' datasets
        :param target_rounds: observed rounds
        :return:
        """
        # get train dataset for attack model
        self.get_train_dataset(
            loaders=shadow_loaders,
            x=all_train_x,
            save=self.args['save_data']
        )
        # update features of training data if load train dataset from local file
        if self.args['load_train_data']:
            all_train_x = self.train_dataset.tensors[0]

        # cluster users based on their observed rounds, an attack model will be trained for each cluster
        client_clusters = self.get_client_clusters(client_rounds_dict)

        final_predicts, final_targets = None, None
        for client_rounds, client_ids in client_clusters.items():
            # get final observed rounds from users' selected rounds and preset rounds, for the fractional aggregation scenario
            if target_rounds is not None:
                client_rounds = list(set(client_rounds).intersection(set(target_rounds)))
            if len(client_rounds) == 0:
                continue
            client_rounds = client_rounds[: self.args['ldia_observed_rounds']]
            # generate train dataset of attack model
            x = all_train_x[:, client_rounds, :]
            self.get_train_dataset(
                loaders=shadow_loaders,
                x=x
            )
            # generate test dataset of attack model
            test_client_loaders = [client_loaders[i] for i in client_ids]
            self.get_test_dataset(client_ids, test_client_loaders, list(client_rounds))
            # get predicted and real distributions
            temp_predicts, temp_targets = self.train_and_test(train=True)
            final_predicts = torch.cat((final_predicts, temp_predicts), dim=0) if final_predicts is not None \
                else temp_predicts
            final_targets = torch.cat((final_targets, temp_targets), dim=0) if final_targets is not None \
                else temp_targets
        # calculate distance between distribution
        return cal_loss(final_targets, final_predicts)

