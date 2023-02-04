#!/usr/bin/env python
# -*- coding: UTF-8 -*-
__license__ = "MIT"
'''
Implementation of LLG, refers to "User-level label leakage from gradients in federated learning"

refers to the implementation on https://github.com/tklab-tud/LLG
'''
import copy
import math

import numpy as np
import torch
from torch import nn
from torch.utils import data

from attack.utils import cal_loss, get_attack_kld_label
from dataset.utils import get_dataset_labels


class LLG:
    def __init__(self, auxiliary_loader, model_arch, args, select_client=False):
        self.args = args
        self.model_arch = copy.deepcopy(model_arch)
        self.auxiliary_loader = auxiliary_loader
        self.select_client = select_client
        self.device = torch.device('cuda:0' if self.args['cuda'] else 'cpu')

    def attack_client(self, client_ids, client_loaders):
        early_round = self.args['rounds']
        targets, predicts = [], []
        global_model = copy.deepcopy(self.model_arch)
        global_model.load(name='target_{}_{}_{}client_global_{}'.format(
            'select' if self.select_client else 'all',
            self.args['target_model'], self.args['n_client'], early_round - 1))
        old_w_params = list((global_model.parameters()))[-2].data

        for index, client_loader in enumerate(client_loaders):
            local_iterations = int(math.ceil(len(client_loader.dataset)/self.args['target_batch_size']))
            # calculate the target gradients in the attack
            client_id = client_ids[index]
            client_model = copy.deepcopy(self.model_arch)
            client_model.load(name='target_{}_{}_{}client_{}_{}'.format(
                'select' if self.select_client else 'all',
                self.args['target_model'], self.args['n_client'], client_id, early_round - 1))
            new_w_params = list((client_model.parameters()))[-2].data
            result_w = old_w_params - new_w_params
            gradients = result_w / self.args['target_epochs'] / self.args['target_learning_rate']
            gradients_for_prediction = torch.sum(gradients, dim=-1).clone()

            # LLGp_prediction
            h1_extraction = []
            # do h1 extraction
            for i_cg, class_gradient in enumerate(gradients_for_prediction):
                if class_gradient < 0:
                    h1_extraction.append((i_cg, class_gradient))
            # create a new setting for impact / offset calculation
            acc_impact = 0
            acc_offset = np.zeros(self.args['num_classes'])
            n = 10
            # calculate bias and impact
            for _ in range(n):
                impact = 0
                for i in range(self.args['num_classes']):
                    temp_model = copy.deepcopy(global_model)
                    if self.args['cuda']:
                        temp_model.set_cuda()
                    criterion = temp_model.loss_function
                    dataset = self.auxiliary_loader.dataset
                    labels = get_dataset_labels(dataset, self.args['dataset'])
                    target_indices = np.where(labels == i)[0]
                    if isinstance(dataset, list):
                        subset = list([dataset[index] for index in target_indices])
                    else:
                        subset = data.Subset(dataset, target_indices)
                    train_loader = data.DataLoader(dataset=subset, batch_size=self.args['target_batch_size'],
                                                   collate_fn=self.auxiliary_loader.collate_fn)

                    # temp_model.train()
                    for batch_idx, (x, y) in enumerate(train_loader):
                        if self.args['cuda']:
                            x, y = x.cuda(), y.cuda()
                        outputs = temp_model.forward(x)
                        if isinstance(criterion, nn.BCEWithLogitsLoss):
                            outputs = outputs.squeeze(1)
                        loss = criterion(outputs, y)
                        loss.backward()
                        # temp_parameters = temp_model.parameters()
                        # temp_grad = torch.autograd.grad(loss, filter(lambda p: p.requires_grad, temp_parameters))
                        break
                    param = list(temp_model.parameters())[-2]
                    temp_gradient = param.grad
                    # temp_gradient = list((_.detach().clone() for _ in temp_grad))[-2]
                    tmp_gradients = torch.sum(temp_gradient, dim=-1).cpu().detach().numpy()
                    impact += torch.sum(temp_gradient, dim=-1)[i].item()
                    for j in range(self.args['num_classes']):
                        if j == i:
                            continue
                        else:
                            acc_offset[j] += tmp_gradients[j]
                impact /= (self.args['num_classes'] * self.args['target_batch_size'])
                acc_impact += impact
            impact = (acc_impact / n) * (1 + 1 / self.args['num_classes']) / local_iterations
            acc_offset = np.divide(acc_offset, n * (self.args['num_classes'] - 1))
            offset = torch.Tensor(acc_offset).to(gradients_for_prediction.device)
            gradients_for_prediction -= offset
            h1_extraction.sort(key=lambda y: y[1])
            # compensate h1 extraction
            prediction = []
            for (i_c, _) in h1_extraction:
                prediction.append(i_c)
                gradients_for_prediction[i_c] = gradients_for_prediction[i_c].add(-impact)
                if len(prediction) >= len(client_loader.dataset):
                    break
            # predict the rest
            for _ in range(len(client_loader.dataset) - len(prediction)):
                # add minimal candidat, likely to be present, to prediction
                min_id = torch.argmin(gradients_for_prediction).item()
                prediction.append(min_id)
                # add the mean value of one occurance to the candidate
                gradients_for_prediction[min_id] = gradients_for_prediction[min_id].add(-impact)

            # generate the ground-truth label distribution from dataloader
            labels = get_dataset_labels(client_loader.dataset, self.args['dataset'])
            all_size = len(labels)
            size_list = [len(np.where(labels == label)[0]) for label in range(self.args['num_classes'])]
            target = get_attack_kld_label(size_list, all_size)
            targets.append(target)

            # generate the inferred label distribution output by LLG+
            prediction = np.array(prediction)
            size_list = [len(np.where(prediction == label)[0]) for label in range(self.args['num_classes'])]
            predict = get_attack_kld_label(size_list, all_size)
            predicts.append(predict)
        return torch.tensor(predicts), torch.tensor(targets)

    def attack(self, client_rounds_dict, client_loaders):
        client_ids = list(range(self.args['n_client']))
        final_predicts, final_targets = self.attack_client(client_ids, client_loaders)
        return cal_loss(final_targets, final_predicts)
