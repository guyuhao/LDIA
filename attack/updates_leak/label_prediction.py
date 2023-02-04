"""
Created on 25 Jan 2019

@author: ahmed.salem

Implementation of Updates-Leak, refers to "Updates-Leak: Data Set Inference and Reconstruction Attacks in Online Learning"

refers to the implementation on https://github.com/AhmedSalem2/Updates-Leak
"""
import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler

from attack.updates_leak.network import labelPredMNIST
from attack.utils import cal_loss, get_attack_kld_label
from common import train_model
from dataset.utils import get_dataset_labels

train_size = 10000

class UpdatesLeak:
    def __init__(self, shadow_loader, auxiliary_loader, model_arch, args, select_client=False):
        self.args = args
        self.model_arch = copy.deepcopy(model_arch)
        self.shadow_loader = shadow_loader
        self.auxiliary_loader = auxiliary_loader
        self.select_client = select_client
        self.device = torch.device("cuda:0" if self.args['cuda'] else "cpu")

    def get_client_clusters(self, client_rounds_dict, client_loaders):
        client_clusters = dict()
        for i in range(self.args['n_client']):
            data_size = len(client_loaders[i].dataset)
            if data_size not in client_clusters.keys():
                client_clusters[data_size] = [i]
            else:
                client_clusters[data_size].append(i)
        return client_clusters

    # Train the attack model.
    def train_attack_model(self, model, train_loader, num_epochs=100, learning_rate=1e-3):
        criterion = nn.KLDivLoss(reduction='batchmean')
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        for epoch in range(num_epochs):
            for mini_batch_data, label in train_loader:
                mini_batch_data = mini_batch_data.to(self.device)
                label = label.to(self.device)
                # ===================forward=====================
                output = model(mini_batch_data)
                loss = criterion(output, label)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return model

    # Test the attack model.
    def test_attack_model(self, model, test_dataset):
        outputs = None
        loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
        # Testing the attack and baseline for the different batches.
        for mini_batch_data in loader:
            mini_batch_data = mini_batch_data.to(self.device)
            output_log = model(mini_batch_data)
            output = torch.exp(output_log)
            outputs = torch.cat((outputs, output), dim=0) if output is not None else output
        return outputs

    def get_diff(self, ori_model, update_model):
        ori_model.eval()
        update_model.eval()

        ori_output_points, update_output_points = None, None
        softmax = nn.Softmax(dim=1)

        criterion = ori_model.loss_function
        with torch.no_grad():
            ori_model = ori_model.to(self.device)
            update_model = update_model.to(self.device)
            for (dataHolder, labelsHolder) in self.auxiliary_loader:
                images, labels = dataHolder.to(self.device), labelsHolder.to(self.device)
                if isinstance(criterion, nn.BCEWithLogitsLoss):
                    ori_outputs = ori_model.forward(images)
                    ori_outputs = torch.sigmoid(ori_outputs)
                    ori_outputs = torch.cat((1 - ori_outputs, ori_outputs), dim=1)
                    ori_output_points = torch.cat((ori_output_points, ori_outputs), dim=0) \
                        if ori_output_points is not None else ori_outputs

                    update_outputs = update_model.forward(images)
                    update_outputs = torch.sigmoid(update_outputs)
                    update_outputs = torch.cat((1 - update_outputs, update_outputs), dim=1)
                    update_output_points = torch.cat((update_output_points, update_outputs), dim=0) \
                        if update_output_points is not None else update_outputs
                else:
                    ori_outputs = softmax(ori_model.forward(images))
                    ori_output_points = torch.cat((ori_output_points, ori_outputs), dim=0) \
                        if ori_output_points is not None else ori_outputs

                    update_outputs = softmax(update_model.forward(images))
                    update_output_points = torch.cat((update_output_points, update_outputs), dim=0) \
                        if update_output_points is not None else update_outputs

        output_diff = ori_output_points - update_output_points
        output_diff = torch.flatten(output_diff)
        output_diff = output_diff.cpu().data
        return output_diff

    def gen_loaders(self, data_size, count):
        all_dataset = self.shadow_loader.dataset
        if isinstance(all_dataset, torch.utils.data.Subset):
            all_indices = np.array(all_dataset.indices)
            all_dataset = all_dataset.dataset
        else:
            all_indices = np.arange(len(all_dataset))

        data_size = min(data_size, len(all_indices))
        result_indices = []
        for i in range(count):
            temp_indices = np.random.choice(all_indices, data_size, replace=False)
            result_indices.append(temp_indices)
        shadow_loaders = []
        for indices in result_indices:
            if isinstance(all_dataset, list):
                dataset = list([all_dataset[index] for index in indices])
                loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.args['target_batch_size'],
                                                     shuffle=True, collate_fn=self.shadow_loader.collate_fn)
            else:
                dataset = torch.utils.data.Subset(all_dataset, indices)
                loader = torch.utils.data.DataLoader(dataset=dataset,
                                                     batch_size=self.args['target_batch_size'], shuffle=True)
            shadow_loaders.append(loader)
        return shadow_loaders

    def gen_train_data(self, ori_model, data_size):
        train_loaders = self.gen_loaders(data_size, train_size)

        diffs, y = None, None
        # updating the model for a "numOfModels" times in parallel, i.e., the model is updated on each data batch independently
        for i in range(0, train_size):
            temp_ori_model = copy.deepcopy(ori_model)

            train_model(model=temp_ori_model,
                        model_type='shadow',
                        train_loader=train_loaders[i],
                        args=self.args,
                        save=False)
            diff = self.get_diff(ori_model=ori_model, update_model=temp_ori_model)
            diff = torch.unsqueeze(diff, dim=0)
            diffs = torch.cat((diffs, diff), dim=0) if diffs is not None else diff

            labels = get_dataset_labels(train_loaders[i].dataset, self.args['dataset'])
            # get target
            size_list = [len(np.where(labels == label)[0]) for label in range(self.args['num_classes'])]
            temp_y = get_attack_kld_label(size_list, len(labels))
            temp_y = torch.unsqueeze(torch.tensor(temp_y, dtype=torch.float32), 0)
            y = torch.cat((y, temp_y), dim=0) if y is not None else temp_y
            del temp_ori_model
        dataset = torch.utils.data.TensorDataset(diffs, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)
        return loader

    def attack_client(self, data_size, client_ids, client_loaders):
        targets = []
        global_model = copy.deepcopy(self.model_arch)
        global_model.load(name='target_{}_{}_{}client_global_{}'.format(
            'select' if self.select_client else 'all',
            self.args['target_model'], self.args['n_client'], self.args['rounds']-1))

        train_loader = self.gen_train_data(ori_model=global_model, data_size=data_size)
        model = labelPredMNIST(attackInput=self.args['num_classes'] * len(self.auxiliary_loader.dataset),
                               numOfClasses=self.args['num_classes'])
        model = model.to(self.device)
        model = self.train_attack_model(model, train_loader, num_epochs=50)

        test_diffs = None
        for i, client_loader in enumerate(client_loaders):
            # logging.warning('data size: {}, client {}'.format(data_size, client_ids[i]))
            labels = get_dataset_labels(client_loader.dataset, self.args['dataset'])
            all_size = len(labels)

            # get target
            size_list = [len(np.where(labels == label)[0]) for label in range(self.args['num_classes'])]
            target = get_attack_kld_label(size_list, all_size)
            targets.append(target)

            client_id = client_ids[i]
            client_model, = copy.deepcopy(self.model_arch),
            client_model.load(name='target_{}_{}_{}client_{}_{}'.format(
                'select' if self.select_client else 'all',
                self.args['target_model'], self.args['n_client'], client_id, self.args['rounds']-1))

            diff = self.get_diff(ori_model=global_model, update_model=client_model)
            diff = torch.unsqueeze(diff, dim=0)
            test_diffs = torch.cat((test_diffs, diff), dim=0) if test_diffs is not None else diff
            del client_model
        predicts = (torch.exp(model(test_diffs.to(self.device)))).cpu().data
        return predicts, torch.tensor(targets)

    def attack(self, client_rounds_dict, client_loaders):
        data_size = self.args['client_train_size']
        client_ids = list(range(self.args['n_client']))
        final_predicts, final_targets = self.attack_client(data_size, client_ids, client_loaders)
        return cal_loss(final_targets, final_predicts)
