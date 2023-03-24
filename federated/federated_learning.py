# -*- coding: utf-8 -*-
"""
Implementation of standalone federated learning
"""

import copy
import logging
from random import randrange

import numpy as np

from attack.ldia.shadow_training import ShadowTrain
from common import train_model, test_model
from defense.Privacy import Privacy_account
from defense.differential_privacy import client_differential_privacy
from federated.utils import fed_aggregation


def federated_train(global_model, rounds, train_loaders, args,
                    test_loader=None, target_clients_dict=dict(), select_client=False, load_client=False,
                    attack=False, shadow_loaders=None, ldia_attack=None,
                    dp=False, gc=False):
    """
    federated training during rounds in rounds

    :param global_model: global model
    :param rounds: rounds of FL
    :param train_loaders: loaders of train datasets of all clients
    :param args: configuration
    :param test_loader: loader of test dataset
    :param target_clients_dict: specify selected clients in each round if valid, the key is round (start from 0)
    :param select_client: whether to aggregate fractional clients in each round, False means complete aggregation
    :param bool load_client: whether to load local model parameters from local file
    :param bool attack: whether to perform LDIA
    :param shadow_loaders: loaders of shadow datasets, valid only if attack is True
    :param ldia_attack: the reference of LdiaAttack, valid only if attack is True
    :param dp: whether to use dp defense
    :param gc: whether to use gc defense
    :return:
        tuple containing with LDIA:
            (1) global_models: empty, not used
            (2) client_model_dict: selected rounds of each user, the key is user-id (start from 0)
            (3) train_x: features of training data for the LDIA attack model
        tuple containing without attack:
            (1) global_models: empty, not used
            (2) client_model_dict: selected rounds of each user, the key is user-id (start from 0)
    """

    global_models = []

    if attack:
        shadow_train = ShadowTrain(shadow_loaders=shadow_loaders, init_model=global_model, select_client=select_client,
                                   args=args, ldia_attack=ldia_attack, dp=dp)

    noise_scale_list = []
    # determine the standard deviation of additive noises for each participant if using dp defense
    if dp:
        threshold_epochs = copy.deepcopy(args['rounds'])
        threshold_epochs_list, noise_list = [], []
        for i in range(args['n_client']):
            noise_scale = copy.deepcopy(Privacy_account(
                args, threshold_epochs, noise_list, 0, 'Origi', len(train_loaders[i].dataset)))
            noise_scale_list.append(noise_scale)

    client_model_dict = {}
    for i in range(args['n_client']):
        client_model_dict[i] = {'rounds': [], 'models': [], 'data_size': 0}
    for round in range(rounds):
        # save global model in current round
        if args['save_model']:
            if select_client:
                model_type = 'target_select_{}_{}client_global_{}'.format(
                    args['target_model'], args['n_client'], round)
            else:
                model_type = 'target_all_{}_{}client_global_{}'.format(
                    args['target_model'], args['n_client'], round)
            global_model.save(name=model_type)

        target_clients = np.arange(args['n_client'])
        # select clients to aggregate in current round for fractional aggregation
        if select_client:
            if round in target_clients_dict.keys():
                target_clients = target_clients_dict[round]
            else:  # randomly select clients if not specified
                target_clients = np.random.choice(target_clients, args['n_selected_client'], replace=False)
                target_clients_dict[round] = target_clients

        client_models = list()

        # local training of all clients
        for index in target_clients:
            # local training of each client
            client_model, client_data_size = client_train(
                global_model=global_model,
                client_index=index,
                round=round,
                select_client=select_client,
                args=args,
                train_loader=train_loaders[index],
                load=load_client,
                dp=dp,
                noise_scale=noise_scale_list[index] if dp else None,
                gc=gc
            )

            # save model parameters, selected rounds and data size for each user
            client_models.append({'index': index, 'model': client_model, 'data_size': client_data_size})
            client_model_dict[index]['rounds'].append(round)
            client_model_dict[index]['data_size'] = client_data_size
        # perform LDIA shadow model training in each round
        if attack:
            if select_client or round < args['ldia_observed_rounds']:
                shadow_train.train_per_round(client_models=client_models, round=round)
        # aggregate to get global model for normal federated training or active global CS-MIA
        global_model = fed_aggregation(
            global_model=global_model,
            client_models=client_models,
            args=args,
            dp=dp
        )

        # evaluate global model if test_loader is provided
        if test_loader is not None:
            if round % 5 == 0 or round == rounds - 1:
                te_loss, te_acc = test_model(model=global_model, test_loader=test_loader, args=args)
                logging.warning('Round {} | test loss {} test acc {}'.format(
                    round, te_loss, te_acc))
        else:
            if round % 5 == 0 or round == rounds - 1:
                logging.warning('Round {}'.format(round))
        del client_models
    if not attack:
        return global_models, client_model_dict
    else:
        return global_models, client_model_dict, shadow_train.train_x


def client_train(global_model, client_index, round, select_client, args, train_loader,
                 load=False, dp=False, noise_scale=None, gc=False):
    """
    local training for each user in each round

    :param global_model: distributed global model in current round
    :param client_index: id of current user
    :param round: current round
    :param bool select_client: whether is fractional aggregation
    :param args: configuration
    :param train_loader: loader of current user's training dataset
    :param bool load: whether to load trained model parameters from local file
    :param bool dp: whether to use dp defense
    :param noise_scale: the standard deviation of additive noises, valid only if dp is True
    :param bool gc: whether to use gradient compression defense
    :return: tuple containing trained local model parameters and data size of current user
    """
    global dimensions
    model = copy.deepcopy(global_model)

    if select_client:
        model_type = 'target_select_{}_{}client_{}_{}'.format(args['target_model'], args['n_client'], client_index,
                                                              round)
    else:
        model_type = 'target_all_{}_{}client_{}_{}'.format(args['target_model'], args['n_client'], client_index,
                                                           round)
    # train local model
    train_model(model=model,
                model_type=model_type,
                train_loader=train_loader,
                test_loader=None,
                args=args,
                load=load,
                dp=dp,
                gc=gc)
    # add noise if using dp defense
    if dp:
        client_differential_privacy(model=model,
                                    args=args,
                                    noise_scale=noise_scale)

    result_dict = model.state_dict()
    # ignore parameters in the embedding layer for text datasets to save memory
    if args['dataset'] in ['ag_news', 'imdb']:
        del result_dict['embed.weight']

    return result_dict, len(train_loader.dataset)



