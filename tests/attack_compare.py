# -*- coding: utf-8 -*-
"""
evaluate LLG+, random and Updates-Leak on datasets
"""
import copy
import logging
import os
import sys
import warnings

from attack.llg.LLG import LLG

sys.path.append(os.path.abspath('%s/..' % sys.path[0]))
warnings.filterwarnings('ignore')

from utils import remove_checkpoints, set_seed
from dataset.base_dataset import get_dataloader
from model.init_model import init_model
from common import get_args
from federated.federated_learning import federated_train
from attack.random import random_attack
from attack.updates_leak.label_prediction import UpdatesLeak


os.environ['FLAGS_eager_delete_tensor_gb'] = '0.0'


def attack_experiment(args, select_client, distribution=None, quantity=None):
    if distribution is not None:
        if distribution == 0:
            args['non_iid'] = 0
        else:
            args['non_iid'] = 1
            args['non_iid_alpha'] = distribution

    train_loaders, test_loader, auxiliary_loader, probe_loader = \
        get_dataloader(args, quantity=quantity)
    init_global_model = init_model(args)

    global_models, client_models_dict = federated_train(
        global_model=init_global_model,
        train_loaders=train_loaders,
        test_loader=test_loader,
        rounds=args['rounds'],
        args=args,
        load_client=bool(args['load_target']),
        select_client=select_client
    )

    if not select_client:
        # random attack
        result = random_attack(train_loaders, args)
        logging.warning('distribution: {}, quantity: {}, select client: {}, '
                        'random attack: kld {}, chebyshev {}, cos {}'.
                        format(distribution, quantity, select_client, result[0], result[1], result[2]))

        # LLG+
        llg = LLG(auxiliary_loader=test_loader,
                  model_arch=copy.deepcopy(init_global_model),
                  args=args)
        result = llg.attack(
            client_rounds_dict=client_models_dict,
            client_loaders=train_loaders
        )
        logging.warning('distribution: {}, quantity: {}, select client: {}, '
                        'llg+ attack: kld {}, chebyshev {}, cos {}'.
                        format(distribution, quantity, select_client, result[0], result[1], result[2]))

        # Updates-Leak
        updates_leak = UpdatesLeak(shadow_loader=auxiliary_loader,
                                   auxiliary_loader=probe_loader,
                                   model_arch=init_global_model,
                                   args=args,
                                   select_client=select_client)
        result = updates_leak.attack(client_rounds_dict=client_models_dict, client_loaders=train_loaders)
        logging.warning('distribution: {}, quantity: {}, select client: {}, '
                        'updates-leak attack: kld {}, chebyshev {}, cos {}'.
                        format(distribution, quantity, select_client, result[0], result[1], result[2]))


def experiment(args, select_client=False):
    if args['num_classes'] == 2:
        # IID and Dir(0.5) distribution
        distribution_list = [0, 0.5]
    else:
        # IID, Dir(0.2), and Dir(0.8) distribution
        distribution_list = [0, 0.2, 0.8]
    for distribution in distribution_list:
        attack_experiment(args, select_client=select_client, distribution=distribution)

    # #C=1 distribution
    attack_experiment(args, select_client=select_client, quantity=1)
    if args['num_classes'] >= 6:
        # #C=3 distribution
        attack_experiment(args, select_client=select_client, quantity=3)
    elif args['num_classes'] >= 4:
        # #C=2 distribution
        attack_experiment(args, select_client=select_client, quantity=2)

    return


if __name__ == '__main__':
    # parse configuration
    args = get_args()

    # compare attack in complete aggregation
    remove_checkpoints(args)
    experiment(args, select_client=False)

