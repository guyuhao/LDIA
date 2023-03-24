# -*- coding: utf-8 -*-
"""
evaluate LDIA under dropout defense with various dropout rate
"""
import logging
import warnings

import numpy as np

warnings.filterwarnings('ignore')

import copy
import os
import sys

sys.path.append(os.path.abspath('%s/..' % sys.path[0]))

from utils import remove_checkpoints
from attack.ldia.shadow_training import get_shadow_loaders
from attack.ldia.ldia_attack import LdiaAttack
from dataset.base_dataset import get_dataloader
from model.init_model import init_model
from common import get_args
from federated.federated_learning import federated_train

dropout_p_list = np.arange(0.1, 1.0, 0.1)

if __name__ == '__main__':
    # parse configuration
    args = get_args()
    args['defense'] = 'dropout'

    train_loaders, test_loader, auxiliary_loader, probe_loader = get_dataloader(args)

    # LDIA attack
    shadow_loaders = get_shadow_loaders(auxiliary_loader, args,
                                        train_size_list=[len(loader.dataset) for loader in train_loaders])

    for i, dropout_p in enumerate(dropout_p_list):
        args['dropout_p'] = dropout_p
        remove_checkpoints(args)
        init_global_model = init_model(args)

        ldia_attack = LdiaAttack(probe_loader, args, model_arch=init_global_model)

        global_models, client_models_dict, train_x = federated_train(
            global_model=copy.deepcopy(init_global_model),
            train_loaders=train_loaders,
            test_loader=test_loader,
            rounds=args['rounds'],
            args=args,
            load_client=bool(args['load_target']),
            attack=True,
            shadow_loaders=shadow_loaders,
            ldia_attack=ldia_attack
        )

        result = ldia_attack.attack(
            shadow_loaders=shadow_loaders,
            all_train_x=train_x,
            client_rounds_dict=client_models_dict,
            client_loaders=train_loaders)

        logging.warning('dropout p: {}, LDIA attack: kld {}, chebyshev {}, cos {}'.format(
            round(dropout_p, 1), result[0], result[1], result[2]))

