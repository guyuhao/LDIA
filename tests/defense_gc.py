# -*- coding: utf-8 -*-
"""
evaluate LDIA under gradient compression defense with various compression ratio
"""
import logging
import warnings

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

percent_list = [0.80, 0.85, 0.90, 0.95, 0.99]

if __name__ == '__main__':
    # parse configuration
    args = get_args()
    args['defense'] = 'gradient_compression'

    train_loaders, test_loader, auxiliary_loader, probe_loader = get_dataloader(args)
    init_global_model = init_model(args)

    # LDIA attack
    ldia_attack = LdiaAttack(probe_loader, args, model_arch=init_global_model)
    shadow_loaders = get_shadow_loaders(auxiliary_loader, args,
                                        train_size_list=[len(loader.dataset) for loader in train_loaders])

    for i, percent in enumerate(percent_list):
        args['gc_percent'] = percent

        remove_checkpoints(args)

        global_models, client_models_dict, train_x = federated_train(
            global_model=copy.deepcopy(init_global_model),
            train_loaders=train_loaders,
            test_loader=test_loader,
            rounds=args['rounds'],
            args=args,
            load_client=bool(args['load_target']),
            attack=True,
            shadow_loaders=shadow_loaders,
            ldia_attack=ldia_attack,
            gc=True
        )

        result = ldia_attack.attack(
            shadow_loaders=shadow_loaders,
            all_train_x=train_x,
            client_rounds_dict=client_models_dict,
            client_loaders=train_loaders)

        logging.warning('percent: {}, LDIA attack: kld {}, chebyshev {}, cos {}'.format(
            round(percent, 2), result[0], result[1], result[2]))

    remove_checkpoints(args)