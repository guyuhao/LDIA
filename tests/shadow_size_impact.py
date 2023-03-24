"""
evaluate the impact of data size of auxiliary set on LDIA
"""
import copy
import logging
import os
import sys

import numpy as np

sys.path.append(os.path.abspath('%s/..' % sys.path[0]))

from utils import remove_checkpoints
from attack.ldia.shadow_training import get_shadow_loaders
from attack.ldia.ldia_attack import LdiaAttack
from dataset.base_dataset import get_dataloader, get_test_auxiliary_loaders
from model.init_model import init_model
from common import get_args
from federated.federated_learning import federated_train

shadow_size_list = np.arange(10, 81, 10)

if __name__ == '__main__':
    # parse configuration
    args = get_args()
    remove_checkpoints(args)

    init_global_model = init_model(args)
    train_loaders, test_loader, auxiliary_loader, probe_loader = get_dataloader(args)

    for i, shadow_size in enumerate(shadow_size_list):
        args['auxiliary_size'] = shadow_size * args['num_classes']
        if i != 0:
            args['load_target'] = 1

        test_loader, auxiliary_loader, probe_loader = get_test_auxiliary_loaders(args)
        ldia_attack = LdiaAttack(probe_loader, args, model_arch=init_global_model)
        shadow_loaders = get_shadow_loaders(auxiliary_loader, args,
                                            train_size_list=[len(loader.dataset) for loader in train_loaders])

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

        logging.warning('LDIA shadow size: {}, LDIA attack: kld {}, chebyshev {}, cos {}'.format(
            shadow_size, result[0], result[1], result[2]))
