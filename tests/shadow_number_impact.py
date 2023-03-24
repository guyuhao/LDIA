"""
evaluate the impact of the number of shadow models on LDIA
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
from dataset.base_dataset import get_dataloader
from model.init_model import init_model
from common import get_args
from federated.federated_learning import federated_train

max_shadow_number = 1200

if __name__ == '__main__':
    # parse configuration
    args = get_args()
    remove_checkpoints(args)

    train_loaders, test_loader, auxiliary_loader, probe_loader = get_dataloader(args)
    init_global_model = init_model(args)

    args['ldia_shadow_number'] = max_shadow_number

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

    shadow_number_list = np.arange(100, max_shadow_number+1, 100)
    for i, shadow_number in enumerate(shadow_number_list):
        if i != 0:
            args['load_test_data'] = 1
        temp_train_x = (copy.deepcopy(train_x))[:shadow_number, ]
        temp_shadow_loaders = shadow_loaders[:shadow_number]

        result = ldia_attack.attack(
            shadow_loaders=temp_shadow_loaders,
            all_train_x=temp_train_x,
            client_rounds_dict=client_models_dict,
            client_loaders=train_loaders)

        logging.warning('shadow number: {}, LDIA attack: kld {}, chebyshev {}, cos {}'.format(
            shadow_number, result[0], result[1], result[2]))

