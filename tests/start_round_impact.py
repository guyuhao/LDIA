"""
evaluate the impact of observed round ranges on LDIA
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

if __name__ == '__main__':
    # parse configuration
    args = get_args()

    remove_checkpoints(args)
    args['ldia_observed_rounds'] = args['rounds']

    train_loaders, test_loader, auxiliary_loader, probe_loader = get_dataloader(args)
    init_global_model = init_model(args)

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

    round_size = 10
    rounds_list = np.arange(0, args['rounds']//round_size, 1)
    # compare attack in complete aggregation
    for i in rounds_list:
        start_round = i * round_size
        end_round = (i+1) * round_size
        result = ldia_attack.attack(
            shadow_loaders=shadow_loaders,
            all_train_x=copy.deepcopy(train_x),
            client_rounds_dict=client_models_dict,
            client_loaders=train_loaders,
            target_rounds=list(range(start_round, end_round))
        )

        logging.warning('start round: {}, LDIA attack: kld {}, chebyshev {}, cos {}'.format(
            start_round, result[0], result[1], result[2]))
