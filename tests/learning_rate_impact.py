"""
evaluate the impact of the learning rate on LDIA in complete aggregation
"""
import copy
import logging
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

lr_list = [5e-2, 5e-3, 1e-3, 5e-4, 1e-4]

if __name__ == '__main__':
    # parse configuration
    args = get_args()

    train_loaders, test_loader, auxiliary_loader, probe_loader = get_dataloader(args)
    shadow_loaders = get_shadow_loaders(auxiliary_loader, args,
                                        train_size_list=[len(loader.dataset) for loader in train_loaders])

    for i, lr in enumerate(lr_list):
        args['target_learning_rate'] = lr

        init_global_model = init_model(args)
        ldia_attack = LdiaAttack(probe_loader, args, select_client=False, model_arch=init_global_model)

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
            select_client=False
        )

        result = ldia_attack.attack(
            shadow_loaders=shadow_loaders,
            all_train_x=train_x,
            client_rounds_dict=client_models_dict,
            client_loaders=train_loaders)

        logging.warning('learning rate: {}, LDIA attack: kld {}, chebyshev {}, cos {}'.format(
            lr, result[0], result[1], result[2]))
