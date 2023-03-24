"""
evaluate the impact of the number of selected participants on LDIA in fractional aggregation
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

# select_client_number_list = range(10, 91, 20)
select_client_number_list = [10, 30, 50, 70, 90]

if __name__ == '__main__':
    # parse configuration
    args = get_args()

    train_loaders, test_loader, auxiliary_loader, probe_loader = get_dataloader(args)
    init_global_model = init_model(args)

    ldia_attack = LdiaAttack(probe_loader, args, select_client=True, model_arch=init_global_model)
    shadow_loaders = get_shadow_loaders(auxiliary_loader, args,
                                        train_size_list=[len(loader.dataset) for loader in train_loaders])

    for i, select_client_number in enumerate(select_client_number_list):
        args['n_selected_client'] = select_client_number

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
            select_client=True
        )

        result = ldia_attack.attack(
            shadow_loaders=shadow_loaders,
            all_train_x=train_x,
            client_rounds_dict=client_models_dict,
            client_loaders=train_loaders)

        logging.warning('select client number: {}, LDIA attack: kld {}, chebyshev {}, cos {}'.format(
            select_client_number, result[0], result[1], result[2]))
