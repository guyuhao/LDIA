"""
evaluate the impact of unbalanced auxiliary set on LDIA
"""
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


shadow_non_iid = 0.8


def attack_experiment(args, distribution=None, quantity=None):
    remove_checkpoints(args)
    if distribution is not None:
        if distribution == 0:
            args['non_iid'] = 0
        else:
            args['non_iid'] = 1
            args['non_iid_alpha'] = distribution

    train_loaders, test_loader, auxiliary_loader, probe_loader = \
        get_dataloader(args, quantity=quantity, shadow_non_iid=shadow_non_iid)
    init_global_model = init_model(args)

    ldia_attack = LdiaAttack(probe_loader, args, model_arch=init_global_model)

    shadow_loaders = get_shadow_loaders(auxiliary_loader, args,
                                        train_size_list=[len(loader.dataset) for loader in train_loaders])
    global_models, client_models_dict, train_x = federated_train(
        global_model=init_global_model,
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

    logging.warning('shadow non_iid, distribution: {}, quantity: {}, '
                    'LDIA attack: kld {}, chebyshev {}, cos {}'.
                    format(distribution, quantity, result[0], result[1], result[2]))


if __name__ == '__main__':
    # parse configuration
    args = get_args()
    if args['num_classes'] == 2:
        distribution_list = [0, 0.5]
    else:
        distribution_list = [0, 0.2, 0.8]

    for distribution in distribution_list:
        attack_experiment(args, distribution=distribution)

    attack_experiment(args, quantity=1)
    if args['num_classes'] >= 6:
        attack_experiment(args, quantity=3)
    elif args['num_classes'] >= 4:
        attack_experiment(args, quantity=2)
