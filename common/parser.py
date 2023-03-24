# -*- coding: utf-8 -*-
"""
Parse configuration file
"""

import argparse
import logging
import yaml

from dataset.base_dataset import get_num_classes


def get_args():
    """
    parse configuration yaml file

    :return: configuration
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    temp = parser.parse_args()
    yaml.warnings({'YAMLLoadWarning': False})
    f = open(temp.config, 'r', encoding='utf-8')
    cfg = f.read()
    args = yaml.load(cfg)
    f.close()
    if 'num_classes' not in args.keys():
        args['num_classes'] = get_num_classes(args['dataset'])

    if args['target_train_size'] is not None:
        args['client_train_size'] = int(args['target_train_size']/args['n_client']) \
            if args['client_train_size'] is None else args['client_train_size']

    if 'ldia_observed_rounds' not in args.keys() or args['ldia_observed_rounds'] is None:
        args['ldia_observed_rounds'] = args['rounds']

    set_logging(args['log'])

    logging.warning(args)
    return args


def set_logging(log_file):
    """
    configure logging INFO messaged located in tests/result

    :param str log_file: path of log file
    """
    logging.basicConfig(
        level=logging.WARNING,
        filename='../tests/result/{}'.format(log_file),
        filemode='w',
        format='[%(asctime)s| %(levelname)s| %(processName)s] %(message)s'
    )
