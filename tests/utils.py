#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Implementation of general functions in experiments
"""
import os
import random
import shutil

import numpy as np
import torch
from torch.backends import cudnn

from common import CHECKPOINT_PATH


def set_seed(seed):
    """
    fix seed in experiments

    :param seed: seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    cudnn.deterministic = True


def remove_checkpoints(args):
    """
    remove existing checkpoints for the new experiment

    :param args: configuration
    """
    path = '{}/{}'.format(CHECKPOINT_PATH, args['dataset'])
    if args['save_model'] == 1 and args['load_target'] == 0 and args['load_shadow'] == 0:
        shutil.rmtree(path)
        os.makedirs(path)
