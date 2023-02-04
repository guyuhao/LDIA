# -*- coding: utf-8 -*-
"""
Preprocess Purchase100 dataset
randomly select 40000 records to build test dataset, and remaining as train dataset
"""
import random
from random import shuffle

import numpy as np
import pandas as pd

DATA_PATH = '../../data/purchase/'
TEST_SIZE = 40000

with open(DATA_PATH+"dataset_purchase", "r") as f:
    dataset = f.readlines()
shuffle(dataset)
print(len(dataset))

test_set = dataset[:TEST_SIZE]
train_set = dataset[TEST_SIZE:]
with open(DATA_PATH+"dataset_purchase_train", "w") as f:
    f.writelines(train_set)
with open(DATA_PATH+"dataset_purchase_test", "w") as f:
    f.writelines(test_set)
