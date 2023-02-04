import os
from collections import Counter

import torch
from nltk import word_tokenize
from torch.nn.utils.rnn import pad_sequence
from torchtext.datasets import AG_NEWS
from torchtext.vocab import Vectors, Vocab

from dataset.utils import auxiliary_test_partition, \
    get_dataset_labels, client_split, generate_loaders, BucketSampler

dataset_path = '../../data/ag_news/'
vocab = None


def collate_batch(batch):
    """
    preprocess samples when training in a batch

    :param batch: batch samples
    :return: pre-processed samples
    """
    fixed_length = 100
    text_transform = lambda x: [vocab[token] for token in word_tokenize(x)]
    label_transform = lambda x: int(x)-1
    label_list, text_list = [], []
    for (_label, _text) in batch:
        label_list.append(label_transform(_label))
        processed_text = torch.tensor(text_transform(_text))[:fixed_length]
        text_list.append(processed_text)
    return pad_sequence(text_list, padding_value=0.0, batch_first=True), torch.tensor(label_list, dtype=torch.int64)


def get_dataloader(args, shadow_non_iid=None, quantity=None):
    """
    generate loader of AG's News dataset

    :param args: configuration
    :param shadow_non_iid: the unbalance level of auxiliary dataset, None means balanced
    :param quantity: the value of n for #C=n data distribution setting
    :return: tuple contains:
        (1) train_loaders: loaders of users' datasets;
        (2) test_loader: loader of test dataset;
        (3) auxiliary_loader: loader of auxiliary dataset
        (4) probe_loader: loader of probe dataset, used by Updates-Leak
    """
    train_loaders = get_train_loaders(args, quantity=quantity)
    test_loader, auxiliary_loader, probe_loader = \
        get_test_auxiliary_loaders(args, shadow_non_iid=shadow_non_iid)
    return train_loaders, test_loader, auxiliary_loader, probe_loader


def get_train_loaders(args, quantity=None):
    """
    generate loaders of users' datasets

    :param args: configuration
    :param quantity: the value of n for #C=n data distribution setting
    :return: loaders of users' datasets
    """
    global vocab
    tokenizer = word_tokenize
    train_iter = AG_NEWS(root=dataset_path, split='train')
    # build vocab from glove if it doesn't exist
    if vocab is None:
        vocab_path = os.path.join(dataset_path, 'vocab.pth')
        if os.path.exists(vocab_path):
            vocab = torch.load(vocab_path)
        else:
            vector_path = os.path.join(dataset_path, 'vectors')
            cache = os.path.join(vector_path, 'vector_cache')
            if not os.path.exists(cache):
                os.mkdir(cache)
            vectors = Vectors(
                name=os.path.join(vector_path, 'glove.840B.300d.txt'),  # torch.Size([2196017, 300])
                cache=cache)
            counter = Counter()
            for (label, line) in train_iter:
                counter.update(tokenizer(line))
            vocab = Vocab(counter, min_freq=2, vectors=vectors)
            del vectors
            torch.save(vocab, vocab_path)
    # load the pre-trained weights of the embedding layer
    args['weight'] = vocab.vectors

    train_dataset = list(train_iter)
    batch_size = args['target_batch_size']
    y = get_dataset_labels(train_dataset, args['dataset'])
    train_datasets = client_split(dataset=train_dataset, x=None, y=y, name='train', args=args, quantity=quantity)
    if isinstance(train_datasets, tuple):
        train_datasets, probability_list = train_datasets
    samplers = []
    for temp in train_datasets:
        samplers.append(BucketSampler(dataset=temp, batch_size=batch_size, tokenizer=tokenizer))
    train_loaders = generate_loaders(train_datasets, 1, shuffle=False, collate_fn=collate_batch, samplers=samplers)
    return train_loaders


def get_test_auxiliary_loaders(args, shadow_non_iid=None):
    """
    generate loaders of test dataset, auxiliary dataset and probe dataset from the original test data

    :param args: configuration
    :param shadow_non_iid: the unbalance level of auxiliary dataset, None means balanced
    :return: tuple contains:
        (1) test_loader: loader of test dataset;
        (2) auxiliary_loader: loader of auxiliary dataset
        (3) probe_loader: loader of probe dataset, used by Updates-Leak
    """
    batch_size = args['target_batch_size']
    test_iter = AG_NEWS(root=dataset_path, split='test')
    test_dataset = list(test_iter)
    auxiliary_dataset, probe_dataset, test_dataset = \
        auxiliary_test_partition(test_dataset, args, shadow_non_iid=shadow_non_iid)
    auxiliary_loader = generate_loaders(
        [auxiliary_dataset], 1, shuffle=False, collate_fn=collate_batch,
        samplers=[BucketSampler(dataset=auxiliary_dataset, batch_size=batch_size, tokenizer=word_tokenize)])[0]
    probe_loader = generate_loaders(
        [probe_dataset], 1, shuffle=False, collate_fn=collate_batch,
        samplers=[BucketSampler(dataset=probe_dataset, batch_size=batch_size, tokenizer=word_tokenize)])[0]
    test_loader = generate_loaders(
        [test_dataset], 1, shuffle=False, collate_fn=collate_batch,
        samplers=[BucketSampler(dataset=test_dataset, batch_size=batch_size, tokenizer=word_tokenize)])[0]
    return test_loader, auxiliary_loader, probe_loader
