# -*- coding: utf-8 -*-
from model.ag_news.ag_news_lstm import ag_news_lstm
from model.cifar.alexnet import alexnet
from model.covertype.covertype_fcn import covertype_fcn
from model.imdb.imdb_cnn import imdb_cnn
from model.mnist.mnist_cnn import mnist_cnn
from model.purchase.purchase_fcn import purchase_fcn

def init_model(args):
    """
    initialize model for parties

    :param args: configuration
    :return: model
    """
    param_dict = {
        'dataset': args['dataset'],
        'num_classes': args['num_classes'],
        'cuda': args['cuda']
    }
    # for target model
    param_dict['lr'] = args['target_learning_rate']
    param_dict['momentum'] = args['target_momentum']
    param_dict['wd'] = args['target_wd']
    # choose model architecture according to dataset
    param_dict['output_dim'] = args['num_classes']
    dataset = args['dataset']
    if dataset == 'purchase':
        return purchase_fcn(param_dict=param_dict,
                            dropout_p=None if (args['defense'] is None or args['defense'] != 'dropout') else args['dropout_p'])
    elif dataset == 'covertype':
        param_dict['optim'] = 'adam'
        return covertype_fcn(param_dict=param_dict)
    elif 'cifar' in dataset:
        return alexnet(param_dict=param_dict)
    elif dataset == 'mnist':
        return mnist_cnn(param_dict=param_dict)
    elif dataset == 'cinic':
        return alexnet(param_dict=param_dict)
    elif dataset == 'ag_news':
        param_dict['weight'] = args['weight']
        del args['weight']
        return ag_news_lstm(param_dict=param_dict)
    elif dataset == 'imdb':
        param_dict['optim'] = 'adam'
        param_dict['weight'] = args['weight']
        param_dict['vocab_size'] = args['vocab_size']
        param_dict['pad_idx'] = args['pad_idx']
        param_dict['unk_idx'] = args['unk_idx']
        del args['weight']
        return imdb_cnn(param_dict=param_dict)
