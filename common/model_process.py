# -*- coding: utf-8 -*-
"""
Implementation of model training, evaluating and so on
"""
import logging
import math

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from common.utils import kl_divergence, var_loss_function
from defense.gradient_compression import GradientCompression


def train_model(model,
                model_type='',
                train_loader=None,
                test_loader=None,
                args=None,
                load=False,
                debug=False,
                save=True,
                dp=False,
                gc=False):
    """
    model training

    :param model: model
    :param model_type: model type, used to judge which model to train (target, shadow, or attack model), the local file name to save model
    :param train_loader: loader of train dataset
    :param test_loader: loader of test dataset, not evaluate test accuracy if is None
    :param args: configuration
    :param load: whether to load model from local file without training
    :param debug: whether to log debug information
    :param save: whether to save model parameters into local file
    :param dp: whether to clip gradients for dp defense during local training
    :param gc: whether to apply gradient compression defense during local training
    :return: accuracy on test dataset if test_loader is provided
    """
    acc = 0.0

    # load model from local file without training
    if load:
        if model.load(name=model_type):
            model.set_cpu()
            return

    # initialize hyper-parameters of target model
    attack = False
    if 'target' in model_type or 'shadow' in model_type:
        epochs = args['target_epochs']
    elif 'ldia' in model_type:
        epochs = args['ldia_epochs']
        attack = True

    if args['cuda']:
        model.set_cuda()
    for epoch in range(0, epochs):
        # train model in one epoch
        train_loss, train_acc = train(train_loader, model, args['cuda'],
                                      attack=attack,
                                      dp_clip=args['clip'] if dp else None,
                                      gc_percent=args['gc_percent'] if gc else None)

        if debug and epoch % 10 == 0:
            logging.warning('epoch: {}, train loss: {}, train acc: {}'.format(epoch, train_loss, train_acc))

        # evaluate performance of current model
        if test_loader is not None:
            test_loss, acc = test_model(
                model=model,
                test_loader=test_loader,
                args=args,
                attack=attack
            )
            if debug:
                logging.warning('epoch: {}, test loss: {}, test acc: {}'.format(epoch, test_loss, acc))
    if save and args['save_model']:
        model.save(name=model_type)
    model.set_cpu()
    return acc


def test_model(model, test_loader, args, attack=False):
    """
    evaluate model performance on test dataset

    :param model: model
    :param test_loader: loader of test dataset
    :param args: configuration
    :param attack: whether to evaluate attack model, True or False
    :return: loss and accuracy
    """
    criterion = model.loss_function

    # switch to evaluate mode
    model.eval()

    y_predict = []
    y_true = []
    result_losses = 0.0

    with torch.no_grad():
        if args['cuda']:
            model.set_cuda()
        for batch_idx, batch in enumerate(test_loader):
            inputs, targets = batch
            if args['cuda']:
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model.forward(inputs)
            if isinstance(outputs, tuple):
                outputs, _ = outputs

            # calculate KL-div for attack model
            if attack:
                for i in range(targets.shape[0]):
                    temp = kl_divergence(targets[i], outputs[i])
                    result_losses += temp.item()
            else:
                if isinstance(criterion, nn.BCEWithLogitsLoss):
                    outputs = outputs.squeeze(1)
                loss = criterion(outputs, targets)
                result_losses += loss.item()*targets.size(0)

                if len(targets.shape) == 1:
                    y_true += targets.data.tolist()
                    if args['num_classes'] == 2:
                        predicted = torch.round(torch.sigmoid(outputs))
                        y_predict += predicted.data.tolist()
                    else:
                        _, predicted = torch.max(outputs.data, 1)
                        y_predict += predicted.tolist()

    if len(y_true) > 0:
        acc = accuracy_score(y_true, y_predict)
        acc = acc.round(5)
    else:
        acc, precision, recall, F1 = None, None, None, None
    loss = None if math.isnan(result_losses) else round(result_losses/len(test_loader.dataset), 5)
    return loss, acc


def train(train_loader, model, use_cuda, attack=False, dp_clip=None, gc_percent=None):
    """
    train model in one epoch

    :param train_loader: loader of train dataset
    :param model: model
    :param use_cuda: whether to use cuda
    :param attack: whether model is attack model
    :param dp_clip: the clipping threshold for dp defense, None means no defense
    :param gc_percent: the compression ratio of gradient compression defense, None means no defense
    :return: model loss and accuracy in current epoch
    """
    # switch to train mode
    model.train()

    criterion = model.loss_function

    losses = 0.0
    y_predict = []
    y_true = []

    kld_loss_function = nn.KLDivLoss(reduction='sum')
    l1_loss_function = nn.SmoothL1Loss(reduction='sum')

    gc = GradientCompression(gc_percent=gc_percent)
    for batch_idx, batch in enumerate(train_loader):
        inputs, targets = batch
        if inputs.shape[0] == 1:
            continue
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model.forward(inputs)
        # train attack model using three losses
        if attack:
            eps = 1e-7
            kld_loss = kld_loss_function((outputs+eps).log(), targets)
            l1_loss = l1_loss_function(outputs, targets)
            var_loss = var_loss_function(targets, outputs)
            loss = 2 * kld_loss + 1 * l1_loss + 3 * var_loss
            loss = loss/targets.shape[0]
        else:
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                outputs = outputs.squeeze(1)
            loss = criterion(outputs, targets)
        losses += loss.item() * targets.size(0)
        loss.backward()

        # clip gradients to threshold if using dp defense
        if dp_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), dp_clip)
        # prune gradients of small magnitudes to zero if using gradient compression defense
        if gc_percent is not None:
            gc.compression(model.parameters())

        model.backward()
    model.step()

    if len(y_true) > 0:
        acc = accuracy_score(y_true, y_predict)
    else:
        acc = None
    return losses/len(train_loader.dataset), acc

