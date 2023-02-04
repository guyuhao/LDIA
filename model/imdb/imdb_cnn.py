#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
CNN model architecture for IMDB dataset
"""
import torch
from torch import nn
import torch.nn.functional as F

from common import CHECKPOINT_PATH
from model.base_model import BaseModel


class ImdbCnn(BaseModel):
    def __init__(self, param_dict):
        super(ImdbCnn, self).__init__(
            dataset=param_dict['dataset'],
            type='cnn',
            param_dict=param_dict
        )
        self.loss_function = nn.BCEWithLogitsLoss()

        vocab_size = param_dict['vocab_size']
        embedding_dim = 100
        output_dim = 1
        pad_idx = param_dict['pad_idx']
        unk_idx = param_dict['unk_idx']
        self.embed = nn.Embedding.from_pretrained(param_dict['weight'], padding_idx=pad_idx)

        n_filters = 100
        filter_sizes = [3, 5, 7]
        dropout_rate = 0.5
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.embed.weight.data[unk_idx] = torch.zeros(embedding_dim)
        self.embed.weight.data[pad_idx] = torch.zeros(embedding_dim)

        self.init_optim(param_dict)

    def init_weight(self):
        for m in self.convs:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, text):
        # text = [batch size, sent len]

        embedded = self.embed(text)

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)

    def set_cuda(self):
        self.embed.cuda()
        self.convs.cuda()
        self.fc.cuda()
        self.dropout.cuda()

    def set_cpu(self):
        self.embed.cpu()
        self.convs.cpu()
        self.fc.cpu()
        self.dropout.cpu()


def imdb_cnn(**kwargs):
    model = ImdbCnn(**kwargs)
    return model
