# -*- coding: utf-8 -*-
"""
Bi-LSTM model architecture for AG's News
"""
import torch
import torch.nn as nn
from torch import optim
from torch.nn import init

from common import CHECKPOINT_PATH
from model.base_model import BaseModel


class AgNewsLSTM(BaseModel):

    def __init__(self, param_dict):
        super(AgNewsLSTM, self).__init__(
            dataset=param_dict['dataset'],
            type='lstm',
            param_dict=param_dict
        )
        self.loss_function = nn.CrossEntropyLoss()

        self.D = 300
        self.C = 4
        self.layers = 1
        self.rnn_drop = 0
        weight_matrix = param_dict['weight']
        self.V = len(weight_matrix)
        self.static = True
        self.bidirectional = True
        self.num_directions = 2 if self.bidirectional else 1
        self.hidden_size = 256
        self.batch_size = 256

        self.embed = nn.Embedding(self.V, self.D)
        self.embed.weight.data.copy_(weight_matrix)
        if self.static:
            self.embed.weight.requires_grad = False

        self.rnn = nn.LSTM(
            input_size=self.D,  # The number of expected features in the input x
            hidden_size=self.hidden_size,  # rnn hidden unit
            num_layers=self.layers,  # number of rnn layers
            batch_first=True,  # set batch first
            dropout=self.rnn_drop,  # dropout probability
            bidirectional=self.bidirectional  # bi-LSTM
        )

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(self.num_directions * self.hidden_size, self.C)

        self.init_optim(param_dict)

        # self.init_weight()

    def init_optim(self, param_dict):
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=param_dict['lr'],
            weight_decay=param_dict['wd'])

    def set_cuda(self):
        self.embed.cuda()
        self.rnn.cuda()
        self.dropout.cuda()
        self.fc.cuda()

    def set_cpu(self):
        self.embed.cpu()
        self.rnn.cpu()
        self.dropout.cpu()
        self.fc.cpu()

    def init_weight(self):
        for name, params in self.rnn.named_parameters():
            # weight: Orthogonal Initialization
            if 'weight' in name:
                nn.init.orthogonal_(params)
            # lstm forget gate bias init with 1.0
            if 'bias' in name:
                b_i, b_f, b_c, b_o = params.chunk(4, 0)
                nn.init.ones_(b_f)
        return

    def forward(self, x):
        # x shape (batch, time_step, input_size), time_step--->seq_len
        # r_out shape (batch, time_step, output_size), out_put_size--->num_directions*hidden_size
        # h_n shape (num_layers*num_directions, batch, hidden_size)
        # c_n shape (num_layers*num_directions, batch, hidden_size)
        # (h_0,c_0), here we use zero initialization
        x = self.embed(x)  # (N, L, D)

        # initialization hidden state
        # 1.zero init
        r_out, (h_n, c_n) = self.rnn(x, None)  # None represents zero initial hidden state

        # choose r_out at the last time step or outputs at every time step
        if self.bidirectional:
            # concatenate normal RNN's last time step(-1) output and reverse RNN's last time step(0) output
            # print(r_out[:, -1, :self.hidden_size].size()) #[B, hidden_size]
            out = torch.cat([r_out[:, -1, :self.hidden_size], r_out[:, 0, self.hidden_size:]], 1)
        else:
            out = r_out[:, -1, :]  # [B, hidden_size*num_directions]

        out = self.fc(self.dropout(out))
        return out


def ag_news_lstm(**kwargs):
    model = AgNewsLSTM(**kwargs)
    return model
