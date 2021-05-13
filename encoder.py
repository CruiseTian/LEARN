import torch
import torch.nn.functional as F

from numpy import arange
from numpy.random import mtrand
import math
import numpy as np
from utils import STEQuantize 
    
class ENC(torch.nn.Module):
    def __init__(self, args):
        super(ENC, self).__init__()

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args
        self.reset_precomp()

        # Encoder

        if args.enc_rnn == 'gru':
            RNN_MODEL = torch.nn.GRU
        elif args.enc_rnn == 'lstm':
            RNN_MODEL = torch.nn.LSTM
        else:
            RNN_MODEL = torch.nn.RNN


        self.enc_rnn       = RNN_MODEL(1, args.enc_num_unit,
                                           num_layers=args.enc_num_layer, bias=True, batch_first=True,
                                           dropout=0)

        self.enc_linear    = torch.nn.Linear(args.enc_num_unit, int(args.code_rate_n/args.code_rate_k))

    def set_precomp(self, mean_scalar, std_scalar):
        self.mean_scalar = mean_scalar.to(self.this_device)
        self.std_scalar  = std_scalar.to(self.this_device)

    # not tested yet
    def reset_precomp(self):
        self.mean_scalar = torch.zeros(1).type(torch.FloatTensor).to(self.this_device)
        self.std_scalar  = torch.ones(1).type(torch.FloatTensor).to(self.this_device)
        self.num_test_block= 0.0

    def enc_act(self, inputs):
        if self.args.enc_act == 'tanh':
            return  F.tanh(inputs)
        elif self.args.enc_act == 'elu':
            return F.elu(inputs)
        elif self.args.enc_act == 'relu':
            return F.relu(inputs)
        elif self.args.enc_act == 'selu':
            return F.selu(inputs)
        elif self.args.enc_act == 'sigmoid':
            return F.sigmoid(inputs)
        elif self.args.enc_act == 'linear':
            return inputs
        else:
            return inputs

    def power_constraint(self, x_input):

        if self.args.no_code_norm:
            return x_input
        else:
            this_mean    = torch.mean(x_input)
            this_std     = torch.std(x_input)

            if self.args.precompute_norm_stats:
                self.num_test_block += 1.0
                self.mean_scalar = (self.mean_scalar*(self.num_test_block-1) + this_mean)/self.num_test_block
                self.std_scalar  = (self.std_scalar*(self.num_test_block-1) + this_std)/self.num_test_block
                x_input_norm = (x_input - self.mean_scalar)/self.std_scalar
            else:
                x_input_norm = (x_input-this_mean)*1.0 / this_std

            if self.args.train_channel_mode == 'block_norm_ste':
                stequantize = STEQuantize.apply
                x_input_norm = stequantize(x_input_norm, self.args)

            if self.args.enc_truncate_limit>0:
                x_input_norm = torch.clamp(x_input_norm, -self.args.enc_truncate_limit, self.args.enc_truncate_limit)

            return x_input_norm

    def forward(self, inputs):
        output, hidden = self.enc_rnn(inputs)
        code = self.enc_act(self.enc_linear(output))
        codes = self.power_constraint(code)
        return codes