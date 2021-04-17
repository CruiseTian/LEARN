import torch.nn.functional as F
from torch.nn import Parameter
import torch

from numpy import arange
from numpy.random import mtrand
import numpy as np

class DEC(torch.nn.Module):
    def __init__(self, args):
        super(DEC, self).__init__()
        self.args = args

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        if args.dec_rnn == 'gru':
            RNN_MODEL = torch.nn.GRU
        elif args.dec_rnn == 'lstm':
            RNN_MODEL = torch.nn.LSTM
        else:
            RNN_MODEL = torch.nn.RNN

        self.dropout = torch.nn.Dropout(args.dropout)

        self.dec1_rnns = RNN_MODEL(int(args.code_rate_n/args.code_rate_k),  args.dec_num_unit,
                                                    num_layers=2, bias=True, batch_first=True,
                                                    dropout=args.dropout)

        self.dec2_rnns = RNN_MODEL(int(args.code_rate_n/args.code_rate_k),  args.dec_num_unit,
                                        num_layers=2, bias=True, batch_first=True,
                                        dropout=args.dropout)

        self.dec_outputs = torch.nn.Linear(2*args.dec_num_unit, 1)

    
    def dec_act(self, inputs):
        if self.args.dec_act == 'tanh':
            return  F.tanh(inputs)
        elif self.args.dec_act == 'elu':
            return F.elu(inputs)
        elif self.args.dec_act == 'relu':
            return F.relu(inputs)
        elif self.args.dec_act == 'selu':
            return F.selu(inputs)
        elif self.args.dec_act == 'sigmoid':
            return F.sigmoid(inputs)
        elif self.args.dec_act == 'linear':
            return inputs
        else:
            return inputs

    def forward(self, received):
        received = received.type(torch.FloatTensor).to(self.this_device)

        rnn_out1,_ = self.dec1_rnns(received)
        rnn_out2,_ = self.dec2_rnns(received)

        for i in range(self.args.block_len):
            if (i>=self.args.block_len-self.args.D-1):
                rt_d = rnn_out2[:,self.args.block_len-1:self.args.block_len,:]
            else:
                rt_d = rnn_out2[:,i+self.args.D:i+self.args.D+1,:]
            rt = rnn_out1[:,i:i+1,:]
            rnn_out = torch.cat((rt, rt_d), dim=2)
            dec_out = self.dec_outputs(rnn_out)
            if i==0:
                final = dec_out
            else:
                final = torch.cat((final,dec_out),dim=1)
        final = torch.sigmoid(final)


        # rnn_out = torch.cat([rnn_out1, rnn_out2], dim = 2)

        # dec_out = torch.sigmoid(self.dec_outputs(rnn_out))

        return final