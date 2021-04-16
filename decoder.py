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

        self.dec1_rnns      = torch.nn.ModuleList()
        self.dec2_rnns      = torch.nn.ModuleList()
        self.dec1_outputs   = torch.nn.ModuleList()
        self.dec2_outputs   = torch.nn.ModuleList()

        for idx in range(args.num_iteration):
            self.dec1_rnns.append(RNN_MODEL(2 + args.num_iter_ft,  args.dec_num_unit,
                                                        num_layers=2, bias=True, batch_first=True,
                                                        dropout=args.dropout, bidirectional=True)
            )

            self.dec2_rnns.append(RNN_MODEL(2 + args.num_iter_ft,  args.dec_num_unit,
                                           num_layers=2, bias=True, batch_first=True,
                                           dropout=args.dropout, bidirectional=True)
            )

            self.dec1_outputs.append(torch.nn.Linear(2*args.dec_num_unit, args.num_iter_ft))

            if idx == args.num_iteration -1:
                self.dec2_outputs.append(torch.nn.Linear(2*args.dec_num_unit, 1))
            else:
                self.dec2_outputs.append(torch.nn.Linear(2*args.dec_num_unit, args.num_iter_ft))

    
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
#GRU
rnn=nn.GRU(10,20,2) # (input_size, hidden_state, num_layers)
input=torch.randn(5,3,10) # input
h0=torch.randn(2,3,20)
output,hn=rnn(input,h0) # GRU训练模型
print(output.size(),hn.size()) #输出
print(output)
print(hn)