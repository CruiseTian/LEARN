import torch.nn.functional as F
from torch.nn import Parameter
import torch
from torch import nn

from numpy import arange
from numpy.random import mtrand
import numpy as np

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(2*dec_hid_dim, dec_hid_dim, bias=False)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, s, enc_output):
        
        # s = [2, batch_size, dec_hid_dim]
        # enc_output = [2, batch_size, src_len, dec_hid_dim]
        
        batch_size = enc_output.shape[1]
        src_len = enc_output.shape[2]
        
        # repeat decoder hidden state src_len times
        # s = [2, batch_size, src_len, dec_hid_dim]
        # enc_output = [2, batch_size, src_len, dec_hid_dim]
        s = s.unsqueeze(2).repeat(1, 1, src_len, 1)
        # energy = [2, batch_size, src_len, dec_hid_dim]
        energy = torch.tanh(self.attn(torch.cat((s, enc_output), dim = 3)))
        
        # attention = [2, batch_size, 1, src_len]
        attention = self.v(energy).transpose(2,3)
        
        return F.softmax(attention, dim=3)


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

        # attention机制实现
        self.attention = Attention(args.enc_num_unit, args.dec_num_unit)
        self.fc = torch.nn.Linear(int(args.code_rate_n/args.code_rate_k), args.enc_num_unit)
        self.fc2 = torch.nn.Linear(2*args.dec_num_unit, args.dec_num_unit)

        self.dec1_rnns = RNN_MODEL(int(args.code_rate_n/args.code_rate_k),  args.dec_num_unit,
                                                    num_layers=args.dec_num_layer, bias=True, batch_first=True,
                                                    dropout=args.dropout)

        self.dec2_rnns = RNN_MODEL(int(args.code_rate_n/args.code_rate_k),  args.dec_num_unit,
                                        num_layers=args.dec_num_layer, bias=True, batch_first=True,
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
        # received = [batch_size, src_len, rate]
        received = received.type(torch.FloatTensor).to(self.this_device)

        enc_output = self.dec_act(self.fc(received))

        hidden1 = torch.zeros([self.args.dec_num_layer,self.args.batch_size,self.args.dec_num_unit]).type(torch.FloatTensor).to(self.this_device)
        hidden2 = torch.zeros([self.args.dec_num_layer,self.args.batch_size,self.args.dec_num_unit]).type(torch.FloatTensor).to(self.this_device)

        for i in range(self.args.block_len):
            # dec_input = [batch_size, 1, rate]
            dec_input = received[:,i:i+1,:]
            dec_output1, hidden1 = self.dec1_rnns(dec_input, hidden1)
            dec_output2, hidden2 = self.dec2_rnns(dec_input, hidden2)
            
            if i==0:
                rnn_out1 = dec_output1
                rnn_out2 = dec_output2
                hiddens1 = hidden1.unsqueeze(2)
                hiddens2 = hidden2.unsqueeze(2)
            else:
                # rnn_out = [batch, block_len, dec_num_unit]
                rnn_out1 = torch.cat((rnn_out1,dec_output1),dim = 1)
                rnn_out2 = torch.cat((rnn_out2,dec_output2),dim = 1)

                # hiddens = [2, batch_size, block_len, dec_num_unit]
                hiddens1 = torch.cat((hiddens1,hidden1.unsqueeze(2)),dim = 2)
                hiddens2 = torch.cat((hiddens2,hidden2.unsqueeze(2)),dim = 2)

                # a = [2, batch_size, 1, src_len]
                a1 = self.attention(hidden1, hiddens1)
                a2 = self.attention(hidden2, hiddens2)

                # c = [2, batch_size, dec_hid_dim]
                c1 = torch.matmul(a1, hiddens1).squeeze(2)
                c2 = torch.matmul(a2, hiddens2).squeeze(2)

                hidden1 = self.fc2(torch.cat((c1, hidden1), dim=2))
                hidden2 = self.fc2(torch.cat((c2, hidden2), dim=2))


        for i in range(self.args.block_len):
            if (i>=self.args.block_len-self.args.D-1):
                rt_d = rnn_out2[:,self.args.block_len-1:self.args.block_len,:]
            else:
                rt_d = rnn_out2[:,i+self.args.D:i+self.args.D+1,:]
            rt = rnn_out1[:,i:i+1,:]
            rnn_out = torch.cat((rt, rt_d), dim=2)
            dec_out = self.dec_act(self.dec_outputs(rnn_out))
            if i==0:
                final = dec_out
            else:
                final = torch.cat((final,dec_out),dim=1)
        final = torch.sigmoid(final)

        return final