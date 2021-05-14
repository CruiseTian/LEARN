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
        self.attn = nn.Linear(2*dec_hid_dim, 1, bias=False)
        
    def forward(self, s, enc_output):
        
        # s = [batch_size, dec_hid_dim]
        # enc_output = [batch_size, src_len, dec_hid_dim]
        
        batch_size = enc_output.shape[0]
        src_len = enc_output.shape[1]
        
        # repeat decoder hidden state src_len times
        # s = [batch_size, src_len, dec_hid_dim]
        # enc_output = [batch_size, src_len, dec_hid_dim]
        s = s.repeat(1, src_len, 1)
        # attention = [batch_size, 1, src_len]
        attention = self.attn(torch.cat((s, enc_output), dim = 2)).transpose(1,2)
        
        return F.softmax(attention, dim=2)


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
        self.fc = torch.nn.Linear(2*args.dec_num_unit, args.dec_num_unit)

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
        # rnn_out = [batch, block_len, dec_num_unit]
        out1,_ = self.dec1_rnns(received)
        out2,_ = self.dec2_rnns(received)

        rnn_out1 = torch.zeros([out1.shape[0],out1.shape[1],out1.shape[2]]).type(torch.FloatTensor).to(self.this_device)
        rnn_out2 = torch.zeros([out2.shape[0],out2.shape[1],out2.shape[2]]).type(torch.FloatTensor).to(self.this_device)
        rnn_out1[:,0,:] = out1[:,0,:]
        rnn_out2[:,0,:] = out2[:,0,:]

        for i in range(self.args.block_len):
            if i>0 and i<5:
                # # a = [batch_size, 1, src_len]
                # a1 = self.attention(out1[:,i:i+1,:], out1[:,:i,:])
                # a2 = self.attention(out2[:,i:i+1,:], out2[:,:i,:])

                # # c = [batch_size, dec_hid_dim]
                # c1 = torch.bmm(a1, out1[:,:i,:])
                # c2 = torch.bmm(a2, out2[:,:i,:])

                # ith_out1 = self.fc(torch.cat((c1,out1[:,i:i+1,:]),dim=2)).squeeze(1)
                # ith_out2 = self.fc(torch.cat((c2,out2[:,i:i+1,:]),dim=2)).squeeze(1)

                # rnn_out1[:,i,:] = ith_out1
                # rnn_out2[:,i,:] = ith_out2
                rnn_out1[:,i,:] = out1[:,i,:]
                rnn_out2[:,i,:] = out2[:,i,:]
            if i>=5:
                # a = [batch_size, 1, src_len]
                a1 = self.attention(out1[:,i:i+1,:], out1[:,i-5:i,:])
                a2 = self.attention(out2[:,i:i+1,:], out2[:,i-5:i,:])

                # c = [batch_size, dec_hid_dim]
                c1 = torch.bmm(a1, out1[:,i-5:i,:])
                c2 = torch.bmm(a2, out2[:,i-5:i,:])

                ith_out1 = self.fc(torch.cat((c1,out1[:,i:i+1,:]),dim=2)).squeeze(1)
                ith_out2 = self.fc(torch.cat((c2,out2[:,i:i+1,:]),dim=2)).squeeze(1)

                rnn_out1[:,i,:] = ith_out1
                rnn_out2[:,i,:] = ith_out2

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