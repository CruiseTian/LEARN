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
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim, bias=False)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, s, enc_output):
        
        # s = [batch_size, dec_hid_dim]
        # enc_output = [src_len, batch_size, enc_hid_dim * 2]
        
        batch_size = enc_output.shape[0]
        src_len = enc_output.shape[1]
        
        # repeat decoder hidden state src_len times
        # s = [batch_size, src_len, dec_hid_dim]
        # enc_output = [batch_size, src_len, enc_hid_dim * 2]
        s = s.unsqueeze(1).repeat(1, src_len, 1)
        enc_output = enc_output.transpose(0, 1)
        
        # energy = [batch_size, src_len, dec_hid_dim]
        energy = torch.tanh(self.attn(torch.cat((s, enc_output), dim = 2)))
        
        # attention = [batch_size, src_len]
        attention = self.v(energy).squeeze(2)
        
        return F.softmax(attention, dim=1)


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
        self.attn = torch.nn.Linear(args.dec_num_unit * 2, 100)
        self.attn_combine = torch.nn.Linear(args.dec_num_unit * 2, args.dec_num_unit)

        self.dec1_rnns = RNN_MODEL(int(args.code_rate_n/args.code_rate_k),  args.dec_num_unit,
                                                    num_layers=2, bias=True, batch_first=True,
                                                    dropout=args.dropout)

        self.dec2_rnns = RNN_MODEL(int(args.code_rate_n/args.code_rate_k),  args.dec_num_unit,
                                        num_layers=2, bias=True, batch_first=True,
                                        dropout=args.dropout)

        self.dec_outputs = torch.nn.Linear(args.dec_num_unit, 1)

    
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

    def forward(self, dec_input, s, enc_output):
        # dec_input = [batch_size]
        # s = [batch_size, dec_hid_dim]
        # enc_output = [src_len, batch_size, enc_hid_dim * 2]
        
        dec_input = dec_input.unsqueeze(1) # dec_input = [batch_size, 1]
        print("shape of dec_input: ",dec_input.shape)

        # attention 部分
        print("shape of s: ",s.shape)

        embedded = dec_input.transpose(0, 1) # embedded = [1, batch_size, emb_dim]

        last_hidden = torch.zeros([2,bs,self.args.dec_num_unit],dtype=torch.float)
        dec_input = input[:,0]
        for i in range(self.args.block_len):
            if i == 0:
                rnn_output, hidden = self.dec1_rnns(dec_input = input[:,0])
            else:
                rnn_output, hidden = self.dec1_rnns(received[:,i,:].reshape(self.args.batch_size,1,-1),hidden)
            print("dec hidden shape: ", hidden.shape)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], input_hidden[0]), 1)), dim=1)
        print("shape of attn_weights: ",attn_weights.shape)

        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 received.unsqueeze(0))
        print("shape of attn_applied: ",attn_applied.shape)
 
        att_out = torch.cat((embedded[0], attn_applied[0]), 1)
        att_out = self.attn_combine(att_out).unsqueeze(0)
        att_out = F.relu(att_out)


        rnn_out1,_ = self.dec1_rnns(att_out)
        rnn_out2,_ = self.dec2_rnns(att_out)

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
        # final = F.log_softmax(final, dim=1)

        return final