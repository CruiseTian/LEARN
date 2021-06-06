import torch
import torch.nn.functional as F
import numpy as np

from utils import STEQuantize as MyQuantize

class Channel_AE(torch.nn.Module):
    def __init__(self, args, enc, dec):
        super(Channel_AE, self).__init__()
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args
        self.enc = enc
        self.dec = dec

    def forward(self, input, fwd_noise):
        codes = self.enc(input)

        if self.args.channel in ['awgn', 't-dist', 'radar']:
            received_codes = codes + fwd_noise
        else:
            print('default AWGN channel')
            received_codes = codes + fwd_noise

        if self.args.rec_quantize:
            myquantize = MyQuantize.apply
            received_codes = myquantize(received_codes, self.args)

        x_dec = self.dec(received_codes)

        return x_dec, codes