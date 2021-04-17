__author__ = '606'

import torch
import torch.optim as optim
import numpy as np
import sys,os
import time
from get_args import get_args
from trainer import train, validate, test

from numpy import arange
from numpy.random import mtrand

from encoder import ENC
from decoder import DEC

# utils for logger
class Logger(object):
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

if __name__ == '__main__':
    #################################################
    # load args & setup logger
    #################################################

    # put all printed things to log file
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    filename = './logs/log'+timestamp+'.txt'
    # if not os.path.isfile(filename):
    #     os.system(r"touch {}".format(filename))#调用系统命令行来创建文件
    logfile = open('./logs/log'+timestamp+'.txt', 'a')
    sys.stdout = Logger('./logs/log'+timestamp+'.txt', sys.stdout)
    # identity = str(np.random.random())[2:8]
    # print('[ID]', identity)
    # logfile = open('./logs/'+identity+'_log.txt', 'a')
    # sys.stdout = Logger('./logs/'+identity+'_log.txt', sys.stdout)

    args = get_args()
    print(args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("use_cuda: ",use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")

    #################################################
    # Setup Channel AE: Encoder, Decoder, Channel
    #################################################
    # choose encoder and decoder.
    # ENC = import_enc(args)
    # DEC = import_dec(args)

    encoder = ENC(args)
    decoder = DEC(args)

    # choose support channels
    from channel_ae import Channel_AE
    model = Channel_AE(args, encoder, decoder).to(device)

    # model = Channel_ModAE(args, encoder, decoder).to(device)

    # weight loading
    if args.init_nw_weight == 'default':
        pass

    else:
        pretrained_model = torch.load(args.init_nw_weight)

        try:
            model.load_state_dict(pretrained_model.state_dict(), strict = False)

        except:
            model.load_state_dict(pretrained_model, strict = False)

        model.args = args

    print(model)


    ##################################################################
    # Setup Optimizers, only Adam and Lookahead for now.
    ##################################################################

    OPT = optim.Adam

    if args.num_train_enc != 0: # no optimizer for encoder
        enc_optimizer = OPT(model.enc.parameters(),lr=args.enc_lr)

    if args.num_train_dec != 0:
        dec_optimizer = OPT(filter(lambda p: p.requires_grad, model.dec.parameters()), lr=args.dec_lr)

    general_optimizer = OPT(filter(lambda p: p.requires_grad, model.parameters()),lr=args.dec_lr)

    #################################################
    # Training Processes
    #################################################
    report_loss, report_ber = [], []

    for epoch in range(1, args.num_epoch + 1):

        if args.joint_train == 1:
            for idx in range(args.num_train_enc+args.num_train_dec):
                train(epoch, model, general_optimizer, args, use_cuda = use_cuda, mode ='encoder')

        else:
            if args.num_train_enc > 0:
                for idx in range(args.num_train_enc):
                    train(epoch, model, enc_optimizer, args, use_cuda = use_cuda, mode ='encoder')

            if args.num_train_dec > 0:
                for idx in range(args.num_train_dec):
                    train(epoch, model, dec_optimizer, args, use_cuda = use_cuda, mode ='decoder')

        this_loss, this_ber  = validate(model, general_optimizer, args, use_cuda = use_cuda)
        report_loss.append(this_loss)
        report_ber.append(this_ber)

    if args.print_test_traj == True:
        print('test loss trajectory', report_loss)
        print('test ber trajectory', report_ber)
        print('total epoch', args.num_epoch)

    #################################################
    # Testing Processes
    #################################################

    torch.save(model.state_dict(), './tmp/torch_model_'+timestamp+'.pt')
    print('saved model', './tmp/torch_model_'+timestamp+'.pt')
    # torch.save(model.state_dict(), './tmp/torch_model_'+identity+'.pt')
    # print('saved model', './tmp/torch_model_'+identity+'.pt')

    if args.is_variable_block_len:
        print('testing block length',args.block_len_low )
        test(model, args, block_len=args.block_len_low, use_cuda = use_cuda)
        print('testing block length',args.block_len )
        test(model, args, block_len=args.block_len, use_cuda = use_cuda)
        print('testing block length',args.block_len_high )
        test(model, args, block_len=args.block_len_high, use_cuda = use_cuda)

    else:
        test(model, args, use_cuda = use_cuda)