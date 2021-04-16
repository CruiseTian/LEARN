__author__ = '606'
import torch
import numpy as np
import math

def errors_ber(y_true, y_pred, positions = 'default'):
    y_true = y_true.view(y_true.shape[0], -1, 1)
    y_pred = y_pred.view(y_pred.shape[0], -1, 1)

    myOtherTensor = torch.ne(torch.round(y_true), torch.round(y_pred)).float()
    if positions == 'default':
        res = sum(sum(myOtherTensor))/(myOtherTensor.shape[0]*myOtherTensor.shape[1])
    else:
        res = torch.mean(myOtherTensor, dim=0).type(torch.FloatTensor)
        for pos in positions:
            res[pos] = 0.0
        res = torch.mean(res)
    return res

def errors_ber_list(y_true, y_pred):
    block_len = y_true.shape[1]
    y_true = y_true.view(y_true.shape[0], -1)
    y_pred = y_pred.view(y_pred.shape[0], -1)

    myOtherTensor = torch.ne(torch.round(y_true), torch.round(y_pred))
    res_list_tensor = torch.sum(myOtherTensor, dim = 1).type(torch.FloatTensor)/block_len

    return res_list_tensor


def errors_ber_pos(y_true, y_pred, discard_pos = []):
    y_true = y_true.view(y_true.shape[0], -1, 1)
    y_pred = y_pred.view(y_pred.shape[0], -1, 1)

    myOtherTensor = torch.ne(torch.round(y_true), torch.round(y_pred)).float()

    tmp =  myOtherTensor.sum(0)/myOtherTensor.shape[0]
    res = tmp.squeeze(1)
    return res

def code_power(the_codes):
    the_codes = the_codes.cpu().numpy()
    the_codes = np.abs(the_codes)**2
    the_codes = the_codes.sum(2)/the_codes.shape[2]
    tmp =  the_codes.sum(0)/the_codes.shape[0]
    res = tmp
    return res

def errors_bler(y_true, y_pred, positions = 'default'):

    y_true = y_true.view(y_true.shape[0], -1, 1)
    y_pred = y_pred.view(y_pred.shape[0], -1, 1)

    decoded_bits = torch.round(y_pred)
    X_test       = torch.round(y_true)
    tp0 = (abs(decoded_bits-X_test)).view([X_test.shape[0],X_test.shape[1]])
    tp0 = tp0.cpu().numpy()

    if positions == 'default':
        bler_err_rate = sum(np.sum(tp0,axis=1)>0)*1.0/(X_test.shape[0])
    else:
        for pos in positions:
            tp0[:, pos] = 0.0
        bler_err_rate = sum(np.sum(tp0,axis=1)>0)*1.0/(X_test.shape[0])

    return bler_err_rate

# note there are a few definitions of SNR. In our result, we stick to the following SNR setup.
def snr_db2sigma(train_snr):
    return 10**(-train_snr*1.0/20)

def snr_sigma2db(train_sigma):
    try:
        return -20.0 * math.log(train_sigma, 10)
    except:
        return -20.0 * torch.log10(train_sigma)

def generate_noise(noise_shape, args, test_sigma = 'default', snr_low = 0.0, snr_high = 0.0, mode = 'encoder'):
    # SNRs at training
    if test_sigma == 'default':
        if args.channel == 'bec':
            if mode == 'encoder':
                this_sigma = args.bec_p_enc
            else:
                this_sigma = args.bec_p_dec

        elif args.channel in ['bsc', 'ge']:
            if mode == 'encoder':
                this_sigma = args.bsc_p_enc
            else:
                this_sigma = args.bsc_p_dec
        else: # general AWGN cases
            this_sigma_low = snr_db2sigma(snr_low)
            this_sigma_high= snr_db2sigma(snr_high)
            # mixture of noise sigma.
            this_sigma = (this_sigma_low - this_sigma_high) * torch.rand(noise_shape) + this_sigma_high

    else:
        if args.channel in ['bec', 'bsc', 'ge']:  # bsc/bec noises
            this_sigma = test_sigma
        else:
            this_sigma = snr_db2sigma(test_sigma)

    # SNRs at testing
    if args.channel == 'awgn':
        fwd_noise  = this_sigma * torch.randn(noise_shape, dtype=torch.float)

    elif args.channel == 't-dist':
        fwd_noise  = this_sigma * torch.from_numpy(np.sqrt((args.vv-2)/args.vv) * np.random.standard_t(args.vv, size = noise_shape)).type(torch.FloatTensor)

    elif args.channel == 'radar':
        add_pos     = np.random.choice([0.0, 1.0], noise_shape,
                                       p=[1 - args.radar_prob, args.radar_prob])

        corrupted_signal = args.radar_power* np.random.standard_normal( size = noise_shape ) * add_pos
        fwd_noise = this_sigma * torch.randn(noise_shape, dtype=torch.float) +\
                    torch.from_numpy(corrupted_signal).type(torch.FloatTensor)

    elif args.channel == 'bec':
        fwd_noise = torch.from_numpy(np.random.choice([0.0, 1.0], noise_shape,
                                        p=[this_sigma, 1 - this_sigma])).type(torch.FloatTensor)

    elif args.channel == 'bsc':
        fwd_noise = torch.from_numpy(np.random.choice([0.0, 1.0], noise_shape,
                                        p=[this_sigma, 1 - this_sigma])).type(torch.FloatTensor)
    elif args.channel == 'ge_awgn':
        #G-E AWGN channel
        p_gg = 0.8         # stay in good state
        p_bb = 0.8
        bsc_k = snr_db2sigma(snr_sigma2db(this_sigma) + 1)          # accuracy on good state
        bsc_h = snr_db2sigma(snr_sigma2db(this_sigma) - 1)   # accuracy on good state

        fwd_noise = np.zeros(noise_shape)
        for batch_idx in range(noise_shape[0]):
            for code_idx in range(noise_shape[2]):

                good = True
                for time_idx in range(noise_shape[1]):
                    if good:
                        if test_sigma == 'default':
                            fwd_noise[batch_idx,time_idx, code_idx] = bsc_k[batch_idx,time_idx, code_idx]
                        else:
                            fwd_noise[batch_idx,time_idx, code_idx] = bsc_k
                        good = np.random.random()<p_gg
                    elif not good:
                        if test_sigma == 'default':
                            fwd_noise[batch_idx,time_idx, code_idx] = bsc_h[batch_idx,time_idx, code_idx]
                        else:
                            fwd_noise[batch_idx,time_idx, code_idx] = bsc_h
                        good = np.random.random()<p_bb
                    else:
                        print('bad!!! something happens')

        fwd_noise = torch.from_numpy(fwd_noise).type(torch.FloatTensor)* torch.randn(noise_shape, dtype=torch.float)

    elif args.channel == 'ge':
        #G-E discrete channel
        p_gg = 0.8         # stay in good state
        p_bb = 0.8
        bsc_k = 1.0        # accuracy on good state
        bsc_h = this_sigma# accuracy on good state

        fwd_noise = np.zeros(noise_shape)
        for batch_idx in range(noise_shape[0]):
            for code_idx in range(noise_shape[2]):

                good = True
                for time_idx in range(noise_shape[1]):
                    if good:
                        tmp = np.random.choice([0.0, 1.0], p=[1-bsc_k, bsc_k])
                        fwd_noise[batch_idx,time_idx, code_idx] = tmp
                        good = np.random.random()<p_gg
                    elif not good:
                        tmp = np.random.choice([0.0, 1.0], p=[ 1-bsc_h, bsc_h])
                        fwd_noise[batch_idx,time_idx, code_idx] = tmp
                        good = np.random.random()<p_bb
                    else:
                        print('bad!!! something happens')

        fwd_noise = torch.from_numpy(fwd_noise).type(torch.FloatTensor)

    else:
        # Unspecific channel, use AWGN channel.
        fwd_noise  = this_sigma * torch.randn(noise_shape, dtype=torch.float)

    return fwd_noise