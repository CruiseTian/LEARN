__author__ = '606'
import argparse

def get_args():
    ################################
    # Setup Parameters and get args
    ################################
    parser = argparse.ArgumentParser()

    ################################################################
    # Channel related parameters
    ################################################################
    parser.add_argument('-channel', choices = ['awgn',             # AWGN
                                               't-dist',           # Non-AWGN, ATN, with -vv associated
                                               'radar',            # Non-AWGN, Radar, with -radar_prob, radar_power, associated
                                               ],
                        default = 'awgn')
    # Channel parameters
    parser.add_argument('-vv',type=float, default=5, help ='only for t distribution channel')

    parser.add_argument('-radar_prob',type=float, default=0.05, help ='only for radar distribution channel')
    parser.add_argument('-radar_power',type=float, default=5.0, help ='only for radar distribution channel')

    # continuous channels training algorithms
    parser.add_argument('-train_enc_channel_low', type=float, default  = 1.0)
    parser.add_argument('-train_enc_channel_high', type=float, default = 1.0)
    parser.add_argument('-train_dec_channel_low', type=float, default  = -1.5)
    parser.add_argument('-train_dec_channel_high', type=float, default = 2.0)

    parser.add_argument('-init_nw_weight', type=str, default='default')

    # code rate is k/n, so that enable multiple code rates. This has to match the encoder/decoder nw structure.
    parser.add_argument('-code_rate_k', type=int, default=1)
    parser.add_argument('-code_rate_n', type=int, default=3)

    ################################################################
    # TurboAE encoder/decoder parameters
    ################################################################
    parser.add_argument('-enc_rnn', choices=['gru', 'lstm', 'rnn'], default='gru')
    parser.add_argument('-dec_rnn', choices=['gru', 'lstm', 'rnn'], default='gru')

    # CNN/RNN related
    parser.add_argument('-enc_num_layer', type=int, default=2)
    parser.add_argument('-dec_num_layer', type=int, default=5)


    parser.add_argument('-dec_num_unit', type=int, default=100, help = 'This is CNN number of filters, and RNN units')
    parser.add_argument('-enc_num_unit', type=int, default=100, help = 'This is CNN number of filters, and RNN units')

    parser.add_argument('-enc_act', choices=['tanh', 'selu', 'relu', 'elu', 'sigmoid', 'linear'], default='elu', help='only elu works')
    parser.add_argument('-dec_act', choices=['tanh', 'selu', 'relu', 'elu', 'sigmoid', 'linear'], default='linear')

    ################################################################
    # Training ALgorithm related parameters
    ################################################################
    parser.add_argument('-joint_train', type=int, default=0, help ='if 1, joint train enc+dec, 0: seperate train')
    parser.add_argument('-num_train_dec', type=int, default=5, help ='')
    parser.add_argument('-num_train_enc', type=int, default=1, help ='')

    parser.add_argument('-dropout',type=float, default=0.0)

    parser.add_argument('-snr_test_start', type=float, default=-1.5)
    parser.add_argument('-snr_test_end', type=float, default=4.0)
    parser.add_argument('-snr_points', type=int, default=12)

    parser.add_argument('-batch_size', type=int, default=100)
    parser.add_argument('-num_epoch', type=int, default=1)
    parser.add_argument('-test_ratio', type=int, default=1,help = 'only for high SNR testing')
    # block length related
    parser.add_argument('-block_len', type=int, default=100)
    parser.add_argument('-block_len_low', type=int, default=10)
    parser.add_argument('-block_len_high', type=int, default=200)
    parser.add_argument('--is_variable_block_len', action='store_true', default=False,
                        help='training with different block length')

    parser.add_argument('-num_block', type=int, default=1000)

    parser.add_argument('-test_channel_mode',
                        choices=['block_norm','block_norm_ste'],
                        default='block_norm')
    parser.add_argument('-train_channel_mode',
                        choices=['block_norm','block_norm_ste'],
                        default='block_norm')
    parser.add_argument('-enc_truncate_limit', type=float, default=0, help='0 means no truncation')

    parser.add_argument('--no_code_norm', action='store_true', default=False,
                        help='the output of encoder is not normalized. Modulation do the work')



    ################################################################
    # STE related parameters
    ################################################################
    parser.add_argument('-enc_quantize_level', type=float, default=2, help = 'only valid for block_norm_ste')
    parser.add_argument('-enc_value_limit', type=float, default=1.0, help = 'only valid for block_norm_ste')
    parser.add_argument('-enc_grad_limit', type=float, default=0.01, help = 'only valid for block_norm_ste')
    parser.add_argument('-enc_clipping', choices=['inputs', 'gradient', 'both', 'none'], default='both',
                        help = 'only valid for ste')

    ################################################################
    # Optimizer related parameters
    ################################################################
    parser.add_argument('-optimizer', choices=['adam', 'lookahead', 'sgd'], default='adam', help = '....:)')
    parser.add_argument('-dec_lr', type = float, default=0.001, help='decoder leanring rate')
    parser.add_argument('-enc_lr', type = float, default=0.001, help='encoder leanring rate')
    # parser.add_argument('-momentum', type = float, default=0.9)

    ################################################################
    # Loss related parameters
    ################################################################

    parser.add_argument('-loss', choices=['bce', 'mse','focal', 'bce_block', 'maxBCE', 'bce_rl', 'enc_rl', 'soft_ber', 'sortBCE'],
                        default='bce', help='only BCE works')

    parser.add_argument('-ber_lambda', type = float, default=1.0, help = 'default 0.0, the more emphasis on BER loss, only for bce_rl')
    parser.add_argument('-bce_lambda', type = float, default=1.0, help = 'default 1.0, the more emphasis on BCE loss, only for bce_rl')


    # focal loss related things
    parser.add_argument('-focal_gamma', type = float, default=0.0, help = 'default gamma=0,BCE; 5 is to emphasis wrongly decoded cases.')
    parser.add_argument('-focal_alpha', type = float, default=1.0, help = 'default alpha=1.0, adjust the loss term')

    # mixture term
    parser.add_argument('-lambda_maxBCE', type = float, default=0.01, help = 'add term to maxBCE loss, only wokr with maxBCE loss')

    ################################################################
    # MISC
    ################################################################
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--rec_quantize', action='store_true', default=False,
                        help='binarize received signal, which will degrade performance a lot')

    parser.add_argument('--print_pos_ber', action='store_true', default=False,
                        help='print positional ber when testing BER')
    parser.add_argument('--print_pos_power', action='store_true', default=False,
                        help='print positional power when testing BER')
    parser.add_argument('--print_test_traj', action='store_true', default=False,
                        help='print test trajectory when testing BER')
    parser.add_argument('--precompute_norm_stats', action='store_true', default=False,
                        help='Use pre-computed mean/std statistics')

    ################################################################
    # Experimental
    ################################################################
    parser.add_argument('--is_k_same_code', action='store_true', default=False,
                        help='train with same code for multiple times')
    parser.add_argument('-k_same_code', type = int, default=2, help = 'add term to maxBCE loss, only wokr with maxBCE loss')

    parser.add_argument('-D', type = int, default=1, help = 'delay')


    args = parser.parse_args()

    return args
