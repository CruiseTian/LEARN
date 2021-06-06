'''
Input file follows the following format. All lines are
mandatory. No comments can be made within the file.
Must be followed strictly.
'''

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import os, sys
import numpy as np

from pylab import rcParams
import matplotlib.pylab as pylab

rcParams['legend.numpoints'] = 1
mpl.style.use('seaborn')
# plt.rcParams['axes.facecolor']='binary'
# print(rcParams.keys())
params = {
    'font.family': 'sans-serif',
    'font.sans-serif': 'Times New Roman',
    'font.weight': 'bold'
}
pylab.rcParams.update(params)

def plot(lr):
    fig, ax = plt.subplots(1, 3, figsize=(15, 6), tight_layout=True)
    ax = ax.ravel()
    s = ['Loss', 'BER', 'BLER']
    snrs =  [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    bler_d1_1_001 = [0.9964999999999999, 0.9902999999999998, 0.9718, 0.9379, 0.8751, 0.7842, 0.6458999999999999, 0.5028, 0.35530000000000006, 0.23699999999999996, 0.1437, 0.07880000000000002]
    bler_d10_1_001 = [0.9942, 0.9829000000000001, 0.9583, 0.9091999999999999, 0.8275, 0.7047999999999999, 0.5689, 0.4257000000000001, 0.2861, 0.185, 0.10680000000000003, 0.05400000000000003]
    bler_d1_5_0001 = [0.9963799999999998, 0.9896399999999999, 0.9720199999999997, 0.9386000000000003, 0.8721799999999998, 0.7729800000000003, 0.6465600000000002, 0.50378, 0.35778, 0.24078000000000008, 0.14485999999999996, 0.08214000000000002]
    bler_d10_5_0001 = [0.9920999999999998, 0.98068, 0.9511600000000004, 0.8996600000000002, 0.8137599999999998, 0.69864, 0.5549600000000001, 0.4095600000000001, 0.2833400000000001, 0.17533999999999994, 0.10410000000000003, 0.05456]

    d1_1_001 = np.loadtxt("./data/data_awgn_lr_0.01_D1_10000.txt",usecols=[1, 2], ndmin=2).T
    d10_1_001 = np.loadtxt("./data/data_awgn_lr_0.01_D10_10000.txt",usecols=[1, 2], ndmin=2).T
    d1_5_0001 = np.loadtxt("./data/data_awgn_lr_0.001_D1_50000.txt",usecols=[1, 2], ndmin=2).T
    d10_5_0001 = np.loadtxt("./data/data_awgn_lr_0.001_D10_50000.txt",usecols=[1, 2], ndmin=2).T
    n = d1_1_001.shape[1]  # number of rows
    x = range(0,n)
    for i in range(2):
        if lr==0.01:
            y1 = d1_1_001[i, x]
            y2 = d10_1_001[i, x]
        else:
            y1 = d1_5_0001[i, x]
            y2 = d10_5_0001[i, x]
        ax[i].plot(x, y1, '-', c='#e41b1b', label="D=1", linewidth=2, markersize=6)
        ax[i].plot(x, y2, '-', c='#377eb8', label="D=10", linewidth=2, markersize=6)
        ax[i].set_title(s[i],fontweight='bold',fontsize=16)
        ax[i].legend(loc='best',fancybox=True, framealpha=0,fontsize=14)
        ax[i].grid(True, linestyle='dotted')  # x坐标轴的网格使用主刻度
        # ax.yaxis.grid(True, linestyle='dotted')  # y坐标轴的网格使用次刻度

    if lr==0.01:
        ax[2].plot(snrs, bler_d1_1_001, '-', c='#e41b1b', label="D=1", linewidth=2, markersize=6)
        ax[2].plot(snrs, bler_d10_1_001, '-', c='#377eb8', label="D=10", linewidth=2, markersize=6)
        ax[2].set_title(s[2],fontweight='bold',fontsize=16)
        ax[2].legend(loc='best',fancybox=True, framealpha=0,fontsize=14)
        ax[2].grid(True, linestyle='dotted')
    else:
        ax[2].plot(snrs, bler_d1_5_0001, '-', c='#e41b1b', label="D=1", linewidth=2, markersize=6)
        ax[2].plot(snrs, bler_d10_5_0001, '-', c='#377eb8', label="D=10", linewidth=2, markersize=6)
        ax[2].set_title(s[2],fontweight='bold',fontsize=16)
        ax[2].legend(loc='best',fancybox=True, framealpha=0,fontsize=14)
        ax[2].grid(True, linestyle='dotted')
    plt.savefig('lr'+str(lr)+'.png', format='png', bbox_inches='tight', transparent=True, dpi=800)
    plt.show()

# plot(0.01)
# plot(0.001)

def plot_attn(lr,D):
    fig, ax = plt.subplots(1, 3, figsize=(15, 6), tight_layout=True)
    ax = ax.ravel()
    s = ['Loss', 'BER', 'BLER']
    filename1 = './data/data_awgn_lr_'+str(lr)+'_D1_10000.txt'
    filename2 = './data/data_awgn_lr_'+str(lr)+'_D10_10000.txt'
    attn_filename1 = './data/attention_data_awgn_lr_'+str(lr)+'_D1_10000.txt'
    attn_filename2 = './data/attention_data_awgn_lr_'+str(lr)+'_D10_10000.txt'

    data1 = np.loadtxt(filename1,usecols=[1, 2], ndmin=2).T
    data2 = np.loadtxt(filename2,usecols=[1, 2], ndmin=2).T
    attn_data1 = np.loadtxt(attn_filename1,usecols=[1, 2], ndmin=2).T
    attn_data2 = np.loadtxt(attn_filename2,usecols=[1, 2], ndmin=2).T

    snrs =  [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    bler1 = [0.9964999999999999, 0.9902999999999998, 0.9718, 0.9379, 0.8751, 0.7842, 0.6458999999999999, 0.5028, 0.35530000000000006, 0.23699999999999996, 0.1437, 0.07880000000000002]
    bler2 = [0.9942, 0.9829000000000001, 0.9583, 0.9091999999999999, 0.8275, 0.7047999999999999, 0.5689, 0.4257000000000001, 0.2861, 0.185, 0.10680000000000003, 0.05400000000000003]
    attn_bler1 = [0.9963, 0.9907, 0.976, 0.9396000000000001, 0.8700999999999999, 0.7794999999999999, 0.6569999999999999, 0.5147, 0.36810000000000004, 0.238, 0.1451, 0.08390000000000003]
    attn_bler2 = [0.9799000000000001, 0.9519999999999997, 0.8960000000000001, 0.7959000000000002, 0.6552, 0.5089999999999999, 0.36280000000000007, 0.24710000000000001, 0.1608, 0.09130000000000002, 0.05030000000000001, 0.029100000000000008]

    n = data1.shape[1]  # number of rows
    x = range(0,n)
    for i in range(2):
        if D==1:
            y1 = data1[i, x]
            y2 = attn_data1[i, x]
        if D==10:
            y1 = data2[i, x]
            y2 = attn_data2[i, x]
        ax[i].plot(x, y1, '--', c='#e41b1b', label="no attention D={}".format(D), linewidth=2, markersize=6)
        # ax[i].plot(x, y2, '--', c='#377eb8', label="no attention D=10", linewidth=2, markersize=6)
        ax[i].plot(x, y2, '-', c='#377eb8', label="with attention D={}".format(D), linewidth=2, markersize=6)
        # ax[i].plot(x, y4, '-', c='#377eb8', label="with attention D=10", linewidth=2, markersize=6)
        ax[i].set_title(s[i],fontweight='bold',fontsize=16)
        ax[i].legend(loc='best',fancybox=True, framealpha=0,fontsize=14)
        ax[i].grid(True, linestyle='dotted')  # x坐标轴的网格使用主刻度
        # ax.yaxis.grid(True, linestyle='dotted')  # y坐标轴的网格使用次刻度

    if D==1:
        ax[2].plot(snrs, bler1, '--', c='#e41b1b', label="no attention D={}".format(D), linewidth=2, markersize=6)
        ax[2].plot(snrs, attn_bler1, '-', c='#377eb8', label="with attention D={}".format(D), linewidth=2, markersize=6)
    if D==10:
        ax[2].plot(snrs, bler2, '--', c='#e41b1b', label="no attention D={}".format(D), linewidth=2, markersize=6)
        ax[2].plot(snrs, attn_bler2, '-', c='#377eb8', label="with attention D={}".format(D), linewidth=2, markersize=6)
    ax[2].set_title(s[2],fontweight='bold',fontsize=16)
    ax[2].legend(loc='best',fancybox=True, framealpha=0,fontsize=14)
    ax[2].grid(True, linestyle='dotted')
    plt.savefig('attention_D'+str(D)+'.png', format='png', bbox_inches='tight', transparent=True, dpi=800)
    plt.show()

plot_attn(0.01,1)
plot_attn(0.01,10)

'''
colors = ['#C95F63', '#F1AD32', '#3B8320', '#516EA9',  '#292DF4', ]
# '#C95F63'红色 '#292DF4'亮蓝
# #['navy', 'firebrick', 'darkgreen', 'darkgoldenrod', 'darkolivegreen',  'darkmagenta']
line_styles = ['-']  # ['-', ':',  '--','-.']
marker_types = ['o', 's', 'v', '^', '*', 'h']

# This lists out all the variables that you can control
# A copy of this dictionary will be generated (deepcopy),
# in case the default values are lost
legend = {
    'title': 'Delay=10 num_block=50000 lr=0.01',
    'xlabel': 'Epoch',
    'ylabel': 'BER',
    'savepath': './data/awgn_lr0.01_D10_50000.png',
    'fig_size': (9,6),
    'label_fontsize': 15,
    'markersize': 10,
    'linewidth': 2,
    'title_size': 20,
    'loc': 5,  # location of legend, see online documentation
    'plot_type': 'line',  # line = line, scatter = scatter, both = line + intervaled marker
    'x_range': 'auto',  # auto, 0 100
    'y_range': 'auto',
    'line_length': 40,
    'markevery': 0.5
}

legend_snr = {
    'title': 'Delay=10 num_block=50000 lr=0.01',
    'xlabel': 'SNR',
    'ylabel': 'BER',
    'savepath': './data/awgn_lr0.01_D10_50000_snr.png',
    'fig_size': (9,6),
    'label_fontsize': 15,
    'markersize': 10,
    'linewidth': 2,
    'title_size': 20,
    'loc': 5,  # location of legend, see online documentation
    'plot_type': 'line',  # line = line, scatter = scatter, both = line + intervaled marker
    'x_range': 'auto',  # auto, 0 100
    'y_range': 'auto',
    'line_length': 40,
    'markevery': 0.5
}
def plot_snr(legend):
    # 读取数据
    X = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    Y = [0.0701569989323616, 0.05043799802660942, 0.0348035953938961, 0.022770600393414497, 0.014346200972795486, 0.008601599372923374, 0.004972000140696764, 0.0027095996774733067, 0.0014349999837577343, 0.0007247999892570078, 0.00036559993168339133, 0.00017820001812651753]
    plt.figure(figsize=legend['fig_size'])
    plt.title(legend['title'], fontsize=legend['title_size'])
    ax = plt.subplot(111)
    plt.xticks(fontsize=legend['label_fontsize']-2)  # x轴字体大小
    plt.yticks(fontsize=legend['label_fontsize']-2)  # y轴字体大小
    l1 = ax.plot(X, Y, color=colors[0], linestyle=line_styles[0], marker=marker_types[0], markersize=legend['markersize'], linewidth=legend['linewidth'], label=legend['ylabel'])
    ax.set_xlabel(legend['xlabel'], fontsize=legend['label_fontsize'])
    ax.set_ylabel(legend['ylabel'], fontsize=legend['label_fontsize'])
    plt.savefig(legend['savepath'])
    plt.show()

def plot(filename, legend):
    # 读取数据
    # X = np.arange(1,121)
    data = np.loadtxt(filename)
    X = data[:,0]
    Y = data[:,2]
    plt.figure(figsize=legend['fig_size'])
    plt.title(legend['title'], fontsize=legend['title_size'])
    ax = plt.subplot(111)
    plt.xticks(fontsize=legend['label_fontsize']-2)  # x轴字体大小
    plt.yticks(fontsize=legend['label_fontsize']-2)  # y轴字体大小
    l1 = ax.plot(X, Y, color=colors[0], linestyle=line_styles[0], linewidth=legend['linewidth'], label=legend['ylabel'])
    ax.set_xlabel(legend['xlabel'], fontsize=legend['label_fontsize'])
    ax.set_ylabel(legend['ylabel'], fontsize=legend['label_fontsize'])
    plt.savefig(legend['savepath'])
    plt.show()
    
    # # 画图
    # plt.figure(figsize=legend['fig_size'])
    # plt.title(legend['title'])
    # ax = plt.subplot(121)
    # plt.xticks(fontsize=legend['label_fontsize']-2)  # x轴字体大小
    # plt.yticks(fontsize=legend['label_fontsize']-2)  # y轴字体大小
    # l1 = ax.plot(x_data, y_data[0], color=colors[0], marker=marker_types[0], linestyle=line_styles[0], markersize=legend['markersize'], markevery=legend['markevery'], linewidth=legend['linewidth'], label=legend['ylabels'][0])
    # ax.set_xlabel(legend['xlabel'], fontsize=legend['label_fontsize'])
    # ax.set_ylabel(legend['ylabels'][0], fontsize=legend['label_fontsize'])

    # ax2 = ax.twinx()
    # plt.yticks(fontsize=legend['label_fontsize']-2)  # y轴字体大小
    # l2 = ax2.plot(x_data, y_data[1], c=colors[1], marker=marker_types[1], linestyle=line_styles[0], markersize=legend['markersize'], markevery=legend['markevery'], linewidth=legend['linewidth'], label=legend['ylabels'][1])
    # ax2.set_ylabel(legend['ylabels'][1], fontsize=legend['label_fontsize'])
    # ax2.grid(None)

    # line = l1 + l2
    # labs = [l.get_label() for l in line]
    # ax.legend(line, labs, loc=legend['loc'], fontsize=legend['label_fontsize'])
    # plt.title(legend['title'], fontsize=legend['title_size'])

    # ax3 = plt.subplot(122)
    # plt.xticks(fontsize=legend['label_fontsize']-2)  # x轴字体大小
    # plt.yticks(fontsize=legend['label_fontsize']-2)  # y轴字体大小
    # l3 = ax3.plot(x_data, y_data[2], c=colors[2], marker=marker_types[2], linestyle=line_styles[0], markersize=legend['markersize'], markevery=legend['markevery'], linewidth=legend['linewidth'], label=legend['ylabels'][2])
    # ax3.set_xlabel(legend['xlabel'], fontsize=legend['label_fontsize'])
    # ax3.set_ylabel(legend['ylabels'][2], fontsize=legend['label_fontsize'])

    # ax4 = ax3.twinx()
    # plt.yticks(fontsize=legend['label_fontsize']-2)  # y轴字体大小
    # l4 = ax4.plot(x_data, y_data[3], c=colors[3], marker=marker_types[3], linestyle=line_styles[0], markersize=legend['markersize'], markevery=legend['markevery'], linewidth=legend['linewidth'], label=legend['ylabels'][3])
    # ax4.set_ylabel(legend['ylabels'][3], fontsize=legend['label_fontsize'])
    # ax4.grid(None)

    # line2 = l3 + l4
    # labs2 = [l.get_label() for l in line2]
    # ax3.legend(line2, labs2, loc=legend['loc'], fontsize=legend['label_fontsize'])
    # plt.title(legend['title'], fontsize=legend['title_size'])
    
    # plt.subplots_adjust(wspace =0.28, hspace =0)#调整子图间距
    # plt.savefig('vgg16-cifar100.png')
    # plt.show()

plot("./data/data_awgn_lr_0.01_D10_50000.txt", legend)
plot_snr(legend_snr)'''