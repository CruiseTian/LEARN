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
plot_snr(legend_snr)