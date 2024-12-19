import argparse
import yaml
from attrdict import AttrDict
import torch
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt
import matplotlib


class WarmupThenDecayScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super(WarmupThenDecayScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Warmup phase: scale the lr linearly
            warmup_factor = float(self.last_epoch) / float(max(1, self.warmup_steps))
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Decay phase: linearly decay the lr to 0
            decay_factor = 1 - (self.last_epoch - self.warmup_steps) / float(
                max(1, self.total_steps - self.warmup_steps))
            return [base_lr * decay_factor for base_lr in self.base_lrs]


def load_configs(train_cfg_path, data_cfg_path, args):
    """
    Load configuration files and update with command line arguments. The command line arguments will overwrite the
    train_cfg and data_cfg files.
    """
    train_cfg = AttrDict(yaml.safe_load(open(train_cfg_path, 'r')))
    data_cfg = AttrDict(yaml.safe_load(open(data_cfg_path, 'r')))
    config = {}
    config.update(train_cfg)
    config.update(data_cfg)
    config.update(vars(args))

    return AttrDict(config)


def update_plot_style():
    matplotlib.rcParams.update({
        'font.family': 'times',
        'font.size': 14.0,
        'lines.linewidth': 2,
        'lines.antialiased': True,
        'axes.facecolor': 'fdfdfd',
        'axes.edgecolor': '777777',
        'axes.linewidth': 1,
        'axes.titlesize': 'medium',
        'axes.labelsize': 'medium',
        'axes.axisbelow': True,
        'xtick.major.size': 0,  # major tick size in points
        'xtick.minor.size': 0,  # minor tick size in points
        'xtick.major.pad': 6,  # distance to major tick label in points
        'xtick.minor.pad': 6,  # distance to the minor tick label in points
        'xtick.color': '333333',  # color of the tick labels
        'xtick.labelsize': 'medium',  # fontsize of the tick labels
        'xtick.direction': 'in',  # direction: in or out
        'ytick.major.size': 0,  # major tick size in points
        'ytick.minor.size': 0,  # minor tick size in points
        'ytick.major.pad': 6,  # distance to major tick label in points
        'ytick.minor.pad': 6,  # distance to the minor tick label in points
        'ytick.color': '333333',  # color of the tick labels
        'ytick.labelsize': 'medium',  # fontsize of the tick labels
        'ytick.direction': 'in',  # direction: in or out
        'axes.grid': False,
        'grid.alpha': 0.3,
        'grid.linewidth': 1,
        'legend.fancybox': True,
        'legend.fontsize': 'Small',
        'figure.figsize': (2.5, 2.5),
        'figure.facecolor': '1.0',
        'figure.edgecolor': '0.5',
        'hatch.linewidth': 0.1,
        'text.usetex': True
    })

    plt.rcParams['text.latex.preamble'] = r'\usepackage{times}'


def get_color_map():
    color_map = {'green': '#009E60', 'orange': '#C04000',
                 'blue': 'C0', 'black': '#3A3B3C',
                 'purple': '#843B62', 'red': '#C41E3A'}
    return color_map