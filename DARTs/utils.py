import os
import numpy as np
import torch
import shutil


class AverageMeter(object):
    """ Keep track of counts, sums and averages."""

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset the counters to zeros."""
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        """ Update the counters based on a running count."""
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target):
    """This will have to undergo some changes"""

    err = torch.mean(torch.abs(output - target))
    return err


def save_checkpoint(state, is_best, save):
    """ Save state based on checkpoint as training progresses."""
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    """ Save model."""
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    """ Load model."""
    model.load_state_dict(torch.load(model_path))


def create_exp_dir(path, scripts_to_save=None):
    """ Create  directory for different experiments performed."""
    if not os.path.exists(path):
        os.mkdir(path)
        print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
