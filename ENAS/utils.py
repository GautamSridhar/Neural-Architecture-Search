from __future__ import print_function

from collections import defaultdict
import collections
from datetime import datetime
import os
import json
import logging

import numpy as np

import torch


##########################
# Torch
##########################

def detach(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(detach(v) for v in h)


# check whether we need this
def batchify(data, bsz, use_cuda):
    # code from https://github.com/pytorch/examples/blob/master/word_language_model/main.py 
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    if use_cuda:
        data = data.cuda()
    return data


##########################
# ETC
##########################

Gene = collections.namedtuple('Gene', ['id', 'name_op'])


class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret

# This can probably go
def to_item(x):
    """Converts x, possibly scalar and possibly tensor, to a Python scalar."""
    if isinstance(x, (float, int)):
        return x

    if float(torch.__version__[0:3]) < 0.4:
        assert (x.dim() == 1) and (len(x) == 1)
        return x[0]

    return x.item()


def get_logger(args=None, name=__file__, level=logging.INFO):

    logger = logging.getLogger(name)

    if getattr(logger, '_init_done__', None):
        logger.setLevel(level)
        return logger

    logger._init_done__ = True
    logger.propagate = False
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(0)

    del logger.handlers[:]
    logger.addHandler(handler)

    return logger


logger = get_logger()


def prepare_dirs(args):
    """Sets the directories for the model, and creates those directories.
    Args:
        args: Parsed from `argparse` in the `config` module.
    """
    if args.load_path:
        if args.load_path.startswith(args.log_dir):
            args.model_dir = args.load_path
        else:
            if args.load_path.startswith(args.dataset):
                args.model_name = args.load_path
            else:
                args.model_name = "{}_{}".format(args.load_path)
    else:
        args.model_name = "{}".format(get_time())

    if not hasattr(args, 'model_dir'):
        args.model_dir = os.path.join(args.log_dir, args.model_name)

    for path in [args.log_dir, args.model_dir]:
        if not os.path.exists(path):
            makedirs(path)

def get_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def save_args(args):
    param_path = os.path.join(args.model_dir, "params.json")

    logger.info("[*] MODEL dir: %s" % args.model_dir)
    logger.info("[*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(args.__dict__, fp, indent=4, sort_keys=True)

def save_dag(args, dag, name):
    save_path = os.path.join(args.model_dir, name)
    logger.info("[*] Save dag : {}".format(save_path))
    json.dump(dag, open(save_path, 'w'))

def load_dag(args):
    load_path = os.path.join(args.dag_path)
    logger.info("[*] Load dag : {}".format(load_path))
    with open(load_path) as f:
        dag = json.load(f)
    dag = {int(k): [Node(el[0], el[1]) for el in v] for k, v in dag.items()}
    draw_network(dag, os.path.join(args.model_dir, "dag.png"))
    return dag          
  
def makedirs(path):
    if not os.path.exists(path):
        logger.info("[*] Make directories : {}".format(path))
        os.makedirs(path)

def remove_file(path):
    if os.path.exists(path):
        logger.info("[*] Removed: {}".format(path))
        os.remove(path)

def backup_file(path):
    root, ext = os.path.splitext(path)
    new_path = "{}.backup_{}{}".format(root, get_time(), ext)

    os.rename(path, new_path)
    logger.info("[*] {} has backup: {}".format(path, new_path))
