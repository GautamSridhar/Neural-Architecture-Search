import sys
import os
import time
import shutil
import logging
import inspect
import pickle
import argparse
import numpy as np
from tqdm import tqdm

from darts_wrapper import DartsWrapper
import matplotlib.pyplot as plt


class Node:
    def __init__(self, parent, arch, node_id, rung):
        self.parent = parent
        self.arch = arch
        self.node_id = node_id
        self.rung = rung
    def to_dict(self):
        out = {'parent':self.parent, 'arch': self.arch, 'node_id': self.node_id, 'rung': self.rung}
        if hasattr(self, 'objective_val'):
            out['objective_val'] = self.objective_val
        return out

class Random_NAS:
    def __init__(self, B, model, seed, save_dir):
        self.save_dir = save_dir

        self.B = B
        self.model = model
        self.seed = seed

        self.iters = 0

        self.arms = {}
        self.node_id = 0

    def get_arch(self):
        arch = self.model.sample_arch()
        self.arms[self.node_id] = Node(self.node_id, arch, self.node_id, 0)
        self.node_id += 1
        return arch

    def save(self):
        to_save = {a: self.arms[a].to_dict() for a in self.arms}
        # Only replace file if save successful so don't lose results of last pickle save
        with open(os.path.join(self.save_dir,'results_tmp.pkl'),'wb') as f:
            pickle.dump(to_save, f)
        shutil.copyfile(os.path.join(self.save_dir, 'results_tmp.pkl'), os.path.join(self.save_dir, 'results.pkl'))

        self.model.save()

    def run(self):
        fig,ax = plt.subplots(figsize=(20,20))
        plt.ion()
        while self.iters < self.B:
            logging.info('Epoch: %03d', self.iters)
            arch = self.get_arch()
            self.model.train_batch(arch,self.iters,ax)
            self.iters += 1
            if self.iters % 500 == 0:
                self.save()
        plt.ioff()
        plt.show()
        self.save()

    def get_eval_arch(self, rounds=None):
        fig,ax = plt.subplots(figsize=(20,20))
        #n_rounds = int(self.B / 7 / 1000)
        if rounds is None:
            n_rounds = max(1,int(self.B/10000))
        else:
            n_rounds = rounds
        best_rounds = []
        for r in range(n_rounds):
            sample_vals = []
            for _ in range(1000):
                arch = self.model.sample_arch()
                try:
                    s = 50
                    ppl = self.model.evaluate(arch,s,ax)
                except Exception as e:
                    ppl = 1000000
                logging.info(arch)
                logging.info('objective_val: %.3f' % ppl)
                sample_vals.append((arch, ppl))

            sample_vals = sorted(sample_vals, key=lambda x:x[1], reverse=False)

            best_rounds.append(sample_vals[0])
        return best_rounds

def main(args):
    # Fill in with root output path
    root_dir = '/home/gautam/Documents/Gaussian_Processes/NAS/Neural-Architecture-Search/Random_Search/Results/'
    if args.save_dir is None:
        args.save_dir = os.path.join(root_dir, 'random/trial%d' % (args.seed))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.eval_only:
        assert args.save_dir is not None

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info(args)

    B = int(args.epochs)
    model = DartsWrapper(args)

    searcher = Random_NAS(B, model, args.seed, args.save_dir)
    logging.info('budget: %d' % (searcher.B))
    if not args.eval_only:
        searcher.run()
        plt.ion()
        archs = searcher.get_eval_arch()
        plt.ioff()
        plt.show()

    else:
        np.random.seed(args.seed+1)
        archs = searcher.get_eval_arch(2)
    logging.info('best architecture')
    logging.info(archs)
    arch = ' '.join([str(a) for a in archs[0][0]])
    with open('/tmp/arch','w') as f:
        f.write(arch)
    return arch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args for SHA with weight sharing')
    parser.add_argument('--dataset', type=str, default='LV', help='dataset to be used')
    parser.add_argument('--train_size', type=int, default=1000, help='size of the training set')
    parser.add_argument('--eval_size', type=int, default=1000, help='size of the validation set')
    parser.add_argument('--test_size', type=int, default=1000, help='size of the test set')
    parser.add_argument('--batch_size', type=int, default=50, help='batch size of data')
    parser.add_argument('--batch_time', type=int, default=25, help='batch time of data')
    parser.add_argument('--integrate_method', type=str, default='dopri5', help='method for numerical integration')
    parser.add_argument('--seed', dest='seed', type=int, default=1)
    parser.add_argument('--epochs', dest='epochs', type=int, default=10)
    parser.add_argument('--save_dir', dest='save_dir', type=str, default=None)
    parser.add_argument('--eval_only', dest='eval_only', type=int, default=0)
    parser.add_argument('--network_inputsize', type=int, default=2, help='input size of the network')
    parser.add_argument('--network_outputsize', type=int, default=2, help='output size of the network')
    parser.add_argument('--max_width', type=int, default=8, help='max width of the network')
    parser.add_argument('--max_depth', type=int, default=2, help='total number of layers in the network')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min lr for scheduler')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--momentume', type=float, default=0.9, help='momentum')
    parser.add_argument('--report_freq', type=int, default=1, help='when to check performance on test set.')
    args = parser.parse_args()

    main(args)