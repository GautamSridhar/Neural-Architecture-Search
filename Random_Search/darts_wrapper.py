import genotypes
from model_search import Network
import utils

import time
import math
import copy
import random
import logging
import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import integrate

from torchdiffeq import odeint_adjoint as odeint
from dataset_def import generate_data,  get_batch, LotkaVolterra, FHN


import matplotlib.pyplot as plt

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class DartsWrapper:
    def __init__(self, args):
        self.args = args

        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        #torch.cuda.set_device(args.gpu)

        # Data #############################

        if args.dataset == 'LV':
            datfunc = LotkaVolterra()
        elif args.dataset == 'FHN':
            datfunc = FHN()

        t_train = torch.linspace(0.,25.,1000)
        t_eval = torch.linspace(0.,100.,1000)
        t_test = torch.linspace(0,200,1000)
        X0 = torch.tensor([10.,5.])

        X_train = generate_data(X0, t_train, method=args.integrate_method, typ=args.dataset)
        X_eval = generate_data(X0, t_eval, method=args.integrate_method, typ=args.dataset)
        X_test = generate_data(X0, t_test, method=args.integrate_method, typ=args.dataset)

        dx_dt_train = datfunc(t=None,x=X_train.numpy().T)
        dx_dt_eval = datfunc(t=None,x=X_eval.numpy().T)
        dx_dt_test = datfunc(t=None,x=X_test.numpy().T)

        # Xtrain_noisy = X_train + 0.75*torch.randn(X_train.shape[0],X_train.shape[1])

        # Xtrain_smooth = np.zeros((true_y_train_node.shape[0],true_y_train_node.shape[1]))

        # xhat0 = scipy.signal.savgol_filter(Xtrain_noisy.numpy()[:,0], 45, 2) # window size 45, polynomial order 2
        # xhat1 = scipy.signal.savgol_filter(Xtrain_noisy.numpy()[:,1], 45, 2) # window size 45, polynomial order 2

        # Xtrain_smooth[:,0] = xhat0
        # Xtrain_smooth[:,1] = xhat1

        # torched_Xtrain_smooth = torch.from_numpy(Xtrain_smooth)

        # dx_dt_train_regress = np.gradient(Xtrain_smooth, t_train, axis=0)

        # torched_X_train = torch.from_numpy(Xtrain_smooth)
        # torched_der_train_regress = torch.from_numpy(der_train_regress)


        self.train_queue = (X_train,dx_dt_train.T)

        self.valid_queue = (X_eval,dx_dt_eval.T)

        self.test_queue = (X_test, dx_dt_test.T)

        criterion = nn.MSELoss()
        self.criterion = criterion

        model = Network(args.network_inputsize, args.network_outputsize, args.max_width, args.max_depth, self.criterion)

        # model = model.cuda()
        self.model = model

        optimizer = torch.optim.Adam(
                                     self.model.parameters(),
                                     args.learning_rate,
                                     # momentum=args.momentum,
                                     weight_decay=args.weight_decay)
        self.optimizer = optimizer

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                    float(args.epochs),
                                                                    eta_min=args.learning_rate_min)

    def train_batch(self, arch, step, ax):
        args = self.args
        self.objs = utils.AverageMeter()
        self.mae = utils.AverageMeter()
        lr = self.scheduler.get_lr()[0]

        weights = self.get_weights_from_arch(arch)
        self.set_model_weights(weights)

        n = self.train_queue[0].shape[0]

        self.optimizer.zero_grad()
        logits = self.model(self.train_queue[0])
        loss = self.criterion(logits, self.train_queue[1])

        loss.backward()
        self.optimizer.step()

        train_mae = utils.accuracy(logits, self.train_queue[1])
        self.objs.update(loss.data.numpy(), n)
        self.mae.update(train_mae.data.numpy(), n)

        valid_err = self.evaluate(arch, step, ax)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f', step, self.objs.avg, self.mae.avg)

        self.scheduler.step()

    def evaluate(self, arch, step, ax):
        # Return error since we want to minimize obj val
        logging.info(arch)
        objs = utils.AverageMeter()
        mae = utils.AverageMeter()

        weights = self.get_weights_from_arch(arch)
        self.set_model_weights(weights)
        logging.info(self.model.genotype(weights))

        with torch.no_grad():
                t_test = torch.linspace(0,200,1000)
                logits = self.model(self.valid_queue[0])
                loss = self.criterion(logits, self.valid_queue[1])

                logits_test = self.model(self.test_queue[0])

                val_mae = utils.accuracy(logits, self.valid_queue[1])
                n = self.valid_queue[0].shape[0]
                objs.update(loss.data.numpy(), n)
                mae.update(val_mae.data.numpy(), n)

                if step % self.args.report_freq == 0:
                    logging.info('valid %03d %e %f', step, objs.avg, mae.avg)
                    ax.cla()
                    # ax[0,1].cla()
                    ax.plot(t_test.numpy(),self.test_queue[1],'g-',label='Test')
                    ax.plot(t_test.numpy(),logits_test.numpy(),'b-',label='Learned')
                    ax.legend()
                    ax.set_title("Learned regression")

                    # ax[0,1].plot(t_train.numpy(),true_y_train_node.numpy(), 'g-',label='Train',)
                    # ax[0,1].plot(t_train.numpy(),pred_y_train.numpy(), 'b-',label='Learned',)
                    # ax[0,1].legend()
                    # ax[0,1].set_title("Regression integrated")
        plt.draw()
        plt.pause(0.1)

        return mae.avg

    def save(self):
        utils.save(self.model, os.path.join(self.args.save_dir, 'weights.pt'))

    def load(self):
        utils.load(self.model, os.path.join(self.args.save_dir, 'weights.pt'))

    def get_weights_from_arch(self, arch):
        num_ops = len(genotypes.PRIMITIVES)
        n_nodes = self.model._max_depth

        alphas_normal = torch.zeros(n_nodes, num_ops)

        for i in range(n_nodes):
            normal1 = arch[i]
            alphas_normal[normal1[0], normal1[1]] = 1

        arch_parameters = [alphas_normal]
        return arch_parameters

    def set_model_weights(self, weights):
      self.model.alphas_normal = weights[0]
      self.model._arch_parameters = [self.model.alphas_normal]

    def sample_arch(self):
        num_ops = len(genotypes.PRIMITIVES)
        n_nodes = self.model._max_depth

        normal = []
        for i in range(n_nodes):
            ops = np.random.choice(range(num_ops), 4)
            normal.extend([(i, ops[0])])

        return (normal)
