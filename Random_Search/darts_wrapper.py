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

import matplotlib.pyplot as plt

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class DartsWrapper:
    def __init__(self, save_path, seed, epochs, network_inputsize, network_outputsize,max_width, max_depth):
        args = {}
        args['epochs'] = epochs
        args['learning_rate'] = 0.025
        args['learning_rate_min'] = 0.001
        args['momentum'] = 0.9
        args['weight_decay'] = 3e-4
        args['network_inputsize'] = network_inputsize
        args['network_outputsize'] = network_outputsize
        args['max_width'] = max_width
        args['max_depth'] = max_depth
        args['seed'] = seed
        args['save'] = save_path
        args['gpu'] = 0
        args['cuda'] = False
        args['report_freq'] = 50
        args = AttrDict(args)
        self.args = args
        self.seed = seed

        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        #torch.cuda.set_device(args.gpu)

        # Data #############################

        a = 1.
        b = 0.1
        c = 1.5
        d = 0.75

        def dX_dt(X, t=0):
            p = np.array([ a*X[0] -   b*X[0]*X[1] ,
                          -c*X[1] + d*b*X[0]*X[1] ])
            return p  

        t_train = torch.linspace(0.,25.,1000)
        t_eval = torch.linspace(0.,100.,1000)
        t_test = torch.linspace(0,200,1000)
        X0 = torch.tensor([10.,5.])
        X_train = integrate.odeint(dX_dt, X0.numpy(), t_train.numpy())
        X_eval = integrate.odeint(dX_dt,X0.numpy(),t_eval.numpy())
        X_test = integrate.odeint(dX_dt, X0.numpy(),t_test.numpy())

        dx_dt_train = dX_dt(X_train.T)
        dx_dt_eval = dX_dt(X_eval.T)
        dx_dt_test = dX_dt(X_test.T)

        noisy_dxdt = dx_dt_train #+ 0.75*np.random.randn(X.shape[1],X.shape[0])

        x_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(noisy_dxdt.T).float()
        self.train_queue = (x_train,y_train)

        x_eval = torch.from_numpy(X_eval).float()
        y_eval = torch.from_numpy(dx_dt_eval.T).float()
        self.valid_queue = (x_eval,y_eval)

        x_test = torch.from_numpy(X_test).float()
        self.test_queue = (x_test, dx_dt_test)

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
                    ax.plot(t_test.numpy(),self.test_queue[1].T,'g-',label='Test')
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
        utils.save(self.model, os.path.join(self.args.save, 'weights.pt'))

    def load(self):
        utils.load(self.model, os.path.join(self.args.save, 'weights.pt'))

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
