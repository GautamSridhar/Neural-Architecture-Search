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

import dataset_def as Dat

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
            # 1

            X0 = torch.tensor([10.,5.])
            theta = [1.0, 0.1, 1.5, 0.75]
            datfunc = Dat.LotkaVolterra(theta)

            t_train = torch.linspace(0.,25.,1000)
            t_eval = torch.linspace(0.,100.,1000)
            t_test = torch.linspace(0,200,100)

        elif args.dataset == 'FHN':
            #2

            X0 = torch.tensor([-1.0, 1.0])
            theta = [0.2,0.2,3.0]
            datfunc = Dat.FHN(theta)

            t_train = torch.linspace(0.,25.,1000)
            t_eval = torch.linspace(0.,100.,1000)
            t_test = torch.linspace(0,200,100)

        elif args.dataset == 'Lorenz63':
            #3

            X0 = torch.tensor([1.0, 1.0, 1.0])
            theta = [10.0, 28.0, 8.0/3.0]
            datfunc = Dat.Lorenz63(theta)

            t_train = torch.linspace(0.,25.,1000) # Need to ask about extents for test case Lorenz
            t_eval = torch.linspace(0.,50.,100)
            t_test = torch.linspace(0.,100.,100)

        # Need X0 and parameters
        # elif args.dataset == 'Lorenz96':
              # 4
        #     X0 = torch.tensor([])
        #     theta = 
        #     datfunc = Lorenz96(theta)

        elif args.dataset == 'ChemicalReactionSimple':
            #5
            X0 = torch.tensor([1., 1.])
            theta = [.5, .8, .4]
            datfunc = Dat.ChemicalReactionSimple(theta)

            t_train = torch.linspace(0.,25.,1000)
            t_eval = torch.linspace(0.,100.,1000)
            t_test = torch.linspace(0,200,100)

        elif args.dataset == 'Chemostat':
            #6
            X0 = torch.tensor([1., 2., 3., 4., 5., 6., 10.])

            Cetas = np.linspace(2., 3., 6,dtype=float)
            VMs = np.linspace(1., 2., 6,dtype=float)
            KMs = np.ones(6,dtype=float)

            theta = np.squeeze(np.concatenate([Cetas.reshape([1, -1]),
                                    VMs.reshape([1, -1]),
                                    KMs.reshape([1, -1])],
                                    axis=1))
            flowrate = 2.
            feedConc = 3.
            datfunc = Dat.Chemostat(6, flowrate, feedConc, theta)

            t_train = torch.linspace(0.,1.,1000) # Ask about the extent here
            t_eval = torch.linspace(0.,2.,1000)
            t_test = torch.linspace(0,5,100)

        elif args.dataset == 'Clock':
            #7
            X0 = torch.tensor([1, 1.2, 1.9, .3, .8, .98, .8])
            theta = np.asarray([.8, .05, 1.2, 1.5, 1.4, .13, 1.5, .33, .18, .26,
                                .28, .5, .089, .52, 2.1, .052, .72])
            datfunc = Dat.Clock(theta)

            t_train = torch.linspace(0.,5.,1000)
            t_eval = torch.linspace(0.,10.,1000)
            t_test = torch.linspace(0,20,100)

        elif args.dataset == 'ProteinTransduction':
            #8
            X0 = torch.tensor([1., 0., 1., 0., 0.])
            theta = [0.07, 0.6, 0.05, 0.3, 0.017, 0.3]
            datfunc = Dat.ProteinTransduction(theta)
            t_train = torch.linspace(0.,25.,1000)
            t_eval = torch.linspace(0.,100.,1000)
            t_test = torch.linspace(0,200,1000)

        self.t_train = t_train
        self.t_eval = t_eval
        self.t_test = t_test


        X_train = Dat.generate_data(datfunc, X0, t_train, method=args.integrate_method)
        X_eval = Dat.generate_data(datfunc, X0, t_eval, method=args.integrate_method)
        X_test = Dat.generate_data(datfunc, X0, t_test, method=args.integrate_method)

        dx_dt_train = datfunc(t=None,x=X_train.numpy().T)
        dx_dt_eval = datfunc(t=None,x=X_eval.numpy().T)
        dx_dt_test = datfunc(t=None,x=X_test.numpy().T)

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
                    ax.plot(self.t_test.numpy(),self.test_queue[1],'g-',label='Test')
                    ax.plot(self.t_test.numpy(),logits_test.numpy(),'b-',label='Learned')
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
