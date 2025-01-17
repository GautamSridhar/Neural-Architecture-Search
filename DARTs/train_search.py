import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F

import matplotlib.pyplot as plt

from scipy import integrate

import dataset_def as Dat
from model_search import Network
from architect import Architect


parser = argparse.ArgumentParser("Regression Search")
parser.add_argument('--dataset', type=str, default='LV', help='dataset to be used')
parser.add_argument('--train_size', type=int, default=1000, help='size of the training set')
parser.add_argument('--eval_size', type=int, default=1000, help='size of the validation set')
parser.add_argument('--test_size', type=int, default=1000, help='size of the test set')
parser.add_argument('--integrate_method', type=str, default='dopri5', help='method for numerical integration')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=500, help='num of training epochs')
parser.add_argument('--network_inputsize', type=int, default=2, help='input size of the network')
parser.add_argument('--network_outputsize', type=int, default=2, help='output size of the network')
parser.add_argument('--max_width', type=int, default=16, help='max width of the network')
parser.add_argument('--max_depth', type=int, default=4, help='total number of layers in the network')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='test-sched-highlr-seed1-unrf', help='experiment name')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=1e-2, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--report_freq', type=float, default=5, help='report frequency')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# Data #############################
if args.dataset == 'LV':
    # 1

    X0 = torch.tensor([10.,5.])
    theta = [1.0, 0.1, 1.5, 0.75]
    datfunc = Dat.LotkaVolterra(theta)

    t_train = torch.linspace(0.,25.,args.train_size)
    t_eval = torch.linspace(0.,100.,args.eval_size)
    t_test = torch.linspace(0,200,args.test_size)

elif args.dataset == 'FHN':
    #2

    X0 = torch.tensor([-1.0, 1.0])
    theta = [0.2,0.2,3.0]
    datfunc = Dat.FHN(theta)

    t_train = torch.linspace(0.,25.,args.train_size)
    t_eval = torch.linspace(0.,100.,args.eval_size)
    t_test = torch.linspace(0,200,args.test_size)

elif args.dataset == 'Lorenz63':
    #3

    X0 = torch.tensor([1.0, 1.0, 1.0])
    theta = [10.0, 28.0, 8.0/3.0]
    datfunc = Dat.Lorenz63(theta)

    t_train = torch.linspace(0.,25.,args.train_size) # Need to ask about extents for test case Lorenz
    t_eval = torch.linspace(0.,50.,args.eval_size)
    t_test = torch.linspace(0.,100.,args.test_size)

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

    t_train = torch.linspace(0.,25.,args.train_size)
    t_eval = torch.linspace(0.,100.,args.eval_size)
    t_test = torch.linspace(0,200,args.test_size)

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

    t_train = torch.linspace(0.,1.,args.train_size) # Ask about the extent here
    t_eval = torch.linspace(0.,2.,args.eval_size)
    t_test = torch.linspace(0,5,args.test_size)

elif args.dataset == 'Clock':
    #7
    X0 = torch.tensor([1, 1.2, 1.9, .3, .8, .98, .8])
    theta = np.asarray([.8, .05, 1.2, 1.5, 1.4, .13, 1.5, .33, .18, .26,
                        .28, .5, .089, .52, 2.1, .052, .72])
    datfunc = Dat.Clock(theta)

    t_train = torch.linspace(0.,5.,args.train_size)
    t_eval = torch.linspace(0.,10.,args.eval_size)
    t_test = torch.linspace(0,20,args.test_size)

elif args.dataset == 'ProteinTransduction':
    #8
    X0 = torch.tensor([1., 0., 1., 0., 0.])
    theta = [0.07, 0.6, 0.05, 0.3, 0.017, 0.3]
    datfunc = Dat.ProteinTransduction(theta)
    t_train = torch.linspace(0.,25.,args.train_size)
    t_eval = torch.linspace(0.,100.,args.eval_size)
    t_test = torch.linspace(0,200,args.test_size)


X_train = Dat.generate_data(datfunc, X0, t_train, method=args.integrate_method)
X_eval = Dat.generate_data(datfunc, X0, t_eval, method=args.integrate_method)
X_test = Dat.generate_data(datfunc, X0, t_test, method=args.integrate_method)

dx_dt_train = datfunc(t=None,x=X_train.numpy().T)
dx_dt_eval = datfunc(t=None,x=X_eval.numpy().T)
dx_dt_test = datfunc(t=None,x=X_test.numpy().T)

train_queue = (X_train,dx_dt_train.T)

valid_queue = (X_eval,dx_dt_eval.T)

fig,ax = plt.subplots(figsize=(20,20))
def main():
    # if not torch.cuda.is_available():
    #     logging.info('no gpu device available')
    #     sys.exit(1)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # logging.info('gpu device = %d' % args.gpu)
    # logging.info("args = %s", args)

    criterion = nn.MSELoss()
    # criterion = criterion.cuda()
    model = Network(args.network_inputsize, args.network_outputsize, args.max_width, args.max_depth, criterion)
    # model = model.cuda()

    optimizer = torch.optim.Adam(
                                model.parameters(),
                                args.learning_rate,
                                #momentum=args.momentum,
                                weight_decay=args.weight_decay
                                )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                                                           optimizer,
                                                           float(args.epochs),
                                                           eta_min=args.learning_rate_min
                                                           )

    architect = Architect(model, args)

    plt.ion()
    for epoch in range(args.epochs):
        lr = scheduler.get_lr()[0]
        # lr = args.learning_rate
        logging.info('epoch %d lr %e', epoch, lr)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        print(F.softmax(model.w_alpha, dim=-1))

        # training
        train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch)
        logging.info('train_acc %f', train_acc)
        scheduler.step()

        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion, epoch)
        logging.info('valid_acc %f', valid_acc)

        utils.save(model, os.path.join(args.save, 'weights.pt'))

        plt.draw()
        plt.pause(0.1)

    plt.ioff()
    plt.show()



def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, step):
    objs = utils.AverageMeter()
    mae = utils.AverageMeter()

    n = train_queue[0].shape[0]

    architect.step(train_queue[0], train_queue[1], valid_queue[0], valid_queue[1], lr, optimizer, unrolled=args.unrolled)

    optimizer.zero_grad()
    logits = model(train_queue[0])
    loss = criterion(logits, train_queue[1])

    loss.backward()
    optimizer.step()

    train_mae = utils.accuracy(logits, train_queue[1])
    objs.update(loss.data.numpy(), n)
    mae.update(train_mae.data.numpy(), n)

    if step % args.report_freq == 0:
        logging.info('train %03d %e %f', step, objs.avg, mae.avg)

    return mae.avg, objs.avg


def infer(valid_queue, model, criterion, step):
    objs = utils.AverageMeter()
    mae = utils.AverageMeter()

    with torch.no_grad():

        logits = model(valid_queue[0])
        loss = criterion(logits, valid_queue[1])

        logits_test = model(X_test)

        val_mae = utils.accuracy(logits, valid_queue[1])
        n = valid_queue[0].shape[0]
        objs.update(loss.data.numpy(), n)
        mae.update(val_mae.data.numpy(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f', step, objs.avg, mae.avg)
            ax.cla()
            # ax[0,1].cla()
            ax.plot(t_test.numpy(),dx_dt_test.T,'g-',label='Test')
            ax.plot(t_test.numpy(),logits_test.numpy(),'b-',label='Learned')
            ax.legend()
            ax.set_title("Learned regression")

            # ax[0,1].plot(t_train.numpy(),true_y_train_node.numpy(), 'g-',label='Train',)
            # ax[0,1].plot(t_train.numpy(),pred_y_train.numpy(), 'b-',label='Learned',)
            # ax[0,1].legend()
            # ax[0,1].set_title("Regression integrated")

    return mae.avg, objs.avg


if __name__ == '__main__':
    main()