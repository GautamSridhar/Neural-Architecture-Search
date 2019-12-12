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
from scipy import integrate

from model_search import Network
from architect import Architect


parser = argparse.ArgumentParser("Regression Search")
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--network_inputsize', type=int, default=2, help='input size of the network')
parser.add_argument('--network_outputsize', type=int, default=2, help='output size of the network')
parser.add_argument('--max_width', type=int, default=16, help='max width of the network')
parser.add_argument('--max_depth', type=int, default=4, help='total number of layers in the network')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='test', help='experiment name')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--unrolled', action='store_true', default=True, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
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
X0 = torch.tensor([10.,5.])
X_train = integrate.odeint(dX_dt, X0.numpy(), t_train.numpy())
X_eval = integrate.odeint(dX_dt,X0.numpy(),t_eval.numpy())
dx_dt_train = dX_dt(X_train.T)
dx_dt_eval = dX_dt(X_eval.T)

noisy_dxdt = dx_dt_train #+ 0.75*np.random.randn(X.shape[1],X.shape[0])

x_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(noisy_dxdt.T).float()
train_queue = (x_train,y_train)

x_eval = torch.from_numpy(X_eval).float()
y_eval = torch.from_numpy(dx_dt_eval.T).float()
valid_queue = (x_eval,y_eval)

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

    optimizer = torch.optim.SGD(
                                model.parameters(),
                                args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay
                                )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                                                           optimizer,
                                                           float(args.epochs),
                                                           eta_min=args.learning_rate_min
                                                           )

    architect = Architect(model, args)

    for epoch in range(args.epochs):
        lr = scheduler.get_lr()[0]
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

        val_mae = utils.accuracy(logits, valid_queue[1])
        n = valid_queue[0].shape[0]
        objs.update(loss.data.numpy(), n)
        mae.update(val_mae.data.numpy(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f', step, objs.avg, mae.avg)

    return mae.avg, objs.avg


if __name__ == '__main__':
    main()