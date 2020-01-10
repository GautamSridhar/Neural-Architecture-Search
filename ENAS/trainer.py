"""The module for training ENAS."""
import contextlib
import glob
import math
import os
from scipy import integrate

import numpy as np
import torch
from torch import nn
import torch.nn.parallel

import models
import utils

import time

from torchdiffeq import odeint_adjoint as odeint
# from dataset_def import generate_data,  get_batch, LotkaVolterra, FHN
import dataset_def as Dat

logger = utils.get_logger()

def _get_optimizer(name):
    if name.lower() == 'sgd':
        optim = torch.optim.SGD
    elif name.lower() == 'adam':
        optim = torch.optim.Adam

    return optim


class Trainer(object):
    """A class to wrap training code."""
    def __init__(self, args):
        """Constructor for training algorithm.
        Args:
            args: From command line, picked up by `argparse`.
            dataset: Currently only `data.text.Corpus` is supported.
        Initializes:
            - Data: train, val and test.
            - Model: shared and controller.
            - Inference: optimizers for shared and controller parameters.
            - Criticism: cross-entropy loss for training the shared model.
        """
        self.args = args
        self.controller_step = 0
        # self.cuda = args.cuda
        self.epoch = 0
        self.shared_step = 0
        self.start_epoch = 0

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

        self.build_model()

        if self.args.load_path:
            self.load_model()

        shared_optimizer = _get_optimizer(self.args.shared_optim)
        controller_optimizer = _get_optimizer(self.args.controller_optim)

        self.shared_optim = shared_optimizer(
            self.shared.parameters(),
            lr=self.args.shared_lr,
            weight_decay=self.args.shared_l2_reg)

        self.controller_optim = controller_optimizer(
            self.controller.parameters(),
            lr=self.args.controller_lr)

        self.criterion_shared = nn.MSELoss()
        self.criterion_controller = nn.L1Loss()

    def build_model(self):
        """Creates and initializes the shared and controller models."""
        if self.args.network_type == 'Net':
            self.shared = models.Network(self.args)
        else:
            raise NotImplementedError(f'Network type '
                                      f'`{self.args.network_type}` is not '
                                      f'defined')
        self.controller = models.Controller(self.args)

        # if self.args.num_gpu == 1:
        #     self.shared.cuda()
        #     self.controller.cuda()
        # elif self.args.num_gpu > 1:
        #     raise NotImplementedError('`num_gpu > 1` is in progress')

    def train(self, single=False):
        """Cycles through alternately training the shared parameters and the
        controller, as described in Section 2.2, Training ENAS and Deriving
        Architectures, of the paper.
        From the paper (for Penn Treebank):
        - In the first phase, shared parameters omega are trained for 400
          steps, each on a minibatch of 64 examples.
        - In the second phase, the controller's parameters are trained for 2000
          steps.
          
        Args:
            single (bool): If True it won't train the controller and use the
                           same dag instead of derive().
        """
        dag = utils.load_dag(self.args) if single else None
        
        if self.args.shared_initial_step > 0: # This has to be set to be set to greater than zero for warmup
            self.train_shared(self.args.shared_initial_step)
            self.train_controller()

        for self.epoch in range(self.start_epoch, self.args.max_epoch):
            # 1. Training the shared parameters omega of the child models
            self.train_shared(dag=dag)

            # 2. Training the controller parameters theta
            if not single:
                self.train_controller()

            if self.epoch % self.args.save_epoch == 0:
                with torch.no_grad():
                    best_dag = dag if dag else self.derive()
                    self.evaluate(best_dag)  # What is max_num?
                self.save_model()

            # Add a learning rate scheduler here.


    def get_loss(self, inputs, targets, dags, mode='Train'):
        """Computes the loss for the same batch for M models.
        This amounts to an estimate of the loss, which is turned into an
        estimate for the gradients of the shared model.
        """
        if not isinstance(dags, list):
            dags = [dags]

        loss = 0
        if mode == 'Train':
            batch_x0, batch_t, batch_x, batch_der = Dat.get_batch(1000, self.args.batch_time, self.args.batch_size,
                                                              inputs, targets, self.t_train)
            batch_regress = batch_x.view(self.args.batch_size, self.args.batch_time, -1)
            batch_der = batch_der.view(self.args.batch_size, self.args.batch_time, -1)

            for dag in dags:
                self.shared.dag = dag
                regress_pred = self.shared(t=None, inputs=batch_regress.float())
                loss_regress = self.criterion_shared(regress_pred, batch_der.float())

                pred_x = odeint(self.shared, batch_x0.float(), batch_t.float(),method=self.args.integrate_method)
                loss_node = self.criterion_shared(pred_x.float(),batch_x.float())

                loss_total = loss_regress + loss_node

                output = self.shared(t=None, inputs=inputs)
                sample_loss = self.criterion_shared(output, targets)
                loss += sample_loss

                return loss_total, loss

        elif mode == 'Valid':
            for dag in dags:
                self.shared.dag = dag
                output = self.shared(t=None, inputs=inputs)
                sample_loss = self.criterion_controller(output, targets)
                loss += sample_loss
                return loss

        assert len(dags) == 1, 'there are multiple `hidden` for multple `dags`'

    def train_shared(self, max_step=None, dag=None):
        """Train the language model for 400 steps of minibatches of 64
        examples.
        Args:
            max_step: Used to run extra training steps as a warm-up.
            dag: If not None, is used instead of calling sample().
        BPTT is truncated at 35 timesteps.
        For each weight update, gradients are estimated by sampling M models
        from the fixed controller policy, and averaging their gradients
        computed on a batch of training data.
        """
        model = self.shared

        if max_step is None:
            max_step = self.args.shared_max_step
        else:
            max_step = min(self.args.shared_max_step, max_step)

        raw_total_loss = 0
        total_loss = 0
        train_idx = 0
        # TODO(brendan): Why - 1 - 1?
        
        # declare training data here
        inputs = self.train_queue[0]
        targets = self.train_queue[1]

        for step in range(max_step):

            with torch.no_grad():
                dags = dag if dag else self.controller.sample(self.args.shared_num_sample)

            loss_total, loss = self.get_loss(inputs, targets, dags, mode='Train')
            raw_total_loss += loss.data

            # update
            self.shared_optim.zero_grad()
            loss_total.backward()

            self.shared_optim.step()

            # Add learning rate scheduler here instead

            total_loss += loss.data

            if ((step % self.args.log_step_shared) == 0) and (step > 0):
                # Need to change summarize to include only loss rather than these
                self._summarize_shared_train(total_loss, raw_total_loss)
                raw_total_loss = 0
                total_loss = 0


    def get_reward(self, dag, entropies):
        """Computes the perplexity of a single sampled model on a minibatch of
        validation data.
        """
        if not isinstance(entropies, np.ndarray):
            entropies = entropies.data.cpu().numpy()

        # Declare validation data here

        inputs = self.valid_queue[0]
        target = self.valid_queue[1]

        valid_ppl = self.get_loss(inputs, target, dag, mode='Valid')
        valid_ppl = utils.to_item(valid_ppl.data)

        # TODO: we don't know reward_c
        if self.args.ppl_square:
            # TODO: but we do know reward_c=80 in the previous paper
            R = self.args.reward_c / valid_ppl ** 2
        else:
            R = self.args.reward_c / valid_ppl

        if self.args.entropy_mode == 'reward':
            rewards = R + self.args.entropy_coeff * entropies
        elif self.args.entropy_mode == 'regularizer':
            rewards = R * np.ones_like(entropies)
        else:
            raise NotImplementedError(f'Unkown entropy mode: {self.args.entropy_mode}')

        return rewards

    def train_controller(self):
        """Fixes the shared parameters and updates the controller parameters.
        The controller is updated with a score function gradient estimator
        (i.e., REINFORCE), with the reward being c/valid_ppl, where valid_ppl
        is computed on a minibatch of validation data.
        A moving average baseline is used.
        The controller is trained for 2000 steps per epoch (i.e.,
        first (Train Shared) phase -> second (Train Controller) phase).
        """
        model = self.controller

        avg_reward_base = None
        baseline = None
        adv_history = []
        entropy_history = []
        reward_history = []

        total_loss = 0
        for step in range(self.args.controller_max_step):
            # sample models
            dags, log_probs, entropies = self.controller.sample(
                with_details=True)

            # calculate reward
            np_entropies = entropies.data.cpu().numpy()
            with torch.no_grad():
                rewards = self.get_reward(dags,np_entropies)

            reward_history.extend(rewards)
            entropy_history.extend(np_entropies)

            # moving average baseline
            if baseline is None:
                baseline = rewards
            else:
                decay = self.args.ema_baseline_decay
                baseline = decay * baseline + (1 - decay) * rewards

            adv = rewards - baseline
            adv_history.extend(adv)

            # policy loss

            loss = -log_probs*torch.tensor(adv)
            if self.args.entropy_mode == 'regularizer':
                loss -= self.args.entropy_coeff * entropies

            loss = loss.sum()  # or loss.mean()

            # update
            self.controller_optim.zero_grad()
            loss.backward()

            if self.args.controller_grad_clip > 0:
                torch.nn.utils.clip_grad_norm(model.parameters(),
                                              self.args.controller_grad_clip)
            self.controller_optim.step()

            total_loss += utils.to_item(loss.data)

            if ((step % self.args.log_step_controller) == 0) and (step > 0):
                self._summarize_controller_train(total_loss,
                                                 adv_history,
                                                 entropy_history,
                                                 reward_history,
                                                 avg_reward_base,
                                                 dags)

                reward_history, adv_history, entropy_history = [], [], []
                total_loss = 0


    def evaluate(self, dag):
        """Evaluate on the validation set.
        NOTE(brendan): We should not be using the test set to develop the
        algorithm (basic machine learning good practices).
        """

        with torch.no_grad():
            total_loss = 0

            inputs = self.test_queue[0]
            targets = self.test_queue[1]
            self.shared.dag = dag[0]
            output = self.shared(t=None, inputs=inputs)
            total_loss = self.criterion_controller(output, targets).data
            test_mae = utils.to_item(total_loss)
            logger.info(f'dag = {dag}')
            logger.info(f'eval | test mae: {test_mae:8.2f}')

    def derive(self, sample_num=None):
        """TODO(brendan): We are always deriving based on the very first batch
        of validation data? This seems wrong...
        """
        if sample_num is None:
            sample_num = self.args.derive_num_sample

        dags, _, entropies = self.controller.sample(sample_num,
                                                    with_details=True)

        max_R = 0
        best_dag = None
        for dag in dags:
            dag = [dag]
            R = self.get_reward(dag, entropies)
            if R.max() > max_R:
                max_R = R.max()
                best_dag = dag

        logger.info(f'best dag = %s', best_dag)
        logger.info(f'derive | max_R: {max_R:8.6f}')

        return best_dag

    @property
    def controller_lr(self):
        return self.args.controller_lr

    @property
    def shared_path(self):
        return f'{self.args.model_dir}/shared_epoch{self.epoch}_step{self.shared_step}.pt'

    @property
    def controller_path(self):
        return f'{self.args.model_dir}/controller_epoch{self.epoch}_step{self.controller_step}.pt'

    def get_saved_models_info(self):
        paths = glob.glob(os.path.join(self.args.model_dir, '*.pth'))
        paths.sort()

        def get_numbers(items, delimiter, idx, replace_word, must_contain=''):
            return list(set([int(
                    name.split(delimiter)[idx].replace(replace_word, ''))
                    for name in basenames if must_contain in name]))

        basenames = [os.path.basename(path.rsplit('.', 1)[0]) for path in paths]
        epochs = get_numbers(basenames, '_', 1, 'epoch')
        shared_steps = get_numbers(basenames, '_', 2, 'step', 'shared')
        controller_steps = get_numbers(basenames, '_', 2, 'step', 'controller')

        epochs.sort()
        shared_steps.sort()
        controller_steps.sort()

        return epochs, shared_steps, controller_steps

    def save_model(self):
        torch.save(self.shared.state_dict(), self.shared_path)
        logger.info(f'[*] SAVED: {self.shared_path}')

        torch.save(self.controller.state_dict(), self.controller_path)
        logger.info(f'[*] SAVED: {self.controller_path}')

        epochs, shared_steps, controller_steps = self.get_saved_models_info()

        for epoch in epochs[:-self.args.max_save_num]:
            paths = glob.glob(
                os.path.join(self.args.model_dir, f'*_epoch{epoch}_*.pt'))

            for path in paths:
                utils.remove_file(path)

    def load_model(self):
        epochs, shared_steps, controller_steps = self.get_saved_models_info()

        if len(epochs) == 0:
            logger.info(f'[!] No checkpoint found in {self.args.model_dir}...')
            return

        self.epoch = self.start_epoch = max(epochs)
        self.shared_step = max(shared_steps)
        self.controller_step = max(controller_steps)

        if self.args.num_gpu == 0:
            map_location = lambda storage, loc: storage
        else:
            map_location = None

        self.shared.load_state_dict(
            torch.load(self.shared_path, map_location=map_location))
        logger.info(f'[*] LOADED: {self.shared_path}')

        self.controller.load_state_dict(
            torch.load(self.controller_path, map_location=map_location))
        logger.info(f'[*] LOADED: {self.controller_path}')

    def _summarize_controller_train(self,
                                    total_loss,
                                    adv_history,
                                    entropy_history,
                                    reward_history,
                                    avg_reward_base,
                                    dags):
        """Logs the controller's progress for this training epoch."""
        cur_loss = total_loss / self.args.log_step_controller

        avg_adv = np.mean(adv_history)
        avg_entropy = np.mean(entropy_history)
        avg_reward = np.mean(reward_history)

        if avg_reward_base is None:
            avg_reward_base = avg_reward
        logger.info(f'dag sampled = {dags}')
        logger.info(
            f'| epoch {self.epoch:3d} | lr {self.controller_lr:.5f} '
            f'| R {avg_reward:.5f} | entropy {avg_entropy:.4f} '
            f'| loss {cur_loss:.5f}')

    def _summarize_shared_train(self, total_loss, raw_total_loss):
        """Logs a set of training steps."""
        cur_loss = utils.to_item(total_loss) / self.args.log_step_shared
        # NOTE(brendan): The raw loss, without adding in the activation
        # regularization terms, should be used to compute ppl.
        cur_raw_loss = utils.to_item(raw_total_loss) / self.args.log_step_shared

        logger.info(f'| epoch {self.epoch:3d} '
                    f'| lr {self.args.shared_lr:.2f} '
                    f'| raw loss {cur_raw_loss:.2f} '
                    f'| loss {cur_loss:.2f} ')