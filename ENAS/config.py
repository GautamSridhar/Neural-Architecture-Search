import argparse
from utils import get_logger

logger = get_logger()


arg_lists = []
parser = argparse.ArgumentParser()

def str2bool(v):
    return v.lower() in ('true')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Network
net_arg = add_argument_group('Network')

# Controller
net_arg.add_argument('--network_type', type=str, default='Net')
net_arg.add_argument('--_max_depth', type=int, default=4)
net_arg.add_argument('--_max_width', type=int, default=16)
net_arg.add_argument('--_network_inputsize', type=int, default=2)
net_arg.add_argument('--_network_outputsize', type=int, default=2)
net_arg.add_argument('--controller_hid', type=int, default=50)


# Shared parameters for PTB
# NOTE(brendan): See Merity config for wdrop
# https://github.com/salesforce/awd-lstm-lm.
net_arg.add_argument('--operations', type=eval,
                     default="['Identity','LinearReLU_2','LinearReLU_4','LinearReLU_8','LinearReLU_16','LinearTanh_2','LinearTanh_4','LinearTanh_8','LinearTanh_16']")

# Training / test parameters
learn_arg = add_argument_group('Learning')
learn_arg.add_argument('--mode', type=str, default='train',
                       choices=['train', 'derive', 'single'],
                       help='train: Training ENAS, derive: Deriving Architectures,\
                       single: training one dag')
learn_arg.add_argument('--max_epoch', type=int, default=150)
learn_arg.add_argument('--entropy_mode', type=str, default='reward', choices=['reward', 'regularizer'])


# Controller
learn_arg.add_argument('--ppl_square', type=str2bool, default=False)
# NOTE(brendan): (Zoph and Le, 2017) page 8 states that c is a constant,
# usually set at 80.
learn_arg.add_argument('--reward_c', type=int, default=500,
                       help="WE DON'T KNOW WHAT THIS VALUE SHOULD BE") # TODO
# NOTE(brendan): irrelevant for actor critic.
learn_arg.add_argument('--ema_baseline_decay', type=float, default=0.95) # TODO: very important
learn_arg.add_argument('--discount', type=float, default=1.0) # TODO
learn_arg.add_argument('--controller_max_step', type=int, default=2000,
                       help='step for controller parameters')
learn_arg.add_argument('--controller_optim', type=str, default='adam')
learn_arg.add_argument('--controller_lr', type=float, default=3.5e-4,
                       help="will be ignored if --controller_lr_cosine=True")
learn_arg.add_argument('--controller_lr_cosine', type=str2bool, default=False)
learn_arg.add_argument('--controller_lr_max', type=float, default=0.05,
                       help="lr max for cosine schedule")
learn_arg.add_argument('--controller_lr_min', type=float, default=0.001,
                       help="lr min for cosine schedule")
learn_arg.add_argument('--controller_grad_clip', type=float, default=0)
learn_arg.add_argument('--tanh_c', type=float, default=2.5)
learn_arg.add_argument('--softmax_temperature', type=float, default=5.0)
learn_arg.add_argument('--entropy_coeff', type=float, default=1e-4)

# Shared Network parameters
learn_arg.add_argument('--shared_initial_step', type=int, default=0)
learn_arg.add_argument('--shared_max_step', type=int, default=100,
                       help='step for shared parameters')
learn_arg.add_argument('--shared_num_sample', type=int, default=1,
                       help='# of Monte Carlo samples')
learn_arg.add_argument('--shared_optim', type=str, default='adam')
learn_arg.add_argument('--shared_lr', type=float, default=0.0001)
learn_arg.add_argument('--shared_l2_reg', type=float, default=1e-7)

learn_arg.add_argument('--derive_num_sample', type=int, default=10)


# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--dataset', type=str, default='LV', help='dataset to be used')
misc_arg.add_argument('--integrate_method', type=str, default='dopri5', help='method for numerical integration')
misc_arg.add_argument('--load_path', type=str, default='')
misc_arg.add_argument('--log_step_shared', type=int, default=5)
misc_arg.add_argument('--log_step_controller', type=int, default=50)
misc_arg.add_argument('--save_epoch', type=int, default=5)
misc_arg.add_argument('--max_save_num', type=int, default=4)
misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
misc_arg.add_argument('--log_dir', type=str, default='logs')
misc_arg.add_argument('--random_seed', type=int, default=1)
misc_arg.add_argument('--dag_path', type=str, default='')

def get_args():
    """Parses all of the arguments above, which mostly correspond to the
    hyperparameters mentioned in the paper.
    """
    args, unparsed = parser.parse_known_args()
    if len(unparsed) > 1:
        logger.info(f"Unparsed args: {unparsed}")
    return args, unparsed