import os 
from datetime import datetime
import argparse
import torch
import torch.nn as nn

# GP
# 2184379 - not inv
# 2184375 - inv
# 2184374/2185002 - inv + contr
# NN
# 2184377 - not inv
# 2184376 - inv
# 2184373/2184531 - inv + contr

from model.model_misc import build_model, train_model, train_mov_mnist
from model.misc import io_utils
from model.misc.torch_utils import seed_everything
from data.data_utils import load_data

SOLVERS   = ["euler", "bdf", "rk4", "midpoint", "adams", "explicit_adams", "fixed_adams", "euler"]
DE_MODELS = ['MLP', 'SVGP']
INV_FNCS  = ['MLP', 'SVGP']
KERNELS   = ['RBF', 'DF']
TASKS     = ['rot_mnist', 'mov_mnist', 'sin']
CNN_ARCHITECTURE = ['cnn', 'dcgan', 'vgg64']
parser = argparse.ArgumentParser('Bayesian Invariant Latent ODE')

# TASK = 'rot_mnist'
# NTRAIN_DEFAULTS     = {'rot_mnist':400, 'mov_mnist':400, 'sin':250}
# BATCH_SIZE_DEFAULTS = {'rot_mnist':25,  'mov_mnist':25,  'sin':50}

#data
parser.add_argument('--task', type=str, default='mov_mnist', choices=TASKS,
                    help="Experiment type")
parser.add_argument('--num_workers', type=int, default=0,
                    help="number of workers")
parser.add_argument('--data_root', type=str, default='data/',
                    help="general data location")
parser.add_argument('--Ntrain', type=int, default=3000,
                    help="Number training data points")
parser.add_argument('--Nvalid', type=int, default=100,
                    help="Number valid data points")
parser.add_argument('--rotrand', type=eval, default=True,
                    help="if True multiple initial rotation angles")
parser.add_argument('--digit', type=int, default=5,
                    help="Rotating MNIST digit (train data)")
parser.add_argument('--seq_len', type=int, default=15,
                    help="For Moving MNIST seq_len for training")
parser.add_argument('--nx', type=int, default=64,
                    help="Frame size")
parser.add_argument('--max_speed', type=int, metavar='SPEED', default=4,
               help='For Moving MNIST only. Digits maximum speed.')
parser.add_argument('--deterministic', type=eval, default=True,
               help='For Moving MNIST only. Whether to consider deterministic, instead of stochastic, bounces.')
parser.add_argument('--ndigits', type=int, metavar='DIGITS', default=2,
               help='For Moving MNIST only. Number of digits.')
parser.add_argument('--subsample', type=int, default=600,
                    help="Subsample styles for Moving MNIST")
parser.add_argument('--seq_len_valid', type=int, default=30,
                    help="For Moving MNIST seq_len for validation")
parser.add_argument('--shuffle', type=eval, default=True,
               help='For Moving MNIST whetehr to shuffle the data')

#de model
parser.add_argument('--ode_latent_dim', type=int, default=10,
                    help="Latent ODE dimensionality")
parser.add_argument('--de', type=str, default='MLP', choices=DE_MODELS,
                    help="Model type to learn the DE")
parser.add_argument('--kernel', type=str, default='RBF', choices=KERNELS,
                    help="ODE solver for numerical integration")
parser.add_argument('--num_features', type=int, default=100,
                    help="Number of Fourier basis functions (for pathwise sampling from GP)")
parser.add_argument('--num_inducing', type=int, default=100,
                    help="Number of inducing points for the sparse GP")
parser.add_argument('--dimwise', type=eval, default=True,
                    help="Specify separate lengthscales for every output dimension")
parser.add_argument('--variance', type=float, default=0.7,
                    help="Initial value for rbf variance")
parser.add_argument('--lengthscale', type=float, default=2.0,
                    help="Initial value for rbf lengthscale")
parser.add_argument('--q_diag', type=eval, default=False,
                    help="Diagonal posterior approximation for inducing variables")
parser.add_argument('--num_layers', type=int, default=2,
                    help="Number of hidden layers in MLP diff func")
parser.add_argument('--num_hidden', type=int, default=200,
                    help="Number of hidden neurons in each layer of MLP diff func")

#inavariance
parser.add_argument('--inv_fnc', type=str, default='MLP', choices=INV_FNCS,
                    help="Invariant function")
parser.add_argument('--inv_latent_dim', type=int, default=16,
                    help="Invariant space dimensionality")
parser.add_argument('--num_inducing_inv', type=int, default=100,
                    help="Number of inducing points for inavariant GP")
parser.add_argument('--contr_loss', type=eval, default=True,
                    help="Contrastive training of the invariant encoder")

#ode stuff
parser.add_argument('--order', type=int, default=1,
                    help="order of ODE")
parser.add_argument('--solver', type=str, default='euler', choices=SOLVERS,
                    help="ODE solver for numerical integration")
parser.add_argument('--dt', type=float, default=0.1,
                    help="numerical solver dt")
parser.add_argument('--use_adjoint', type=eval, default=False,
                    help="Use adjoint method for gradient computation")

#vae
parser.add_argument('--n_filt', type=int, default=16,
                    help="Number of filters in the cnn")
parser.add_argument('--frames', type=int, default=5,
                    help="Number of timesteps used for encoding velocity") 
parser.add_argument('--decoder_H', type=int, default=100,
                    help="Number of hidden neurons in MLP decoder") 
parser.add_argument('--rnn_hidden', type=int, default=10,
                    help="Encoder RNN latent dimensionality") 
parser.add_argument('--dec_act', type=str, default='relu',
                    help="MLP Decoder activation") 
parser.add_argument('--cnn_arch', type=str, default='dcgan', choices=CNN_ARCHITECTURE,
                    help="CNN architecture type") 
                    

#training 
parser.add_argument('--Nepoch', type=int, default=2500,
                    help="Number of gradient steps for model training")
parser.add_argument('--batch_size', type=int, default=25,
                    help="batch size")
parser.add_argument('--lr', type=float, default=0.002,
                    help="Learning rate for model training")
parser.add_argument('--seed', type=int, default=121,
                    help="Global seed for the training run")
parser.add_argument('--continue_training', type=eval, default=False,
                    help="If set to True continoues training of a previous model")
parser.add_argument('--plot_every', type=int, default=100,
                    help="How often plot the training")
parser.add_argument('--plotL', type=int, default=1,
                    help="Number of MC draws for plotting")

#log 
parser.add_argument('--save', type=str, default='results/',
                    help="Directory name for saving all the model outputs")


if __name__ == '__main__':
    args = parser.parse_args()

    ######### setup output directory and logger ###########
    # if running in the cluster in Tuebingen
    if 'cyildiz40' in os.getcwd():
        from pathlib import Path
        path = 'figs'
        try:
            p1 = os.environ['SLURM_ARRAY_JOB_ID']
            p2 = os.environ['SLURM_ARRAY_TASK_ID']
            path = os.path.join(path, p1, p2)
        except:
            p1 = os.environ['SLURM_JOB_ID']
            path = os.path.join(path,p1)
        Path(path).mkdir(parents=True, exist_ok=True)
        args.save = path
    elif 'cagatay' in os.getcwd():
        args.save = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.save, args.task)
    else:
        args.save = os.path.join(os.path.abspath(os.path.dirname(__file__)), \
            args.save+args.task+'/'+datetime.now().strftime('%d_%m_%Y-%H:%M'), '')
    io_utils.makedirs(args.save)
    io_utils.makedirs(os.path.join(args.save, 'plots'))
    logger = io_utils.get_logger(logpath=os.path.join(args.save, 'logs.txt'))
    logger.info('Results stored in {}'.format(args.save))

    ########## set global random seed ###########
    seed_everything(args.seed)

    ########## dtype #########
    dtype = torch.float64
    logger.info('********** Float type is {} ********** '.format(dtype))

    ########## plotter #######
    from model.misc.plot_utils import Plotter
    save_path = os.path.join(args.save, 'plots')
    plotter   = Plotter(save_path, args.task)

    ########### device #######
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('********** Running model on {} ********** '.format(device))

    ########### data ############ ``
    trainset, validset = load_data(args, device, dtype)
    logger.info('********** {} dataset with loaded ********** '.format(args.task))

    ########### model ###########
    invodevae = build_model(args, device, dtype)
    invodevae.to(device)
    invodevae.to(dtype)

    logger.info('********** Model Built {} ODE **********'.format(args.de))
    if args.task == 'mov_mnist': 
        if args.de == 'SVGP': 
            logger.info('Model parameters: subsample style {} | Ndata {} | cnn architecture {} | ODE latent_dim {} |inv_latent_dim {}| lr {} | order {} | dt {}| num features {} | num inducing {}  |  kernel {} |   variance {} | lengthscale {} | solver {}|  inv_fnc {}'.format(
                args.subsample, args.Ntrain, args.cnn_arch, args.ode_latent_dim, args.inv_latent_dim, args.lr, args.order, args.dt, args.num_features, args.num_inducing, args.kernel, args.variance, args.lengthscale, args.solver, args.inv_fnc))               
        elif args.de == 'MLP':
            logger.info('Model parameters: subsample style {} | Ndata {} |  cnn architecture {} | ODE latent_dim {} |inv_latent_dim {}| lr {} | order {} | dt {} | solver {} | inv_fnc {}'.format(
                args.subsample, args.Ntrain, args.cnn_arch, args.ode_latent_dim, args.inv_latent_dim, args.lr, args.order, args.dt, args.solver, args.inv_fnc))               
    
    elif args.task == 'rot_mnist':
        logger.info('Model parameters: num features {} | num inducing {} | num epochs {} | lr {} | order {} | dt {} | kernel {} | ODE latent_dim {} | inv_latent_dim {} | variance {} | lengthscale {} | rotated initial angle {}| cnn architecture {}'.format(
                    args.num_features, args.num_inducing, args.Nepoch,args.lr, args.order, args.dt, args.kernel, args.ode_latent_dim, args.inv_latent_dim, args.variance, args.lengthscale, args.rotrand, args.cnn_arch))

    logger.info(invodevae)
    if args.continue_training:
        fname = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.save, 'invodevae.pth')
        invodevae.load_state_dict(torch.load(fname,map_location=torch.device(device)))
        logger.info('********** Resume training for model {} ********** '.format(fname))

    # train_model(args, invodevae, plotter, trainset, testset, logger)
    train_mov_mnist(args, invodevae, plotter, trainset, validset, logger)





    ### additional experiments
    # fname = '/Users/cagatay/Nextcloud/InvOdeVaeOriginal/results/2196031/invodevae.pth'
    # fname = '/mnt/qb/work/bethge/cyildiz40/InvOdeVae/figs/2180009/invodevae.pth'
    # invodevae.load_state_dict(torch.load(fname,map_location=torch.device(device)))
    # train_model(args, invodevae, plotter, trainset, testset, logger, freeze_dyn=True)

