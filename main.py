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

from model.model_misc import build_model, train_model #, train_mov_mnist
from model.misc import io_utils
from model.misc.torch_utils import seed_everything
from data.data_utils import load_data

SOLVERS   = ["euler", "bdf", "rk4", "midpoint", "adams", "explicit_adams", "fixed_adams", "euler"]
DE_MODELS = ['MLP', 'SVGP']
INV_FNCS  = ['MLP', 'SVGP']
KERNELS   = ['RBF', 'DF']
TASKS     = ['rot_mnist', 'mov_mnist', 'sin', 'spiral', 'lv']
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
parser.add_argument('--Ntrain', type=int, default=500, 
                    help="Number training data points")
parser.add_argument('--Nvalid', type=int, default=50, 
                    help="Number valid data points")
parser.add_argument('--Ntest', type=int, default=50, 
                    help="Number of testing data points")
parser.add_argument('--seq_len', type=int, default=15, 
                    help="Number of training frames for Moving Mnist train reconstruction")
parser.add_argument('--rotrand', type=eval, default=True,
                    help="if True multiple initial rotation angles")
parser.add_argument('--style', type=list, default=[0,1,2,3,4,5,6,7,8,9], 
                    help="List of styles (digits) to use")
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
parser.add_argument('--T_inv', type=int, default=25,
                    help="Time frames to select for RNN based Encoder for Invariance")

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
parser.add_argument('--T_in', type=int, default=10,
                    help="Time frames to select for RNN based Encoder for intial state")
                    

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
parser.add_argument('--forecast_tr',type=int, default=2, #for moving mnist 2
                    help="Number of forecast steps for plotting train")
parser.add_argument('--forecast_te',type=int, default=2, #for moving mnist 2 
                    help="Number of forecast steps for plotting test")

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
    trainset, validset, testset = load_data(args, device, dtype)
    logger.info('********** {} dataset with loaded ********** '.format(args.task))

    ########### model ###########
    invodevae = build_model(args, device, dtype)
    invodevae.to(device)
    invodevae.to(dtype)

    logger.info('********** Model Built {} ODE with invariance {} and contrastive loss {} **********'.format(args.de, args.inv_latent_dim, args.contr_loss))
    logger.info('********** Training Augemented Dynamics: {} **********'.format(invodevae.aug))
    for arg, value in sorted(vars(args).items()):
        logger.info("Argument %s: %r", arg, value)
    logger.info(invodevae)

    if args.continue_training:
        fname = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.save, 'invodevae.pth')
        invodevae.load_state_dict(torch.load(fname,map_location=torch.device(device)))
        logger.info('********** Resume training for model {} ********** '.format(fname))

    train_model(args, invodevae, plotter, trainset, validset, testset, logger)


