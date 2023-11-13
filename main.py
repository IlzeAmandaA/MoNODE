import os 
import argparse, time
import numpy as np
import torch
from datetime import datetime
from model.build_model import build_model
from model.model_misc import train_model 
from model.misc import io_utils
from model.misc.torch_utils import seed_everything, count_params
from data.data_utils import load_data

SOLVERS   = ["euler", "bdf", "rk4", "midpoint", "adams", "explicit_adams", "fixed_adams", "dopri5"]
TASKS     = ['rot_mnist', 'rot_mnist_ou', 'sin', 'bb', 'lv', 'mocap', 'mocap_shift']
MODELS     = ['node', 'sonode', 'hbnode']
GRADIENT_ESTIMATION = ['no_adjoint', 'adjoint', 'ac_adjoint']
parser = argparse.ArgumentParser('MoNODE')

#data
parser.add_argument('--task', type=str, default='mov_mnist', choices=TASKS,
                    help="Experiment type")
parser.add_argument('--noise', type=float, default=None,
                    help="set noise level for noise robustness experiments")  
parser.add_argument('--Nobj', type=int, default=1,
                    help="param that can be used for multiple object set-up")                 
parser.add_argument('--num_workers', type=int, default=0,
                    help="number of workers")
parser.add_argument('--data_root', type=str, default='data/',
                    help="general data location")
parser.add_argument('--shuffle', type=eval, default=True,
               help='For Moving MNIST whetehr to shuffle the data')

#de model
parser.add_argument('--model', type=str, default='node', choices=MODELS,
                    help='node model type')
parser.add_argument('--ode_latent_dim', type=int, default=10,
                    help="Latent ODE dimensionality")
parser.add_argument('--de_L', type=int, default=2,
                    help="Number of hidden layers in MLP diff func")
parser.add_argument('--de_H', type=int, default=100,
                    help="Number of hidden neurons in each layer of MLP diff func")


#invariance
parser.add_argument('--inv_fnc', type=str, default='MLP',
                    help="Invariant function")
parser.add_argument('--modulator_dim', type=int, default=0,
                    help = 'dim of the dynamics modulator variable')
parser.add_argument('--content_dim', type=int, default=0,
                    help = 'dim of the content variable')
parser.add_argument('--T_inv', type=int, default=5,
                    help="Time frames to select for RNN based Encoder for Invariance")
parser.add_argument('--cnn_filt_inv', type=int, default=16,
                    help="Nfilt invariant encoder cnn")


#ode stuff
parser.add_argument('--order', type=int, default=1,
                    help="order of ODE")
parser.add_argument('--solver', type=str, default='euler', choices=SOLVERS,
                    help="ODE solver for numerical integration")
parser.add_argument('--dt', type=float, default=0.1,
                    help="numerical solver dt")
parser.add_argument('--use_adjoint', type=str, default='no_adjoint', choices=GRADIENT_ESTIMATION, #we used False
                    help="Use adjoint method for gradient computation")

#vae 
parser.add_argument('--T_in', type=int, default=10,
                    help="Time frames to select for RNN based Encoder for intial state")
parser.add_argument('--cnn_filt_enc', type=int, default=16,
                    help="Number of filters in the cnn encoder")
parser.add_argument('--cnn_filt_de', type=int, default=16,
                    help="Number of filters in the cnn decoder")
parser.add_argument('--rnn_hidden', type=int, default=10,
                    help="Encoder RNN latent dimensionality") 
parser.add_argument('--dec_H', type=int, default=100,
                    help="Number of hidden neurons in MLP decoder") 
parser.add_argument('--dec_L', type=int, default=2,
                    help="Number of hidden layers in MLP decoder") 
parser.add_argument('--dec_act', type=str, default='relu',
                    help="MLP Decoder activation") 
parser.add_argument('--enc_H', type=int, default=50,
                    help="Encoder hidden dimensionality for GRU unit") 
parser.add_argument('--sonode_v', type=str, default='MLP', choices=['MLP','RNN'],
                    help="velocity encoder for SONODE") 

#training 
parser.add_argument('--Nepoch', type=int, default=600,
                    help="Number of gradient steps for model training")
parser.add_argument('--Nincr', type=int, default=10,
                    help="Number of sequential increments of the sequence length")
parser.add_argument('--batch_size', type=int, default=25,
                    help="batch size")
parser.add_argument('--lr', type=float, default=0.002,
                    help="Learning rate for model training")
parser.add_argument('--seed', type=int, default=121,
                    help="Global seed for the training run")
parser.add_argument('--continue_training', type=eval, default=False,
                    help="If set to True continoues training of a previous model")
parser.add_argument('--plot_every', type=int, default=20,
                    help="How often plot the training")
parser.add_argument('--plotL', type=int, default=1,
                    help="Number of MC draws for plotting")
parser.add_argument('--forecast_tr',type=int, default=2, 
                    help="Number of forecast steps for plotting train")
parser.add_argument('--forecast_vl',type=int, default=2,
                    help="Number of forecast steps for plotting test")
parser.add_argument('--exp_id', type=int, default=0,
                    help = 'exp ID for directory')

#log 
parser.add_argument('--save', type=str, default='results/',
                    help="Directory name for saving all the model outputs")


if __name__ == '__main__':
    args = parser.parse_args()

    ######### setup output directory and logger ###########
    args.save = os.path.join(os.path.abspath(os.path.dirname(__file__)), \
        args.save+args.task+'/'+args.model+'/'+datetime.now().strftime('%d_%m_%Y-%H:%M-')+str(args.exp_id), '')
    
    ############################
    io_utils.makedirs(args.save)
    io_utils.makedirs(os.path.join(args.save, 'plots'))
    io_utils.makedirs(os.path.join(args.save, 'plots', 'fit'))
    io_utils.makedirs(os.path.join(args.save, 'plots', 'latents'))
    logger = io_utils.get_logger(logpath=os.path.join(args.save, 'logs.txt'))
    logger.info('Results stored in {}'.format(args.save))

    ########## set global random seed ###########
    if args.seed==-1:
        args.seed = int(time.time()*np.random.random()/1000)
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
    trainset, validset, testset, params = load_data(args, device, dtype)
    logger.info('********** {} dataset with loaded ********** '.format(args.task))
    logger.info('data params: {}'.format(params[args.task]))

    ########### model ###########
    model = build_model(args, device, dtype)
    model.to(device)
    model.to(dtype)

    logger.info('********** Built {} model with dynamics modulator dim {} and  content variable dim {}**********'.format(args.model, args.modulator_dim, args.content_dim))
    logger.info('********** Number of parameters: {} **********'.format(count_params(model)))
    logger.info('********** Augmented Dynamics: {} **********'.format(model.aug))
    for arg, value in sorted(vars(args).items()):
        logger.info("Argument %s: %r", arg, value)
    logger.info(model)

    if args.continue_training:
        fname = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.save, 'model.pth')
        model.load_state_dict(torch.load(fname,map_location=torch.device(device)))
        logger.info('********** Resume training for model {} ********** '.format(fname))

    train_model(args, model, plotter, trainset, validset, testset, logger, params[args.task])


