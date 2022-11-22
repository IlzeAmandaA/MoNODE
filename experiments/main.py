import os 
import time
from datetime import datetime, timedelta
import argparse
import torch
import torch.nn as nn

from model.create_model import build_model, compute_loss, compute_MSE
from model.create_plots import plot_results
from model.misc import io_utils
from model.misc.torch_utils import seed_everything
from model.misc import log_utils 
from data.wrappers import load_data

SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", "adams", "explicit_adams", "fixed_adams"]
DE_MODELS = ['MLP', 'SVGP']
KERNELS = ['RBF', 'DF']
parser = argparse.ArgumentParser('Bayesian Invariant Latent ODE')

#data
parser.add_argument('--task', type=str, default='rot_mnist',
                    help="Experiment type")
parser.add_argument('--num_workers', type=int, default=0,
                    help="number of workers")
parser.add_argument('--data_root', type=str, default='data/',
                    help="general data location")
parser.add_argument('--Ntrain', type=int, default=360,
                    help="Number training data points")
parser.add_argument('--Nvalid', type=int, default=40,
                    help="Number valid data points")
parser.add_argument('--rotrand', type=eval, default=True,
                    help="if True multiple initial rotatio angles")
parser.add_argument('--batch', type=int, default=20,
                    help="batch size")
parser.add_argument('--value', type=int, default=3,
                    help="training choice")

#de model
parser.add_argument('--de', type=str, default='MLP', choices=DE_MODELS,
                    help="Model type to learn the DE")
parser.add_argument('--kernel', type=str, default='RBF', choices=KERNELS,
                    help="ODE solver for numerical integration")
parser.add_argument('--num_features', type=int, default=256,
                    help="Number of Fourier basis functions (for pathwise sampling from GP)")
parser.add_argument('--num_inducing', type=int, default=100,
                    help="Number of inducing points for the sparse GP")
parser.add_argument('--dimwise', type=eval, default=True,
                    help="Specify separate lengthscales for every output dimension")
parser.add_argument('--variance', type=float, default=0.7,
                    help="Initial value for rbf variance")
parser.add_argument('--lengthscale', type=float, default=2.0,
                    help="Initial value for rbf lengthscale")

#inavariance gp
parser.add_argument('--num_inducing_inv', type=int, default=100,
                    help="Number of inducing points for inavariant GP")


#ode solver
parser.add_argument('--ode', type=int, default=1,
                    help="order of ODE")
parser.add_argument('--solver', type=str, default='euler', choices=SOLVERS,
                    help="ODE solver for numerical integration")
parser.add_argument('--D_in', type=int, default=6,
                    help="ODE f(x) input dimensionality")
parser.add_argument('--D_out', type=int, default=6,
                    help="ODE f(x) output dimensionality")
parser.add_argument('--dt', type=float, default=0.1,
                    help="numerical solver dt")
parser.add_argument('--use_adjoint', type=eval, default=False,
                    help="Use adjoint method for gradient computation")

#vae
parser.add_argument('--latent_dim', type=int, default=6,
                    help="Latent space dimensionality")
parser.add_argument('--n_filt', type=int, default=8,
                    help="Number of filters in the cnn")
parser.add_argument('--frames', type=int, default=5,
                    help="Number of timesteps used for encoding velocity") 

#training 
parser.add_argument('--Nepoch', type=int, default=5000,
                    help="Number of gradient steps for model training")
parser.add_argument('--lr', type=float, default=0.001,
                    help="Learning rate for model training")
parser.add_argument('--seed', type=int, default=121,
                    help="Global seed for the training run")
parser.add_argument('--continue_training', type=eval, default=False,
                    help="If set to True continoues training of a previous model")


#log 
parser.add_argument('--save', type=str, default='results/',
                    help="Directory name for saving all the model outputs")



if __name__ == '__main__':
    args = parser.parse_args()

    ######### setup output directory and logger ###########
    args.save = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.save+args.task+'/'+datetime.now().strftime('%d_%m_%Y-%H:%M'), '')
    io_utils.makedirs(args.save)
    io_utils.makedirs(os.path.join(args.save, 'plots'))
    logger = io_utils.get_logger(logpath=os.path.join(args.save, 'logs'))
    logger.info('Results stored in {}'.format(args.save))

    ########## set global random seed ###########
    seed_everything(args.seed)

    ########### device #######
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Running model on {}'.format(device))

    ########### data ############ 
    trainset, testset = load_data(args, device)

    ########### model ###########
    invodevae = build_model(args, device)
    invodevae.to(device)

    logger.info('********** Model Built {} ODE **********'.format(args.de))
    logger.info('Model parameters: num features {} | num inducing {} | num epochs {} | lr {} | ode {} | D_in {} | D_out {} | dt {} | kernel {} | latent_dim {} | variance {} |lengthscale {} | rotated initial angle {}'.format(
                    args.num_features, args.num_inducing, args.Nepoch,args.lr, args.ode, args.D_in, args.D_out, args.dt, args.kernel, args.latent_dim, args.variance, args.lengthscale, args.rotrand))

    if args.continue_training:
        fname = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.model_path, 'odegpvae_mnist.pth')
        invodevae.load_state_dict(torch.load(fname,map_location=torch.device(device)))
        logger.info('Resume training for model {}'.format(fname))

    ########### log loss values ########
    elbo_meter = log_utils.CachedRunningAverageMeter(10)
    nll_meter = log_utils.CachedRunningAverageMeter(10)
    reg_kl_meter = log_utils.CachedRunningAverageMeter(10)
    inducing_kl_meter = log_utils.CachedRunningAverageMeter(10)
    mse_meter = log_utils.CachedAverageMeter()
    time_meter = log_utils.CachedAverageMeter()

    ########### train ###########
    optimizer = torch.optim.Adam(invodevae.parameters(),lr=args.lr)


    logger.info('********** Started Training **********')
    begin = time.time()
    global_itr = 0
    for ep in range(args.Nepoch):
        L = 1 if ep<args.Nepoch//2 else 5 
        for itr,local_batch in enumerate(trainset):
            minibatch = local_batch.to(device) # B x T x 1 x 28 x 28 (batch, time, image dim)
            loss, nlhood, kl_reg, kl_u, Xrec_tr, ztL_tr = compute_loss(invodevae, minibatch, L)

            # if torch.isnan(loss):
            #     cache_results(logger, args, odegpvae, trainset, testset, elbo_meter, nll_meter, reg_kl_meter, inducing_kl_meter,  hyperparam_meter) 
            #     sys.exit()

            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()

            #store values 
            elbo_meter.update(loss.item(), global_itr)
            nll_meter.update(nlhood.item(), global_itr)
            reg_kl_meter.update(kl_reg.item(), global_itr)
            inducing_kl_meter.update(kl_u.item(), global_itr)
            time_meter.update(time.time() - begin, global_itr)
            global_itr +=1

            if itr % args.log_freq == 0 :
                logger.info('Iter:{:<2d} | Time {} | elbo {:8.2f}({:8.2f}) | nlhood:{:8.2f}({:8.2f}) | kl_reg:{:<8.2f}({:<8.2f}) | kl_u:{:8.5f}({:8.5f})'.\
                    format(itr, timedelta(seconds=time_meter.val), 
                                elbo_meter.val, elbo_meter.avg,
                                nll_meter.val, nll_meter.avg,
                                reg_kl_meter.val, reg_kl_meter.avg,
                                inducing_kl_meter.val, inducing_kl_meter.avg)) 
            
        with torch.no_grad():
            mse_meter.reset()
            for itr_test,test_batch in enumerate(testset):
                test_batch = test_batch.to(device)
                test_elbo, nlhood, kl_reg, kl_gp, Xrec_te, ztL_te = compute_loss(invodevae, test_batch, L)
                Xrec_te = Xrec_te.squeeze(0) #N,T,d,nc,nc
                test_mse = compute_MSE(test_batch, Xrec_te)
                torch.save(invodevae.state_dict(), os.path.join(args.save, 'invodevae.pth'))
                mse_meter.update(test_mse.item(),itr_test)
                break
        logger.info('Epoch:{:4d}/{:4d}| tr_elbo:{:8.2f}({:8.2f}) | test_elbo {:5.3f} |test_mse:{:5.3f})\n'.format(ep, args.Nepoch, elbo_meter.val, elbo_meter.avg, test_elbo.item(), mse_meter.val))
        plot_results(args, ztL_tr, Xrec_tr, minibatch, ztL_te, Xrec_te, test_batch, elbo_meter, nll_meter, reg_kl_meter, inducing_kl_meter)



