import os 
import time
from datetime import datetime, timedelta
import argparse
import torch
import torch.nn as nn

# 2168544, 2168604 - gp, euler
# 2168567 - gp, dopri5
# 2168606 - nn, euler
# 2168608 - nn, dopri5

# 2168643 - inv, gp, euler
# 2168644 - inv, nn, euler

from model.create_model import build_model, compute_loss, compute_MSE
from model.misc import io_utils
from model.misc.torch_utils import seed_everything
from model.misc import log_utils 
from model.misc.data_utils import load_data
from model.misc.plot_utils import plot_results

SOLVERS   = ["euler", "bdf", "rk4", "midpoint", "adams", "explicit_adams", "fixed_adams", "euler"]
DE_MODELS = ['MLP', 'SVGP', 'SGP']
KERNELS   = ['RBF', 'DF']
TASKS     = ['rot_mnist', 'mov_mnist', 'sin']
parser = argparse.ArgumentParser('Bayesian Invariant Latent ODE')

#data
parser.add_argument('--task', type=str, default='rot_mnist', choices=TASKS,
                    help="Experiment type")
parser.add_argument('--aug', type=eval, default=False,
                    help="augmented ODE system or not")
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
parser.add_argument('--batch_size', type=int, default=20,
                    help="batch size")
parser.add_argument('--digit', type=int, default=3,
                    help="Rotating MNIST digit (train data)")

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
parser.add_argument('--q_diag', type=eval, default=False,
                    help="Diagonal posterior approximation for inducing variables")
parser.add_argument('--num_layers', type=int, default=2,
                    help="Number of hidden layers in MLP diff func")
parser.add_argument('--num_hidden', type=int, default=200,
                    help="Number of hidden neurons in each layer of MLP diff func")

#inavariance gp
parser.add_argument('--inv_latent_dim', type=int, default=5,
                    help="Invariant space dimensionality")
parser.add_argument('--num_inducing_inv', type=int, default=100,
                    help="Number of inducing points for inavariant GP")

#ode solver
parser.add_argument('--order', type=int, default=1,
                    help="order of ODE")
parser.add_argument('--solver', type=str, default='euler', choices=SOLVERS,
                    help="ODE solver for numerical integration")
parser.add_argument('--dt', type=float, default=0.1,
                    help="numerical solver dt")
parser.add_argument('--use_adjoint', type=eval, default=False,
                    help="Use adjoint method for gradient computation")

#vae
parser.add_argument('--ode_latent_dim', type=int, default=8,
                    help="Latent ODE dimensionality")
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
parser.add_argument('--plot_every', type=int, default=100,
                    help="How often plot the training")

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
    else:
        args.save = os.path.join(os.path.abspath(os.path.dirname(__file__)), \
            args.save+args.task+'/'+datetime.now().strftime('%d_%m_%Y-%H:%M'), '')
    io_utils.makedirs(args.save)
    io_utils.makedirs(os.path.join(args.save, 'plots'))
    logger = io_utils.get_logger(logpath=os.path.join(args.save, 'logs'))
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

    ########### data ############ 
    trainset, testset = load_data(args, device, dtype)
    logger.info('********** {} dataset with loaded ********** '.format(args.task))

    ########### model ###########
    invodevae = build_model(args, device, dtype)
    invodevae.to(device)
    invodevae.to(dtype)

    logger.info('********** Model Built {} ODE **********'.format(args.de))
    logger.info('Model parameters: num features {} | num inducing {} | num epochs {} | lr {} | order {} | dt {} | kernel {} | ODE latent_dim {} | inv_latent_dim {} | variance {} | lengthscale {} | rotated initial angle {}'.format(
                    args.num_features, args.num_inducing, args.Nepoch,args.lr, args.order, args.dt, args.kernel, args.ode_latent_dim, args.inv_latent_dim, args.variance, args.lengthscale, args.rotrand))
    logger.info(invodevae)
    if args.continue_training:
        fname = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.save, 'invodevae.pth')
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
            tr_minibatch = local_batch.to(device) # B x T x 1 x 28 x 28 (batch, time, image dim)
            loss, nlhood, kl_reg, kl_u, Xrec_tr, ztL_tr = compute_loss(invodevae, tr_minibatch, L)

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

        with torch.no_grad():
            mse_meter.reset()
            for itr_test,test_batch in enumerate(testset):
                test_batch = test_batch.to(device)
                test_elbo, nlhood, kl_reg, kl_gp, Xrec_te, ztL_te = compute_loss(invodevae, test_batch, L=1)
                Xrec_te = Xrec_te.squeeze(0) #N,T,d,nc,nc
                test_mse = compute_MSE(test_batch, Xrec_te)
                torch.save(invodevae.state_dict(), os.path.join(args.save, 'invodevae.pth'))
                mse_meter.update(test_mse.item(),itr_test)
            logger.info('Epoch:{:4d}/{:4d}| tr_elbo:{:8.2f}({:8.2f}) | test_elbo {:5.3f} |test_mse:{:5.3f})'.format(ep, args.Nepoch, elbo_meter.val, elbo_meter.avg, test_elbo.item(), mse_meter.val))   

            if ep % args.plot_every==0:
                Xrec_tr, ztL_tr, _, _ = invodevae(tr_minibatch, L=1, T_custom=2*tr_minibatch.shape[1])
                Xrec_te, ztL_te, _, _ = invodevae(test_batch,   L=1, T_custom=2*test_batch.shape[1])

                plot_results(plotter, args, ztL_tr[0,:,:,:], Xrec_tr[0,:,:,:], tr_minibatch, ztL_te[0,:,:,:], \
                    Xrec_te.squeeze(0), test_batch, elbo_meter, nll_meter, reg_kl_meter, inducing_kl_meter)
    


