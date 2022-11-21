import os 
from datetime import datetime
import argparse
import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from model.create_model import build_model, compute_loss, compute_MSE
from model.create_plots import plot_results
from model.misc import io_utils
from model.misc.torch_utils import seed_everything
from model.misc import log_utils 
from data.wrappers import load_data
from odevae import ODEVAE
from invodevae import INVODEVAE
from utilities import plot_rot_mnist, get_dataset, plot_latent_traj

SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", "adams", "explicit_adams", "fixed_adams"]
DE_MODELS = ['MLP', 'SVGP']
parser = argparse.ArgumentParser('Bayesian Invariant Latent ODE')

#data
parser.add_argument('--task', type=str, default='rot_mnist/',
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

#model 
parser.add_argument('--de', type=str, default='MLP', choices=DE_MODELS,
                    help="Model type to learn the DE")

#ode solver
parser.add_argument('--solver', type=str, default='euler', choices=SOLVERS,
                    help="ODE solver for numerical integration")

#training 
parser.add_argument('--Nepoch', type=int, default=5000,
                    help="Number of gradient steps for model training")
parser.add_argument('--lr', type=float, default=0.001,
                    help="Learning rate for model training")
parser.add_argument('--device', type=str, default='cpu',
                    help="device")

#log 
parser.add_argument('--save', type=str, default='results/',
                    help="Directory name for saving all the model outputs")



if __name__ == '__main__':
    args = parser.parse_args()

    ######### setup output directory and logger ###########
    args.save = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.save+args.task+datetime.now().strftime('_%d_%m_%Y-%H:%M'), '')
    io_utils.makedirs(args.save)
    io_utils.makedirs(os.path.join(args.save, 'plots'))
    logger = io_utils.get_logger(logpath=os.path.join(args.save, 'logs'))
    logger.info('Results stored in {}'.format(args.save))

    ########## set global random seed ###########
    seed_everything(args.seed)

    ########### device #######
    logger.info('Running model on {}'.format(args.device))

    ########### data ############ 
    trainset, testset = load_data(args, plot=True)

    ########### model ###########
    invodevae = build_model(args)
    invodevae.to(args.device)

    logger.info('********** Model Built {} ODE **********'.format(args.de))
    logger.info('Model parameters: num features {} | num inducing {} | num epochs {} | lr {} | ode {} | D_in {} | D_out {} | dt {} | kernel {} | latent_dim {} | variance {} |lengthscale {} | rotated initial angle {}'.format(
                    args.num_features, args.num_inducing, args.Nepoch,args.lr, args.ode, args.D_in, args.D_out, args.dt, args.kernel, args.latent_dim, args.variance, args.lengthscale, args.rotrand))

    if args.continue_training:
        fname = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.model_path, 'odegpvae_mnist.pth')
        invodevae.load_state_dict(torch.load(fname,map_location=torch.device(args.device)))
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
            minibatch = local_batch.to(args.device) # B x T x 1 x 28 x 28 (batch, time, image dim)
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
                test_batch = test_batch.to(args.device)
                test_elbo, nlhood, kl_reg, kl_gp, Xrec_te, ztL_te = compute_loss(invodevae, test_batch, L)
                Xrec_te = Xrec_te.squeeze(0) #N,T,d,nc,nc
                test_mse = compute_MSE(test_batch, Xrec_te)

                plot_rot_mnist(test_batch, Xrec, False, fname=os.path.join(args.save, 'plots/rot_mnist.png'))#TODO 
                torch.save(invodevae.state_dict(), os.path.join(args.save, 'invodevae.pth'))
                mse_meter.update(test_mse.item(),itr_test)
                break
        logger.info('Epoch:{:4d}/{:4d}| tr_elbo:{:8.2f}({:8.2f}) | test_elbo {:5.3f} |test_mse:{:5.3f})\n'.format(ep, args.Nepoch, elbo_meter.val, elbo_meter.avg, test_elbo.item(), mse_meter.val))
        plot_results(args, ztL_tr, Xrec_tr, minibatch, ztL_te, Xrec_te, test_batch, elbo_meter, nll_meter, reg_kl_meter, inducing_kl_meter)



