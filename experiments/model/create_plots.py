import os

from model.misc.plot_utils import *

def plot_results(args, ztl_tr, tr_rec, trainset, ztl_te, te_rec, testset, elbo_meter, nll_meter, reg_kl_meter, inducing_kl_meter):
    #plot reconstructions (data space)
    plot_rot_mnist(trainset, tr_rec, False, fname=os.path.join(args.save, 'plots/rec_tr_rot_mnist.png'))
    plot_rot_mnist(testset, te_rec, False, fname=os.path.join(args.save, 'plots/rec_te_rot_mnist.png'))
    
    #plot latent trajectories (latent space)
    plot_latent_traj(ztl_tr,   fname=os.path.join(args.save, 'plots/latent_tr_rot_mnist.png'))
    plot_latent_traj(ztl_te, fname=os.path.join(args.save, 'plots/latent_te_rot_mnist.png'))

    #plot loss 
    plot_trace(elbo_meter, nll_meter, reg_kl_meter, inducing_kl_meter, args) # logpL_meter, logztL_meter, args)


def plot_results_caca(plotter, args, ztl_tr, tr_rec, trainset, ztl_te, te_rec, \
    testset, elbo_meter, nll_meter, reg_kl_meter, inducing_kl_meter):
    #plot reconstructions (data space)

    plotter.plot_fit(trainset, tr_rec, 'tr')
    plotter.plot_fit(testset,  te_rec, 'test')

    plotter.plot_latent(ztl_tr, 'tr')
    plotter.plot_latent(ztl_te, 'test')

    # plot_trace(elbo_meter, nll_meter, reg_kl_meter, inducing_kl_meter, args) # logpL_meter, logztL_meter, args)




