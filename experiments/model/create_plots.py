import os

from model.misc.plot_utils import *

def plot_results(args, ztl_tr, tr_rec, trainset, ztl_te, te_rec, testset, elbo_meter, nll_meter, reg_kl_meter, inducing_kl_meter):
    #plot reconstructions (data space)
    plot_rot_mnist(trainset, tr_rec, False, fname=os.path.join(args.save, 'plots/train_rec_rot_mnist.png'))
    plot_rot_mnist(testset, te_rec, False, fname=os.path.join(args.save, 'plots/test_rec_rot_mnist.png'))
    
    #plot latent trajectories (latent space)
    plot_latent_traj(ztl_tr,   fname='rot_mnist_latents_tr.png')
    plot_latent_traj(ztl_te, fname='rot_mnist_latents_test.png')

    #plot loss 
    plot_trace(elbo_meter, nll_meter, reg_kl_meter, inducing_kl_meter, args) # logpL_meter, logztL_meter, args)






