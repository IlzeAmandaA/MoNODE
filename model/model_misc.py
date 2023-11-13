import os, numpy as np
from datetime import datetime
import torch
from torch.distributions import kl_divergence as kl

from model.misc import log_utils 
from model.misc.plot_utils import plot_results


def elbo(model, X, Xrec, s0_mu, s0_logv, v0_mu, v0_logv,L):
    ''' Input:
            qz_m        - latent means [N,2q]
            qz_logv     - latent logvars [N,2q]
            X           - input images [L,N,T,nc,d,d]
            Xrec        - reconstructions [L,N,T,nc,d,d]
        Returns:
            likelihood
            kl terms
    '''
    # KL reg
    q = model.vae.encoder.q_dist(s0_mu, s0_logv, v0_mu, v0_logv)
    kl_z0 = kl(q, model.vae.prior).sum(-1) #N

    #Reconstruction log-likelihood
    lhood = model.vae.decoder.log_prob(X,Xrec,L) #L,N,T,d,nc,nc
    idx   = list(np.arange(X.ndim+1)) # 0,1,2,...
    lhood = lhood.sum(idx[2:]).mean(0) #N

    return lhood.mean(), kl_z0.mean() 


def contrastive_loss(C):
    ''' 
    C - invariant embeddings [N,T,q] or [L,N,T,q] 
    '''
    C = C.mean(0) if C.ndim==4 else C
    C = C / C.pow(2).sum(-1,keepdim=True).sqrt() # N,Tinv,q
    N_,T_,q_ = C.shape
    C = C.reshape(N_*T_,q_) # NT,q
    Z   = (C.unsqueeze(0) * C.unsqueeze(1)).sum(-1) # NT, NT
    idx = torch.meshgrid(torch.arange(T_),torch.arange(T_))
    idxset0 = torch.cat([idx[0].reshape(-1)+ n*T_ for n in range(N_)])
    idxset1 = torch.cat([idx[1].reshape(-1)+ n*T_ for n in range(N_)])
    pos = Z[idxset0,idxset1].sum()
    return -pos

def compute_mse(model, data, T_train, L=1, task=None):

    T_start = 0
    T_max = 0
    T = data.shape[1]
    #run model    
    Xrec, ztL, (s0_mu, s0_logv), (v0_mu, v0_logv), C, c, m = model(data, L, T)
    
    dict_mse = {}
    while T_max < T:
        if task == 'rot_mnist':
            T_max += T_train
            mse = torch.mean((Xrec[:,:,T_start:T_max]-data[:,T_start:T_max])**2)
            dict_mse[str(T_max)] = mse
            T_start += T_train 
            T_max += T_train
        else:
            T_max += T_train
            mse = torch.mean((Xrec[:,:,:T_max]-data[:,:T_max])**2)
            dict_mse[str(T_max)] = mse 
    return dict_mse 


def compute_loss(model, data, L, num_observations):
    """
    Compute loss for optimization
    @param model: mo/node  
    @param data: true observation sequence 
    @param L: number of MC samples
    @return: loss, nll, regularizing_kl, inducing_kl
    """
    T = data.shape[1]

    #run model    
    Xrec, ztL, (s0_mu, s0_logv), (v0_mu, v0_logv), C, c, m = model(data, L, T_custom=T)

    #compute loss
    if model.model =='sonode':
        mse   = torch.mean((Xrec-data)**2)
        loss = mse 
        return loss, 0.0, 0.0, Xrec, ztL, mse, c, m
    
    elif model.model =='node' or model.model == 'hbnode':
        lhood, kl_z0 = elbo(model, data, Xrec, s0_mu, s0_logv, v0_mu, v0_logv,L)
        
        lhood = lhood * num_observations
        kl_z0 = kl_z0 * num_observations
        loss  = - lhood + kl_z0
        mse   = torch.mean((Xrec-data)**2)
        return loss, -lhood, kl_z0, Xrec, ztL, mse, c, m
    

def freeze_pars(par_list):
    for par in par_list:
        try:
            par.requires_grad = False
        except:
            print('something wrong!')
            raise ValueError('This is not a parameter!?')


def train_model(args, model, plotter, trainset, validset, testset, logger, params, freeze_dyn=False):

    loss_meter  = log_utils.CachedRunningAverageMeter(0.97)
    tr_mse_meter   = log_utils.CachedRunningAverageMeter(0.97)
    vl_mse_rec  = log_utils.CachedRunningAverageMeter(0.97)
    vl_mse_for = log_utils.CachedRunningAverageMeter(0.97)
    time_meter = log_utils.CachedRunningAverageMeter(0.97)

    if args.model == 'node' or args.model=='hbnode':
        nll_meter   = log_utils.CachedRunningAverageMeter(0.97)
        kl_z0_meter = log_utils.CachedRunningAverageMeter(0.97)
        

    logger.info('********** Started Training **********')
    if freeze_dyn:
        freeze_pars(model.flow.parameters())

    ############## build the optimizer ############
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ########## Training loop ###########
    start_time=datetime.now()
    global_itr = 0
    best_valid_loss = None
    test_mse = 0.0
    #increase the data set length in N increments sequentally 
    T_train = params['train']['T']
    if args.task == 'rot_mnist':
        T_train = params['train']['T'] - 1 #compute mse for one loop
    ep_inc_c = args.Nepoch // args.Nincr
    ep_inc_v = T_train // args.Nincr
    T_ = ep_inc_v
    best_valid_loss = 1e9

    for ep in range(args.Nepoch):
        #no latent space to sample from
        if args.model == 'sonode':
            L=1
        else:
            L = 1 if ep<args.Nepoch//2 else 5 

        if (ep != 0) and (ep % ep_inc_c == 0):
            T_ += ep_inc_v

        for itr,local_batch in enumerate(trainset):
            tr_minibatch = local_batch.to(model.device) # N,T,...
            if args.task=='sin' or args.task=='spiral' or args.task=='lv' or 'mocap' in args.task: #slowly increase sequence length
                [N,T] = tr_minibatch.shape[:2]

                N_  = int(N*(T//T_))
                if T_ < T:
                    t0s = torch.randint(0,T-T_,[N_])  #select a random initial point from the sequence
                else:
                    t0s = torch.zeros([N_]).to(int)
                tr_minibatch = tr_minibatch.repeat([N_,1,1])
                tr_minibatch = torch.stack([tr_minibatch[n,t0:t0+T_] for n,t0 in enumerate(t0s)]) # N*ns,T//2,d
                
            loss, nlhood, kl_z0, Xrec_tr, ztL_tr, tr_mse, _, _ = \
                compute_loss(model, tr_minibatch, L, num_observations = params['train']['N'])

            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()

            #store values 
            loss_meter.update(loss.item(), global_itr)
            tr_mse_meter.update(tr_mse.item(), global_itr)
            if args.model == 'node' or args.model=='hbnode':
                nll_meter.update(nlhood.item(), global_itr)
                kl_z0_meter.update(kl_z0.item(), global_itr)
            global_itr +=1

            time_val = datetime.now()-start_time
            
        with torch.no_grad():
            
            dict_valid_mses = {}
            for valid_batch in validset:
                valid_batch = valid_batch.to(model.device)
                dict_mse = compute_mse(model, valid_batch, T_train)
                for key,val in dict_mse.items():
                        if key not in dict_valid_mses:
                            dict_valid_mses[key] = []
                        dict_valid_mses[key].append(val.item())
            T_rec = list(dict_valid_mses.keys())[0]
            T_for  = list(dict_valid_mses.keys())[-1]
            valid_mse_rec = np.mean(dict_valid_mses[T_rec])
            valid_mse_for = np.mean(dict_valid_mses[T_for])

            logger.info('Epoch:{:4d}/{:4d} | tr_loss:{:8.2f}({:8.2f}) | valid_mse T={} :{:5.3f} | valid_mse T={} :{:5.3f} '.\
                    format(ep, args.Nepoch, loss_meter.val, loss_meter.avg, T_rec, valid_mse_rec, T_for, valid_mse_for)) 
                
            # update valid loggers
            vl_mse_rec.update(valid_mse_rec,ep)
            vl_mse_for.update(valid_mse_for, ep) 
            time_meter.update(time_val.seconds, ep)
    
            #compare validation error seen so far
            if best_valid_loss > valid_mse_rec: #we want as smaller mse
                best_valid_loss = valid_mse_rec

                torch.save({
                    'args': args,
                    'state_dict': model.state_dict(),
                }, os.path.join(args.save, 'model.pth'))
                            
                #compute test error for this model 
                dict_test_mses = {}
                # test_mse = {}
                for test_batch in testset:
                    test_batch = test_batch.to(model.device)
                    dict_mse = compute_mse(model, test_batch, T_train, L=1, task=args.task)
                    for key,val in dict_mse.items():
                        if key not in dict_test_mses:
                            dict_test_mses[key] = []
                        dict_test_mses[key].append(val.item())

                logger.info('********** Current Best Model based on validation error ***********')
                logger.info('Epoch:{:4d}/{:4d}'.format(ep, args.Nepoch))
                for key, val in dict_test_mses.items():
                    logger.info('T={} test_mse {:5.3f}({:5.3f})'.format(key, np.mean(dict_test_mses[key]), np.std(dict_test_mses[key])))

            if ep % args.plot_every==0 or (ep+1) == args.Nepoch:
                Xrec_tr, ztL_tr, _, _, C_tr, _, _ = model(tr_minibatch, L=args.plotL, T_custom=args.forecast_tr*tr_minibatch.shape[1])
                Xrec_vl, ztL_vl, _, _, C_vl, _, _ = model(valid_batch,  L=args.plotL, T_custom=args.forecast_vl*valid_batch.shape[1])
                
                if args.model == 'node' or args.model == 'hbnode':
                    plot_results(plotter, \
                                Xrec_tr, tr_minibatch, Xrec_vl, valid_batch, \
                                {"plot":{'Loss(-elbo)': loss_meter, 'Nll' : nll_meter, 'KL-z0': kl_z0_meter, "train-MSE": tr_mse_meter}, "valid-MSE-rec": vl_mse_rec, "valid-MSE-for": vl_mse_for, "iteration": ep, "time": time_meter}, \
                                ztL_tr,  ztL_vl,  C_tr, C_vl,)
                
                elif args.model == 'sonode':
                    plot_results(plotter, \
                                Xrec_tr, tr_minibatch, Xrec_vl, valid_batch,\
                                {"plot":{"Loss" : loss_meter, "valid-MSE-rec": vl_mse_rec, "valid-MSE-for":vl_mse_for}, "time" : time_meter, "iteration": ep})


    logger.info('Epoch:{:4d}/{:4d} | time: {} | train_elbo: {:8.2f} | train_mse: {:5.3f} | valid_mse_rec: {:5.3f}) | valid_mse_for: {:5.3f})  | best_valid_mse: {:5.3f})'.\
                format(ep, args.Nepoch, datetime.now()-start_time, loss_meter.val, tr_mse_meter.avg, vl_mse_rec.val, vl_mse_for.val, best_valid_loss))

    for key, val in dict_test_mses.items():
        logger.info('T={} test_mse {:5.3f}({:5.3f})'.format(key, np.mean(dict_test_mses[key]), np.std(dict_test_mses[key])))

    

    


