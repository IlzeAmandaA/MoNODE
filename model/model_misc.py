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

    # KL inudcing 
    if model.flow.odefunc.diffeq.type == 'SVGP':
        kl_gp = model.flow.kl()
    else:
        kl_gp = 0.0 * torch.zeros(1).to(X.device)

    if model.inv_enc is not None:
        kl_gp_2 = model.inv_enc.kl().to(X.device) + kl_gp
    else:
        kl_gp_2 = kl_gp

    return lhood.mean(), kl_z0.mean(), kl_gp_2 


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

def compute_mse(model, data, T_train, L=1):

    T_max = 0
    T = data.shape[1]
    #run model    
    Xrec, ztL, (s0_mu, s0_logv), (v0_mu, v0_logv), C = model(data, L, T)
    
    dict_mse = {}
    while T_max < T:
        T_max += T_train
        mse = torch.mean((Xrec[:,:,:T_max]-data[:,:T_max])**2)
        dict_mse[str(T_max)] = mse 
    return dict_mse 


def compute_loss(model, data, L, num_observations, contr_loss=False, sc_lambda=1.0):
    """
    Compute loss for optimization
    @param model: a odegpvae objectb 
    @param data: true observation sequence 
    @param L: number of MC samples
    @param contr_loss: whether to compute contrastive loss or not
    @return: loss, nll, regularizing_kl, inducing_kl
    """
    T = data.shape[1]

    #run model    
    Xrec, ztL, (s0_mu, s0_logv), (v0_mu, v0_logv), C = model(data, L, T_custom=T)

    #compute loss
    if model.model =='sonode':
        mse   = torch.mean((Xrec-data)**2)

        if contr_loss and model.is_inv:
            contr_learn_loss = contrastive_loss(C)
        else:
            contr_learn_loss = torch.zeros_like(mse)

        loss = mse + sc_lambda * contr_learn_loss
        return loss, 0.0, 0.0, 0.0, 0.0, 0.0, mse, contr_learn_loss
    
    elif model.model =='node':
        lhood, kl_z0, kl_gp = elbo(model, data, Xrec, s0_mu, s0_logv, v0_mu, v0_logv,L)
        if contr_loss and model.is_inv:
            contr_learn_loss = contrastive_loss(C)
        else:
            contr_learn_loss = torch.zeros_like(lhood)
            
        lhood = lhood * num_observations
        kl_z0 = kl_z0 * num_observations
        loss  = - lhood + kl_z0 + kl_gp + sc_lambda*contr_learn_loss
        mse   = torch.mean((Xrec-data)**2)
        return loss, -lhood, kl_z0, kl_gp, Xrec, ztL, mse, contr_learn_loss


def freeze_pars(par_list):
    for par in par_list:
        try:
            par.requires_grad = False
        except:
            print('something wrong!')
            raise ValueError('This is not a parameter!?')


def train_model(args, invodevae, plotter, trainset, validset, testset, logger, params, freeze_dyn=False):

    loss_meter  = log_utils.CachedRunningAverageMeter(0.97)
    vl_loss_meter  = log_utils.CachedRunningAverageMeter(0.97)
    time_meter = log_utils.CachedRunningAverageMeter(0.97)
    tr_mse_meter   = log_utils.CachedRunningAverageMeter(0.97)
    contr_meter = log_utils.CachedRunningAverageMeter(0.97)
    vl_mse_meter = log_utils.CachedRunningAverageMeter(0.97)

    if args.model == 'node':
        nll_meter   = log_utils.CachedRunningAverageMeter(0.97)
        kl_z0_meter = log_utils.CachedRunningAverageMeter(0.97)
        
        
        

    logger.info('********** Started Training **********')
    if freeze_dyn:
        freeze_pars(invodevae.flow.parameters())

    ############## build the optimizer ############
    gp_pars  = [par for name,par in invodevae.named_parameters() if 'SVGP' in name]
    rem_pars = [par for name,par in invodevae.named_parameters() if 'SVGP' not in name]
    assert len(gp_pars)+len(rem_pars) == len(list(invodevae.parameters()))
    optimizer = torch.optim.Adam([
                    {'params': rem_pars, 'lr': args.lr},
                    {'params': gp_pars, 'lr': args.lr*10}
                    ],lr=args.lr)


    ########## Training loop ###########
    start_time=datetime.now()
    global_itr = 0
    best_valid_loss = None
    test_elbo, test_mse, test_std = 0.0, 0.0, 0.0
    #increase the data set length in N increments sequentally 
    T_train = params['train']['T']
    ep_inc_c = args.Nepoch // args.Nincr
    ep_inc_v = T_train // args.Nincr
    T_ = ep_inc_v

    for ep in range(args.Nepoch):
        if args.model == 'sonode':
            L=1
        else:
            L = 1 if ep<args.Nepoch//2 else 5 

        if (ep != 0) and (ep % ep_inc_c == 0):
            T_ += ep_inc_v

        for itr,local_batch in enumerate(trainset):
            tr_minibatch = local_batch.to(invodevae.device) # N,T,...
            if args.task=='sin' or args.task=='spiral' or args.task=='lv': #slowly increase sequence length
                [N,T] = tr_minibatch.shape[:2]

                N_  = int(N*(T//T_))
                if T_ < T:
                    t0s = torch.randint(0,T-T_,[N_])  #select a random initial point from the sequence
                else:
                    t0s = torch.zeros([N_]).to(int)
                tr_minibatch = tr_minibatch.repeat([N_,1,1])
                tr_minibatch = torch.stack([tr_minibatch[n,t0:t0+T_] for n,t0 in enumerate(t0s)]) # N*ns,T//2,d
                
            loss, nlhood, kl_z0, kl_u, Xrec_tr, ztL_tr, tr_mse, contr_learn_cost = \
                compute_loss(invodevae, tr_minibatch, L, num_observations = params['train']['N'], contr_loss=args.contr_loss, sc_lambda=args.lambda_contr)

            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()

            #store values 
            loss_meter.update(loss.item(), global_itr)
            tr_mse_meter.update(tr_mse.item(), global_itr)
            contr_meter.update(contr_learn_cost.item(), global_itr)
            if args.model == 'node':
                nll_meter.update(nlhood.item(), global_itr)
                kl_z0_meter.update(kl_z0.item(), global_itr)
            global_itr +=1

            val = datetime.now()-start_time
            
        with torch.no_grad():
            

            valid_losses,valid_mses = [],[]
            for itr_test,valid_batch in enumerate(validset):

                valid_batch = valid_batch.to(invodevae.device)
                loss, _, _, _, _, _, valid_mse, _ = compute_loss(invodevae, valid_batch, L=1, num_observations = params['valid']['N'], contr_loss=args.contr_loss, sc_lambda=args.lambda_contr) 
                valid_losses.append(loss.item())
                valid_mses.append(valid_mse.item())
            valid_loss, valid_mse, valid_std = np.mean(np.array(valid_losses)),np.mean(np.array(valid_mses)),np.std(np.array(valid_mses))


            logger.info('Epoch:{:4d}/{:4d} | tr_loss:{:8.2f}({:8.2f}) | valid_loss {:5.3f} | valid_mse:{:5.3f} | contr_loss:{:5.3f}({:5.3f})'.\
                    format(ep, args.Nepoch, loss_meter.val, loss_meter.avg, valid_loss, valid_mse, contr_meter.val, contr_meter.avg)) 

                
            # update valid loggers
            vl_loss_meter.update(valid_loss,ep)
            time_meter.update(val.seconds, ep)
            vl_mse_meter.update(valid_mse, ep, valid_std) 

                                
            #compare validation error seen so far
            if best_valid_loss is None:
                best_valid_loss = valid_mse

            elif best_valid_loss > valid_mse: #we want as smaller mse
                best_valid_loss = valid_mse

                torch.save({
                    'args': args,
                    'state_dict': invodevae.state_dict(),
                }, os.path.join(args.save, 'invodevae.pth'))
                            
                #compute test error for this model 
                dict_mses = {}
                test_mse = {}
                for itr_test,test_batch in enumerate(testset):
                    test_batch = test_batch.to(invodevae.device)
                    dict_mse = compute_mse(invodevae, test_batch, T_train)
                    for key,val in dict_mse.items():
                        if key not in dict_mses:
                            dict_mses[key] = []
                        dict_mses[key].append(val.item())
                for key in dict_mses:
                    test_mse[key] = (np.mean(dict_mses[key]), np.std(dict_mses[key]))

                logger.info('********** Current Best Model based on validation error ***********')
                logger.info('Epoch:{:4d}/{:4d}'.format(ep, args.Nepoch))
                for key, val in test_mse.items():
                    logger.info('T={} test_mse {:5.3f}({:5.3f})'.format(key, val[0], val[1]))

            if ep % args.plot_every==0:
                Xrec_tr, ztL_tr, _, _, C_tr = invodevae(tr_minibatch, L=args.plotL, T_custom=args.forecast_tr*tr_minibatch.shape[1])
                Xrec_vl, ztL_vl, _, _, C_vl = invodevae(valid_batch,  L=args.plotL, T_custom=args.forecast_vl*valid_batch.shape[1])
                
                if args.model == 'node':
                    plot_results(plotter, \
                                Xrec_tr, tr_minibatch, Xrec_vl, valid_batch, \
                                {"plot":{'Loss(-elbo)': loss_meter, 'Nll' : nll_meter, 'KL-z0': kl_z0_meter, "train-MSE": tr_mse_meter}, "valid-MSE": vl_mse_meter, "valid(-elbo)": vl_loss_meter, "iteration": ep, "time": time_meter}, \
                                ztL_tr,  ztL_vl,  C_tr, C_vl,)
                
                elif args.model == 'sonode':
                    plot_results(plotter, \
                                Xrec_tr, tr_minibatch, Xrec_vl, valid_batch,\
                                {"plot":{"Loss" : loss_meter, "validation-mse": vl_loss_meter}, "time" : time_meter, "iteration": ep})


    if args.model == 'node':
        logger.info('Epoch:{:4d}/{:4d} | time: {} | train_elbo: {:8.2f} | valid_elbo: {:8.2f}| valid_mse: {:5.3f})'.\
                    format(ep, args.Nepoch, datetime.now()-start_time, loss_meter.val, vl_loss_meter.val, vl_mse_meter.val))
    elif args.model == 'sonode':
        logger.info('Epoch:{:4d}/{:4d} | time: {} | train_loss: {:8.2f} | train_mse  {:5.3f}  | valid_mse: {:5.3f}'.\
                    format(ep, args.Nepoch, datetime.now()-start_time, loss_meter.val, tr_mse_meter.avg, vl_mse_meter.val))
    
    for key, val in test_mse.items():
        logger.info('T={} test_mse {:5.3f}({:5.3f})'.format(key, val[0], val[1]))

    torch.save({
		'args': args,
		'state_dict': invodevae.state_dict(),
	}, os.path.join(args.save, 'invodevae_'+str(ep+1)+'_.pth'))
    

    


