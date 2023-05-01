from einops import rearrange
from torchdiffeq import odeint

from data.data_utils import __load_data
from data.data_utils import load_data
from model.misc import log_utils 
from model.misc import io_utils
from model.misc.torch_utils import seed_everything
from model.misc.plot_utils import plot_results

import torch
import os, numpy as np
import argparse
import torch.nn as nn
import time, datetime

dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TASKS  = ['rot_mnist', 'rot_mnist_ou',  'mov_mnist', 'sin', 'bb', 'lv']
parser = argparse.ArgumentParser('HBNODE')
parser.add_argument('--data_root', type=str, default='data/',
                    help="general data location")
parser.add_argument('--task', type=str, default='sin', choices=TASKS,
                    help="Experiment type")               
parser.add_argument('--num_workers', type=int, default=0,
                    help="number of workers")
parser.add_argument('--shuffle', type=eval, default=True,
               help='For Moving MNIST whetehr to shuffle the data')
parser.add_argument('--batch_size', type=int, default=25,
                    help="batch size")
parser.add_argument('--nhid', type=int, default=4,
                    help="hidden rnn dim")

class tempf(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.actv   = nn.Tanh()
        self.dense1 = nn.Linear(in_channels, in_channels)
        # self.dense2 = nn.Linear(in_channels, out_channels)
        # self.dense3 = nn.Linear(out_channels, out_channels)

    def forward(self, h, x):
        out = self.dense1(x)
        # out = self.actv(out)
        # out = self.dense2(out)
        # out = self.actv(out)
        # out = self.dense3(out)
        return out


class temprnn(nn.Module):
    def __init__(self, in_channels, out_channels, nhidden, res=False, cont=False):
        super().__init__()
        #in_channels 17 
        self.actv = nn.Tanh()
        self.dense1 = nn.Linear(in_channels + 2 * nhidden, 2 * nhidden)
        self.dense2 = nn.Linear(2 * nhidden, 2 * nhidden)
        self.dense3 = nn.Linear(2 * nhidden, 2 * out_channels)
        self.cont = cont
        self.res  = res

    def forward(self, h, x):
        # print(h[:, 0].shape) # 10, 24
        # print(h[:, 1].shape) # 10, 24
        # print(x.shape) # 10, 1
        out = torch.cat([h[:, 0], h[:, 1], x], dim=1) #10, 49 
        # print('out', out.shape)
        out = self.dense1(out)
        out = self.actv(out)
        out = self.dense2(out)
        out = self.actv(out)
        out = self.dense3(out).reshape(h.shape)
        out = out + h
        return out

class Parameter(nn.Module):
    def __init__(self, val, frozen=False):
        super().__init__()
        val = torch.Tensor(val)
        self.val = val
        self.param = nn.Parameter(val)
        self.frozen = frozen

    def forward(self):
        if self.frozen:
            self.val = self.val.to(self.param.device)
            return self.val
        else:
            return self.param

    def freeze(self):
        self.val = self.param.detach().clone()
        self.frozen = True

    def unfreeze(self):
        self.frozen = False

    def __repr__(self):
        return "val: {}, param: {}".format(self.val.cpu(), self.param.detach().cpu())
     
    
class NODE(nn.Module):
    def __init__(self, df=None, **kwargs):
        super(NODE, self).__init__()
        self.__dict__.update(kwargs)
        self.df = df
        self.nfe = 0
        self.elem_t = None

    def forward(self, t, x):
        self.nfe += 1
        if self.elem_t is None:
            return self.df(t, x)
        else:
            return self.elem_t * self.df(self.elem_t, x)

    def update(self, elem_t):
        self.elem_t = elem_t.view(*elem_t.shape, 1)


class ODE_RNN_with_Grad_Listener(nn.Module):
    def __init__(self, ode, rnn, nhid, ic, rnn_out=False, both=False, tol=1e-7):
        super().__init__()
        self.ode = ode
        self.t = torch.Tensor([0, 1])
        self.nhid = [nhid] if isinstance(nhid, int) else nhid
        self.rnn = rnn
        self.tol = tol
        self.rnn_out = rnn_out
        self.ic = ic
        self.both = both

    def forward(self, t, x, outnet=None, multiforecast=None, retain_grad=False):
        """
        --
        :param t: [time, batch]
        :param x: [time, batch, ...]
        :return: [time, batch, *nhid]
        """
        assert t.shape[0]>=x.shape[0] or outnet is not None, 'we need an output network if input sequence is shorter than pred horizon'
        n_t, n_b = t.shape
        h_ode = [None] * (n_t + 1)
        h_rnn = [None] * (n_t + 1)
        h_ode[-1] = h_rnn[-1] = torch.zeros(n_b, *self.nhid)

        if self.ic:
            h_ode[0] = h_rnn[0] = self.ic(rearrange(x, 't b c -> b (t c)')).view((n_b, *self.nhid))
        else:
            h_ode[0] = h_rnn[0] = torch.zeros(n_b, *self.nhid, device=x.device)
        if self.rnn_out:
            for i in range(n_t):
                self.ode.update(t[i])
                h_ode[i] = odeint(self.ode, h_rnn[i], self.t, atol=self.tol, rtol=self.tol)[-1]
                try:
                    h_rnn[i + 1] = self.rnn(h_ode[i], x[i])
                except:
                    h_rnn[i + 1] = self.rnn(h_ode[i], outnet(h_rnn[i]))
            out = (h_rnn,)
        else:
            for i in range(n_t):
                self.ode.update(t[i])
                try:
                    h_rnn[i] = self.rnn(h_ode[i], x[i])
                except:
                    h_rnn[i] = self.rnn(h_ode[i], outnet(h_ode[i]))
                h_ode[i + 1] = odeint(self.ode, h_rnn[i], self.t, atol=self.tol, rtol=self.tol)[-1]
            out = (h_ode,)

        if self.both:
            out = (h_rnn, h_ode)

        out = [torch.stack(h, dim=0) for h in out]

        if multiforecast is not None:
            self.ode.update(torch.ones_like((t[0])))
            forecast = odeint(self.ode, out[-1][-1], multiforecast * 1.0, atol=self.tol, rtol=self.tol)
            out = (*out, forecast)

        if retain_grad:
            self.h_ode = h_ode
            self.h_rnn = h_rnn
            for i in range(n_t + 1):
                if self.h_ode[i].requires_grad:
                    self.h_ode[i].retain_grad()
                if self.h_rnn[i].requires_grad:
                    self.h_rnn[i].retain_grad()

        return out
    
    # old and simple runner
    # def forward(self, t, x, multiforecast=None, retain_grad=False):
    #     """
    #     --
    #     :param t: [time, batch]
    #     :param x: [time, batch, ...]
    #     :return: [time, batch, *nhid]
    #     """
    #     n_t, n_b = t.shape
    #     h_ode = [None] * (n_t + 1)
    #     h_rnn = [None] * (n_t + 1)
    #     h_ode[-1] = h_rnn[-1] = torch.zeros(n_b, *self.nhid)

    #     if self.ic:
    #         h_ode[0] = h_rnn[0] = self.ic(rearrange(x, 't b c -> b (t c)')).view((n_b, *self.nhid))
    #     else:
    #         h_ode[0] = h_rnn[0] = torch.zeros(n_b, *self.nhid, device=x.device)
    #     if self.rnn_out:
    #         for i in range(n_t):
    #             self.ode.update(t[i])
    #             h_ode[i] = odeint(self.ode, h_rnn[i], self.t, atol=self.tol, rtol=self.tol)[-1]
    #             h_rnn[i + 1] = self.rnn(h_ode[i], x[i])
    #         out = (h_rnn,)
    #     else:
    #         for i in range(n_t):
    #             self.ode.update(t[i])
    #             h_rnn[i] = self.rnn(h_ode[i], x[i])
    #             h_ode[i + 1] = odeint(self.ode, h_rnn[i], self.t, atol=self.tol, rtol=self.tol)[-1]
    #         out = (h_ode,)

    #     if self.both:
    #         out = (h_rnn, h_ode)

    #     out = [torch.stack(h, dim=0) for h in out]

    #     if multiforecast is not None:
    #         self.ode.update(torch.ones_like((t[0])))
    #         forecast = odeint(self.ode, out[-1][-1], multiforecast * 1.0, atol=self.tol, rtol=self.tol)
    #         out = (*out, forecast)

    #     if retain_grad:
    #         self.h_ode = h_ode
    #         self.h_rnn = h_rnn
    #         for i in range(n_t + 1):
    #             if self.h_ode[i].requires_grad:
    #                 self.h_ode[i].retain_grad()
    #             if self.h_rnn[i].requires_grad:
    #                 self.h_rnn[i].retain_grad()

    #     return out

class HBNODE(NODE):
    def __init__(self, df, actv_h=None, gamma_guess=-3.0, gamma_act='sigmoid', corr=-100, corrf=True, sign=1):
        super().__init__(df)
        # Momentum parameter gamma
        self.gamma = Parameter([gamma_guess], frozen=False)
        self.gammaact = nn.Sigmoid() if gamma_act == 'sigmoid' else gamma_act
        self.corr = Parameter([corr], frozen=corrf)
        self.sp = nn.Softplus()
        self.sign = sign # Sign of df
        self.actv_h = nn.Identity() if actv_h is None else actv_h # Activation for dh, GHBNODE only

    def forward(self, t, x):
        """
        Compute [theta' m' v'] with heavy ball parametrization in
        $$ h' = -m $$
        $$ m' = sign * df - gamma * m $$
        based on paper https://www.jmlr.org/papers/volume21/18-808/18-808.pdf
        :param t: time, shape [1]
        :param x: [theta m], shape [batch, 2, dim]
        :return: [theta' m'], shape [batch, 2, dim]
        """
        self.nfe += 1
        h, m = torch.split(x, 1, dim=1)
        dh = self.actv_h(- m)
        dm = self.df(t, h) * self.sign - self.gammaact(self.gamma()) * m
        dm = dm + self.sp(self.corr()) * h
        out = torch.cat((dh, dm), dim=1)
        if self.elem_t is None:
            return out
        else:
            return self.elem_t * out

    def update(self, elem_t):
        self.elem_t = elem_t.view(*elem_t.shape, 1, 1)


class MODEL(nn.Module):
    def __init__(self, data_dim, res=False, nhid=4, cont=False):
        super(MODEL, self).__init__()
        self.cell = HBNODE(tempf(nhid, nhid), corr=0, corrf=True)
        self.rnn = temprnn(data_dim, nhid, nhid, res=res, cont=cont)
        # self.cell = HeavyBallNODE(tempf(nhid, nhid))
        # self.rnn = temprnn(nhid, nhid, nhid, res=res, cont=cont)
        self.ode_rnn = ODE_RNN_with_Grad_Listener(self.cell, self.rnn, (2, nhid), None, tol=1e-7)
        # self.outlayer = nn.Linear(nhid, nhid)
        self.outlayer = nn.Linear(nhid, data_dim)
    
    @property
    def model(self):
        return 'hbnode'
    
    @property
    def is_inv(self):
        return False

    def forward(self, X, L=1, T_custom=None):
        ''' 
            X - [N,T,d]
        '''
        N, T, D = X.shape
        self.cell.nfe = 0
        Tode = T if T_custom is None else T_custom 
        ts = (0.1 * torch.arange(Tode,dtype=torch.float).to(X.device)).repeat(N,1).transpose(1,0) # T,N
        X = X.transpose(1,0) # T,N,D
        out = self.ode_rnn(ts, X, retain_grad=True)[0] # T+1, N, 2, nhid
        out = self.outlayer(out[:, :, 0])[1:]
        self.cell.nfe = 0
        Xrec = out.transpose(1,0) # N,T,D
        return Xrec, None, (None, None), (None, None), None 
        # Xrec, ztL, (s0_mu, s0_logv), (v0_mu, v0_logv), C


def simple_main():
    args = parser.parse_args()
    
    (trainset, validset, testset),_ = __load_data(args, device, dtype, args.task)
    data_dim = trainset.dataset.Xtr.shape[-1]
    lr_dict = {0: 0.001, 50: 0.003}
    torch.manual_seed(0)
    model = MODEL(data_dim, res=True, nhid=args.nhid, cont=True).to(device) #.to(0)
    print(model.__str__())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_dict[0])
    val_mse = []

    for epoch in range(500):
        if epoch in lr_dict:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr_dict[epoch])

        #for b_n in range(0, data.train_x.shape[1], batchsize):
        for batch in trainset:
            batch = batch.to(device) # N,T,D
            Xrec = model(batch)[0] # N,T,D
            loss = (batch-Xrec).pow(2).mean([0,1]).sum()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        validation_vals= []
        for validbatch in validset:
            validbatch = validbatch.to(device)
            Xval = model(validbatch)[0] # N,T,D
            val_loss = (validbatch-Xval).pow(2).mean([0,1]).sum()
            validation_vals.append(val_loss.item())

        val_mse.append(np.mean(validation_vals))


def similar_to_our_main():
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
    
    ############################
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
    trainset, validset, testset, params = load_data(args, device, dtype)
    logger.info('********** {} dataset with loaded ********** '.format(args.task))
    logger.info('data params: {}'.format(params[args.task]))

    ########### model ###########
    # invodevae = build_model(args, device, dtype, params)
    invodevae = MODEL(trainset.dataset.Xtr.shape[-1], res=True, nhid=args.nhid, cont=True).to(device).to(dtype) 
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

    train_model(args, invodevae, plotter, trainset, validset, testset, logger, params[args.task])






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


def compute_loss(model, data, L, num_observations, contr_loss=False, T_valid=None, sc_lambda=1.0):
    """
    Compute loss for optimization
    @param model: a odegpvae objectb 
    @param data: true observation sequence 
    @param L: number of MC samples
    @param contr_loss: whether to compute contrastive loss or not
    @return: loss, nll, regularizing_kl, inducing_kl
    """
    T = data.shape[1]
    if T_valid != None:
        in_data = data[:,:T_valid]
        T= T_valid
    else:
        in_data = data 

    #run model    
    Xrec, ztL, (s0_mu, s0_logv), (v0_mu, v0_logv), C = model(in_data, L, T_custom=T)

    #compute loss
    if model.model =='sonode' or model.model =='hbnode':
        mse   = torch.mean((Xrec-in_data)**2)

        if contr_loss and model.is_inv:
            contr_learn_loss = contrastive_loss(C)
        else:
            contr_learn_loss = torch.zeros_like(mse)

        loss = mse + sc_lambda * contr_learn_loss
        return loss, 0.0, 0.0, 0.0, 0.0, 0.0, mse, contr_learn_loss
    
    elif model.model =='node':
        lhood, kl_z0, kl_gp = elbo(model, in_data, Xrec, s0_mu, s0_logv, v0_mu, v0_logv,L)
        if contr_loss and model.is_inv:
            contr_learn_loss = contrastive_loss(C)
        else:
            contr_learn_loss = torch.zeros_like(lhood)
            
        lhood = lhood * num_observations
        kl_z0 = kl_z0 * num_observations
        loss  = - lhood + kl_z0 + kl_gp + sc_lambda*contr_learn_loss
        mse   = torch.mean((Xrec-in_data)**2)
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
    T = params['train']['T']
    ep_inc_c = args.Nepoch // args.Nincr
    ep_inc_v = T // args.Nincr
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
                loss, _, _, _, _, _, valid_mse, _ = compute_loss(invodevae, valid_batch, L=1, num_observations = params['valid']['N'], contr_loss=args.contr_loss, sc_lambda=args.lambda_contr) #, T_valid=valid_batch.shape[1]//2)
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
                test_elbos,test_mses = [],[]

                for itr_test,test_batch in enumerate(testset):
                    test_batch = test_batch.to(invodevae.device)
                    test_elbo, _, _, _, _, _, test_mse, _ = compute_loss(invodevae, test_batch, L=1, num_observations=params['test']['N'], contr_loss=args.contr_loss, T_valid=valid_batch.shape[1], sc_lambda=args.lambda_contr)
                    test_elbos.append(test_elbo.item())
                    test_mses.append(test_mse.item())
                test_elbo, test_mse, test_std = np.mean(np.array(test_elbos)),np.mean(np.array(test_mses)), np.std(np.array(test_mses))
                logger.info('********** Current Best Model based on validation error ***********')
                logger.info('Epoch:{:4d}/{:4d} | test_elbo:{:8.2f} | test_mse {:5.3f}({:5.3f}) '.\
                format(ep, args.Nepoch, test_elbo, test_mse, test_std)) 

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
                                Xrec_tr.unsqueeze(0), tr_minibatch, Xrec_vl.unsqueeze(0), valid_batch,\
                                {"plot":{"Loss" : loss_meter, "validation-mse": vl_loss_meter}, "time" : time_meter, "iteration": ep})


    if args.model == 'node':
        logger.info('Epoch:{:4d}/{:4d} | time: {} | train_elbo: {:8.2f} | valid_elbo: {:8.2f}| valid_mse: {:5.3f} | test_elbo: {:8.2f} | test_mse: {:5.3f}({:5.3f}) '.\
                    format(ep, args.Nepoch, datetime.now()-start_time, loss_meter.val, vl_loss_meter.val, vl_mse_meter.val, test_elbo, test_mse, test_std)) 
    elif args.model == 'sonode':
        logger.info('Epoch:{:4d}/{:4d} | time: {} | train_loss: {:8.2f} | train_mse  {:5.3f}  | valid_mse: {:5.3f} | test_mse {:5.3f}({:5.3f})'.\
                    format(ep, args.Nepoch, datetime.now()-start_time, loss_meter.val, tr_mse_meter.avg, vl_mse_meter.val, test_mse, test_std))
        

    torch.save({
		'args': args,
		'state_dict': invodevae.state_dict(),
	}, os.path.join(args.save, 'invodevae_'+str(ep+1)+'_.pth'))
    

if __name__ == '__main__':
    simple_main()
    # similar_to_our_main()
