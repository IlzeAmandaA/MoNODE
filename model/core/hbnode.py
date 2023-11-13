from einops import rearrange
from torchdiffeq import odeint
from model.core.mlp import MLP

import torch
import torch.nn as nn

"""
Code based from:
Heavy Ball Neural Ordinary Differential Equations
https://github.com/hedixia/HeavyBallNODE
Author: Hedi Xia
"""

class tempf(nn.Module):
    def __init__(self, in_channels, out_channels=100):
        super().__init__()
        self.actv   = nn.modules.activation.Softplus() #nn.Tanh()
        self.dense1 = nn.Linear(in_channels, out_channels)
        self.dense2 = nn.Linear(out_channels, out_channels)
        self.dense3 = nn.Linear(out_channels, in_channels)

    def forward(self, h, x):
        out = self.dense1(x)
        out = self.actv(out)
        out = self.dense2(out)
        out = self.actv(out)
        out = self.dense3(out)
        return out

 
class temprnn(nn.Module):
    def __init__(self, in_channels, out_channels, nhidden, res=False, cont=False):
        super().__init__()
        self.actv = nn.Tanh()
        self.dense1 = nn.Linear(in_channels + 2 * nhidden, 2 * nhidden)
        self.dense2 = nn.Linear(2 * nhidden, 2 * nhidden)
        self.dense3 = nn.Linear(2 * nhidden, 2 * out_channels)
        self.cont = cont
        self.res  = res

    def forward(self, h, x):
        out = torch.cat([h[:, 0], h[:, 1], x], dim=1) #100, 9 
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
        self.diffeq = df
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
                    x_i = outnet(h_ode[i])[:,0] 
                    h_rnn[i] = self.rnn(h_ode[i], x_i)
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

class HBNODE_BASE(NODE):
    def __init__(self, df, actv_h=None, gamma_guess=-3.0, gamma_act='sigmoid', corr=-100, corrf=True, sign=1):
        super().__init__(df)
        # Momentum parameter gamma 
        self.gamma = Parameter([gamma_guess], frozen=False)
        self.gammaact = nn.Sigmoid() if gamma_act == 'sigmoid' else gamma_act
        self.corr = Parameter([corr], frozen=corrf)
        self.sp = nn.Softplus()
        self.sign = sign # Sign of df
        self.actv_h = nn.Identity() if actv_h is None else actv_h # Activation for dh, GHBNODE only

    @property
    def device(self):
        return list(self.diffeq.parameters())[0].device

    @property
    def model(self):
        return 'hbnode'
    
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
        hc = self.concat_zc(h)
        dh = self.actv_h(- m)
        dm = self.diffeq(hc) * self.sign - self.gammaact(self.gamma()) * m
        dm = dm + self.sp(self.corr()) * h
        out = torch.cat((dh, dm), dim=1)
        if self.elem_t is None:
            return out
        else:
            return self.elem_t * out

    def update(self, elem_t):
        self.elem_t = elem_t.view(*elem_t.shape, 1, 1)

    def augment(self, zc=None):
        self.zc = zc

    def concat_zc(self, s):
        if self.zc is not None:
            return torch.cat([s,self.zc],-1) 
        else:
            return s


class HBNODE(nn.Module):
    def __init__(self, data_dim, res=False, nhid=4, cont=False, tol=1e-7, H_dec=128, L_dec=0, dec_act='relu'):
        super(HBNODE, self).__init__()
        self.cell = HBNODE_BASE(tempf(nhid), corr=0, corrf=True)
        self.rnn = temprnn(data_dim, nhid, nhid, res=res, cont=cont)
        self.ode_rnn = ODE_RNN_with_Grad_Listener(self.cell, self.rnn, (2, nhid), None, tol=tol)
        self.outlayer = MLP(nhid, data_dim, L=L_dec, H=H_dec, act=dec_act)
    
    @property
    def device(self):
        return list(self.outlayer.parameters())[0].device

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
        out = self.ode_rnn(ts, X, outnet=self.outlayer, retain_grad=True)[0] # T+1, N, 2, nhid
        out = self.outlayer(out[:, :, 0])[1:]
        self.cell.nfe = 0
        Xrec = out.transpose(1,0) # N,T,D
        return Xrec

