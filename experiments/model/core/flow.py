import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint as odeint_nonadjoint


class ODEfunc(nn.Module):
    def __init__(self, diffeq, order):
        """
        Defines the ODE function:
            mainly calls layer.build_cache() method to fix the draws from random variables.
        Modified from https://github.com/rtqichen/ffjord/

        @param diffeq: Layer of GPODE/npODE/neuralODE
        @param order: Which order ODE to use (1,2)
        """
        super(ODEfunc, self).__init__()
        self.diffeq = diffeq
        self.order = order
        self.register_buffer("_num_evals", torch.tensor(0.))

    def before_odeint(self, rebuild_cache):
        self._num_evals.fill_(0)
        if rebuild_cache:
            self.diffeq.build_cache()

    def num_evals(self):
        return self._num_evals.item()

    def first_order(self, sv):       
        dsv = self.diffeq(sv) # 25, 2q
        return dsv

    def second_order(self, sv):
        q = sv.shape[1]//2
        ds = sv[:,q:]  # N,q
        dv = self.diffeq(sv) # N,q        
        return torch.cat([ds,dv],1) # N,2q  

    def forward(self, t, sv): 
        self._num_evals += 1
        if self.order == 1:
            return self.first_order(sv)
        elif self.order == 2:
            return self.second_order(sv)


class Flow(nn.Module):
    def __init__(self, diffeq, order = 2, solver='dopri5', atol=1e-6, rtol=1e-6, use_adjoint=False):
        """
        Defines an ODE flow:
            mainly defines forward() method for forward numerical integration of an ODEfunc object
        See https://github.com/rtqichen/torchdiffeq for more information on numerical ODE solvers.

        @param diffeq: Layer of GPODE/npODE/neuralODE
        @param solver: Solver to be used for ODE numerical integration
        @param atol: Absolute tolerance for the solver
        @param rtol: Relative tolerance for the solver
        @param use_adjoint: Use adjoint method for computing loss gradients, calls odeint_adjoint from torchdiffeq
        """
        super(Flow, self).__init__()
        self.odefunc = ODEfunc(diffeq, order)
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.use_adjoint = use_adjoint

    def forward(self, z0, ts):
        """
        Numerical solution of an IVP
        @param z0: Initial latent state (N,2q)
        @param logp0: Initial distirbution (N)
        @param ts: Time sequence of length T, first value is considered as t_0
        @return: zt, logp: (N,T,2q) tensor, (N,T) tensor
        """
        odeint = odeint_adjoint if self.use_adjoint else odeint_nonadjoint
        if self.odefunc.diffeq.type == 'SVGP' :
            self.odefunc.before_odeint(rebuild_cache=True)
            odef = self.odefunc

        elif self.odefunc.diffeq.type == 'SGP':
            self.odefunc.before_odeint(rebuild_cache=True)
            self.odefunc.diffeq.fix_gpytorch_cache(0)
            self.odefunc.diffeq.draw_posterior_function()
            
        zt = odeint(
            self.odefunc,
            z0,
            ts,
            atol=self.atol,
            rtol=self.rtol,
            method=self.solver
        )
        return zt.permute([1,0,2])


    def num_evals(self):
        return self.odefunc.num_evals()

    def kl(self):
        """
        Calls KL() computation from the diffeq layer
        """
        return self.odefunc.diffeq.kl() #.sum()

    def log_prior(self):
        """
        Calls log_prior() computation from the diffeq layer
        """
        return self.odefunc.diffeq.log_prior().sum()

