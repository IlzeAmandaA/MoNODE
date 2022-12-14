import torch
import torch.nn as nn

 
class INVODEVAE(nn.Module):
    def __init__(self, flow, vae, num_observations, order, steps, dt, inv_gp=None) -> None:
        super().__init__()

        self.flow = flow #Dynamics 
        self.num_observations = num_observations
        self.vae = vae
        self.inv_gp = inv_gp
        self.dt = dt
        self.v_steps = steps
        self.order = order

    @property
    def is_inv(self):
        return self.inv_gp is not None

    def build_decoding(self, ztL, dims, inv_z=None):
        """
        Given a mean of the latent space decode the input back into the original space.

        @param ztL: latent variable (L,N,T,q)
        @param inv_z: invaraiant latent variable (L,N,q)
        @param dims: dimensionality of the original variable 
        @return Xrec: reconstructed in original data space (L,N,T,nc,d,d)
        """
        if self.order == 1:
            st_muL = ztL
        elif self.order == 2:
            q = ztL.shape[-1]//2
            st_muL = ztL[:,:,:,:q] # L,N,T,q Only the position is decoded

        if inv_z is not None:
            inv_z_L = torch.stack([inv_z]*ztL.shape[2],-2) # L,N,T,q
            st = torch.cat([st_muL, inv_z_L], -1) #L,N,T,2q
        else:
            st = st_muL
        Xrec = self.vae.decoder(st, dims) # L,N,T,...
        return Xrec
    
    def sample_trajectories(self, z0, T, L=1):
        ztL = []
        ts  = self.dt * torch.arange(T,dtype=torch.float).to(z0.device)
        #sample L trajectories
        for l in range(L):
            zt = self.flow(z0, ts) # N,T,2q 
            ztL.append(zt.unsqueeze(0)) # 1,N,T,2q
        ztL   = torch.cat(ztL,0) # L,N,T,2q
        return ztL

    def forward(self, X, L=1, T_custom=None):
        if self.is_inv:
            self.inv_gp.build_cache()

        [N,T,nc,d,d] = X.shape
        T_orig = T
        if T_custom:
            T = T_custom

        #encode dynamics
        s0_mu, s0_logv = self.vae.encoder(X[:,0]) # N,q
        z0 = self.vae.encoder.sample(s0_mu, s0_logv)
        v0_mu, v0_logv = None, None
        if self.order == 2:
            v0_mu, v0_logv = self.vae.encoder_v(torch.squeeze(X[:,0:self.v_steps]))
            v0 = self.vae.encoder_v.sample(v0_mu, v0_logv)
            z0 = torch.concat([z0,v0],dim=1) #N, 2q

        #encode content (invariance)
        if self.is_inv:
            qz_st = self.vae.inv_encoder(X.reshape(N*T_orig, nc,d,d)) # NT,q
            mean,var = self.inv_gp.build_conditional(qz_st)
            dist = torch.distributions.Normal(mean,var)
            inv_z_st = dist.rsample(torch.Size([L])).reshape(L,N,T_orig,-1).mean(-2) # L,N,q
        else:
            inv_z_st = None


        #sample ODE trajectories 
        ztL = self.sample_trajectories(z0,T,L) # L,N,T,2q

        #decode
        N = X.shape[0]
        out_dims = [L,N,T,*X.shape[2:]]
        Xrec = self.build_decoding(ztL, out_dims, inv_z_st)

        return Xrec, ztL, (s0_mu, s0_logv), (v0_mu, v0_logv)
