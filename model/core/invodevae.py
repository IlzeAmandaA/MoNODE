import torch
import torch.nn as nn

 
class INVODEVAE(nn.Module):
    def __init__(self, flow, vae, num_observations, order, steps, dt, inv_enc=None, aug=False) -> None:
        super().__init__()

        self.flow = flow #Dynamics 
        self.num_observations = num_observations
        self.vae = vae
        self.inv_enc = inv_enc
        self.dt = dt
        self.v_steps = steps
        self.order = order
        self.aug = aug

    @property
    def device(self):
        return self.flow.device

    @property
    def is_inv(self):
        return self.inv_enc is not None

    def build_decoding(self, ztL, dims, c=None):
        """
        Given a mean of the latent space decode the input back into the original space.

        @param ztL: latent variable (L,N,T,q)
        @param inv_z: invaraiant latent variable (L,N,q)
        @param dims: dimensionality of the original variable 
        @param c: invariant code (L,N,q)
        @return Xrec: reconstructed in original data space (L,N,T,nc,d,d)
        """
        assert c is None or c.ndim==3, 'wrong input dimensionality!'
        if self.order == 1:
            stL = ztL
        elif self.order == 2:
            q = ztL.shape[-1]//2
            stL = ztL[:,:,:,:q] # L,N,T,q Only the position is decoded

        if c is not None:
            cL = torch.stack([c]*ztL.shape[2],-2) # L,N,T,q
            stL = torch.cat([stL, cL], -1) #L,N,T,2q

        Xrec = self.vae.decoder(stL, dims) # L,N,T,...
        return Xrec
    
    def sample_trajectories(self, z0, T, L=1):
        ts  = self.dt * torch.arange(T,dtype=torch.float).to(z0.device)
        if z0.ndim==2:
            z0L = torch.stack([z0]*L)
        else:
            z0L = z0
        ztL = [self.flow(z0, ts) for z0 in z0L] # sample L trajectories
        return torch.stack(ztL) # L,N,T,2q

    def sample_augmented_trajectories(self, z0, zc, T, L=1):
        '''
            z0 - initial values [N,q]  or [L,N,q]
            zc - invariant code [N,q2] or [L,N,q2]
        '''
        ts  = self.dt * torch.arange(T,dtype=torch.float).to(z0.device)
        z0L = torch.stack([z0]*L) if z0.ndim==2 else z0
        zcL = torch.stack([zc]*L) if zc.ndim==2 else zc
        ztL = [self.flow(z0, ts, zc) for z0,zc in zip(z0L,zcL)] # sample L trajectories
        return torch.stack(ztL) # L,N,T,2q

    def forward(self, X, L=1, T_custom=None):
        try:
            self.inv_enc.last_layer_gp.build_cache()
        except:
            pass

        [N,T] = X.shape[:2]
        if T_custom:
            T = T_custom

        #encode dynamics
        s0_mu, s0_logv = self.vae.encoder(X) # N,q
        z0 = self.vae.encoder.sample(s0_mu, s0_logv, L=L) # N,q or L,N,q
        v0_mu, v0_logv = None, None
        if self.order == 2:
            assert not self.aug, 'sorry, second order systems + augmented dynamics not implemented yet'
            v0_mu, v0_logv = self.vae.encoder_v(X)
            v0 = self.vae.encoder_v.sample(v0_mu, v0_logv, L=L) # N,q or L,N,q
            z0 = torch.concat([z0,v0],dim=1) #N, 2q

        #encode content (invariance)
        if self.is_inv:
            C = self.inv_enc(X, L=L) # embeddings [L,N,T,q]
            c = C.mean(2) # time-invariant code [L,N,q]
        else:
            C,c = None, None

        if self.aug:
            ztL  = self.sample_augmented_trajectories(z0, c, T, L) # L,N,T,2q
            Xrec = self.build_decoding(ztL, [L,N,T,-1])
        else:
            #sample ODE trajectories 
            ztL  = self.sample_trajectories(z0,T,L) # L,N,T,2q
            Xrec = self.build_decoding(ztL, [L,N,T,*X.shape[2:]], c)

        return Xrec, ztL, (s0_mu, s0_logv), (v0_mu, v0_logv), C
