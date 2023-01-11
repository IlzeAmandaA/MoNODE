import torch
import torch.nn as nn

 
class INVODEVAE(nn.Module):
    def __init__(self, flow, vae, num_observations, order, steps, dt, inv_enc=None, aug=False, nobj=1) -> None:
        super().__init__()

        self.flow = flow #Dynamics 
        self.num_observations = num_observations
        self.vae = vae
        self.inv_enc = inv_enc
        self.dt = dt
        self.v_steps = steps
        self.order = order
        self.aug = aug
        self.Nobj = nobj

    @property
    def device(self):
        return self.flow.device
    
    @property
    def dtype(self):
        return list(self.parameters())[0].dtype

    @property
    def is_inv(self):
        return self.inv_enc is not None

    def build_decoding(self, ztL, dims, cT=None):
        """
        Given a mean of the latent space decode the input back into the original space.

        @param ztL: latent variable (L,N,T,q)
        @param inv_z: invaraiant latent variable (L,N,q)
        @param dims: dimensionality of the original variable 
        @param cT: invariant code (L,N,T,q)
        @return Xrec: reconstructed in original data space (L,N,T,nc,d,d)
        """
        if self.order == 1:
            stL = ztL
        elif self.order == 2:
            q = ztL.shape[-1]//2
            stL = ztL[:,:,:,:q] # L,N,T,q Only the position is decoded

        if cT is not None:
            stL = torch.cat([stL, cT], -1) #L,N,T,2q

        Xrec = self.vae.decoder(stL, dims) # L,N,T,...
        return Xrec
    
    def sample_trajectories(self, z0L, T, L=1):
        '''
        @param z0L - initial latent encoding of shape [L,N,nobj,q]
        '''
        ts  = self.dt * torch.arange(T,dtype=torch.float).to(z0L.device)
        # sample L trajectories
        ztL = torch.stack([self.flow(z0, ts) for z0 in z0L]) # [L,N,T,nobj,q]
        return ztL # L,N,T,Nobj,q)

    def sample_augmented_trajectories(self, z0L, cL, T, L=1):
        '''
        @param z0L - initial latent encoding of shape [L,N, Nobj,q]
        @param cL - invariant code  [L,N,Nobj,q]
        '''
        ts  = self.dt * torch.arange(T,dtype=torch.float).to(z0L.device)
        ztL = [self.flow(z0, ts, zc) for z0,zc in zip(z0L,cL)] # sample L trajectories
        return torch.stack(ztL) # L,N,T,nobj,2q

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
        z0 = z0.unsqueeze(0) if z0.ndim==2 else z0 # L,N,q

        #if multiple object separate latent vector (but shared dynamics)
        q  = z0.shape[-1]
        z0 = z0.reshape(L,N,self.Nobj,q//self.Nobj) # L,N,nobj,q_

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
            cT = torch.stack([c]*T,2) # [L,N,T,q]
        else:
            C,cT = None, None

        #sample trajectories
        if self.aug:
            c = c.reshape((L,N,self.Nobj,-1)) #L,N,Nobj,q
            ztL  = self.sample_augmented_trajectories(z0, c, T, L) # L,N,T,Nobj, 2q
            ztL = ztL.reshape(L,N,T,-1) # L,T,N, nobj*2q
            Xrec = self.build_decoding(ztL, [L,N,T,-1]) 
        else:
            #sample ODE trajectories 
            ztL  = self.sample_trajectories(z0,T,L) # L,T,N,nobj,q
            ztL = ztL.reshape(L,N,T,-1) # L,T,N, nobj*q
            Xrec = self.build_decoding(ztL, [L,N,T,*X.shape[2:]], cT)

        return Xrec, ztL, (s0_mu, s0_logv), (v0_mu, v0_logv), C
