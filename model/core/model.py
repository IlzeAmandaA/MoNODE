import torch
import torch.nn as nn

 
class MoNODE(nn.Module):
    def __init__(self, model, flow, vae, order, dt, inv_enc=None, aug=False, nobj=1, Tin=5) -> None:
        super().__init__()

        self.flow = flow 
        self.vae = vae
        self.inv_enc = inv_enc
        self.dt = dt
        self.order = order
        self.aug = aug
        self.Nobj = nobj
        self.model = model
        self.Tin = Tin

    @property
    def device(self):
        return self.flow.device
    
    @property
    def dtype(self):
        return list(self.parameters())[0].dtype

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

        if self.model =='sonode':
            Xrec =  ztL[:,:,:,:ztL.shape[-1]//2] #Only position is used for reconstructions, N, T, 1
            return Xrec 
        
        elif self.model == 'node' or self.model =='hbnode':
            if self.order == 1:
                stL = ztL
            elif self.order == 2:
                q = ztL.shape[-1]//2
                stL = ztL[:,:,:,:q] # L,N,T,q Only the position is decoded

        if c is not None:
            cT = torch.stack([c]*ztL.shape[2],-2) # L,N,T,q
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
        @param cL - invariant code  [L,N,q]
        '''
        ts  = self.dt * torch.arange(T,dtype=torch.float).to(z0L.device)
        ztL = [self.flow(z0, ts, zc) for z0,zc in zip(z0L,cL)] # sample L trajectories
        return torch.stack(ztL) # L,N,T,nobj,2q
    
    def sample_hbnode_trajectories(self, X, T, L=1):
        ''' 
            X - [N,T,d]
            Xrec - [N,T,d]
        '''
        hbnode = self.flow.odefunc.diffeq
        Xrec   = hbnode(X, L, T_custom=T) # N,T,D
        XrecL  = torch.stack(L*[Xrec]) # L,N,T,D
        return XrecL

    def forward(self, X, L=1, T_custom=None):

        try:
            self.inv_enc.last_layer_gp.build_cache()
        except:
            pass

        [N,T] = X.shape[:2]
        if T_custom:
            T = T_custom

        #condition on
        in_data = X[:,:self.Tin]
        # if self.model == 'node':
        if self.model == 'node' or self.model == 'hbnode':
            s0_mu, s0_logv = self.vae.encoder(in_data) # N,q
            z0 = self.vae.encoder.sample(s0_mu, s0_logv, L=L) # N,q or L,N,q
            z0 = z0.unsqueeze(0) if z0.ndim==2 else z0 # L,N,q

            #if multiple object separate latent vector (but shared dynamics)
            q  = z0.shape[-1]
            z0 = z0.reshape(L,N,self.Nobj,q//self.Nobj) # L,N,nobj,q_

            v0_mu, v0_logv = None, None
            if self.order == 2:
                v0_mu, v0_logv = self.vae.encoder_v(in_data)
                v0 = self.vae.encoder_v.sample(v0_mu, v0_logv, L=L) # N,q or L,N,q
                v0 = v0.unsqueeze(0) if v0.ndim==2 else v0

                #if multiple object separate latent vector (but shared dynamics)
                q  = v0.shape[-1]
                v0 = v0.reshape(L,N,self.Nobj,q//self.Nobj) # L,N,nobj,q_
                if self.model == 'hbnode':
                    z0 = torch.concat([z0,v0],dim=2) #L, N, 2, q
                else:
                    z0 = torch.concat([z0,v0],dim=-1)  #L, N, 1, 2q

      
        elif self.model =='sonode':
            #compute velocity and concatenate to position
            s0_mu, s0_logv,v0_mu, v0_logv = None, None, None, None
            v0 = self.vae(in_data) #N, dim
            x0 = in_data[:,0,:] #N, dim
            z0 = torch.stack((x0, v0),dim=1).reshape(1,N,self.Nobj,X.shape[-1]*2) #L, N, 1, 2*dim

        #encode content (invariance), pass whole sequence length 
        if self.is_inv:
            InvMatrix = self.inv_enc(X, L=L) # embeddings [L,N,T,q] or [L,N,ns,q]
            inv_var = InvMatrix.mean(2) # time-invariant code [L,N,q]
            c, m = inv_var[:,:,:self.inv_enc.content_dim], inv_var[:,:,self.inv_enc.content_dim:]
        else:
            InvMatrix,c,m = None, None, None

        #sample trajectories
        if self.aug:
            mL = m.reshape((L,N,self.Nobj,-1)) #L,N,Nobj,q
            ztL  = self.sample_augmented_trajectories(z0, mL, T, L) # L,N,T,Nobj, 2q
            ztL = ztL.reshape(L,N,T,-1) # L,T,N, nobj*2q
            Xrec = self.build_decoding(ztL, [L,N,T,*X.shape[2:]], c)
        else:
            ztL = self.sample_trajectories(z0,T,L) # L,T,N,nobj,q
            ztL = ztL.reshape(L,N,T,-1) # L,T,N, nobj*q
            Xrec = self.build_decoding(ztL, [L,N,T,*X.shape[2:]], c)
        
        return Xrec, ztL, (s0_mu, s0_logv), (v0_mu, v0_logv), InvMatrix, c, m
