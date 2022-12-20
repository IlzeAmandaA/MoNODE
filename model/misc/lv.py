import torch 
import torch.nn as nn 


def rk4_step(f,h,t,x):
	k1 = h * (f(t, x))
	k2 = h * (f((t+h/2), (x+k1/2)))
	k3 = h * (f((t+h/2), (x+k2/2)))
	k4 = h * (f((t+h), (x+k3)))
	return (k1+2*k2+2*k3+k4) / 6


def forward_simulate(f, x0, ts):
	''' f(t,x) '''
	X  = [x0]
	ode_steps = len(ts)-1
	for i in range(ode_steps):
		h  = ts[i+1]-ts[i]
		t  = ts[i]
		x  = X[i]
		x_next = x + rk4_step(f,h,t,x)
		X.append(x_next)
	X = torch.stack(X) # T,N,d
	return X


class LotkaVolterra(nn.Module):
	def __init__(self, alpha, beta, delta, gamma):
		'''
		Parameters
		----------
		alpha : float
			prey growth parameter.
		beta : float
			determines the rate of predation.
		delta : float
			the growth parameter of the predator population.
		gamma : float
			predator extinction parameter.

		Returns
		-------
		None.

		'''
		super().__init__()
		self.alpha = self.__convert_to_tensor(alpha)
		self.beta  = self.__convert_to_tensor(beta)
		self.delta = self.__convert_to_tensor(delta)
		self.gamma = self.__convert_to_tensor(gamma)
	
	def __convert_to_tensor(self,x):
		return x if isinstance(x,torch.Tensor) else torch.tensor(x)
	
	def vectorize_if_necessary(self, ndim):
		dim_diff = ndim - self.alpha.ndim
		squeeze_fnc = torch.unsqueeze if dim_diff>0 else torch.squeeze 
		for i in range(abs(dim_diff)):
			self.alpha = squeeze_fnc(self.alpha,-1)
			self.beta  = squeeze_fnc(self.beta, -1)
			self.delta = squeeze_fnc(self.delta,-1)
			self.gamma = squeeze_fnc(self.gamma,-1)
				
	def forward(self, t, state):
		x,y = state.split([1,1],dim=-1) # M,N,1 & M,N,1
		dx = self.alpha*x   - self.beta*x*y # N,1
		dy = self.delta*x*y - self.gamma*y  # N,1
		return torch.cat([dx,dy],-1)

	def forward_simulate(self, x0, ts):
		self.vectorize_if_necessary(x0.ndim)
		return forward_simulate(self, x0, ts) # T,NM,d