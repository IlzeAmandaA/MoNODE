import torch
import numpy as np
from torchdiffeq import odeint
from data.mnist import MovingMNIST, RotatingMNIST
from model.misc.plot_utils import plot_mnist, plot_sin, plot_2d_origin, plot_bb, plot_bb_V
from data.bb import BouncingBallsSim


def gen_sin_data(data_path, params, flag, task='sin'):
	N = params[task][flag]['N']
	T = params[task][flag]['T']
	plot = True if flag=='train' else False
	phis = torch.rand(N,1) #
	fs = torch.rand(N,1) * .5 + .5 # N,1, [0.5, 1.0]
	A  = torch.rand(N,1) * 2 + 1   # N,1, [1.0, 3.0]
	ts = torch.arange(T) * params[task]['dt'] # T
	ts = torch.stack([ts]*N)  # N,T
	ts = (ts*fs+phis) * 2*np.pi # N,T
	X  = ts.sin() * A
	X += torch.randn_like(X)*params[task]['sig']
	X = X.unsqueeze(-1) # N,T,1
	plot_sin(X,X.unsqueeze(0),fname='data/sin/example_sin_'+flag)
	torch.save(X, data_path)


def gen_lv_data(data_path, params, flag, task='lv'):
	N = params[task][flag]['N']
	T = params[task][flag]['T'] 
	if params[task]['clean']:
		N_add = 200 #add aditional data samples as some might be discarded
		N_old = N
		N += N_add

	alpha = torch.rand([N,1]) / .3 + .1
	gamma = torch.rand([N,1]) / .3 + .1

	def odef(t,state,alpha,beta,gamma,delta):
		x,y = state.split([1,1],dim=-1) # M,1 & M,1
		dx = alpha*x   - beta*x*y # M,1
		dy = delta*x*y - gamma*y  # M,1
		return torch.cat([dx,dy],-1)
	
	def _check_valid(x0, xt, N, T, DIFF):
		valid_samples = []
		invalid_samples = []
		for i in range(N):
			initial = x0[i]
			for m in range(T):
				if m != 0: #the initial will always be the same
					frame = xt[m,i]
					for x in torch.abs(frame - initial): #should pass close to the origin dynamics 
						if x< DIFF:
							if i not in valid_samples:
								valid_samples.append(i)
								
			if i not in valid_samples:
				invalid_samples.append(i)
		return valid_samples, invalid_samples

	odef_ = lambda t,x: odef(t,x,alpha,params[task]['beta'],gamma,params[task]['delta'])

	X0 = torch.tensor(np.random.uniform(low=1.0, high=5.0, size=(N,2))) 
	X0 = X0 + 1*torch.rand_like(X0)
	ts = torch.arange(T)*params[task]['dt']

	Xt = odeint(odef_, X0, ts, method='dopri5') # T,N, 2
	
	if params[task]['clean']:
		valid_s, invalid_s = _check_valid(X0,Xt, N, T, params[task]['DIFF'])
		#disregard invalid samples (trajectory incomplete)
		Xt  = Xt[:,valid_s] # T, N, 2
		Xt = Xt[:,:N_old]  # T, N, 2

	Xt = Xt.permute(1,0,2) #N,T,2
	filename = 'data/lv/example_lv_clean' if params[task]['clean'] else 'data/lv/example_lv'
	plot_2d_origin(Xt,fname=filename + flag,N=10)
	torch.save(Xt, data_path)

def gen_rmnist_data(data_path, params, flag, task='rot_mnist'):
	N = params[task][flag]['N']
	T = params[task][flag]['T'] 
	#load MNIST digits
	data = RotatingMNIST('data/', data_n = N, n_angles = T, digit=params[task]['digit'])

	data._gen_angles()
	data._sample_digit()
	videos = data._sample_rotation()

	#normalize
	Xt = data._collate_fn(videos) #N,T,1,dim,dim

	#random initial angle
	T = Xt.shape[1]
	Xt  = torch.cat([Xt,Xt[:,1:]],1) # N,2T,1,d,d
	t0s   = torch.randint(0,T,[Xt.shape[0]])
	Xt   = torch.stack([Xt[i,t0:t0+T]   for i,t0 in enumerate(t0s)])
	plot_mnist(Xt[:15], Xt[15:], fname='data/rot_mnist/example_rmnist' + flag)
	#save
	torch.save(Xt, data_path)

def gen_rmnist_ou_data(data_path, params, flag, task='rot_mnist_ou'):
	N = params[task][flag]['N']
	T = params[task][flag]['T'] 
	#load MNIST digits
	data = RotatingMNIST('data/', data_n = N, n_angles = T, digit=params[task]['digit'])
	
	# new stochastic setup
	data._gen_angles_stochastic()
	data._sample_digit()
	videos = data._sample_rotation_sequentially()

	#normalize
	Xt = data._collate_fn(videos) #N,T,1,dim,dim

	plot_mnist(Xt, Xt, fname='data/rot_mnist_ou/example_rmnist_ou' + flag)
	#save
	torch.save(Xt, data_path)

def gen_mmnist_data(data_path, params, flag, task='mov_mnist'):
	N = params[task][flag]['N']
	T = params[task][flag]['T'] 
	#load MNIST digits
	mmnist = MovingMNIST('data/', params[task]['style'] ,seq_len=T, num_digits=params[task]['ndigits'])
	
	#subsample styles
	mmnist._sample_style_idx()

	#subsample data
	mmnist._subsample_data()

	#generate moving sequences
	videos = []
	for n in range(N):
		videos.append(mmnist._sample_sequence())
	
	#normalize to [0,1] range
	Xt = mmnist._collate_fn(videos) #N,T,1,dim,dim
	filename = 'data/mov_mnist/example_mmnist_' + str(T)
	plot_mnist(Xt[:15], Xt[15:], fname=filename + flag)
	torch.save(Xt, data_path)

	
def gen_bb_data(data_path, params, flag, task='bb'):
	N = params[task][flag]['N']
	T = params[task][flag]['T']
	nballs = params[task]['nballs']
	bb = BouncingBallsSim()
	outs = []
	for i in range(N):
		outs.append(bb.sample_trajectory(A=nballs, T=T))
	
	X,V,fr = [np.stack(x[i] for x in outs) for i in range(3)]
	torch.save(V, data_path)
	torch.save([torch.tensor(x) for x in [X,V,fr]], 'data/bb/all_var_'+flag+'_.pkl')
	plot_bb(X, fname = 'data/bb/example_bb_' + flag)
	plot_bb_V(V, fname = 'data/bb/example_bb_V_' + flag)


def gen_spiral_data(data_path, N, **kwargs): 
	'''
	Note: keep T fixed
	'''
	def odef(t, x, A):
		return (x**3) @ A

	#coefficient for every data point
	A  = torch.tensor(np.array([[-0.1, 2.0], [-2.0, -0.1]])) + torch.rand((N,2,2))*0.1

	#ode
	odef_ = lambda t,x: odef(t,x,A)

	#random starting point for every sequence 
	X0 = torch.tensor(np.random.uniform(low=1, high=3, size=(N,1,2))) 
	X0 = X0 + 1*torch.rand_like(X0)
	ts = torch.arange(kwargs['T'])*kwargs['dt']

	#generate sequences
	Xt = odeint(odef_, X0, ts, method='dopri5') # T,N,1,2
	Xt = Xt.permute(1,0,2,3) #N,T,1,2
	Xt = Xt.reshape(N,kwargs['T'],2)
	if kwargs['plot']: plot_2d_origin(Xt,fname='data/spiral/example_spiral.png',N=10)
	torch.save(Xt, data_path)


