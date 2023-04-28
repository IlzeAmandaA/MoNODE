import torch
import numpy as np
from torchdiffeq import odeint
from data.mnist import MovingMNIST, RotatingMNIST
from model.misc.plot_utils import plot_mnist, plot_sin_gt, plot_2d_gt, plot_bb, plot_bb_V
from data.bb import BouncingBallsSim

def add_noise(traj_list, time_steps, noise_weight):
	n_samples = traj_list.size(0)

	# Add noise to all the points except the first point
	n_tp = time_steps - 1
	noise = np.random.sample((n_samples, n_tp))
	noise = torch.Tensor(noise).to(traj_list.device)


	traj_list_w_noise = traj_list.clone()
	# Dimension [:,:,0] is a time dimension -- do not add noise to that
	traj_list_w_noise[:,1:] += noise_weight * noise
	return traj_list_w_noise


def add_noise_lv(traj_list, time_steps, noise_weight):
	n_samples = traj_list.size(0)

	# Add noise to all the points except the first point
	n_tp = time_steps - 1
	noise_x1 = np.random.sample((n_samples, n_tp))
	noise_x1 = torch.Tensor(noise_x1).to(traj_list.device)
	noise_x2 = np.random.sample((n_samples, n_tp))
	noise_x2 = torch.Tensor(noise_x2).to(traj_list.device)

	traj_list_w_noise = traj_list.clone()
	# Dimension [:,:,0] is a time dimension -- do not add noise to that
	traj_list_w_noise[:,1:,0] += noise_weight * noise_x1
	traj_list_w_noise[:,1:,1] += noise_weight * noise_x2
	return traj_list_w_noise

def gen_sin_data(data_path, params, flag, task='sin'):
	N = params[task][flag]['N']
	T = params[task][flag]['T']
	noise = params[task]['noise']
	plot = True if flag=='train' else False
	phis = torch.rand(N,1) #
	fs = torch.rand(N,1) * .5 + .5 # N,1, [0.5, 1.0]
	A  = torch.rand(N,1) * 3 + 1   # N,1, [1.0, 3.0]
	ts = torch.arange(T) * params[task]['dt'] # T
	ts = torch.stack([ts]*N)  # N,T
	ts = (ts*fs+phis) * 2*np.pi # N,T
	X  = ts.sin() * A
	if noise > 0.0:
		X = add_noise(X, X.shape[1], noise)
	X = X.unsqueeze(-1) # N,T,1
	plot_sin_gt(X,fname='data/sin/example_sin_'+flag)
	torch.save(X, data_path)


def gen_lv_data(data_path, params, flag, task='lv'):
	N = params[task][flag]['N']
	T = params[task][flag]['T'] 
	noise = params[task]['noise']

	alpha = torch.rand([N,1]) / .4 + params[task]['alpha']
	gamma = torch.rand([N,1]) / .4 + params[task]['gamma']

	delta = torch.rand([N,1]) * 0.1  + params[task]['delta']
	beta = torch.rand([N,1]) * 0.1  + params[task]['beta']


	def odef(t,state,alpha,beta,gamma,delta):
		x,y = state.split([1,1],dim=-1) # M,1 & M,1
		dx = alpha*x   - beta*x*y # M,1
		dy = delta*x*y - gamma*y  # M,1
		return torch.cat([dx,dy],-1)
	

	odef_ = lambda t,x: odef(t,x,alpha,beta,gamma,delta)

	X0 = torch.tensor(np.random.uniform(low=1.0, high=5.0, size=(N,2))) 
	ts = torch.arange(T)*params[task]['dt']

	Xt = odeint(odef_, X0, ts, method=params[task]['solver']) # T,N, 2

	Xt = Xt.permute(1,0,2) #N,T,2
	if noise > 0.0:
		Xt = add_noise_lv(Xt, Xt.shape[1], noise)

	filename = 'data/lv/example_lv'
	plot_2d_gt(Xt,fname=filename + flag,N=10)
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


