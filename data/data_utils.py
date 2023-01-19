import os
import numpy as np
import torch
import torch.nn as nn
from   torch.utils import data
from torchdiffeq import odeint

from data.mnist import MovingMNIST, RotatingMNIST
from model.misc import io_utils
from model.misc.plot_utils import plot_2d, plot_mnist, plot_sin, plot_2d_origin


def load_data(args, device, dtype):
	if args.task=='rot_mnist':
		trainset, valset, testset = load_rmnist_data(args, device, dtype)
	elif args.task=='mov_mnist':
		trainset, valset, testset  = load_mmnist_data(args, dtype, dtype)
	elif args.task=='sin':
		trainset, valset, testset  = load_sin_data(args, device, dtype)
	elif args.task=='lv':
		trainset, valset, testset  = load_lv_data(args, device, dtype)
	elif args.task == 'spiral':
		trainset, valset, testset  = load_spiral_data(args,device,dtype)
	else:
		return ValueError(r'Invalid task {arg.task}')
	return trainset, valset, testset  #, N, T, D

def load_rmnist_data(args, device, dtype):
	return __load_data(args,device,dtype,'rot_mnist')

def load_mmnist_data(args,device,dtype):
	return __load_data(args,device,dtype,'mov_mnist')


def load_sin_data(args, device, dtype):
	return __load_data(args, device, dtype, 'sin')


def load_lv_data(args, device, dtype):
	return __load_data(args, device, dtype, 'lv')


def load_spiral_data(args, device,dtype):
	return __load_data(args, device, dtype, 'spiral')


def __load_data(args, device, dtype, dataset=None):
	assert dataset=='sin' or dataset=='lv'or dataset=='spiral' or dataset =='mov_mnist' or dataset=='rot_mnist'
	io_utils.makedirs(args.data_root + '/' + args.task)
	data_path_tr = os.path.join(args.data_root + '/' + args.task,f'{dataset}-tr-data.pkl')
	data_path_vl = os.path.join(args.data_root + '/' + args.task,f'{dataset}-vl-data.pkl')
	data_path_te = os.path.join(args.data_root + '/' + args.task,f'{dataset}-te-data.pkl')
	try:
		Xtr = torch.load(data_path_tr)
		Xvl = torch.load(data_path_vl)
		Xte = torch.load(data_path_te)
	except:
		if dataset=='sin':
			data_loader_fnc = gen_sin_data
		elif dataset == 'lv':
			data_loader_fnc = gen_lv_data
		elif dataset == 'spiral':
			data_loader_fnc = gen_spiral_data
		elif dataset == 'rot_mnist':
			data_loader_fnc = gen_rmnist_data
		elif dataset == 'mov_mnist':
			data_loader_fnc = gen_mmnist_data

		data_loader_fnc(data_path_tr, args.Ntrain, style = args.style,plot=True)
		data_loader_fnc(data_path_vl, args.Nvalid, style = args.style)
		if dataset == 'sin':
			data_loader_fnc(data_path_te, args.Ntest, T=150, style =  args.style)
		else:
			data_loader_fnc(data_path_te, args.Ntest, style =  args.style)

		Xtr = torch.load(data_path_tr)
		Xvl = torch.load(data_path_vl)
		Xte = torch.load(data_path_te)


	Xtr = Xtr[:args.Ntrain] if Xtr.shape[0] != args.Ntrain else Xtr
	Xvl = Xvl[:args.Nvalid] if Xvl.shape[0] != args.Nvalid else Xvl
	Xte = Xte[:args.Ntest] if Xte.shape[0] != args.Ntest else Xte
		
	Xtr = Xtr.to(device).to(dtype)
	Xvl = Xvl.to(device).to(dtype)
	Xte = Xte.to(device).to(dtype)

	if dataset == 'mov_mnist':
		Xtr = Xtr[:,:args.seq_len]
		Xvl = Xvl[:,:args.seq_len]

	return __build_dataset(args.num_workers, args.batch_size, Xtr, Xvl, Xte)


class Dataset(data.Dataset):
	def __init__(self, Xtr):
		self.Xtr = Xtr # N,T,_
	def __len__(self):
		return len(self.Xtr)
	def __getitem__(self, idx):
		return self.Xtr[idx]
	@property
	def shape(self):
		return self.Xtr.shape


def __build_dataset(num_workers, batch_size, Xtr, Xvl, Xte, shuffle=True):
	# Data generators
	if num_workers>0:
		from multiprocessing import Process, freeze_support
		torch.multiprocessing.set_start_method('spawn', force="True")

	params = {'batch_size': batch_size, 'shuffle': shuffle, 'num_workers': num_workers, 'drop_last': True}
	trainset = Dataset(Xtr)
	trainset = data.DataLoader(trainset, **params)
	validset  = Dataset(Xvl)
	validset  = data.DataLoader(validset, **params)
	testset  = Dataset(Xte)
	testset  = data.DataLoader(testset, **params)
	return trainset, validset, testset


def gen_sin_data(data_path, N, T=50, dt=0.1, sig=.1, style=None, plot=False): 
	phis = torch.rand(N,1) #
	fs = torch.rand(N,1) * .5 + .5 # N,1, [0.5, 1.0]
	A  = torch.rand(N,1) * 2 + 1   # N,1, [1.0, 3.0]
	ts = torch.arange(T) * dt # T
	ts = torch.stack([ts]*N)  # N,T
	ts = (ts*fs+phis) * 2*np.pi # N,T
	X  = ts.sin() * A
	X += torch.randn_like(X)*sig
	X = X.unsqueeze(-1) # N,T,1
	if plot: plot_sin(X,X.unsqueeze(0),fname='data/sin/example_sin.png')
	torch.save(X, data_path)


def gen_spiral_data(data_path, N=1000, T=1000, dt=0.01, style=None, plot=False): 
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
	ts = torch.arange(T)*dt

	#generate sequences
	Xt = odeint(odef_, X0, ts, method='dopri5') # T,N,1,2
	Xt = Xt.permute(1,0,2,3) #N,T,1,2
	Xt = Xt.reshape(N,T,2)
	if plot: plot_2d_origin(Xt,fname='data/spiral/example_spiral.png',N=10)
	torch.save(Xt, data_path)

def gen_lv_data(data_path, N=5, T=200, dt=.1, DIFF=.01, beta=0.5, delta=0.2, style=None,plot=False):
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

	odef_ = lambda t,x: odef(t,x,alpha,beta,gamma,delta)

	X0 = torch.tensor(np.random.uniform(low=1.0, high=5.0, size=(N,2))) 
	X0 = X0 + 1*torch.rand_like(X0)
	ts = torch.arange(T)*dt

	Xt = odeint(odef_, X0, ts, method='dopri5') # T,N, 2

	valid_s, invalid_s = _check_valid(X0,Xt, N, T, DIFF)
	#disregard invalud samples (trajectory incomplete)
	Xt_valid  = Xt[:,valid_s] # T, N, 2
	Xt_valid = Xt_valid[:,:N_old]  # T, N, 2
	Xt_valid = Xt_valid.permute(1,0,2) #N,T,2
	if plot: plot_2d_origin(Xt_valid,fname='data/lv/example_lv.png',N=10)
	torch.save(Xt_valid, data_path)

def gen_mmnist_data(data_path, N=10, seq_len=30, ndigits=2, style=None, plot=False): 
	#load MNIST digits
	mmnist = MovingMNIST('data/', style ,seq_len=seq_len, num_digits=ndigits)
	
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
	if plot: plot_mnist(Xt[:15], Xt[15:], fname='data/mov_mnist/example_mmnist.png')
	torch.save(Xt, data_path)

def gen_rmnist_data(data_path, N=10, n_angles=16, digit=3, style=None, plot=False):
	#load MNIST digits
	data = RotatingMNIST('data/', data_n = N, n_angles = n_angles, digit=digit)
	data._gen_angles()
	data._sample_digit()

	#generate rotation sequences
	videos = data._sample_rotation()

	#normalize
	Xt = data._collate_fn(videos) #N,T,1,dim,dim

	#random initial angle
	T = Xt.shape[1]
	Xt  = torch.cat([Xt,Xt[:,1:]],1) # N,2T,1,d,d
	t0s   = torch.randint(0,T,[Xt.shape[0]])
	Xt   = torch.stack([Xt[i,t0:t0+T]   for i,t0 in enumerate(t0s)])
	if plot: plot_mnist(Xt[:15], Xt[15:], fname='data/rot_mnist/example_rmnist.png')
	#save
	torch.save(Xt, data_path)


