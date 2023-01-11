import os, numpy as np, scipy.io as sio

import torch
import torch.nn as nn
from   torch.utils import data

from   data.lv import LotkaVolterra
from torchdiffeq import odeint
from data.mmnist import MovingMNIST
from model.misc import io_utils


def load_data(args, device, dtype):
	if args.task=='rot_mnist':
		trainset, valset = load_rot_mnist_data(args, device, dtype)
	elif args.task=='mov_mnist':
		trainset, valset = load_mov_mnist_data(args, dtype)
	elif args.task=='sin':
		trainset, valset = load_sin_data(args, device, dtype)
	elif args.task=='lv':
		trainset, valset = load_lv_data(args, device, dtype)
	elif args.task == 'spiral':
		trainset, valset = load_spiral_data(args,device,dtype)
	else:
		return ValueError(r'Invalid task {arg.task}')
	return trainset, valset #, N, T, D


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


def __build_dataset(num_workers, batch_size, Xtr, Xtest, shuffle=True):
	# Data generators
	if num_workers>0:
		from multiprocessing import Process, freeze_support
		torch.multiprocessing.set_start_method('spawn', force="True")

	params = {'batch_size': batch_size, 'shuffle': shuffle, 'num_workers': num_workers}
	trainset = Dataset(Xtr)
	trainset = data.DataLoader(trainset, **params)
	testset  = Dataset(Xtest)
	testset  = data.DataLoader(testset, **params)
	return trainset, testset

def __build_dataloader(dataset, params):
	if params['num_workers']>0:
		from multiprocessing import Process, freeze_support
		torch.multiprocessing.set_start_method('spawn', force="True")
	return data.DataLoader(dataset, **params)

def load_rot_mnist_data(args, device, dtype):
	fullname = os.path.join(args.data_root, "rot-mnist.mat")
	dataset = sio.loadmat(fullname)
	
	X = dataset['X'].squeeze()
	if args.digit:
		Y = dataset['Y'].squeeze() 
		X = X[Y==args.digit,:,:]
	T = X.shape[1]

	N = args.Ntrain #train
	Nt = args.Nvalid + N # valid
	Xtr   = torch.tensor(X[:N],   device=device, dtype=dtype).view([args.Ntrain,T,1,28,28])
	Xtest = torch.tensor(X[N:Nt], device=device, dtype=dtype).view([args.Nvalid,T,1,28,28])

	if args.rotrand:
		Xtr   = torch.cat([Xtr,Xtr[:,1:]],1) # N,2T,1,d,d
		Xtest = torch.cat([Xtest,Xtest[:,1:]],1) # N,2T,1,d,d
		t0s_tr   = torch.randint(0,T,[Xtr.shape[0]])
		t0s_test = torch.randint(0,T,[Xtest.shape[0]])
		Xtr   = torch.stack([Xtr[i,t0:t0+T]   for i,t0 in enumerate(t0s_tr)])
		Xtest = torch.stack([Xtest[i,t0:t0+T] for i,t0 in enumerate(t0s_test)])

	# Generators
	return __build_dataset(args.num_workers, args.batch_size, Xtr, Xtest)


def load_mov_mnist_data(args, dtype):
	dataset = MovingMNIST.make_dataset(args.data_root, args.nx, args.seq_len,args.max_speed,
                                        args.deterministic, args.ndigits, args.subsample, args.Ntrain, dtype)
	trainset = dataset.get_fold('train')
	valset = dataset.get_fold('val')
	# Change validation sequence length, if specified
	if args.seq_len_valid is not None:
		valset.change_seq_len(args.seq_len_valid)

	params = {'batch_size': args.batch_size, 'collate_fn': dataset.collate_fn, 'sampler': None, 'drop_last':True,
				 'shuffle': args.shuffle, 'pin_memory':True, 'num_workers': args.num_workers}
	return __build_dataloader(trainset, params), __build_dataloader(valset, params)

	# N  = args.Ntrain #train
	# Nt = args.Nvalid + N # valid
	# data = np.load(os.path.join(args.data_root,'mov-mnist.npy')).transpose([1,0,2,3])[:Nt,:args.seq_len] # N,T,d,d
	# data = torch.tensor(data).to(device).to(dtype).unsqueeze(2) / 255.0 # N,T,1,d,d
	# Xtr, Xtest = data[:N], data[N:]
	#return __build_dataset(args.num_workers, args.batch_size, Xtr, Xtest)


def __load_data(args, device, dtype, dataset='sin'):
	assert dataset=='sin' or dataset=='lv'or dataset=='spiral'
	io_utils.makedirs(args.data_root + '/' + args.task)
	data_path = os.path.join(args.data_root + '/' + args.task,f'{dataset}-data.pkl')
	try:
		X = torch.load(data_path)
	except:
		if dataset=='sin':
			data_loader_fnc = gen_sin_data
		elif dataset == 'lv':
			data_loader_fnc = gen_lv_data
		elif dataset == 'spiral':
			data_loader_fnc = gen_spiral_data
		data_loader_fnc(data_path, args.Ntrain+args.Nvalid)
		X = torch.load(data_path)
	X = X.to(device).to(dtype)
	return __build_dataset(args.num_workers, args.batch_size, X[:args.Ntrain], X[args.Ntrain:])


def load_sin_data(args, device, dtype):
	return __load_data(args, device, dtype, 'sin')


def load_lv_data(args, device, dtype):
	return __load_data(args, device, dtype, 'lv')


def load_spiral_data(args, device,dtype):
	return __load_data(args, device, dtype, 'spiral')


def gen_sin_data(data_path, N, T=50, dt=0.1, sig=.1): 
	phis = torch.rand(N,1) #
	fs = torch.rand(N,1) * .5 + .5 # N,1, [0.5, 1.0]
	A  = torch.rand(N,1) * 2 + 1   # N,1, [1.0, 3.0]
	ts = torch.arange(T) * dt # T
	ts = torch.stack([ts]*N)  # N,T
	ts = (ts*fs+phis) * 2*np.pi # N,T
	X  = ts.sin() * A
	X += torch.randn_like(X)*sig
	X = X.unsqueeze(-1) # N,T,1
	torch.save(X, data_path)

def gen_spiral_data(data_path, N, T, dt, ): #TODO implement
	pass


def gen_lv_data(data_path, N=5, T=50, dt=.2, sig=.01, w=10):
	d  = 2 # state dim

	alpha = torch.rand([N,1]) / .3 + .1
	gamma = torch.rand([N,1]) / .3 + .1
	beta  = 0.5
	delta = 0.2

	def odef(t,state,alpha,beta,gamma,delta):
		x,y = state.split([1,1],dim=-1) # M,1 & M,1
		dx = alpha*x   - beta*x*y # M,1
		dy = delta*x*y - gamma*y  # M,1
		return torch.cat([dx,dy],-1)

	odef_ = lambda t,x: odef(t,x,alpha,beta,gamma,delta)
	
	x0 = torch.tensor([5.0,2.5]) + w*torch.rand([N,d])
	ts = torch.arange(T) * dt
	xt = odeint(odef_, x0, ts, method='dopri5') # T,N,n

	X = X.reshape(T,N,M,d).permute(2,1,0,3) # M,N,T,d
	std = (X.max(2)[0] - X.min(2)[0]).sqrt().unsqueeze(2) # N,M,1,d
	X  += torch.randn([M,N,T,d]) * std * sig
	torch.save(X, data_path)