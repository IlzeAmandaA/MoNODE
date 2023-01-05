import os, numpy as np, scipy.io as sio

import torch
import torch.nn as nn
from   torch.utils import data

from   model.misc.lv import LotkaVolterra


def load_data(args, device, dtype):
	if args.task=='rot_mnist':
		trainset, testset = load_rot_mnist_data(args, device, dtype)
	elif args.task=='mov_mnist':
		trainset, testset = load_mov_mnist_data(args, device, dtype)
	elif args.task=='sin':
		trainset, testset = load_sin_data(args, device, dtype)
	elif args.task=='lv':
		trainset, testset = load_lv_data(args, device, dtype)
	else:
		return ValueError(r'Invalid task {arg.task}')
	return trainset, testset #, N, T, D


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


def load_mov_mnist_data(args, device, dtype):
	N  = args.Ntrain #train
	Nt = args.Nvalid + N # valid
	data = np.load(os.path.join(args.data_root,'mov-mnist.npy')).transpose([1,0,2,3])[:Nt,:args.seq_len] # N,T,d,d
	data = torch.tensor(data).to(device).to(dtype).unsqueeze(2) / 255.0 # N,T,1,d,d
	Xtr, Xtest = data[:N], data[N:]
	return __build_dataset(args.num_workers, args.batch_size, Xtr, Xtest)


def __load_data(args, device, dtype, dataset='sin'):
	assert dataset=='sin' or dataset=='lv'
	data_path = os.path.join(args.data_root,f'{dataset}-data.pkl')
	try:
		X = torch.load(data_path)
	except:
		data_loader_fnc = gen_sin_data if dataset=='sin' else gen_lv_data
		data_loader_fnc(data_path, args.Ntrain+args.Nvalid)
		X = torch.load(data_path)
	X = X.to(device).to(dtype)
	return __build_dataset(args.num_workers, args.batch_size, X[:args.Ntrain], X[args.Ntrain:])


def load_sin_data(args, device, dtype):
	return __load_data(args, device, dtype, 'sin')


def load_lv_data(args, device, dtype):
	return __load_data(args, device, dtype, 'lv')


# def load_sin_data(args, device, dtype):
# 	data_path = os.path.join(args.data_root,'sin-data.pkl')
# 	try:
# 		X = torch.load(data_path)
# 	except:
# 		gen_sin_data(data_path, args.Ntrain+args.Nvalid)
# 		X = torch.load(data_path)
# 	try:
# 		X = X.to(device).to(dtype)
# 	except:
# 		X = X[0].to(device).to(dtype)
# 		torch.save(X,data_path)
# 	return __build_dataset(args.num_workers, args.batch_size, X[:args.Ntrain], X[args.Ntrain:])


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


def gen_lv_data(data_path, M, N=5, T=50, dt=.2, sig=.01, w=10):
	d  = 2 # state dim

	# alpha = (1+torch.arange(M)) / M / .5
	# gamma = (1+torch.arange(M)) / M / .5 
	alpha = torch.rand([M]) / .3 + .1
	gamma = torch.rand([M]) / .3 + .1
	alpha = alpha.repeat([N]) # NM
	gamma = gamma.repeat([N]) # NM
	beta  = 0.5
	delta = 0.2
	lotka_volterra = LotkaVolterra(alpha, beta, delta, gamma)
	
	x0 = torch.tensor([5.0,2.5]) + w*torch.rand([N*M,d])
	ts = torch.arange(T) * dt
	
	with torch.no_grad():
		X = lotka_volterra.forward_simulate(x0,ts) # T,NM,d
		X = X.reshape(T,N,M,d).permute(2,1,0,3) # M,N,T,d
	std = (X.max(2)[0] - X.min(2)[0]).sqrt().unsqueeze(2) # N,M,1,d
	X  += torch.randn([M,N,T,d]) * std * sig
	torch.save(X, data_path)