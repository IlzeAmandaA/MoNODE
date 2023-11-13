import torch
import os, numpy as np
import re
from torchdiffeq import odeint
from data.mnist import RotatingMNIST
from model.misc.plot_utils import plot_mnist, plot_sin_gt, plot_2d_gt, plot_bb, plot_bb_V
from data.bb import BouncingBallsSim
from data.mocap import read_amc


def _adjust_name(data_path, substr, insertion):
	idx = data_path.index(substr)
	return data_path[:idx] + '-' + insertion + data_path[idx:]

def add_noise(traj_list, time_steps, noise_weight):
	'''
	Credits go to:
	# Latent ODEs for Irregularly-Sampled Time Series
	# Author: Yulia Rubanova
	'''
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
	'''
	Credits go to:
	# Latent ODEs for Irregularly-Sampled Time Series
	# Author: Yulia Rubanova
	'''
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
	fname = data_path[:[m.start() for m in re.finditer('/', data_path)][-1]+1] + 'example_sin_' + flag
	plot_sin_gt(X,fname=fname)
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
	Xt_labels = data.labels

	#random initial angle
	T = Xt.shape[1]
	Xt  = torch.cat([Xt,Xt[:,1:]],1) # N,2T,1,d,d
	t0s   = torch.randint(0,T,[Xt.shape[0]])
	Xt   = torch.stack([Xt[i,t0:t0+T]   for i,t0 in enumerate(t0s)])

	if params[task][flag]["rep"] > 1:
		T_loop=T-1
		Xt = Xt[:,:T_loop,]
		Xt = Xt.repeat(1,params[task][flag]["rep"],1,1,1)
	
	plot_mnist(Xt[:15], Xt[15:], fname='data/rot_mnist/example_rmnist' + flag)
	#save
	torch.save(Xt, data_path)
	data_path_label = _adjust_name(data_path, '.pkl', 'labels')
	torch.save(Xt_labels, data_path_label)

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


def gen_mocap_data(data_path, params, flag, task='mocap'):
	N = params[task][flag]['N']
	T = params[task][flag]['T']
	plot = True if flag=='train' else False

	DATA_ROOT = "data/mocap/"
	Xs = []
	min_len = 300
	fnames = os.listdir(DATA_ROOT)
	for fname in fnames:
		fname = os.path.join(DATA_ROOT,fname)
		if fname.endswith('amc') or fname.endswith('txt'):
			X = read_amc(fname, crop=True) # T,D
			if X.shape[0]>min_len:
				Xs.append(X[-min_len::2])
			else:
				print(f'{fname} is skipped since too short')

	Xs = torch.tensor(np.stack(Xs),dtype=torch.float32) # 56,T,D
	if flag=='train':
		torch.save(Xs[:-10],os.path.join(DATA_ROOT,'mocap-tr-data.pkl'))
	elif flag=='valid':
		torch.save(Xs[-10:-5],os.path.join(DATA_ROOT,'mocap-vl-data.pkl'))
	elif flag=='test':
		torch.save(Xs[-5:],os.path.join(DATA_ROOT,'mocap-te-data.pkl'))

 
def gen_mocap_shift_data(data_path, params, flag, task='mocap_shift'):
	N = params[task][flag]['N']
	T = params[task][flag]['T']
	plot = True if flag=='train' else False
	DATA_ROOT = "data/mocap/"
	Xtr,Xts = [],[]
	min_len = 300
	fnames = os.listdir(DATA_ROOT)
	for fname in fnames:
		fname = os.path.join(DATA_ROOT,fname)
		if fname.endswith('amc') or fname.endswith('txt'):
			X = read_amc(fname, crop=True) # T,D
			# import matplotlib.pyplot as plt
			# fig, axs = plt.subplots(10,5,figsize=(40,20))
			# for j in range(50):
			# 	axs[j%10][j//10].plot(X[-min_len:,j])
			# print(fname)
			# plt.savefig(f'{fname}.png')
			# plt.close()
			if X.shape[0]>min_len:
				if '16_' in fname:
					Xts.append(X[-min_len::2])
				else:
					Xtr.append(X[-min_len::2])
			else:
				print(f'{fname} is skipped since too short')

	Xtr = torch.tensor(np.stack(Xtr),dtype=torch.float32) # 48,T,D
	Xts = torch.tensor(np.stack(Xts),dtype=torch.float32) # 8,T,D
	if flag=='train':
		torch.save(Xtr[:-5],os.path.join(DATA_ROOT,'mocap_shift-tr-data.pkl'))
	elif flag=='valid':
		torch.save(Xtr[-5:],os.path.join(DATA_ROOT,'mocap_shift-vl-data.pkl'))
	elif flag=='test':
		torch.save(Xts,os.path.join(DATA_ROOT,'mocap_shift-te-data.pkl'))