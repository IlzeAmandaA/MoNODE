import os
import yaml
import numpy as np
import torch
import torch.nn as nn
from   torch.utils import data
from data.data_gen import gen_sin_data, gen_lv_data, gen_rmnist_data, gen_rmnist_ou_data, gen_mmnist_data, gen_bb_data


from model.misc import io_utils

def _adjust_name(data_path, substr, insertion):
	idx = data_path.index(substr)
	return data_path[:idx] + '-' + insertion + data_path[idx:]


def load_data(args, device, dtype):
	if args.task in ['rot_mnist', 'rot_mnist_ou', 'mov_mnist', 'sin', 'lv', 'spiral', 'bb'] :
		(trainset, valset, testset), params = __load_data(args, device, dtype, args.task)
	else:
		return ValueError(r'Invalid task {arg.task}')
	return trainset, valset, testset, params  #, N, T, D, data_settings


def __load_data(args, device, dtype, dataset=None):
	#create data folder/files
	io_utils.makedirs(os.path.join(args.data_root,args.task))
	data_path_tr = os.path.join(args.data_root, args.task,f'{dataset}-tr-data.pkl')
	data_path_vl = os.path.join(args.data_root, args.task,f'{dataset}-vl-data.pkl')
	data_path_te = os.path.join(args.data_root, args.task,f'{dataset}-te-data.pkl')

	#load data parameters
	with open("data/config.yml", 'r') as stream:
		try:
			params = yaml.safe_load(stream)
		except yaml.YAMLError as exc:
			print(exc)

	#adjust name if specifc configuration
	if dataset == 'lv' and params[dataset]['clean']:
		data_path_tr = _adjust_name(data_path_tr, '.pkl', 'clean')
		data_path_vl = _adjust_name(data_path_vl, '.pkl', 'clean')
		data_path_te = _adjust_name(data_path_te, '.pkl', 'clean')
	elif dataset == 'bb':
		data_path_tr = _adjust_name(data_path_tr, '.pkl', str(params[dataset]['nballs']))
		data_path_vl = _adjust_name(data_path_vl, '.pkl', str(params[dataset]['nballs']))
		data_path_te = _adjust_name(data_path_te, '.pkl', str(params[dataset]['nballs']))

	#load or generate data
	try:
		Xtr = torch.load(data_path_tr)
		Xvl = torch.load(data_path_vl)
		Xte = torch.load(data_path_te)
		#if loaded data does not match the parameter settings assert and re generate the data 
		assert Xtr.shape[0] == params[dataset]['train']['N'] and Xtr.shape[1] == params[dataset]['train']['T'] 
		assert Xvl.shape[0] == params[dataset]['valid']['N'] and Xvl.shape[1] == params[dataset]['valid']['T']
		assert Xte.shape[0] == params[dataset]['test']['N'] and Xte.shape[1] == params[dataset]['test']['T']
			
	except:
		print(dataset)
		if dataset=='sin':
			data_loader_fnc = gen_sin_data
		elif dataset == 'lv':
			data_loader_fnc = gen_lv_data
		elif dataset == 'rot_mnist':
			data_loader_fnc = gen_rmnist_data
		elif dataset == 'rot_mnist_ou':
			data_loader_fnc = gen_rmnist_ou_data
		elif dataset == 'mov_mnist':
			data_loader_fnc = gen_mmnist_data
		elif dataset == 'bb':
			data_loader_fnc = gen_bb_data

		data_loader_fnc(data_path_tr, params, flag='train')
		data_loader_fnc(data_path_vl, params, flag = 'valid')
		data_loader_fnc(data_path_te, params, flag='test')

		Xtr = torch.load(data_path_tr)
		Xvl = torch.load(data_path_vl)
		Xte = torch.load(data_path_te)

	if dataset == 'bb':
		Xtr = torch.Tensor(Xtr).unsqueeze(2)
		Xvl = torch.Tensor(Xvl).unsqueeze(2)
		Xte = torch.Tensor(Xte).unsqueeze(2)

	Xtr = Xtr.to(device).to(dtype)
	Xvl = Xvl.to(device).to(dtype)
	Xte = Xte.to(device).to(dtype)

	return __build_dataset(args.num_workers, args.batch_size, Xtr, Xvl, Xte), params


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

