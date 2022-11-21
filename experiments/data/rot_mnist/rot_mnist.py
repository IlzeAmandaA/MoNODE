import os
import scipy.io as sio
import numpy as np
import torch
from torch.utils import data
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from .utils import Dataset

def load_rotmnist_data(args, plot=True):
	fullname = os.path.join(args.data_root, "rot_mnist", "rot-mnist.mat")
	dataset = sio.loadmat(fullname)
	
	X = dataset['X'].squeeze()
	if args.mask:
		Y = dataset['Y'].suqeeze() 
		X = X[Y==args.value,:,:]

	N = args.Ntrain #train
	Nt = args.Nvalid + N # valid
	T = args.T #16
	Xtr   = torch.tensor(X[:N],dtype=torch.float32, device=args.device).view([N,T,1,28,28])
	Xtest = torch.tensor(X[N:Nt],dtype=torch.float32, device=args.device).view([-1,T,1,28,28])

	if args.rotrand:
		Xtr   = torch.cat([Xtr,Xtr[:,1:]],1) # N,2T,1,d,d
		Xtest = torch.cat([Xtest,Xtest[:,1:]],1) # N,2T,1,d,d
		t0s_tr   = torch.randint(0,T,[Xtr.shape[0]])
		t0s_test = torch.randint(0,T,[Xtest.shape[0]])
		Xtr   = torch.stack([Xtr[i,t0:t0+T]   for i,t0 in enumerate(t0s_tr)])
		Xtest = torch.stack([Xtest[i,t0:t0+T] for i,t0 in enumerate(t0s_test)])

	# Generators
	params = {'batch_size': args.batch, 'shuffle': True, 'num_workers': args.num_workers} #25
	trainset = Dataset(Xtr)
	trainset = data.DataLoader(trainset, **params)
	testset  = Dataset(Xtest)
	testset  = data.DataLoader(testset, **params)

	if plot:
		x = next(iter(trainset))
		plt.figure(1,(20,8))
		for j in range(6):
			for i in range(16):
				plt.subplot(7,20,j*20+i+1)
				plt.imshow(np.reshape(x[j,i,:],[28,28]), cmap='gray');
				plt.xticks([]); plt.yticks([])
		plt.savefig(os.path.join(args.save, 'plots/data.png'))
		plt.close()
	return trainset, testset