from data.rot_mnist.rot_mnist import load_rotmnist_data

def load_data(args, device):
	if args.task=='rot_mnist':
		trainset, testset = load_rotmnist_data(args, device)
	return trainset, testset #, N, T, D