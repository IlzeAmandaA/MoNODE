from data.rot_mnist.rot_mnist import load_rotmnist_data

def load_data(args):
	if args.task=='rot_mnist':
		trainset, testset = load_rotmnist_data(args)
	return trainset, testset #, N, T, D