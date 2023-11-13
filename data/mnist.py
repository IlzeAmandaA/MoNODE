import numpy as np
from scipy.ndimage import rotate
from torchvision import datasets
import torch

class MNIST(object):
    def __init__(self, data_path, dtype) -> None:
        self.mnist = datasets.MNIST(data_path, train=True, download=True)
        self.dtype = dtype

    def _collate_fn(self, videos):
        """
        Collate function for the PyTorch data loader.

        Merges all batch videos in a tensor with shape (length, batch, channels, width, height) and converts their pixel
        values to [0, 1].

        Parameters
        ----------
        videos : list
            List of uint8 NumPy arrays representing videos with shape (length, batch, width, height, channels).

        Returns
        -------
        torch.*.Tensor
            Batch of videos with shape (batch,length,channels, width, height) and float values lying in [0, 1].
        """
        seq_len = len(videos[0])
        batch_size = len(videos)
        nc = 1 if videos[0].ndim == 3 else 3
        w = videos[0].shape[1]
        h = videos[0].shape[2]
        tensor = torch.zeros((batch_size,seq_len, nc, h, w), dtype = self.dtype)
        for i, video in enumerate(videos):
            if nc == 1:
               # tensor[:, i, 0] += torch.from_numpy(video)
                tensor[i,:,0]  += torch.from_numpy(video)
            if nc == 3:
                tensor[:, i] += torch.from_numpy(np.moveaxis(video, 3, 1))
        tensor = tensor.type(self.dtype) #tensor.float()
        tensor = tensor / 255
        return tensor

class RotatingMNIST(MNIST):
    def __init__(self, data_path, data_n, n_angles, digit=None, frame_size=28, dtype=torch.float64) -> None:
        super().__init__(data_path, dtype)
        self.digit = digit
        self.n_angles = n_angles
        self.data_n = data_n
        self.frame_size = frame_size

    def _sample_digit(self):
        if self.digit != 'None':
            self.data_digit_idx  = torch.where(self.mnist.targets == self.digit)
            data_digit_imgs = self.mnist.data[self.data_digit_idx]
        else:
            data_digit_imgs = self.mnist.data
        idxs = np.random.randint(0, data_digit_imgs.shape[0], self.data_n)
        self.data_digit_imgs = data_digit_imgs[idxs]
        self.labels = self.mnist.targets[idxs]

    def _gen_angles(self):
        angles = np.linspace(0, 2 * np.pi, self.n_angles)[1:]
        self.angles = np.rad2deg(angles)
    
    def _sample_rotation(self):
        """ 
        Rotate the input MNIST image in angles specified 
        Credits go to:
        Scalable Inference in SDEs by Direct Matching of the Fokker–Planck–Kolmogorov Equation
        Author: Solin, A.
        """
        rotated_imgs = np.array(self.data_digit_imgs).reshape((-1, 1, self.frame_size, self.frame_size))
        for a in self.angles:
            rotated_imgs = np.concatenate(
                (
                    rotated_imgs,
                    rotate(self.data_digit_imgs, a, axes=(1, 2), reshape=False).reshape((-1, 1, self.frame_size, self.frame_size)),
                ),
                axis=1,
            )
        return rotated_imgs

    def _gen_angles_stochastic(self):
        import math
        x0 = torch.rand([self.data_n]) * 30 - 15
        mu = torch.zeros_like(x0) # (torch.arange(1,6)-2).repeat(20,1).reshape(-1) * 3

        theta = 1.0
        sig   = math.sqrt(0.1) 

        def f(x):
            return theta*(x-mu) # + 5*theta*(math.pi*(x-mu)).sin()

        def g(x):
            return sig * math.sqrt(2*theta) * torch.randn_like(x)

        X  = [x0]
        dt = 0.01
        L = 10
        for t in range(L*self.n_angles-1):
            x_next = X[t] - dt*f(X[t]) + math.sqrt(dt)*g(X[t])
            X.append(x_next)
        
        self.angles = torch.stack(X[::L]).numpy().T 

    def _sample_rotation_sequentially(self):
        """ Rotate the input MNIST image in angles specified """
        # self.data_digit_imgs is [N,28,28]
        # self.angles is [N,T]
        self.data_digit_imgs = self.data_digit_imgs.numpy()
        rot_imgs = []
        for it,(img,angles) in enumerate(zip(self.data_digit_imgs,self.angles)):
            rot_imgs_ = []
            angles = angles / 2 / np.pi * 360
            # angles = angles - angles[0]
            for angle in angles:
                rot_img = rotate(img.reshape(1,28,28), angle, axes=(1, 2), reshape=False)[0]
                rot_imgs_.append(rot_img.copy())
            rot_imgs.append(np.stack(rot_imgs_))
        rot_imgs = np.stack(rot_imgs) # N,T,28,28
        return rot_imgs

