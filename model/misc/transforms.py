from model.misc.settings import settings

import numpy as np
import torch
import torch.nn.functional as F


def invsoftplus(x):
    lower = 1e-12
    xs = torch.max(x - lower, torch.tensor(torch.finfo(x.dtype).eps).to(x))
    return xs + torch.log(-(torch.exp(-xs)-1))

def softplus(x):
    return F.softplus(x)

class Identity:
    def __init__(self):
        pass

    def __str__(self):
        return 'Identity transformation'

    def forward_tensor(self, x):
        return x

    def backward_tensor(self, y):
        return y

    def forward(self, x):
        return x

    def backward(self, y):
        return y


class SoftPlus:
    def __init__(self, lower=1e-12):
        self._lower = lower

    def __str__(self):
        return 'Softplus transformation'

    def forward(self, x):
        return np.logaddexp(0, x) + self._lower

    def forward_tensor(self, x):
        return F.softplus(x) + self._lower

    def backward_tensor(self, y):
        ys = torch.max(y - self._lower, torch.tensor(torch.finfo(y.dtype).eps).to(y))
        return ys + torch.log(-torch.expm1(-ys))

    def backward(self, y):
        ys = np.maximum(y - self._lower, np.finfo(settings.numpy_float).eps)
        return ys + np.log(-np.expm1(-ys))


class LowerTriangular:
    def __init__(self, N, num_matrices=1, device='cpu', dtype=torch.float32):
        self.N = N
        self.num_matrices = num_matrices  # We need to store this for reconstruction.
        self.device = device
        self.dtype  = dtype

    def __str__(self):
        return 'Lower cholesky transformation'

    def forward(self, x):
        fwd = torch.zeros(self.num_matrices, self.N, self.N, dtype=self.dtype).numpy()
        # fwd = np.zeros((self.num_matrices, self.N, self.N), dtype=self.dtype)
        indices = np.tril_indices(self.N, 0)
        z = np.zeros(len(indices[0])).astype(int)
        for i in range(self.num_matrices):
            fwd[(z + i,) + indices] = x[i, :]
        return fwd

    def backward(self, y):
        ind = np.tril_indices(self.N)
        return np.vstack([y_i[ind] for y_i in y])

    def forward_tensor(self, x):
        fwd = torch.zeros((self.num_matrices, self.N, self.N), device=self.device, dtype=self.dtype)
        indices = np.tril_indices(self.N, 0)
        z = np.zeros(len(indices[0])).astype(int)
        for i in range(self.num_matrices):
            fwd[(z + i,) + indices] = x[i, :]
        return fwd

    def backward_tensor(self, y):
        ind = np.tril_indices(self.N)
        return torch.stack([y_i[ind] for y_i in y])


class StackedLowerTriangular:
    def __init__(self, N, num_n, num_m, device='cpu', dtype=torch.float32):
        self.N = N
        self.num_n = num_n  # We need to store this for reconstruction.
        self.num_m = num_m
        self.device = device
        self.dtype  = dtype

    def __str__(self):
        return 'Lower cholesky transformation for stack sequence of covariance matrices'

    def forward(self, x):
        fwd = np.zeros((self.num_n, self.num_m, self.N, self.N), dtype=settings.numpy_float)
        indices = np.tril_indices(self.N, 0)
        z = np.zeros(len(indices[0])).astype(int)
        for i in range(self.num_n):
            for j in range(self.num_m):
                fwd[(z + i, z + j,) + indices] = x[i, j, :]
        return fwd

    def backward(self, y):
        ind = np.tril_indices(self.N)
        return np.stack([np.stack([y_i[ind] for y_i in y_j]) for y_j in y])

    def forward_tensor(self, x):
        fwd = torch.zeros((self.num_n, self.num_m, self.N, self.N), device=self.device, dtype=self.dtype)
        indices = np.tril_indices(self.N, 0)
        z = np.zeros(len(indices[0])).astype(int)
        for i in range(self.num_n):
            for j in range(self.num_m):
                fwd[(z + i, z + j,) + indices] = x[i, j, :]
        return fwd

    def backward_tensor(self, y):
        ind = np.tril_indices(self.N)
        return torch.stack([torch.stack([y_i[ind] for y_i in y_j]) for y_j in y])
