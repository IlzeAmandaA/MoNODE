from re import S
import numpy as np

class CachedHyperparametrs(object):
    """Stored the hyperparameter values of the GP"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.vals = []
        self.iters = []
    
    def update(self, val, iter):
        #print(val)
        self.vals.append(val)
       # print(self.vals)
        self.iters.append(iter)


class CachedRunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0
        self.vals = []
        self.iters = []

    def update(self, val, iter):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
        self.vals.append(val)
        self.iters.append(iter)

class CachedAverageMeter(object):
    """Computes and stores the average and current value over optimization iterations"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.std = 0
        self.vals = []
        self.iters = []

    def update(self, val, iter, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.iters.append(iter)