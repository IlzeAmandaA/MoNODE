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
    """Computes and stores the weighted moving average (WMA) and current value over optimization iterations"""

    def __init__(self, period=10):
        self.period = period
        self.compute_weights()
        self.reset()

    def compute_weights(self):
        normalize = (self.period * (self.period + 1))//2
        self.weights = np.array([self.period - t for t in range(self.period)])/normalize

    def reset(self):
        self.val = None
        self.avg = 0
        self.vals = np.array([])
        self.iters = []

    def update(self, val, iter):
        if self.val is None:
            self.avg = val
        elif len(self.vals)<self.period:
            self.avg = np.mean(np.array(self.vals))
        else:
            self.avg = np.average(np.flip(self.vals[-self.period:]),weights=self.weights)
        self.val = val
        self.vals = np.append(self.vals, val)
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