import numpy as np
from .strategy import Strategy

class RandomSampling(Strategy):
    def __init__(self, dataset):
        super(RandomSampling, self).__init__(dataset)

    def query(self, n, net=None):
        return np.random.choice(np.where(self.dataset.labeled_idxs==0)[0], n, replace=False)
