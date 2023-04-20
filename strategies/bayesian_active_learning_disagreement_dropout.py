import numpy as np
import torch
from .strategy import Strategy

class BALDDropout(Strategy):
    def __init__(self, dataset , n_drop=10):
        super(BALDDropout, self).__init__(dataset)
        self.n_drop = n_drop

    def query(self, n, net=None):
        unlabeled_idxs, unlabeled_data_X, unlabeled_data_y = self.dataset.get_unlabeled_data()
        probs = net.predict_prob_dropout_split(unlabeled_data_X, unlabeled_data_y, n_drop=self.n_drop, method="sigmoid", threshold_method="quantile", num=0.95)
        pb = probs.mean(0)
        entropy1 = (-pb*torch.log(pb)).sum(1)
        entropy2 = (-probs*torch.log(probs)).sum(2).mean(0)
        uncertainties = entropy2 - entropy1
        return unlabeled_idxs[uncertainties.sort()[1][:n]]
