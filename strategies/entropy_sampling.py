import numpy as np
import torch
from .strategy import Strategy

class EntropySampling(Strategy):
    def __init__(self, dataset):
        super(EntropySampling, self).__init__(dataset)

    def query(self, n, net=None):
        unlabeled_idxs, unlabeled_data_X, unlabeled_data_y = self.dataset.get_unlabeled_data()
        probs = net.predict_prob(unlabeled_data_X, method="sigmoid", threshold_method="quantile", num=0.95)
        log_probs = torch.log(probs)
        uncertainties = (probs*log_probs).sum(1)
        return unlabeled_idxs[uncertainties.sort()[1][:n]]
