import numpy as np
import torch
from .strategy import Strategy
from scipy.stats import mode

class VarRatio(Strategy):
    def __init__(self, dataset):
        super(VarRatio, self).__init__(dataset)

    def query(self, n, net=None):
        unlabeled_idxs, unlabeled_data_X, unlabeled_data_y = self.dataset.get_unlabeled_data()
        probs = net.predict_prob(unlabeled_data_X, method="sigmoid", threshold_method="quantile", num=0.95)
        preds = torch.max(probs, 1)[0]
        uncertainties = 1.0 - preds
        return unlabeled_idxs[uncertainties.sort(descending=True)[1][:n]]
