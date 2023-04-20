import numpy as np
from .strategy import Strategy

class MarginSampling(Strategy):
    def __init__(self, dataset):
        super(MarginSampling, self).__init__(dataset)

    def query(self, n, net=None):
        unlabeled_idxs, unlabeled_data_X, unlabeled_data_y = self.dataset.get_unlabeled_data()
        
        probs = net.predict_prob(unlabeled_data_X, method="sigmoid", threshold_method="quantile", num=0.95)
        probs_sorted, idxs = probs.sort(descending=True)
        uncertainties = probs_sorted[:, 0] - probs_sorted[:,1]
        return unlabeled_idxs[uncertainties.sort()[1][:n]]
