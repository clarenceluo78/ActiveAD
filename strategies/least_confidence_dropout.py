import numpy as np
from .strategy import Strategy

class LeastConfidenceDropout(Strategy):
    def __init__(self, dataset, n_drop=10):
        super(LeastConfidenceDropout, self).__init__(dataset)
        self.n_drop = n_drop

    def query(self, n, net=None):
        unlabeled_idxs, unlabeled_data_X, unlabeled_data_y = self.dataset.get_unlabeled_data()
        probs = net.predict_prob_dropout(unlabeled_data_X, unlabeled_data_y, n_drop=self.n_drop, method="sigmoid", threshold_method="quantile", num=0.95)
        uncertainties = probs.max(1)[0]
        return unlabeled_idxs[uncertainties.sort()[1][:n]]
