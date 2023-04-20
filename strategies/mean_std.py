import numpy as np
import torch
from .strategy import Strategy

class MeanSTD(Strategy):
    def __init__(self, dataset, n_drop = 10):
        super(MeanSTD, self).__init__(dataset)
        self.n_drop = n_drop

    def query(self, n, net=None):
        unlabeled_idxs, unlabeled_data_X, unlabeled_data_y = self.dataset.get_unlabeled_data()
        probs = net.predict_prob_dropout_split(unlabeled_data_X, unlabeled_data_y, n_drop=self.n_drop, method="sigmoid", threshold_method="quantile", num=0.95).numpy()
        sigma_c = np.std(probs, axis=0)
        uncertainties = torch.from_numpy(np.mean(sigma_c, axis=-1))
        return unlabeled_idxs[uncertainties.sort(descending=True)[1][:n]]
