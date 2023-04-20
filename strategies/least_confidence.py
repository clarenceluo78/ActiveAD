import numpy as np
import torch
from .strategy import Strategy

class LeastConfidence(Strategy):
    def __init__(self, dataset):
        super(LeastConfidence, self).__init__(dataset)

    def query(self, n, net=None):
        unlabeled_idxs, unlabeled_data_X, unlabeled_data_y = self.dataset.get_unlabeled_data()
        # probs = net.predict_prob(unlabeled_data_X)
        # uncertainties = probs.max(1)[0]
        # print('uncertainties',uncertainties)
        
        scores, threshold = net.predict_score(unlabeled_data_X, return_threshold=True, quantile_num=0.95)
        uncertainties = torch.from_numpy(np.squeeze(np.absolute(scores-threshold)))
        return unlabeled_idxs[uncertainties.sort()[1][:n]]
