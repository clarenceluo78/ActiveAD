import numpy as np
import torch
from .strategy import Strategy

import collections
import sys
import math

from .random_sampling import RandomSampling
from .least_confidence import LeastConfidence
from .margin_sampling import MarginSampling
from .entropy_sampling import EntropySampling
from .least_confidence_dropout import LeastConfidenceDropout
from .margin_sampling_dropout import MarginSamplingDropout
from .entropy_sampling_dropout import EntropySamplingDropout
from .kmeans_sampling import KMeansSampling
from .kcenter_greedy import KCenterGreedy
from .kcenter_greedy_pca import KCenterGreedyPCA
from .bayesian_active_learning_disagreement_dropout import BALDDropout
from .badge_sampling import BadgeSampling
from .var_ratio import VarRatio
from .mean_std import MeanSTD

'''
This method allows multiple DAL approaches as basic approach.
use -a CEALSampling+[target approach name] like -a CEALSampling+EntropySampling.
Please cite the original paper if you use this method.
@article{wang2016cost,
  title={Cost-effective active learning for deep image classification},
  author={Wang, Keze and Zhang, Dongyu and Li, Ya and Zhang, Ruimao and Lin, Liang},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  volume={27},
  number={12},
  pages={2591--2600},
  year={2016},
  publisher={IEEE}
}
'''

class CEALSampling(Strategy):
	def __init__(self, dataset, delta=5 * 1e-5, n_drop=10):
		super(CEALSampling, self).__init__(dataset)
		self.n_drop = n_drop
		self.delta = delta

		self.strategy_list = []
		self.strategy_list.append(RandomSampling(dataset))
		self.strategy_list.append(LeastConfidence(dataset))
		self.strategy_list.append(MarginSampling(dataset))
		self.strategy_list.append(EntropySampling(dataset))
		self.strategy_list.append(LeastConfidenceDropout(dataset))
		self.strategy_list.append(MarginSamplingDropout(dataset))
		self.strategy_list.append(EntropySamplingDropout(dataset))
		# self.strategy_list.append(KMeansSampling(dataset))
		# self.strategy_list.append(KCenterGreedy(dataset))
		# self.strategy_list.append(KCenterGreedyPCA(dataset))
		self.strategy_list.append(BALDDropout(dataset))
		# self.strategy_list.append(BadgeSampling(dataset))
		self.strategy_list.append(VarRatio(dataset))
		self.strategy_list.append(MeanSTD(dataset))
		self.n_strategy = len(self.strategy_list)
		self.strategy_name = ['RandomSampling', 'LeastConfidence','MarginSampling','EntropySampling',
			'LeastConfidenceDropout','MarginSamplingDropout','EntropySamplingDropout','KMeansSampling',
			'KCenterGreedy','KCenterGreedyPCA','BALDDropout','CoreSet','CoreSetPCA','BadgeSampling',
			'BatchBALD', 'VarRatio', 'MeanSTD']
		self.strategy_dict = dict(zip(self.strategy_name, range(len(self.strategy_list))))

	def query(self, n, t=1, net=None, option = 'EntropySampling'):
		unlabeled_idxs, unlabeled_X, unlabeled_y= self.dataset.get_unlabeled_data()
		if option not in self.strategy_name:
			print('No support mode.')
			sys.exit()
		strategy_id = self.strategy_dict[option]
		# self.strategy_list[strategy_id].net.clf = self.net.clf
		q_idxs = self.strategy_list[strategy_id].query(n, net)
		self.delta = self.delta - 0.033 * 1e-5 * t

		# self.update(q_idxs)

		after_unlabeled_idxs, after_unlabeled_X, after_unlabeled_y = self.dataset.get_unlabeled_data()
		probs = net.predict_prob(after_unlabeled_X, y=after_unlabeled_y)
		pred_label = net.predict(after_unlabeled_X, y=after_unlabeled_y)

		entropy = (-1.0 * probs * np.log(probs)).sum(1)
		high_confident_idx = np.where(entropy < self.delta, True, False)
		real_idx = after_unlabeled_idxs[high_confident_idx]
		#print(collections.Counter(high_confident_idx))
		new_X = self.dataset.get_unlabeled_data_by_idx(high_confident_idx)
		new_Y = torch.tensor(np.array(pred_label)[high_confident_idx]).type(torch.int8)
		new_Y = torch.squeeze(new_Y)
		# new_Y = torch.tensor(np.array(pred_label)[high_confident_idx]).type(self.dataset.Y_train.dtype)
		labeled_idxs, labeled_X, labeled_Y = self.dataset.get_labeled_data()

		# labeled_X, labeled_Y = self.dataset.get_data_by_idx(labeled_idxs)
		all_X = cat_two(labeled_X, new_X)
		all_Y = cat_two(labeled_Y, new_Y)
		all_data = self.dataset.get_new_data(all_X, all_Y)
		# return q_idxs
		return q_idxs, all_data

def cat_two(x, y):
	if type(x) is np.ndarray:
		if type(y) is not np.ndarray:
			y = y.numpy().astype(x.dtype)
		return np.concatenate((x,y), axis = 0)
	else:
		if type(y) is np.ndarray:
			y = torch.from_numpy(y).type(x.dtype)
		return torch.cat((x, y), 0)
