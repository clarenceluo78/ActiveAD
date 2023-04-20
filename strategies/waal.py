import numpy as np
from .strategy import Strategy
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append("..")

from data import ALDataset

'''
This implementation is with reference of https://github.com/cjshui/WAAL.
Please cite the original paper if you plan to use this method.
@inproceedings{shui2020deep,
  title={Deep active learning: Unified and principled method for query and training},
  author={Shui, Changjian and Zhou, Fan and Gagn{\'e}, Christian and Wang, Boyu},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={1308--1318},
  year={2020},
  organization={PMLR}
}
'''
class WAAL(Strategy):
	def __init__(self, dataset):
		super(WAAL, self).__init__(dataset)
		self.selection = 10
		self.batch_size = 1000

	def query(self, n, net=None):
		unlabeled_idxs, unlabeled_data_X, unlabeled_data_y = self.dataset.get_unlabeled_data()
		probs = net.predict_prob(unlabeled_data_X)

		uncertainly_score = 0.5* self.L2_upper(probs) + 0.5* self.L1_upper(probs)
		# prediction output discriminative score
		dis_score = self.pred_dis_score_waal(unlabeled_data_X, unlabeled_data_y, net=net)
		# computing the decision score
		total_score = uncertainly_score - self.selection * dis_score
		b = total_score.sort()[1][:n]

		return unlabeled_idxs[total_score.sort()[1][:n]]

	def L2_upper(self, probas):
		value = torch.norm(torch.log(probas),dim=1)
		return value


	def L1_upper(self, probas):
		value = torch.sum(-1*torch.log(probas),dim=1)
		return value

	def pred_dis_score_waal(self, X, y, net=None):
		data = ALDataset(X, y)
				
		loader_te = DataLoader(data, self.batch_size , shuffle=False)

		net.clf.eval()
		net.dis.eval()

		with torch.no_grad():
			for batch_id, (x, y) in enumerate(loader_te):
				x, y = x.cuda(), y.cuda()
				_, latent = net.clf(x, return_embedding=True)
				out = net.dis(latent).cpu()
				scores = out if batch_id == 0 else torch.cat((scores, out), 0)

		return scores
