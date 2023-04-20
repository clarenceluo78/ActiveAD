import numpy as np
import torch
from .strategy import Strategy
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

import sys
sys.path.append("..")

from data import ALDataset
'''
This implementation is with reference of https://github.com/Mephisto405/Learning-Loss-for-Active-Learning.
Please cite the original paper if you use this method.
@inproceedings{yoo2019learning,
  title={Learning loss for active learning},
  author={Yoo, Donggeun and Kweon, In So},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={93--102},
  year={2019}
}
'''
class LossPredictionLoss(Strategy):
	def __init__(self, dataset):
		super(LossPredictionLoss, self).__init__(dataset)
		self.batch_size = 1000

	def query(self, n, net=None):
		unlabeled_idxs, unlabeled_data_X, unlabeled_data_y = self.dataset.get_unlabeled_data()
		uncertainties = self.unc_lpl(unlabeled_data_X, unlabeled_data_y, net=net)
		return unlabeled_idxs[uncertainties.sort(descending=True)[1][:n]]

	def unc_lpl(self, X, y, net=None):
		data = ALDataset(X, y)
		loader = DataLoader(data, self.batch_size , shuffle=False)

		net.clf.eval()
		net.clf_lpl.eval()
		uncertainty = torch.tensor([]).cuda()
		with torch.no_grad():
			for x, y in loader:
				x, y = x.cuda(), y.cuda()
				out, feature = net.clf(x, return_embedding=True)
				pred_loss = net.clf_lpl(feature)
				pred_loss = pred_loss.view(pred_loss.size(0))
				uncertainty = torch.cat((uncertainty, pred_loss), 0)

		uncertainty = uncertainty.cpu()
		return uncertainty