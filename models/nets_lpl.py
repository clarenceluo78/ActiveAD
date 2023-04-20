import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.autograd import Variable
from copy import deepcopy
from tqdm import tqdm
import sys
sys.path.append("..")
from data import ALDataset
# LossPredictionLoss
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    
	assert len(input) % 2 == 0, 'the batch size is not even.'
	assert input.shape == input.flip(0).shape

	input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], batch size = 2B
	target = (target - target.flip(0))[:len(target)//2]
	target = target.detach()

	one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors
	
	if reduction == 'mean':
		loss = torch.sum(torch.clamp(margin - one * input, min=0))
		loss = loss / input.size(0) # Note that the size of input is already haved
	elif reduction == 'none':
		loss = torch.clamp(margin - one * input, min=0)
	else:
		NotImplementedError()
	
	return loss

class Net_LPL:
	def __init__(self, model_name='LPL', config=None, net=None, net_lpl=None):
		self.config = config
		self.config['optimizer'] = 'Adam'
		self.batch_size = config['model_batch_size']
		self.nb_batch = 20
		self.net = net if isinstance(net, nn.Module) else lpl
		self.net_lpl = net_lpl if isinstance(net_lpl, nn.Module) else LossNet
		self.device = torch.device("cuda" if config['use_cuda'] else "cpu")
		
	def fit(self, X_train, y_train,  X_unlabeled=None, weight = 1.0, margin = 1.0 , lpl_epoch = 20):
		# n_epoch = self.config['n_epoch']
		n_epoch = lpl_epoch + self.config['n_epoch']
		epoch_loss = lpl_epoch

		dim = X_train.shape[1]
		self.clf = self.net(input_dim = dim, num_classes=2).to(self.device)
		self.clf_lpl = self.net_lpl().to(self.device)

		#self.clf.train()
		if self.config['optimizer'] == 'Adam':
			optimizer = optim.Adam(self.clf.parameters(), lr=0.001, weight_decay=1e-5)
		elif self.config['optimizer'] == 'SGD':
			optimizer = optim.SGD(self.clf.parameters(), lr=0.001, weight_decay=1e-5)
		else:
			raise NotImplementedError
		
		optimizer_lpl = optim.Adam(self.clf_lpl.parameters(), lr = 0.01)


		data = ALDataset(X_train, y_train)
		loader = DataLoader(data, self.batch_size , shuffle=True)
 
		self.clf.train()
		self.clf_lpl.train()
		for epoch in tqdm(range(1, n_epoch+1), ncols=100):
			for batch_idx, (x, y) in enumerate(loader):
				x, y = x.to(self.device), y.to(self.device)
				optimizer.zero_grad()
				optimizer_lpl.zero_grad()
				out, feature = self.clf(x, return_embedding=True)
				out, e1 = self.clf(x, return_embedding=True)
				y=y.unsqueeze(1)
				zero_prob = 1-y
				gt_probs = torch.cat((zero_prob,y),1)
				cross_ent = nn.CrossEntropyLoss(reduction='none')
				target_loss = cross_ent(out,gt_probs)
				if epoch >= epoch_loss:
					feature[0] = feature[0].detach()
					feature[1] = feature[1].detach()
					feature[2] = feature[2].detach()
					feature[3] = feature[3].detach()
				pred_loss = self.clf_lpl(feature)
				pred_loss = pred_loss.view(pred_loss.size(0))

				backbone_loss = torch.sum(target_loss) / target_loss.size(0)
				module_loss = LossPredLoss(pred_loss, target_loss, margin)
				loss = backbone_loss + weight * module_loss
				loss.backward()
				optimizer.step()
				optimizer_lpl.step()

	def predict_prob(self, X, y=None, method="linear", threshold_method="quantile", num=0.95):
		self.clf.eval()
		with torch.no_grad():
			X = torch.from_numpy(X).to(self.device)
			out = self.clf(X.float())
			prob = F.softmax(out, dim=1).cpu().detach()
		return prob

	def predict_score(self, X, y=None, return_threshold=False, quantile_num=0.95):
		prob = self.predict_prob(X).numpy()
		score = prob[:, 1]
		if return_threshold:
			print('quanitile:')
			print(np.quantile(score,[i/10 for i in range(0,11)]))
			threshold = np.quantile(score, quantile_num)
			return score, threshold
		else:
			return score

	def predict(self, X, y=None, threshold=0.5):
		prob = self.predict_prob(X)
		label = prob.max(1)[1]
		return label

	def predict_prob_dropout(self, X, y=None, n_drop=10):
		probs = torch.zeros([len(X), len(np.unique(y))])
		for i in range(n_drop):
			probs += self.predict_prob(X)
		probs /= n_drop
		return probs
	
	def predict_prob_dropout_split(self, data, n_drop=10):
		probs = torch.zeros([n_drop, len(X), len(np.unique(y))])
		for i in range(n_drop):
			probs[i] = self.predict_prob(X)
		return probs
	
	def get_model(self):
		return self.clf


class lpl(nn.Module):
	def __init__(self, input_dim, embSize=[512, 256, 128, 64], num_classes=1):
		# embSize=[64, 32, 16, 8]
		super(lpl, self).__init__()
		self.input_layer = nn.Sequential(
				nn.Linear(input_dim, embSize[0]),
				nn.ReLU(),
				nn.Dropout(p=0.01)
		)
		self.blocks = nn.ModuleList([
			nn.Sequential(
				nn.Linear(embSize[l], embSize[l+1]),
				nn.ReLU(),
				nn.Dropout(p=0.01)
			) for l in range(len(embSize)-1)
		])	  
		self.clf = nn.Linear(embSize[-1], num_classes)
		self.emb_dim= embSize[0]

	def forward(self, x, return_embedding=False):
		embbedings = []
		x = self.input_layer(x)
		embbedings.append(x)
		for layer in self.blocks:
			x = layer(x)
			embbedings.append(x)

		output = self.clf(x)

		if return_embedding:
			return output, embbedings
		else:
			return output
		
	def get_embedding_dim(self):
		return self.emb_dim



class LossNet(nn.Module):
	def __init__(self, embSize=[512, 256, 128, 64], interm_dim=128):
		super(LossNet, self).__init__()

		self.blocks = nn.ModuleList([
			nn.Sequential(
				nn.Linear(embSize[l], interm_dim),
				nn.ReLU(),
				nn.Dropout(p=0.01)
			) for l in range(len(embSize))
		])

		self.linear = nn.Linear(4 * interm_dim, 1)
	
	def forward(self, features):
		n=0
		embs=[]
		for layer in self.blocks:
			x = layer(features[n])
			embs.append(x)
			n+=1
		out = self.linear(torch.cat(embs, 1))
		return out


# def get_lossnet(name):
# 	if name == 'PneumoniaMNIST':
# 		return LossNet(feature_sizes=[224, 112, 56, 28], num_channels=[64, 128, 256, 512], interm_dim=128)
# 	elif 'MNIST' in name:
# 		return LossNet(feature_sizes=[14, 7, 4, 2], num_channels=[64, 128, 256, 512], interm_dim=128) 
# 	elif 'CIFAR' in name:
# 		return LossNet(feature_sizes=[32, 16, 8, 4], num_channels=[64, 128, 256, 512], interm_dim=128)
# 	elif 'ImageNet' in name:
# 		return LossNet(feature_sizes=[64, 32, 16, 8], num_channels=[64, 128, 256, 512], interm_dim=128)
# 	elif 'BreakHis' in name:
# 		return LossNet(feature_sizes=[224, 112, 56, 28], num_channels=[64, 128, 256, 512], interm_dim=128)
# 	elif 'waterbirds' in name:
# 		return LossNet(feature_sizes=[128, 64, 32, 16], num_channels=[64, 128, 256, 512], interm_dim=128)
# 	else:
# 		raise NotImplementedError

