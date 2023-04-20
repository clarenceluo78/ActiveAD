import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.autograd import Variable, grad
from copy import deepcopy
from tqdm import tqdm
import torch.nn.init as init
from .DevNet import SemiADNet
import sys
sys.path.append("..")
from data import ALDataset
class waal_clf(nn.Module):
	def __init__(self, input_dim, embSize=320, num_classes=1):
		super(waal_clf, self).__init__()
		self.layer1 = nn.Sequential(
				nn.Linear(input_dim, embSize),
				nn.ReLU(),
				nn.Dropout(p=0.01)
		)
		self.layer2 = nn.Sequential(
				nn.Linear(embSize, 50),
				nn.ReLU(),
				nn.Dropout(p=0.01)
		)
		self.layer3 = nn.Linear(50, num_classes)
		self.embSize = embSize

	def forward(self, X, return_embedding=False):
		emb = self.layer1(X)
		X = self.layer2(emb)
		X = self.layer3(X)
		if return_embedding:
			return X, emb
		else:
			return X
		
	def get_embedding_dim(self):
		return self.embSize
	
class Net_WAAL:
	def __init__(self, model_name='WAAL', config=None, net_clf=None, net_dis=None):
		self.config = config
		self.batch_size = config['model_batch_size']
		self.nb_batch = 20
		self.device = torch.device("cuda" if config['use_cuda'] else "cpu")
		self.net_clf = net_clf if isinstance(net_clf, nn.Module) else waal_clf
		self.net_dis = net_dis if isinstance(net_dis, nn.Module) else Discriminator
		self.use_imbalence_train = False
  
	def fit(self, X_train, y_train, X_unlabeled=None, ratio=None, X_valid=None, y_valid=None, alpha=1e-3):
		n_epoch = self.config['n_epoch']
		dim = X_train.shape[1]
		outlier_indices = np.where(y_train == 1)[0]
		inlier_indices = np.where(y_train == 0)[0]

		self.clf = self.net_clf(input_dim=dim, num_classes=2).to(self.device)
		self.dis = self.net_dis(dim = self.clf.get_embedding_dim()).to(self.device)
		# setting three optimizers
		self.opt_clf = optim.Adam(self.clf.parameters(), lr=0.001, weight_decay=1e-5)
		self.opt_dis = optim.Adam(self.dis.parameters(), lr=0.001, weight_decay=1e-5)
		# computing the unbalancing ratio, a value betwwen [0,1]
		#gamma_ratio = X_labeled.shape[0]/X_unlabeled.shape[0]
		gamma_ratio = X_train.shape[0]/X_unlabeled.shape[0]
		print(gamma_ratio)
		self.clf.train()
		self.dis.train()
		if not self.use_imbalence_train:
			data = ALDataset(X_train, y_train)
			loader = DataLoader(data, self.batch_size , shuffle=True)

		for epoch in range(n_epoch):
			if self.use_imbalence_train:
				for i in range(self.nb_batch):
					label_x, label_y = self.input_batch_generation_sup(X_train, outlier_indices, inlier_indices, self.batch_size)
					label_x, label_y = label_x.to(self.device), label_y.to(self.device)
					unlabel_x =  self.sample_unlabeled(X_unlabeled, self.batch_size).to(self.device)
					self.train(label_x, label_y , unlabel_x, gamma_ratio=gamma_ratio,alpha=alpha)
			else:
				for batch_idx, (label_x, label_y) in enumerate(loader):
					label_y = label_y.unsqueeze(1)
					label_x, label_y = label_x.to(self.device), label_y.to(self.device)
					unlabel_x =  self.sample_unlabeled(X_unlabeled, len(label_x) ).to(self.device)
					self.train(label_x, label_y , unlabel_x, gamma_ratio=gamma_ratio,alpha=alpha)				

	def train(self, label_x, label_y , unlabel_x, gamma_ratio=None,alpha=1e-3):
		# training feature extractor and predictor
		self.set_requires_grad(self.clf,requires_grad=True)
		self.set_requires_grad(self.dis,requires_grad=False)
		
		self.opt_clf.zero_grad()
		lb_out, lb_z = self.clf(label_x, return_embedding=True)
		_, unlb_z = self.clf(unlabel_x, return_embedding=True)

		# prediction loss (deafult we use F.cross_entropy)
		zero_prob = 1-label_y
		gt_probs = torch.cat((zero_prob,label_y),1)

		pred_loss = torch.mean(F.cross_entropy(lb_out,gt_probs))
		# pred_loss = self.deviation_loss(label_y,  lb_out)

		# Wasserstein loss (unbalanced loss, used the redundant trick)
		wassertein_distance = self.dis(unlb_z).mean() - gamma_ratio * self.dis(lb_z).mean()

		with torch.no_grad():
			_, lb_z = self.clf(label_x, return_embedding=True)
			_, unlb_z = self.clf(unlabel_x, return_embedding=True)

		gp = self.gradient_penalty(self.dis, unlb_z, lb_z)
		loss = pred_loss + alpha * wassertein_distance + alpha * gp * 5
		# for CIFAR10 the gradient penality is 5
		# for SVHN the gradient penality is 2

		loss.backward()
		# clip_grad_norm_(self.clf.parameters(), 1.0)
		self.opt_clf.step()

		# Then the second step, training discriminator
		self.set_requires_grad(self.clf, requires_grad=False)
		self.set_requires_grad(self.dis, requires_grad=True)

		with torch.no_grad():
			_, lb_z = self.clf(label_x, return_embedding=True)
			_, unlb_z = self.clf(unlabel_x, return_embedding=True)

		for _ in range(1):
			# gradient ascent for multiple times like GANS training
			gp = self.gradient_penalty(self.dis, unlb_z, lb_z)

			wassertein_distance = self.dis(unlb_z).mean() - gamma_ratio * self.dis(lb_z).mean()

			dis_loss = -1 * alpha * wassertein_distance - alpha * gp * 2

			self.opt_dis.zero_grad()
			dis_loss.backward()
			self.opt_dis.step()

	def input_batch_generation_sup(self, X_train, outlier_indices, inlier_indices, batch_size):
		'''
		batchs of samples. This is for csv data.
		Alternates between positive and negative pairs.
		'''
		n_inliers = len(inlier_indices)
		n_outliers = len(outlier_indices)
		
		sample_num = batch_size//2
		
		inlier_idx = np.random.choice([i for i in range(n_inliers)], sample_num, replace=True)
		outlier_idx = np.random.choice([i for i in range(n_outliers)], sample_num, replace=True)
		
		sampled_X = np.concatenate((X_train[inlier_indices[inlier_idx]], X_train[outlier_indices[outlier_idx]]), axis=0)
		sampled_y = np.concatenate((np.expand_dims(np.zeros(sample_num), axis=1), np.expand_dims(np.ones(sample_num), axis=1)), axis=0)
		# print(sampled_X.shape)
		return torch.from_numpy(sampled_X).float(), torch.from_numpy(sampled_y).float()
	
	def sample_unlabeled(self, X_unlabeled, batch_size):
		# is_replace = True if len(X_unlabeled)<batch_size else False
		is_replace = True
		idx = np.random.choice([i for i in range(len(X_unlabeled))], batch_size, replace=is_replace)
		sampled = X_unlabeled[idx]
		return torch.from_numpy(sampled).float()
	
	def deviation_loss(self, y_true, y_pred):
		'''
		z-score-based deviation loss
		'''
		confidence_margin = 5.
		self.ref = torch.normal(mean=0., std=torch.full([5000], 1.)).cuda()
		dev = (y_pred - torch.mean(self.ref)) / torch.std(self.ref)
		inlier_loss = torch.abs(dev)
		outlier_loss = torch.abs((confidence_margin - dev).clamp_(min=0.))
		return torch.mean((1 - y_true) * inlier_loss + y_true * outlier_loss)

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


	
	def single_worst(self, probas):

		"""
		The single worst will return the max_{k} -log(proba[k]) for each sample
		:param probas:
		:return:  # unlabeled \times 1 (tensor float)
		"""

		value,_ = torch.max(-1*torch.log(probas),1)

		return value

	# setting gradient values
	def set_requires_grad(self, model, requires_grad=True):
		"""
		Used in training adversarial approach
		:param model:
		:param requires_grad:
		:return:
		"""

		for param in model.parameters():
			param.requires_grad = requires_grad


	# setting gradient penalty for sure the lipschitiz property
	def gradient_penalty(self, critic, h_s, h_t):
		''' Gradeitnt penalty approach'''
		alpha = torch.rand(h_s.size(0), 1).to(self.device)
		differences = h_t - h_s
		interpolates = h_s + (alpha * differences)
		interpolates = torch.cat([interpolates, h_s, h_t]).requires_grad_()
		# interpolates.requires_grad_()
		preds = critic(interpolates)
		gradients = grad(preds, interpolates,
						 grad_outputs=torch.ones_like(preds),
						 retain_graph=True, create_graph=True)[0]
		gradient_norm = gradients.norm(2, dim=1)
		gradient_penalty = ((gradient_norm - 1)**2).mean()

		return gradient_penalty 

	
	def get_model(self):
		return self.clf

	def get_embeddings(self, data):
		self.clf.eval()
		embeddings = torch.zeros([len(data), self.clf.get_embedding_dim()])
		loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
		with torch.no_grad():
			for x, y, idxs in loader:
				x, y = x.to(self.device), y.to(self.device)
				out, e1 = self.clf(x)
				embeddings[idxs] = e1.cpu()
		return embeddings
	
class Discriminator(nn.Module):
		"""Adversary architecture(Discriminator) for WAE-GAN."""
		def __init__(self, dim=20):
				super(Discriminator, self).__init__()
				self.dim = np.prod(dim)
				self.net = nn.Sequential(
						nn.Linear(self.dim, 512),
						nn.ReLU(True),
						nn.Linear(512, 512),
						nn.ReLU(True),
						nn.Linear(512,1),
						nn.Sigmoid(),
				)
				self.weight_init()

		def weight_init(self):
				for block in self._modules:
						for m in self._modules[block]:
								kaiming_init(m)

		def forward(self, z):
				return self.net(z).reshape(-1)

def kaiming_init(m):
		if isinstance(m, (nn.Linear, nn.Conv2d)):
				init.kaiming_normal_(m.weight)
				if m.bias is not None:
						m.bias.data.fill_(0)
		elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
				m.weight.data.fill_(1)
				if m.bias is not None:
						m.bias.data.fill_(0)

def normal_init(m, mean, std):
		if isinstance(m, (nn.Linear, nn.Conv2d)):
				m.weight.data.normal_(mean, std)
				if m.bias.data is not None:
						m.bias.data.zero_()
		elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
				m.weight.data.fill_(1)
				if m.bias.data is not None:
						m.bias.data.zero_()
