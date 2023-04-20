import numpy as np
import torch
import random
import os
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score
#   python demo.py -a RandomSampling -s 100 -q 1000 -b 100 -d CIFAR10 --seed 4666 -t 3 -g 0
class Data:
    def __init__(self, X_train, Y_train, X_test, Y_test):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

        self.n_pool = len(X_train)
        self.n_test = len(X_test)

        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)

    def initialize_labels(self, num):
        """generate initial labeled pool"""
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        if type(num) == int:  
            self.labeled_idxs[tmp_idxs[:num]] = True
        elif type(num) == float:
            init = int(num * self.n_pool) 
            init = init if init%2==0 else init+1
            self.labeled_idxs[:init] = True
        else:
            raise ValueError('initial labels should be int or float')
    
    def get_unlabeled_data_by_idx(self, idx):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return self.X_train[unlabeled_idxs][idx]
    
    def get_data_by_idx(self, idx):
        return self.X_train[idx], self.Y_train[idx]

    def get_new_data(self, X, Y):
        return X, Y

    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return labeled_idxs, self.X_train[labeled_idxs], self.Y_train[labeled_idxs]
    
    def get_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs]
    
    def get_train_data(self):
        return self.labeled_idxs.copy(), self.X_train, self.Y_train

    def get_test_data(self):
        return self.X_test, self.Y_test
    
    def get_partial_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return self.X_train[labeled_idxs], self.Y_train[labeled_idxs]
    
    def get_partial_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs]

    def cal_test_acc(self, preds):
        return 1.0 * (self.Y_test==preds).sum().item() / self.n_test

    def metric(self, y_score, pos_label=1):
        """Return metrics of anomaly detection"""
        aucroc = roc_auc_score(y_true=self.Y_test, y_score=y_score)
        aucpr = average_precision_score(y_true=self.Y_test, y_score=y_score, pos_label=1)
        return {'aucroc':aucroc, 'aucpr':aucpr}

class ALDataset(Dataset):
    def __init__(self, X, y):
        self.data = torch.from_numpy(X).float()
        self.target = torch.from_numpy(y).float()

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)