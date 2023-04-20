import numpy as np
from tqdm import tqdm
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from torch.autograd import Variable
from copy import deepcopy
# from keras import backend as K

class SemiADNet(nn.Module):
    def __init__(self, input_dim, embSize=20, num_classes=1):
        super(SemiADNet, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Linear(input_dim, embSize),
                nn.ReLU(),
                nn.Dropout(p=0.01)
        )
        self.layer2 = nn.Sequential(
                nn.Linear(embSize, num_classes),
                nn.ReLU(),
                nn.Dropout(p=0.01)
        )
        self.embSize = embSize

    def forward(self, X, return_embedding=False):
        emb = self.layer1(X)
        X= self.layer2(emb)
        if return_embedding:
            return X, emb
        else:
            return X
        
    def get_embedding_dim(self):
        return self.embSize

class DevNet():
    def __init__(self, model_name='DevNet', config=None, network_depth=2, num_classes=1):
        self.seed = config['seed']
        self.MAX_INT = np.iinfo(np.int32).max
        self.batch_size = config['model_batch_size']
        self.nb_batch = 20
        self.network_depth = network_depth
        self.epochs = config['n_epoch']
        self.patience = 50
        # self.patience = config['patience']
        self.disable_tqdm = config['disable_tqdm']
        self.device = torch.device("cuda" if config['use_cuda'] else "cpu")
        self.ref = None
        self.model = SemiADNet(input_dim=config['input_dim']).to(self.device)

        # self.criterion = build_criterion(args.criterion)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)


    def fit(self, X_train, y_train, X_unlabeled=None, ratio=None, X_valid=None, y_valid=None):
        # #index
        outlier_indices = np.where(y_train == 1)[0]
        inlier_indices = np.where(y_train == 0)[0]

        #start time
        self.input_shape = X_train.shape[1:]
        epochs = self.epochs
        batch_size = self.batch_size

        
        self.model.train()
        self.best_val = np.inf
        patience = self.patience

        for epoch in range(epochs):
            for i in range(self.nb_batch):
                data, y_true = self.input_batch_generation_sup(X_train, outlier_indices, inlier_indices, batch_size)
                self.train(data, y_true)

            if isinstance(X_valid, np.ndarray):
                val_metric = self.eval_on(X_valid, y_valid)
                print(f'Epoch {epoch}, validation metric {val_metric}')

                if val_metric < self.best_val:
                    self.best_val = val_metric
                    self.best_model = copy.deepcopy(self.model)
                    patience = self.patience
                else:
                    patience -= 1

                if patience == 0:
                    break
        

    def train(self, data, y_true):
        train_loss = 0.0
        
        data = data.to(self.device)
        y_true = y_true.to(self.device)

        y_pred = self.model(data)

        loss = self.deviation_loss(y_true , y_pred)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        train_loss += loss.item()
        # print(train_loss)
        # self.scheduler.step()

    def eval_on(self, X, y):
        self.model.eval()
        
        with torch.no_grad():
            X = torch.from_numpy(X).to(self.device)
            y_pred = self.model(X.float())
        loss = self.deviation_loss(torch.from_numpy(y).to(self.device) , y_pred)
        return loss
    
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
        
    def predict_score(self, X, y=None, return_threshold=False, quantile_num=0.95):
        self.model.eval()
        with torch.no_grad():
            X = torch.from_numpy(X).to(self.device)
            score = self.model(X.float()).cpu().detach().numpy()

        if return_threshold:
            print('quanitile:')
            print(np.quantile(score,[i/10 for i in range(0,11)]))
            threshold = np.quantile(score, quantile_num)
            return score, threshold
        else:
            return score

    def predict(self, X, y=None, threshold=0.5):
        score = self.predict_score(X)

        label = np.where(score > threshold, 1, 0)
        return label

    def trans_prob(self, probs, method="linear", threshold_method="quantile", num=0.95):
        # default: linear
        if method == "linear":
            new_probs = (probs - probs.min()) /(probs.max() - probs.min() ) 
        elif method == "sigmoid":
            if threshold_method == "quantile":
                if isinstance(probs, np.ndarray):
                    threshold = np.quantile(probs, num)
                    new_probs = 1/ (1+ np.exp(-(probs-threshold)))
                else:
                    threshold = torch.quantile(probs, num)
                    new_probs = 1/ (1+ torch.exp(-(probs-threshold)))
            else:
                threshold = num
                new_probs = 1/ (1+ np.exp(-(probs-threshold)))

            print('threshold:',threshold)

        else:
            raise NotImplementedError

        return new_probs

    def predict_prob(self, X, y=None, method="linear", threshold_method="quantile", num=0.95):
        one_prob = self.predict_score(X)
        # one_prob : [[0.1][0.2]...]
        one_prob = self.trans_prob(one_prob, method=method, threshold_method=threshold_method, num=num)

        one_prob = torch.from_numpy(one_prob)
        zero_prob = 1-one_prob
        probs = torch.cat((zero_prob,one_prob),1)
        return probs

    def predict_prob_dropout(self, X, y,  n_drop=10, method="linear", threshold_method="quantile", num=0.95):
        probs = torch.zeros([len(X), len(np.unique(y))])
        for i in range(n_drop):
            one_prob = self.predict_score(X)
            # one_prob : [[0.1][0.2]...]
            one_prob = self.trans_prob(one_prob, method=method, threshold_method=threshold_method, num=num)
            one_prob = torch.from_numpy(one_prob)
            zero_prob = 1-one_prob
            probs += torch.cat((zero_prob,one_prob),1)
        probs /= n_drop
        return probs
    
    def predict_prob_dropout_split(self, X, y, n_drop=10, method="linear", threshold_method="quantile", num=0.95):
        probs = torch.zeros([n_drop, len(X), len(np.unique(y))])
        # loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        for i in range(n_drop):
            one_prob = self.predict_score(X)
            # one_prob : [[0.1][0.2]...]
            
            one_prob = self.trans_prob(one_prob, method=method, threshold_method=threshold_method, num=num)
            one_prob = torch.from_numpy(one_prob)
            zero_prob = 1-one_prob
            probs[i] = torch.cat((zero_prob,one_prob),1)
        return probs
    
    def get_embeddings(self, X):
        self.model.eval()
        with torch.no_grad():
            X = torch.from_numpy(X).to(self.device)
            score, embeddings = self.model(X.float(),return_embedding=True)
       
        return embeddings.cpu()
    
    def get_grad_embeddings(self, X):
        self.model.eval()
        embDim = self.model.get_embedding_dim()
        # embeddings = np.zeros([len(data), embDim * nLab])
        with torch.no_grad():
            X = torch.from_numpy(X).to(self.device)
            score, embeddings = self.model(X.float(),return_embedding=True)
            score = score.cpu()
            embeddings = embeddings.cpu()
            
        one_prob = self.trans_prob(score.numpy(), method="linear")
        one_prob = torch.from_numpy(one_prob)
        zero_prob = 1 - one_prob
        
        one_embedding = deepcopy(embeddings) * (1 - one_prob) * -1.0
        zero_embedding = deepcopy(embeddings) * (-1 * zero_prob) * -1.0
        grad_embedding = np.concatenate((zero_embedding , one_embedding), axis=1)
        return grad_embedding

    def save_weights(self, filename):
        torch.save(self.model.state_dict(), os.path.join(args.experiment_dir, filename))

    def deviation_loss(self, y_true, y_pred):
        '''
        z-score-based deviation loss
        '''
        confidence_margin = 5.
        # size=5000 is the setting of l in algorithm 1 in the paper
        # if self.ref is None:
        #     self.ref = torch.normal(mean=0., std=torch.full([5000], 1.)).cuda()
        # else:
        #     self.ref = self.ref.cuda()
        self.ref = torch.normal(mean=0., std=torch.full([5000], 1.)).cuda()
        dev = (y_pred - torch.mean(self.ref)) / torch.std(self.ref)
        inlier_loss = torch.abs(dev)
        outlier_loss = torch.abs((confidence_margin - dev).clamp_(min=0.))
        return torch.mean((1 - y_true) * inlier_loss + y_true * outlier_loss)