import numpy as np
import torch
import torch.nn.functional as F
from .strategy import Strategy
from tqdm import tqdm

class AdversarialBIM(Strategy):
    def __init__(self, dataset, eps=0.05, max_iter=5000):
        super(AdversarialBIM, self).__init__(dataset)
        self.eps = eps
        self.max_iter = max_iter
        
    def query(self, n, net=None):
        unlabeled_idxs, unlabeled_data_X, unlabeled_data_y = self.dataset.get_unlabeled_data()

        net.model.cpu()
        net.model.eval()
        dis = np.zeros(unlabeled_idxs.shape)

        for i in tqdm(range(len(unlabeled_idxs)), ncols=100):
            x, y, idx = unlabeled_data_X[i], unlabeled_data_y[i], unlabeled_idxs[i]
            dis[i] = self.cal_dis(torch.from_numpy(x).float(), net)

        net.model.cuda()

        return unlabeled_idxs[dis.argsort()[:n]]
    

        
    def cal_dis(self, x, net=None):
        nx = torch.unsqueeze(x, 0)
        nx.requires_grad_()
        eta = torch.zeros(nx.shape)

        out, e1 = net.model(nx+eta, return_embedding=True)
        out = self.score_to_prob(out, net)
        py = out.max(1)[1]
        ny = out.max(1)[1]
        i_iter=0
        if out.max(1)[0]>=0.7:
            return np.inf
        while py.item() == ny.item() and i_iter < self.max_iter:
            loss = F.cross_entropy(out, ny)
            loss.backward()

            eta += self.eps * torch.sign(nx.grad.data)
            nx.grad.data.zero_()

            out, e1 = net.model(nx+eta, return_embedding=True)
            out = self.score_to_prob(out, net)
            py = out.max(1)[1]
            i_iter+=1

        return (eta*eta).sum()

    def score_to_prob(self, score, net):
        # one_prob = net.trans_prob(score, method="sigmoid")
        one_prob = score/10
        zero_prob = 1-one_prob
        probs = torch.cat((zero_prob,one_prob),1)
        return probs
        


