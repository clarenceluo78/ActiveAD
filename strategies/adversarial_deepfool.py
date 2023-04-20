import numpy as np
import torch
import torch.nn.functional as F
from .strategy import Strategy
from tqdm import tqdm

class AdversarialDeepFool(Strategy):
    def __init__(self, dataset, max_iter=50):
        super(AdversarialDeepFool, self).__init__(dataset)
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
        n_class = out.shape[1]
        py = out.max(1)[1].item()
        ny = out.max(1)[1].item()

        i_iter = 0

        while py == ny and i_iter < self.max_iter:
            out[0, py].backward(retain_graph=True)
            grad_np = nx.grad.data.clone()
            value_l = np.inf
            ri = None

            for i in range(n_class):
                if i == py:
                    continue

                nx.grad.data.zero_()
                out[0, i].backward(retain_graph=True)
                grad_i = nx.grad.data.clone()

                wi = grad_i - grad_np
                fi = out[0, i] - out[0, py]
                value_i = np.abs(fi.item()) / np.linalg.norm(wi.numpy().flatten())
                ri = value_i/np.linalg.norm(wi.numpy().flatten()) * wi
                # if value_i < value_l:
                #     ri = value_i/np.linalg.norm(wi.numpy().flatten()) * wi
                
            if isinstance(ri, torch.Tensor):
                eta += ri.clone()
            nx.grad.data.zero_()
            out, e1 = net.model(nx+eta, return_embedding=True)
            out = self.score_to_prob(out, net)
            py = out.max(1)[1].item()
            i_iter += 1

        return (eta*eta).sum()
    
    def score_to_prob(self, score, net):
        # one_prob = net.trans_prob(score, method="sigmoid")
        one_prob = score/10
        zero_prob = 1-one_prob
        probs = torch.cat((zero_prob,one_prob),1)
        return probs




