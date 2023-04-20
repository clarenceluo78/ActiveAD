import numpy as np
from .strategy import Strategy
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from tqdm import tqdm

class KCenterGreedyPCA(Strategy):
    def __init__(self, dataset):
        super(KCenterGreedyPCA, self).__init__(dataset)

    def query(self, n, net=None):
        labeled_idxs, train_data_X, train_data_y = self.dataset.get_train_data()
        embeddings = net.get_embeddings(train_data_X).numpy()

        #downsampling embeddings if feature dim > 50
        if len(embeddings[0]) > 50:
            pca = PCA(n_components=50)
            embeddings = pca.fit_transform(embeddings)
        embeddings = embeddings.astype(np.float16)

        dist_mat = np.matmul(embeddings, embeddings.transpose())
        sq = np.array(dist_mat.diagonal()).reshape(len(labeled_idxs), 1)
        dist_mat *= -2
        dist_mat += sq
        dist_mat += sq.transpose()
        dist_mat = np.sqrt(dist_mat)

        mat = dist_mat[~labeled_idxs, :][:, labeled_idxs]

        for i in tqdm(range(n), ncols=100):
            mat_min = mat.min(axis=1)
            q_idx_ = mat_min.argmax()
            q_idx = np.arange(self.dataset.n_pool)[~labeled_idxs][q_idx_]
            labeled_idxs[q_idx] = True
            mat = np.delete(mat, q_idx_, 0)
            mat = np.append(mat, dist_mat[~labeled_idxs, q_idx][:, None], axis=1)
            
        return np.arange(self.dataset.n_pool)[(self.dataset.labeled_idxs ^ labeled_idxs)]
