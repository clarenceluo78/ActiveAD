import argparse
import numpy as np
import pandas as pd
import os
from data_generator import DataGenerator
from data import Data
from utils import get_strategy, dump_json
import typing as ty

from models.PyOD import PYOD
from models.DevNet import DevNet
from models.nets_waal import Net_WAAL
from models.nets_lpl import Net_LPL

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='data name', required=True)
parser.add_argument('--strategy', type=str, help='strategy name', required=True)
parser.add_argument('--model', type=str, help='model name', required=True)
parser.add_argument('--init_labeled', '-i', default=0.1, type=float, help='Initial pool of labeled data')
parser.add_argument('--budget', '-b', default=0.5, type=float, help='quota (budget) of active learning')
# parser.add_argument('--batch_size', default=64, type=int, help='batch size in one active learning iteration')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--is_dump', type=bool, help='whether dump the result to json', default=True)
from sklearn.metrics import f1_score
args = parser.parse_args()

# python pipeline.py --dataset 2_annthyroid --model DevNet --strategy LeastConfidence
# python pipeline.py --dataset 3_backdoor --model DevNet --strategy RandomSampling

class Pipeline():
    def __init__(self, args):
        model_dict = {'DevNet': DevNet, 'XGBOD': PYOD, 'WAAL' : Net_WAAL, 'LPL':Net_LPL}
        self.args = args
        self.config = {
            'dataset': args.dataset,
            'strategy': args.strategy,
            'model': args.model,
            'seed': args.seed,
            'disable_tqdm': False,
            'use_cuda' : True,
            'n_epoch' : 50
        }

        # generate data wrapper
        datagenerator = DataGenerator() # data generator
        datagenerator.dataset = self.config['dataset'] # specify the dataset name
        data = datagenerator.generator(la=None, realistic_synthetic_mode=None, noise_type=None)
        dataset = Data(X_train=data['X_train'], Y_train=data['y_train'], X_test=data['X_test'], Y_test=data['y_test'])
        dataset.initialize_labels(args.init_labeled)

        # update config here
        self.config['init_labeled'] = sum(dataset.labeled_idxs).item()
        self.config['budget'] = round(int(args.budget * len(dataset.X_train))/10) //2 * 2 *10  # round budget to nearest 10
        self.config['budget_ratio'] = args.budget
        self.config['batch_size'] = self.config['budget'] // 10 if self.config['budget'] > 320 else 32  #TODO
        self.config['model_batch_size'] = 512  #TODO
        self.config['input_dim'] = data['X_train'].shape[1]
        self.config['train_size'] = len(dataset.X_train)
        self.config['test_size'] = len(dataset.X_test)
        self.config['all_train_outliers'] = self.outlier_count(dataset.Y_train)
        print('Train size: %d, Test size: %d' % (dataset.X_train.shape[0], dataset.X_test.shape[0]))
        print('Initial labeled: %d, Budget: %d, Batch size: %d' % (self.config['init_labeled'], self.config['budget'], self.config['batch_size']))
        
        # Note that dataset inputted to strategy and the
        # dataset used in pipeline (for performance test) are different
        self.dataset = dataset 
        
        # get base model and strategy
        model = model_dict.get(self.config['model'])
        strategy = get_strategy(self.config['strategy'], dataset=dataset)  # load strategy
        self.model = model(model_name=self.config['model'], config=self.config)
        self.strategy = strategy
        
        # get output directory
        self.output_dir = f'output/{self.config["dataset"]}/{self.config["strategy"]}/{self.config["model"]}/{self.config["budget_ratio"]}'
        
    #TODO: put this function to utils or data?
    def outlier_count(self, y):
        outlier_indices = np.where(y == 1)[0]
        inlier_indices = np.where(y == 0)[0]
        n_outliers = len(outlier_indices)
        return n_outliers

    def train_model(self):
        _ ,X_train, y_train =  self.dataset.get_labeled_data()
        _ ,X_unlabeled, y_unlabeled = self.dataset.get_unlabeled_data()
        n_outliers = self.outlier_count(y_train)
        print("Training size: %d, No. outliers: %d" % (X_train.shape[0], n_outliers))
        
        # change Devnet to semi-supervised mode
        if self.config['model'] == 'DevNet':
            # _ ,X_unlabeled, y_unlabeled = self.dataset.get_unlabeled_data()
            X_train = np.concatenate((X_train, X_unlabeled), axis=0)
            y_train = np.concatenate((y_train, np.zeros(len(y_unlabeled))), axis=0)

        # train base model
        self.model.fit(X_train, y_train, X_unlabeled=X_unlabeled)
        
        # test model and return metrics
        X_test, y_test = self.dataset.get_test_data()
        preds, threshold = self.model.predict_score(X_test, return_threshold=True)
        print('threshold:',threshold)
        metrics = self.dataset.metric(preds)
        print('metrics:', metrics)
        
        pred_label = self.model.predict(X_test, threshold=threshold)
        f1 = f1_score(y_test, pred_label, average=None)
        print('f1-score',f1)
        
        # return metrics in the process
        self.results.append([X_train.shape[0], n_outliers, self.config['batch_size'], n_outliers-self.last_outlier_count, 
                             round(metrics['aucroc'],4), round(metrics['aucpr'],4), np.array2string(f1.round(4), separator=',',suppress_small=True)])
        self.last_outlier_count = n_outliers
        
    
    def train(self):
        self.results = []
        self.last_outlier_count = 0
        round = self.config['budget'] // self.config['batch_size']
        
        self.train_model()  # train test model on initial labeled data
        
        # round 1 to rd
        for rd in range(1, round+1):
            print('Round {}'.format(rd))
            high_confident_idx = []
            high_confident_pseudo_label = []
            # query
            q_idxs = self.strategy.query(self.config['batch_size'], net=self.model)
            self.strategy.update(q_idxs)
            self.train_model()

        df = pd.DataFrame(self.results, columns=['all_X', 'all_outliers', 'new_X','new_outliers' ,'aucroc','aucpr','f1']) 
        print(df)
        
        # dump result to json
        stats: ty.Dict[str, ty.Any] = {
            'process': {
                f'round_{i}' : self.results[i] for i in range(len(self.results))
            },
            'metrics': {
                'F-aucroc': df.iloc[-1]['aucroc'],
                'F_aucpr': df.iloc[-1]['aucpr'],
                'F_f1': df.iloc[-1]['f1'],
            },
        }
        self.config.update(stats)
        

    def dump(self, is_dump=True):
        if is_dump:
            os.system('mkdir -p output')
            os.system(f'mkdir -p {self.output_dir}')
            dump_json(self.config, f'{self.output_dir}/stats.json', indent=4)
        else:
            pass

if __name__ == '__main__':
    pipe = Pipeline(args)
    pipe.train()
    pipe.dump(is_dump=args.is_dump)