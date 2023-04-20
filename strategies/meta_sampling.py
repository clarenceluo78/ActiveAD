from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from .strategy import Strategy
import os
import time
from .meta_aad.env import make_train_env, make_eval_env
from .meta_aad.ppo import PPO, evaluate_policy  # pytorch version
from .meta_aad.utils import generate_csv_writer
# from .meta_aad.ppo2 import PPO2, evaluate_policy  # tf version
# from stable_baselines.common.vec_env import VecEnv

class MetaSampling(Strategy):
    def __init__(self, dataset):
        super(MetaSampling, self).__init__(dataset)
        self.is_trained = False
        self.is_first_eval = True
        self.eval_env = make_eval_env(self.dataset, is_train=False)

    def train(self):
        env = make_train_env(self.dataset)
        
        #TODO: add to args or config
        anomaly_curve_log = os.path.join('log', 'anomaly_curves')
        eval_log_interval = 100
        total_timesteps = 200000
        log_interval = 10
        
        #TODO: visualize anomaly curve
        # train_eval_env = make_eval_env(self.dataset, is_train=True)
        
        model = PPO('MlpPolicy', env, verbose=1)
        # model.set_eval(train_eval_env, eval_log_interval)
        model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
        model_log = os.path.join('model_log')
        if not os.path.exists(model_log):
            os.makedirs(model_log)
            
        #TODO: currently just save the last model
        model.save(os.path.join('model_log', 'model'))
        
    def evaluate(self, model, env, n_eval_episodes=1, eval_interval=64, deterministic=True,
                    render=False, callback=None, reward_threshold=None,
                    return_episode_rewards=False, use_batch=True, is_first_eval=True):
        """Evaluate meta policy's sampling performance in each batch

        Args:
            model (_type_): trained PPO model
            env (_type_): self-defined environment
            n_eval_episodes (int, optional): _description_. Defaults to 1.
            eval_interval (int, optional): _description_. Defaults to 64.
            deterministic (bool, optional): _description_. Defaults to True.
            render (bool, optional): _description_. Defaults to False.
            callback (_type_, optional): _description_. Defaults to None.
            reward_threshold (_type_, optional): _description_. Defaults to None.
            return_episode_rewards (bool, optional): _description_. Defaults to False.
            use_batch (bool, optional): _description_. Defaults to True.
            is_first_eval (bool, optional): _description_. Defaults to True.

        Returns:
            list: quried indices in each batch
        """
        if is_first_eval:
            obs, legal = env.reset()
        else:
            obs, legal = env._obs(), env.legal
        
        counter = 0
        results = []
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        
        new_labeled_indices = []
        
        while not done:
            if not use_batch:
                action, state = model.predict(obs, state=state, deterministic=deterministic)
                obs, reward, done, _info = env.step(action)
                new_labeled_indices.append(action)
            else:
                action, state = model.predict_batch(obs, legal, deterministic=deterministic)
                obs, legal, reward, done, _info = env.step(action)
                new_labeled_indices.append(action)
                counter += 1
            episode_reward += reward
            
            if counter == eval_interval:
                return env.get_quried_indices(new_labeled_indices)

    def query(self, n, net=None):
        
        if not self.is_trained:
            start_time = time.time()
            self.train()
            end_time = time.time()
            print('Meta training time: ', end_time - start_time)
            self.is_trained = True
        
        model = PPO.load(os.path.join('model_log', 'model'))
        
        query_idx = self.evaluate(model, self.eval_env, n_eval_episodes=n, use_batch=True, is_first_eval=self.is_first_eval)
        self.is_first_eval = False  #TODO: get rid of this eval indicator
        
        return query_idx
