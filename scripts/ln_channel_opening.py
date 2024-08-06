import os
import sys

current_file_directory = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(current_file_directory, '..'))
sys.path.append(project_root)


from utils import load_data, make_agent, make_env, load_model
from stable_baselines3 import SAC, TD3, PPO
from numpy import load
import gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import random


class EarlyStoppingCallback(BaseCallback):
    def __init__(self, check_freq: int, n_steps_without_progress: int, verbose=0):
        super(EarlyStoppingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.n_steps_without_progress = n_steps_without_progress
        self.best_reward = -np.inf
        self.n_steps_since_best_reward = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Get the current reward estimate
            x, y = self.model.ep_info_buffer.pop() #infos.get("episode")
            if y > self.best_reward:
                self.best_reward = y
                self.n_steps_since_best_reward = 0
            else:
                self.n_steps_since_best_reward += 1

            if self.n_steps_since_best_reward > self.n_steps_without_progress:
                print("Stopping training because the reward has not improved for {} steps.".format(self.n_steps_without_progress))
                return False
        return True

def train(env_params, train_params, tb_log_dir, tb_name, log_dir, seed):

    data = load_data(env_params['data_path'], env_params['merchants_path'], env_params['local_size'],
                    env_params['n_channels'],env_params['local_heads_number'], env_params["max_capacity"])
    
    env = make_env(data, env_params, seed, multiple_env=True)
    model = make_agent(env, train_params['algo'], train_params['device'], tb_log_dir)
    # model = load_model("PPO", env_params,"plotting/tb_results/trained_model/PPO_tensorboard_fixed_graph_50nodes_5lengthEpisode_mlp_complex_6featureVersion")
    # model.set_env(env)

    #Add Callback for early stopping
    callback = EarlyStoppingCallback(check_freq=10, n_steps_without_progress=1000)
    model.learn(total_timesteps=train_params['total_timesteps'], tb_log_name=tb_name, log_interval=5)

    # model.learn(total_timesteps=train_params['total_timesteps'], tb_log_name=tb_name)
    model.save(log_dir+tb_name)


def main():
    """
    amounts:   in satoshi
    fee_rate and fee_base:  in data {mmsat, msat}
    capacity_upper_scale bound:  upper bound for action range(capacity)
    maximum capacity:   in satoshi
    local_heads_number: number of heads when creating subsamples
    sampling_stage, sampling_k:    parameters of snowball_sampling
    """

    import argparse
    parser = argparse.ArgumentParser(description='Lightning network environment for multichannel')
    parser.add_argument('--algo', choices=['PPO', 'TRPO', 'SAC', 'TD3', 'A2C', 'DDPG'], required=True)
    parser.add_argument('--data_path', default='data/data.json')
    parser.add_argument('--merchants_path', default='data/merchants.json')
    parser.add_argument('--tb_log_dir', default='plotting/tb_results')
    parser.add_argument('--tb_name', required=True)
    parser.add_argument('--log_dir', default='plotting/tb_results/trained_model/')
    parser.add_argument('--n_seed', type=int, default=1) # 5
    parser.add_argument('--total_timesteps', type=int, default=300000)
    parser.add_argument('--max_episode_length', type=int, default=5)
    parser.add_argument('--local_size', type=int, default=50)
    parser.add_argument('--counts', default=[200, 200, 200], type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--amounts', default=[10000, 50000, 100000], type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--epsilons', default=[.6, .6, .6], type=lambda s: [float(item) for item in s.split(',')])
    parser.add_argument('--device', default='auto')
    parser.add_argument('--max_capacity', type = int, default=1e7) #SAT
    parser.add_argument('--n_channels', type=int, default=5)
    parser.add_argument('--mode', type=str, default='channel_openning')#TODO: add this arg to all scripts
    parser.add_argument('--capacity_upper_scale_bound', type=int, default=10)
    parser.add_argument('--local_heads_number', type=int, default=5)
    parser.add_argument('--sampling_k', type=int, default=4)
    parser.add_argument('--sampling_stages', type=int, default=4)

    

    
    args = parser.parse_args()

    train_params = {'algo': args.algo,
                    'total_timesteps': args.total_timesteps,
                    'device': args.device}

    env_params = {'mode' : args.mode,
                  'data_path': args.data_path,
                  'merchants_path': args.merchants_path,
                  'max_episode_length': args.max_episode_length,
                  'local_size': args.local_size,
                  'counts': args.counts,
                  'amounts': args.amounts,
                  'epsilons': args.epsilons,
                  'max_capacity': args.max_capacity,
                  'n_channels': args.n_channels,
                  'capacity_upper_scale_bound': args.capacity_upper_scale_bound,
                  'local_heads_number':args.local_heads_number,
                  'sampling_k':args.sampling_k,
                  'sampling_stages':args.sampling_stages}

    

    for seed in range(args.n_seed):
        train(env_params, train_params,
              tb_log_dir=args.tb_log_dir, log_dir=args.log_dir, tb_name=args.tb_name,
              seed=np.random.randint(low=0, high=1000000))

if __name__ == '__main__':
    main()
