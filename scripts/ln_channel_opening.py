from utils import load_data, make_agent, make_env, load_model
from stable_baselines3 import SAC, TD3, PPO
from numpy import load
import gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import random
import os
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from model import Transformer_policy



def make_multiple_env(rank, data, env_params, seed):
    def _init():
        env = make_env(data, env_params, seed, multiple_env=True)
        env.seed(seed + rank)
        return env
    return _init

def train(env_params, train_params, tb_log_dir, tb_name, log_dir, seed):

    data = load_data(env_params['data_path'], env_params['merchants_path'], env_params['local_size'],
                    env_params['n_channels'],env_params['local_heads_number'], env_params["max_capacity"])
    
    envs = make_env(data, env_params, seed, multiple_env=False)
    ## NOTE: uncomment when using multiprocessing
    # num_cpu = 10  # Number of processes to use
    # envs = SubprocVecEnv([make_multiple_env(i, data, env_params, seed+i) for i in range(num_cpu)])
    
    Transformer_policy.MAX_POSITION_EMBEDDING = env_params['local_size'] # do not change
    Transformer_policy.MODEL = train_params['model'] # do not change

    model = make_agent(envs, train_params['algo'], train_params['device'], train_params["model"], tb_log_dir)

    # NOTE: uncomment when loading a pretrained model
    # model = load_model("PPO", env_params,"plotting/tb_results/trained_model/PPO_tensorboard_50nodes_5channel_DeepMLP_dynamic_15M_Episode_editedLog")
    # model.set_env(envs)


    # NOTE: uncomment when best model is needed. set the desired threshold
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=0.9, verbose=1)
    eval_callback = EvalCallback(envs, callback_on_new_best=callback_on_best, verbose=1, n_eval_episodes=100,eval_freq = 1500, best_model_save_path = r"plotting\tb_results\trained_model")
    model.learn(total_timesteps=train_params['total_timesteps'], tb_log_name=tb_name, log_interval=10, callback = eval_callback)

    # model.learn(total_timesteps=train_params['total_timesteps'], tb_log_name=tb_name, log_interval=10) 
    
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
    parser.add_argument('--model', choices=['GNN', 'MLP', 'Transformer', 'GNN+Transformer'], required=True)
    parser.add_argument('--data_path', default='data/data.json')
    parser.add_argument('--merchants_path', default='data/merchants.json')
    parser.add_argument('--tb_log_dir', default='plotting/tb_results')
    parser.add_argument('--tb_name', required=True)
    parser.add_argument('--log_dir', default="plotting/tb_results/trained_model/")
    parser.add_argument('--n_seed', type=int, default=1) # 5
    parser.add_argument('--total_timesteps', type=int, default=100000)
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
                    'device': args.device,
                    'model':args.model}

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
                  'sampling_stages':args.sampling_stages,
                  'model':args.model}

    

    for seed in range(args.n_seed):
        train(env_params, train_params,
              tb_log_dir=args.tb_log_dir, log_dir=args.log_dir, tb_name=args.tb_name,
              seed=np.random.randint(low=0, high=1000000))

if __name__ == '__main__':
    main()
