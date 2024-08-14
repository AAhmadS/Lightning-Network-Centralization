import os
import sys

current_file_directory = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(current_file_directory, '..'))
sys.path.append(project_root)


from utils import load_data, make_env
import numpy as np
from model.GATmodule import GNN_Agent

def make_GNN_agent(env, device, features_dim):
    return GNN_Agent(env, device, features_dim)
    
def train(env_params, train_params, seed):

    data = load_data(env_params['data_path'], env_params['merchants_path'], env_params['local_size'],
                    env_params['n_channels'],env_params['local_heads_number'], env_params["max_capacity"])
    
    env = make_env(data, env_params, seed, multiple_env=False)
    
    model = make_GNN_agent(env, train_params['device'], features_dim=64)

    model.train(total_timesteps=train_params['total_timesteps'])

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
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max_capacity', type = int, default=1e7) #SAT
    parser.add_argument('--n_channels', type=int, default=5)
    parser.add_argument('--mode', type=str, default='channel_openning')
    parser.add_argument('--capacity_upper_scale_bound', type=int, default=10)
    parser.add_argument('--local_heads_number', type=int, default=5)
    parser.add_argument('--sampling_k', type=int, default=4)
    parser.add_argument('--sampling_stages', type=int, default=4)

    
    args = parser.parse_args()

    train_params = {'total_timesteps': args.total_timesteps,
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
        train(env_params, train_params, seed=np.random.randint(low=0, high=1000000))

if __name__ == '__main__':
    main()
