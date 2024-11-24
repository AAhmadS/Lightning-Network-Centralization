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
from simulator.preprocessing import get_directed_edges, make_LN_graph,get_providers
from model import Transformer_policy
import json
from networkx.readwrite import json_graph
import networkx as nx


def initialize_graph(data_path, providers_path):
    
    directed_edges = get_directed_edges(data_path)
    providers = get_providers(providers_path)
    G = make_LN_graph(directed_edges, providers)
    return G

def update_graph(graph, updates, node_id):    
    trgs = updates.keys()
    src = str(node_id)
    for trg in trgs:
        [cap_src, fee_base_src, fee_rate_src, bal_src], [cap_trg, fee_base_trg, fee_rate_trg, bal_trg]   = updates[trg]
        graph.add_edge(trg, src, capacity = cap_trg, fee_base = fee_base_trg, fee_rate = fee_rate_trg, balance = bal_trg, channel_id = None)
        graph.add_edge(src, trg, capacity = cap_src, fee_base = fee_base_src, fee_rate = fee_rate_src, balance = bal_src, channel_id = None) 
    graph.nodes[src]["feature"] = [0, 0, 0, 0]
    return graph
        
def apply_action(model, env, graph, node_id):
    done = False
    state = env.reset()
    random.seed()
    while not done:
        action, _state = model.predict(state)

        print("ACTION:",action)
        state, reward, done, info = env.step(action)

    updates = env.get_updates()
    
    graph = update_graph(graph, updates, node_id)
    return graph
    
def make_envs_models(env_params, log_dir, tb_names):
    
    n_channels = env_params['n_channels']
    local_sizes = env_params['local_size']
    max_episode_lengths = env_params['max_episode_length']
    
    envs = []
    models = []

    for n_channel, local_size, max_episode_length, tb_name in zip(n_channels, local_sizes, max_episode_lengths, tb_names):
        print("creating env, model for:", local_size)
        environ_params = env_params.copy()
        environ_params["n_channels"] = n_channel
        environ_params["local_size"] = local_size
        environ_params["max_episode_length"] = max_episode_length
        
        data = load_data(env_params['data_path'], env_params['merchants_path'], local_size,
                n_channel ,env_params['local_heads_number'], env_params['max_capacity'])
        
        seed = np.random.randint(low=0, high=1000000)
        env = make_env(data, environ_params, seed, multiple_env = False)
        envs.append(env)
        Transformer_policy.MAX_POSITION_EMBEDDING = local_size
        Transformer_policy.MODEL = env_params['model']
        models.append(PPO.load(os.path.join(log_dir, tb_name),env))

    
    ln_graph = initialize_graph(env_params['data_path'], env_params['merchants_path'])
    
    return models, envs, ln_graph
   
def analyze(env_params, batch, tb_names, log_dir, duration):
    num_models = len(tb_names)
    weights = [0.55, 0.3, 0.15]
    # Initializing environment for each setting
    models, envs, graph = make_envs_models(env_params, log_dir, tb_names)
    print("Models, Envs, and main Graph loaded")
    node_id = 0
    dyn_graph = graph.copy()
    print("Primary grpah size data:\n")
    print(f"Env graphs updated: {len(dyn_graph.nodes())}")
    print(f"Env graphs updated: {len(dyn_graph.edges())}")
    for _ in range(duration // batch):
        print("In the analyzing loop")
        index = int(np.random.choice(np.arange(num_models), 1, p=weights))
        env, model = envs[index], models[index]

        for i in range(batch):
            dyn_graph = apply_action(model, env, dyn_graph, node_id)
            node_id+=1
    
        for env in envs: 
            env.LN_graph = dyn_graph
            env.set_undirected_attributed_LN_graph()
        print(f"Env graphs updated: {len(dyn_graph.nodes())}")
        print(f"Env graphs updated: {len(dyn_graph.edges())}")
            

    import pickle
    with open('Evoluted_LN_2000.pkl', 'wb') as f:
        pickle.dump(dyn_graph, f)
    with open('Evoluted_LN_2000.pkl', 'rb') as f:
        G = pickle.load(f)
    
    print(len(G.nodes()))

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
    parser.add_argument('--algo', choices=['PPO', 'TRPO', 'SAC', 'TD3', 'A2C', 'DDPG'], default="PPO")
    parser.add_argument('--data_path', default='data/data.json')
    parser.add_argument('--merchants_path', default='data/merchants.json')
    parser.add_argument('--tb_name',  type=lambda s: [item for item in s.split(',')], default=["Transformer-50-5", "Transformer-100-10", "Transformer-200-15"]) # Best to design a config file
    parser.add_argument('--log_dir', default='plotting/tb_results/trained_model/')
    parser.add_argument('--duration', type=int, default=2000) 
    parser.add_argument('--max_episode_length', default=[5, 10 , 15], type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--batch', type=int, default=10)
    parser.add_argument('--local_size', default=[50, 100, 200],  type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--counts', default=[200, 200, 200], type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--amounts', default=[10000, 50000, 100000], type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--epsilons', default=[.6, .6, .6], type=lambda s: [float(item) for item in s.split(',')])
    parser.add_argument('--max_capacity', type = int, default=1e7) #SAT
    parser.add_argument('--n_channels', default=[5, 10 , 15], type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--capacity_upper_scale_bound', type=int, default=10)
    parser.add_argument('--local_heads_number', type=int, default=5)
    parser.add_argument('--sampling_k', type=int, default=4)
    parser.add_argument('--sampling_stages', type=int, default=4)

    

    
    args = parser.parse_args()

    env_params = {'data_path': args.data_path,
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
                  'model':'Transformer'}

    
    n_seed = 1
    for _ in range(n_seed):
        analyze(env_params, batch = args.batch, 
                tb_names=args.tb_name, log_dir=args.log_dir, 
                duration=args.duration)

if __name__ == '__main__':
    main()
