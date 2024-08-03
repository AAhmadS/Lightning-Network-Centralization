import sys
import os

project_root = os.path.dirname(os.path.realpath(__file__))
stable_path = os.path.join(project_root, "stable-baselines3")
sys.path.append(stable_path)

import numpy as np
import stable_baselines3
import sb3_contrib
import secrets
from simulator import preprocessing
from env.multi_channel import FeeEnv
import networkx as nx
import os
import pickle
import graph_embedding_processing
from sklearn.model_selection import train_test_split
from model.GATv2_feature_extractor import CustomGATv2Extractor
from model.custom_buffer import MyCustomDictRolloutBuffer


def make_agent(env, algo, device, tb_log_dir):
    #NOTE: You must use `MultiInputPolicy` when working with dict observation space, not MlpPolicy
    # policy = "MlpPolicy"
    policy = "MultiInputPolicy"
    # policy = Custom_policy
    # create model
    if algo == "PPO":
        from stable_baselines3 import PPO
        # Create the custom policy
        policy_kwargs = dict(
            features_extractor_class=CustomGATv2Extractor,
            features_extractor_kwargs=dict(features_dim=64),
        )
        # Instantiate the PPO agent with the custom policy
        model = PPO(policy, env, device=device, tensorboard_log=tb_log_dir,rollout_buffer_class = MyCustomDictRolloutBuffer, policy_kwargs=policy_kwargs, verbose=1)
        # model = PPO(policy, env, verbose=1, device=device, tensorboard_log=tb_log_dir)
    elif algo == "TRPO":
        from sb3_contrib import TRPO
        model = TRPO(policy, env, verbose=1, device=device, tensorboard_log=tb_log_dir)
    elif algo == "SAC":
        from stable_baselines3 import SAC
        model = SAC(policy, env, verbose=1, device=device, tensorboard_log=tb_log_dir)
    elif algo == "DDPG":
        from stable_baselines3 import DDPG
        model = DDPG(policy, env, verbose=1, device=device, tensorboard_log=tb_log_dir)
    elif algo == "TD3":
        from stable_baselines3 import TD3
        model = TD3(policy, env, verbose=1, device=device, tensorboard_log=tb_log_dir)
    elif algo == "A2C":
        from stable_baselines3 import A2C
        model = A2C(policy, env, verbose=1, device=device, tensorboard_log=tb_log_dir)
    elif algo == "TQC":
        from sb3_contrib import TQC
        model = TQC(policy, env, verbose=1, device=device, tensorboard_log=tb_log_dir)
    elif algo == "ARS":
        from sb3_contrib import ARS
        model = ARS(policy, env, verbose=1, device=device, tensorboard_log=tb_log_dir)
    else:
        raise NotImplementedError()

    return model


def make_env(data, env_params, seed):

    assert len(env_params['counts']) == len(env_params['amounts']) and len(env_params['counts']) == len(
        env_params['epsilons']), "number of transaction types missmatch"
    
    directed_edges = preprocessing.get_directed_edges(env_params['data_path'])
    providers = data['providers']

    G = preprocessing.make_LN_graph(directed_edges, providers, env_params['manual_balance'], data["src"]
    , data["trgs"], data["channel_ids"], env_params['capacities'], env_params['initial_balances'])

   

    env = FeeEnv(env_params["mode"],data,env_params['max_capacity'], env_params['fee_base_upper_bound']
                , env_params['max_episode_length'],len(env_params['counts']),env_params['counts'],
                env_params['amounts'], env_params['epsilons'],env_params['capacity_upper_scale_bound'],
                seed,G)


    return env
    


def get_or_create_list_of_sub_nodes(G, src, local_heads_number, providers, local_size, list_size = 500, train_filename='train_list_of_sub_nodes.pkl', test_filename='test_list_of_sub_nodes.pkl'):
    # If the train file exists, load the list from the file
    if os.path.exists(train_filename):
        with open(train_filename, 'rb') as f:
            return pickle.load(f)
    
    # If the train file doesn't exist, create the list, split it, save the train and test lists to the file, and then return the train list
    else:
        list_of_sub_nodes = preprocessing.create_list_of_sub_nodes(G, src, local_heads_number, providers, local_size, list_size)
        train_list, test_list = train_test_split(list_of_sub_nodes, test_size=0.2, random_state=42)
        
        with open(train_filename, 'wb') as f:
            pickle.dump(train_list, f)
        
        with open(test_filename, 'wb') as f:
            pickle.dump(test_list, f)
        
        return train_list

def load_data(mode, node, directed_edges_path, providers_path, local_size, manual_balance, initial_balances, capacities, n_channels,local_heads_number, capacities_upperbound):
    """
    :return:
    data = dict{mode: Levin is used for different modes: fee_selection, channel_openning (default: String)
                src: node chosen for simulation  (default: int)
                trgs: nodes src is connected to  (default: list)
                channel_ids: channel ids of src node [NOT USED YET]  (default: list)
                initial_balances: initial distribution of capacity of each channel  (default: list)
                capacities: capacity of each channel  (default: list)
                node_variables: ???  (default: )
                                nodes of the localized subgraph
                providers: Merchants of the whole network  (default: ?)
                active_providers: Merchants of the local network around src  (default: ?)
                active_channels: channel which their balances are being updated each timestep  (default: ?)
                                In other words channels connected to our node
                network_dictionary: whole network data  (default: dict)
                n_channels: number of channels within data (default:int)
            }
    """
    print('==================Loading Network Data==================')
    data = {}
    
    # for fee selection mode
    # src_index = node
    
    #TODO #6: check if the src is actually new 
    src = generate_hex_string(66)
    
    # subgraph_radius = 2
    data['providers'] = preprocessing.get_providers(providers_path)
    data['n_channels'] = n_channels
    
    directed_edges = preprocessing.get_directed_edges(directed_edges_path)
    
    
    # for fee selection mode
    # data['src'], data['trgs'], data['channel_ids'], n_channels = preprocessing.select_node(directed_edges, src_index)
    # data['capacities'] = [153243, 8500000, 4101029, 5900000, 2500000, 7000000]
    # data['initial_balances'] = [153243 / 2, 8500000 / 2, 4101029 / 2, 5900000 / 2, 2500000 / 2, 7000000 / 2]
    
    data['src'], data['trgs'], data['channel_ids'] = preprocessing.create_node(directed_edges, src, n_channels)
    
    #NOTE: this part is for fee selection mode
    # channels = []
    # for trg in data['trgs']:
    #     channels.append((data['src'], trg))
    data['fee_policy'] = preprocessing.create_fee_policy_dict(directed_edges,src)
    # data['active_channels'], \
    # data['network_dictionary'], \
    # data['node_variables'], \
    # data['active_providers'], \
    # data['initial_balances'], \
    # data['capacities'], \
    # data['fee_policy'],\
    # data['nodes']= preprocessing.get_init_parameters(data['providers'],
    #                                                        directed_edges,
    #                                                        data['src'], data['trgs'],
    #                                                        data['channel_ids'],
    #                                                        channels,
    #                                                        local_size,
    #                                                        manual_balance, initial_balances, capacities,mode,local_heads_number)
    
    data["capacity_max"] = max(directed_edges["capacity"].max(), capacities_upperbound)
    data["fee_base_max"] = directed_edges["fee_base_msat"].max()
    data["fee_rate_max"] = directed_edges["fee_rate_milli_msat"].max()
    return data


#NOTE : you can use this method for fee setting in static way too.
def get_static_fee(directed_edges, node_index, number_of_channels):
    # action = get_original_fee(directed_edges, node_index)
    # action = get_mean_fee(directed_edges, number_of_channels)
    action = get_constant_fee(alpha=541.62316304877, beta=1760.82436708861, number_of_channels=number_of_channels)
    return action

def get_proportional_fee(state, number_of_channels, directed_edges, node_index):
    balances = state[0:number_of_channels]
    capacities = get_capacities(directed_edges, node_index)
    fee_rates = []
    for i in range(len(balances)):
        b = balances[i]
        c = capacities[i]
        f = -1 + 2*(1-(b/c))
        fee_rates.append(f)
    base_fees = [-1]*number_of_channels     # = zero after rescaling
    return fee_rates+base_fees

def get_match_peer_fee(directed_edges, node_index):
    action = get_peer_fee(directed_edges, node_index)
    return action


def get_fee_based_on_strategy(state, strategy, directed_edges, node_index):
    number_of_channels = get_number_of_channels(directed_edges, node_index)
    rescale = True
    if strategy == 'static':
        action = get_static_fee(directed_edges, node_index, number_of_channels)
        rescale = False
    elif strategy == 'proportional':
        action = get_proportional_fee(state, number_of_channels, directed_edges, node_index)
        action = np.array(action)
        rescale = True
    elif strategy == 'match_peer':
        action = get_match_peer_fee(directed_edges, node_index)
        rescale = False
    else:
        raise NotImplementedError
    return action, rescale

def get_channels_and_capacities_based_on_strategy(strategy,capacity_upper_scale_bound,n_channels,n_nodes, src,graph_nodes, graph):
    if strategy == 'random':
        action = get_random_channels_and_capacities(capacity_upper_scale_bound,n_channels,n_nodes)
    if strategy == 'top_k_betweenness':
        action = get_top_k_betweenness(capacity_upper_scale_bound,n_channels,src,graph_nodes,graph)
    #TODO: define basline strategy for random choose channels and capacities index.

    return action

def get_top_k_betweenness(scale, n_channels, src, graph_nodes, graph,alpha=2):
     nodes_by_betweenness = nx.betweenness_centrality(graph)
     if src in nodes_by_betweenness:
         del nodes_by_betweenness[src]
     sorted_by_betweenness = dict(sorted(nodes_by_betweenness.items(), key=lambda item: item[1]))
     top_k_betweenness = list(sorted_by_betweenness.keys())[:n_channels]
     top_k_betweenness = [graph_nodes.index(item) for item in top_k_betweenness if item in graph_nodes]
    #  top_k_capacity = list(sorted_by_betweenness.values())[-n_channels:]
    #  top_k_capacity = [round(scale*(elem+alpha*max(top_k_capacity))/(sum(top_k_capacity)+n_channels*alpha*max(top_k_capacity))) for elem in top_k_capacity]

     top_k_capacity  = [scale] * n_channels
     
     return top_k_betweenness + top_k_capacity
     
def get_random_channels_and_capacities(capacity_upper_scale_bound,n_channels,n_nodes):
    if n_nodes < n_channels:
        raise "Error: n_nodes must be greater than or equal to n_channels"
    
    # Create a vector of zeros of size n_nodes
    vector1 = np.random.randint(0, n_nodes, n_channels).tolist()
    
    # Create a vector of size n_channels with random integers between 0 and 50
    vector2 = np.random.randint(0, capacity_upper_scale_bound + 1, n_channels).tolist()

    return vector1+ vector2

def get_mean_fee(directed_edges, number_of_channels):
    mean_alpha = directed_edges['fee_rate_milli_msat'].mean()
    mean_beta = directed_edges['fee_base_msat'].mean()
    return [mean_alpha]*number_of_channels + [mean_beta]*number_of_channels

def get_constant_fee(alpha, beta, number_of_channels):
    return [alpha]*number_of_channels + [beta]*number_of_channels

def get_original_fee(directed_edges, node_index):
    src = directed_edges.loc[node_index]['src']
    fee_rates = list(directed_edges[directed_edges['src'] == src]['fee_rate_milli_msat'])
    base_fees = list(directed_edges[directed_edges['src'] == src]['fee_base_msat'])
    return fee_rates + base_fees

def get_peer_fee(directed_edges, node_index):
    src = directed_edges.loc[node_index]['src']
    fee_rates = list(directed_edges[directed_edges['trg'] == src]['fee_rate_milli_msat'])
    base_fees = list(directed_edges[directed_edges['trg'] == src]['fee_base_msat'])
    return fee_rates + base_fees

def get_capacities(directed_edges, node_index):
    src = directed_edges.loc[node_index]['src']
    capacities = list(directed_edges[directed_edges['src'] == src]['capacity'])
    return capacities

def get_number_of_channels(directed_edges, node_index):
    src = directed_edges.loc[node_index]['src']
    number_of_channels = len(directed_edges[directed_edges['src'] == src])
    return number_of_channels


def get_discounted_reward(rewards, gamma):
    discounted_reward = 0
    for i in range(len(rewards)):
        coeff = pow(gamma, i)
        r = coeff*rewards[i]
        discounted_reward += r
    return discounted_reward


def load_model(algo, env_params, path):
    node_index = env_params['node_index']
    if algo == 'DDPG':
        from stable_baselines3 import DDPG
        model = DDPG.load(path=path)
    elif algo == 'PPO':
        from stable_baselines3 import PPO
        model = PPO.load(path=path)
    elif algo == 'TRPO':
        from sb3_contrib import TRPO
        model = TRPO.load(path=path)
    elif algo == 'TD3':
        from stable_baselines3 import TD3
        model = TD3.load(path=path)
    elif algo == 'A2C':
        from stable_baselines3 import A2C
        print("LOAD:")
        model = A2C.load(path=path)

    else:
        raise NotImplementedError

    return model



def load_localized_model(radius, path):
    from stable_baselines3 import PPO
    model = PPO.load(path=path)
    return model

##NOTE: This function is needed for generating node hash string which is 
# used in data frame for adding new channels.
# (use src hash for origin of channel and trg dor destination)

def generate_hex_string(length):
    return secrets.token_hex(length)