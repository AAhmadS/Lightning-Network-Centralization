import numpy as np
from env.multi_channel import FeeEnv
from simulator import preprocessing
from utils import load_data, make_env, get_fee_based_on_strategy, get_discounted_reward,get_channels_and_capacities_based_on_strategy



def evaluate(mode,strategy, env, env_params, gamma):
    directed_edges = preprocessing.get_directed_edges(env_params['data_path'])
    node_index = env_params['node_index']
    done = False
    state = env.reset()
    rewards = []
    while not done:
        if mode == 'fee selection':
            action, rescale = get_fee_based_on_strategy(state, strategy, directed_edges, node_index)
            state, reward, done, info = env.step(action, rescale)
            state = state*1000
        else:
            #NOTE: Add env_params["scale"], apply to amounts when generating amount graphs
            if strategy == "top_k_betweenness":
                graph = env.get_local_graph(scale = 230000/(10000/env_params["amounts"][0]))
            else: graph = None
            #NOTE: should define a evaluation functions here to use as balinese evaluation for channel selection fee.
            action = get_channels_and_capacities_based_on_strategy(strategy,env_params['capacity_upper_scale_bound']
                                                                   ,env_params['n_channels'],env_params['local_size'], env.src,env.graph_nodes,graph)
            print("ACTION",action)
            state, reward, done, info = env.step(action)
        rewards.append(reward)
        if reward!=0:
            print("REWARD",reward)
        # print(reward)

    discounted_reward = get_discounted_reward(rewards, gamma)
    return discounted_reward




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Baselines')
    parser.add_argument('--strategy', choices=['static', 'proportional', 'match_peer','random','top_k_betweenness'], default='static', required=False)
    parser.add_argument('--data_path', default='data/data.json')
    parser.add_argument('--merchants_path', default='data/merchants.json')
    # parser.add_argument('--tb_log_dir', default='plotting/tb_results')
    # parser.add_argument('--tb_name', required=False)
    parser.add_argument('--node_index', type=int, default=76620) #97851
    parser.add_argument('--log_dir', default='plotting/tb_results/trained_model/')
    parser.add_argument('--n_seed', type=int, default=1) # 5
    parser.add_argument('--fee_base_upper_bound', type=int, default=100)
    parser.add_argument('--total_timesteps', type=int, default=100000)
    parser.add_argument('--max_episode_length', type=int, default=10)
    parser.add_argument('--local_size', type=int, default=100)
    parser.add_argument('--counts', default=[10, 10, 10], type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--amounts', default=[10000, 50000, 100000], type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--epsilons', default=[.6, .6, .6], type=lambda s: [float(item) for item in s.split(',')])
    parser.add_argument('--manual_balance', default=False)
    parser.add_argument('--initial_balances', default=[], type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--capacities', default=[],type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--device', default='auto')
    parser.add_argument('--max_capacity', type = int, default=1e7) 
    parser.add_argument('--n_channels', type=int, default=3)
    parser.add_argument('--mode', type=str, default='channel_openning')#TODO: add this arg to all scripts
    parser.add_argument('--capacity_upper_scale_bound', type=int, default=25)
    parser.add_argument('--local_heads_number', type=int, default=5)



    args = parser.parse_args()

    # strategy = args.strategy
    env_params = {'mode' : args.mode,
                  'data_path': args.data_path,
                  'merchants_path': args.merchants_path,
                  'node_index': args.node_index,
                  'fee_base_upper_bound': args.fee_base_upper_bound,
                  'max_episode_length': args.max_episode_length,
                  'local_size': args.local_size,
                  'counts': args.counts,
                  'amounts': args.amounts,
                  'epsilons': args.epsilons,
                  'manual_balance': args.manual_balance,
                  'initial_balances': args.initial_balances,
                  'capacities': args.capacities,
                  'max_capacity': args.max_capacity,
                  'n_channels': args.n_channels,
                  'capacity_upper_scale_bound': args.capacity_upper_scale_bound,
                  'local_heads_number':args.local_heads_number}


    strategy = args.strategy
    reward_list = []

    for s in range(args.n_seed):
        seed = np.random.randint(low=0, high=1000000)
        # data = load_data(env_params['node_index'], env_params['data_path'], env_params['merchants_path'], env_params['local_size'],
        #                  env_params['manual_balance'], env_params['initial_balances'], env_params['capacities'])
        data = load_data(env_params['mode'],env_params['node_index'], env_params['data_path'], env_params['merchants_path'], env_params['local_size'],
                     env_params['manual_balance'], env_params['initial_balances'], env_params['capacities'],env_params['n_channels'],env_params['local_heads_number'],env_params["max_capacity"])
        env = make_env(data, env_params, seed)
        for i in range(200):
            discounted_reward = evaluate(env_params['mode'],strategy, env, env_params, gamma=1)
            reward_list.append(discounted_reward)


    import statistics
    mean = statistics.mean(reward_list)
    print('mean: ', mean)
