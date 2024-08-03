import numpy as np
from env.multi_channel import FeeEnv
from simulator import preprocessing
from utils import load_data, make_env, get_fee_based_on_strategy, get_discounted_reward, load_model
import random


def evaluate(model, env, gamma):
    done = False
    state = env.reset()
    rewards = []
    random.seed()
    while not done:
        action, _state = model.predict(state)
        action = np.array(action)[0]

        # print("a1",action)


        # action = np.random.randint(0,100,3).tolist() + np.random.randint(0,25,3).tolist()
        # print("a2",action)


        print("ACTION:",action)
        state, reward, done, info = env.step(action)
        rewards.append(reward)
        # print(reward)

    discounted_reward = get_discounted_reward(rewards, gamma)
    # print("ACTION:",action)
    if discounted_reward!=0:
        print("DISCOUNTED REWARD:",discounted_reward)
    return discounted_reward




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Baselines')
    parser.add_argument('--algo', choices=['PPO', 'TRPO', 'SAC', 'TD3', 'A2C', 'DDPG'], default='PPO')
    parser.add_argument('--log_dir', default='')
    parser.add_argument('--data_path', default='data/data.json')
    parser.add_argument('--merchants_path', default='data/merchants.json')
    parser.add_argument('--fee_base_upper_bound', type=int, default=100)
    parser.add_argument('--max_episode_length', type=int, default=10)
    parser.add_argument('--n_seed', type=int, default=1)  # 5
    parser.add_argument('--local_size', type=int, default=100)
    parser.add_argument('--node_index', type=int, default=97851)  # 97851
    parser.add_argument('--counts', default=[10, 10, 10], type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--amounts', default=[10000, 50000, 100000],
                        type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--epsilons', default=[.6, .6, .6], type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--manual_balance', default=True)
    parser.add_argument('--initial_balances', default=[], type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--capacities', default=[],type=lambda s: [int(item) for item in s.split(',')])
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

    algos = ['PPO', 'TRPO', 'SAC', 'TD3', 'A2C', 'DDPG']
    algos = ['PPO']

    algo_reward_dict = dict()
    for algo in algos:
        algo_reward_dict[algo] = []

    for s in range(args.n_seed):
        seed = np.random.randint(low=0, high=1000000)
        data = load_data(env_params['mode'],env_params['node_index'], env_params['data_path'], env_params['merchants_path'], env_params['local_size'],
                     env_params['manual_balance'], env_params['initial_balances'], env_params['capacities'],env_params['n_channels'],env_params['local_heads_number'],env_params["max_capacity"])
        for algo in algos:
            env = make_env(data, env_params, seed)
            model = load_model(algo, env_params,"plotting/tb_results/trained_model/PPO_tensorboard")
            model.set_env(env)

            for i in range(200):
                discounted_reward = evaluate(model, env, gamma=1)
                algo_reward_dict[algo].append(discounted_reward)


    import statistics
    algo_mean_reward_dict = dict()
    for algo in algos:
        algo_mean_reward_dict[algo] = statistics.mean(algo_reward_dict[algo])

    print('_____________________________________________________')
    print(algo_mean_reward_dict)
