import gym
from gym import spaces
from gym import *
from gym.utils import seeding
import numpy as np

from simulator.simulator import simulator
from simulator.preprocessing import generate_transaction_types


class FeeEnv(gym.Env):
    """
    ### Description

    This environment corresponds to the LIGHTNING NETWORK simulation. A source node is chosen and a local network
    around that node with radius 2 is created and at each time step, a certain number of transitions are being simulated.

    ### Scales

    We are using the following scales for simulating the real world Lightning Network:

    - Fee Rate: msat                                      - Base Fee: msat
    - Transaction amounts: sat                            - Reward(income): msat
    - Capacity: sat                                       - Balance: sat

    ### Action Space

    The action is a `ndarray` with shape `(2*n_channel,)` which can take values `[0,upper bound]`
    indicating the fee rate and base fee of each channel starting from source node.

    | dim       | action                 | dim        | action                |
    |-----------|------------------------|------------|-----------------------|
    | 0         | fee rate channel 0     | 0+n_channel| fee base channel 0    |
    | ...       |        ...             | ...        |         ...           |
    | n_channel | fee rate last channel  | 2*n_channel| fee base last channel |

    ### Observation Space

    The observation is a `ndarray` with shape `(2*n_channel,)` with the values corresponding to the balance of each
    channel and also accumulative transaction amounts in each time steps.

    | dim       | observation            | dim        | observation                 |
    |-----------|------------------------|------------|-----------------------------|
    | 0         | balance channel 0      | 0+n_channel| sum trn amount channel 0    |
    | ...       |          ...           | ...        |            ...              |
    | n_channel | balance last channel   | 2*n_channel| sum trn amount last channel |

    ### Rewards

    Since the goal is to maximize the return in long term, reward is sum of incomes from fee payments of each channel.
    Reward scale is Sat in order to control the upperbound.

    ***Note:
    We are adding the income from each payment to balance of the corresponding channel.
    """

    def __init__(self, data, fee_base_upper_bound, max_episode_length, number_of_transaction_types, counts, amounts, epsilons, seed):
        # Source node
        self.src = data['src'] 
        
        
        #NOTE: added attribute
        self.n_nodes = len(data['trgs'].unique())
        
        
        self.trgs = data['trgs']
        self.n_channel = len(self.trgs)
        print('actione dim:', 2 * self.n_channel)

        #NOTE: following lines are for fee selection mode
        # # Base fee and fee rate for each channel of src
        # self.action_space = spaces.Box(low=-1, high=+1, shape=(2 * self.n_channel,), dtype=np.float32)
        # self.fee_rate_upper_bound = 1000
        # self.fee_base_upper_bound = fee_base_upper_bound

        # # Balance and transaction amount of each channel
        # self.observation_space = spaces.Box(low=0, high=np.inf, shape=(2 * self.n_channel,), dtype=np.float32)

         ## defining action space & edit of step function in multi channel
        ## first n_channels are id's  of connected nodes and the seconds are corresponidg  capacities
        ## add self.capacities to the fields of env class

        #TODO #8:
        self.capacities = [50000, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000,
                                1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 9000000, 10000000] # mSAT

        self.action_space = MultiDiscrete([self.n_nodes for _ in range(self.n_channel)] + [len(self.capacities) for _ in range(self.n_channel)])


        #NOTE: defining observation space
        # The observation is a ndarray with shape (n_nodes + 2*n_channels,). The first part of the observation space is 2*n_nodes with the values of 0 or 1, indicating whether we connect a channel with it or not.
        #The second part is 2*n_channels with the values corresponding to the balance of each channel and also accumulative transaction amounts in each time step.
        #Please note that the dimensions for balance and transaction amounts start from n_nodes and n_nodes + n_channels respectively. This allows us to separate the node connection information from the channel balance and transaction amounts.
        
        maximum_balance = ... # maximum balance
        max_transaction_amount = ... # maximum transaction amount

        # Define the bounds for each part of the observation space
        node_bounds = [2] * (self.n_nodes) # values can be 0 or 1
        # Create the observation space
        multi_discrete_space = MultiDiscrete(node_bounds)
        box_space = Box(low=0, high=np.inf, shape=(2 * self.n_channel,), dtype=np.float32)
        self.observation_space = Dict({
            'multi_discrete': multi_discrete_space,
            'box': box_space
            })
            
        # Initial values of each channel for fee selection mode
        # self.initial_balances = data['initial_balances']
        # self.capacities = data['capacities']
        # self.state = np.append(self.initial_balances, np.zeros(shape=(self.n_channel,)))
        self.state = np.append(np.zeros(shape=(self.n_nodes,)),np.zeros(shape=(self.n_channel,)), np.zeros(shape=(self.n_channel,)))

        self.time_step = 0
        self.max_episode_length = max_episode_length
        # for fee selection
        # self.balance_ratio = 0.1

        # Simulator
        transaction_types = generate_transaction_types(number_of_transaction_types, counts, amounts,
                                                       epsilons)
        self.simulator = simulator(src=data['src'],
                                   trgs=data['trgs'],
                                   channel_ids=data['channel_ids'],
                                   active_channels=data['active_channels'],
                                   network_dictionary=data['network_dictionary'],
                                   merchants=data['providers'],
                                   transaction_types=transaction_types,
                                   node_variables=data['node_variables'],
                                   active_providers=data['active_providers'],
                                   fixed_transactions=False)

        self.seed(seed)


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action, rescale=True):
        # Rescaling the action vector (fee selection mode)
        # if rescale:
        #     action[0:self.n_channel] = .5 * self.fee_rate_upper_bound * action[0:self.n_channel] + \
        #                                .5 * self.fee_rate_upper_bound
        #     action[self.n_channel:2 * self.n_channel] = .5 * self.fee_base_upper_bound * action[
        #                                                                                  self.n_channel:2 * self.n_channel] + \
        #                                                 .5 * self.fee_base_upper_bound

        # Running simulator for a certain time interval
        balances, transaction_amounts, transaction_numbers = self.simulate_transactions(action)
        self.time_step += 1

        reward = 1e-6 * np.sum(np.multiply(action[0:self.n_channel], transaction_amounts) + \
                        np.multiply(action[self.n_channel:2 * self.n_channel], transaction_numbers))

        info = {'TimeLimit.truncated': True if self.time_step >= self.max_episode_length else False}

        done = self.time_step >= self.max_episode_length

        self.state = np.append(balances, transaction_amounts)/1000

        return self.state, reward, done, info

    def simulate_transactions(self, action):
        self.simulator.set_channels_fees(action)

        output_transactions_dict = self.simulator.run_simulation(action)
        balances, transaction_amounts, transaction_numbers = self.simulator.get_simulation_results(action,
                                                                                                   output_transactions_dict)

        return balances, transaction_amounts, transaction_numbers
    
    #NOTE: first commit:
    '''
    def simulate_transactions(self, action):
    
        self.simulator.set_channel_fees(action)
        
        output_transactions_dict = self.simulator.run_simulation(action, fees)
        balances, transaction_amounts, transaction_numbers = self.simulator.get_simulation_results(action,
                                                                                                   output_transactions_dict)

    '''

    def reset(self):
        print('episode ended!')
        self.time_step = 0
        self.state = np.append(self.initial_balances, np.zeros(shape=(self.n_channel,)))

        return np.array(self.state, dtype=np.float64)

