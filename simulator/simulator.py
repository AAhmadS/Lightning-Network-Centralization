import numpy as np
import networkx as nx
import pandas as pd
import math
import copy
import time
from . import generating_transactions


#environment has an object of simulator
class simulator():
  def __init__(self,
               src,
               network_dictionary,
               merchants,
               transaction_types, # = [(count, amount, epsilon),...]
               node_variables,
               active_providers,
               fee_policy,
               fixed_transactions = True,
               support_onchain_rebalancing = False,
               graph_nodes = None,
               current_graph = None):
    
    self.src = src
    self.transaction_types = transaction_types
    self.merchants = merchants #list of merchants
    self.node_variables = node_variables
    self.active_providers = active_providers
    self.fee_policy = fee_policy
    self.network_dictionary = network_dictionary
    self.fixed_transactions = fixed_transactions
    self.support_onchain_rebalancing = support_onchain_rebalancing
    self.graph_nodes = graph_nodes
    self.current_graph = current_graph

    self.src_fee_base, self.src_fee_rate = fee_policy[src]
    self.number_of_transaction_types = len(transaction_types)
    self.graphs_dict = self.generate_graphs_dict(transaction_types)

    self.shares = {}
    self.trgs = []
    self.nodes_cumulative_trs_amounts = np.zeros(len(self.graph_nodes))
    self.map_nodes_to_id = dict(zip(self.graph_nodes, np.arange(len(self.graph_nodes))))


    if fixed_transactions : 
      self.transactions_dict = self.generate_transactions_dict(src, transaction_types, node_variables, active_providers)
    else :
      self.transactions_dict = None


 

  def calculate_weight(self,edge,amount): 
    return edge[2]*1000 + edge[1]*amount 


  def generate_transactions_dict(self, src, transaction_types, node_variables, active_providers):
    transactions_dict = dict()
    for (count, amount, epsilon) in transaction_types :
        trs = generating_transactions.generate_transactions(src, amount, count, node_variables, epsilon, active_providers, verbose=False, exclude_src=True)
        transactions_dict[amount] = trs
    return transactions_dict


  def generate_graph(self, amount):
    '''Description: for each transaction_type, outputs a subgraph
                    in which the relevant transactions can flow.

    '''
    graph = nx.DiGraph()
    for key in self.network_dictionary :
      val = self.network_dictionary[key]
      # val[0] represents balance key[0] src and key[1] target, val[1] is fee_rate and val[2] is fee_base

      if val[0] > amount :
          graph.add_edge(key[0],key[1],weight = val[1]*amount + val[2]*1000) #to change the base rate to milli msat we multiply it to 1000
    
    return graph



  def generate_graphs_dict(self, transaction_types):
    '''
    Description: generates subgraph for each transaction_type.
    ]            This is done during each iteration.
                 
    '''
    graph_dicts = dict()
    for (count, amount, epsilon) in transaction_types :
        graph = self.generate_graph(amount)
        graph_dicts[amount] = graph
    return graph_dicts


  #NOTE: why are just active channels updated in the amount graphs,\
    #it was not plausible, so we changed it :)
  def update_graphs(self, src, trg, transaction_amount):
      """
      Updates the current graph and the graphs dictionary based on the changes in the network dictionary.
      
      Args:
          src (str): The source node of the channel.
          trg (str): The target node of the channel.
          transaction_amount (float): The amount of the transaction.
      
      Returns:
          None
      """
            
      src_trg = self.network_dictionary[(src,trg)]
      src_trg_balance = src_trg[0] - transaction_amount
      
      trg_src = self.network_dictionary[(trg,src)]
      trg_src_balance = trg_src[0] + transaction_amount
      
      if src != self.src and trg != self.src:
        self.network_dictionary[(src,trg)][0] = src_trg_balance
        self.network_dictionary[(trg,src)][0] = trg_src_balance
        self.current_graph[src][trg]['balance'] = src_trg_balance
        self.current_graph[trg][src]['balance'] = trg_src_balance
      
      for (_, amount, _) in self.transaction_types:
          graph = self.graphs_dict[amount]  
          
          if (src_trg_balance <= amount) and (graph.has_edge(src, trg)):
            graph.remove_edge(src, trg)
          
          if (trg_src_balance > amount) and (not graph.has_edge(trg,src)): 
            graph.add_edge(trg, src, weight = self.calculate_weight(trg_src, amount))

          self.graphs_dict[amount] = graph
  


        
  def update_network_and_active_channels(self, action, prev_action):
    """
    Updates the network dictionary and active channels based on the differences between the current action and the previous action.
    
    Args:
        action (list): The current action, containing target channels and their balances.
        prev_action (list): The previous action, containing target channels and their balances.
    
    Returns:
        tuple: A tuple containing two lists:
            - The additive targets and balances in the current action.
            - The channels that need to be omitted.
    """

    additive_channels, omitting_channels = self.delete_previous_action_differences(action, prev_action)

    self.add_channels_to_network_dictionary(additive_channels)

    return additive_channels, omitting_channels
  
  # def update_network_and_active_channels(self, action):
  #       trg = action[0]
  #       bal = action[1]
  #       # [balance, fee_base, fee_rate, capacity]
  #       if trg in self.trgs:
  #         self.network_dictionary[(self.src, trg)][0] += bal
  #         self.network_dictionary[(trg, self.src)][0] += bal
  #         self.network_dictionary[(trg, self.src)][3] += 2*bal
  #         self.network_dictionary[(self.src, trg)][3] += 2*bal
  #         ommitive_channels = [trg]
          
  #       else:
  #         self.trgs.append(trg)
  #         self.network_dictionary[(self.src, trg)] = [bal, None, None, 2* bal]
  #         self.network_dictionary[(trg, self.src)] = [bal, None, None, 2* bal]
  #         ommitive_channels = []
  #       self.active_channels[(self.src, trg)] = self.network_dictionary[(self.src, trg)]
  #       self.active_channels[(trg, self.src)] = self.network_dictionary[(trg, self.src)]

  #       return [trg, self.network_dictionary[(self.src, trg)][0]] , ommitive_channels
     
  
  def delete_previous_action_differences(self, action, prev_action):
    """
    Deletes the differences between the previous action and the current action.
    
    Args:
        action (list): The current action.
        prev_action (list): The previous action.
    
    Returns:
        tuple: A tuple containing two lists:
            - The additive targets and balances in the current action.
            - The channels that need to be omitted.
    """
    
    midpoint_prev_action = len(prev_action) // 2

    if midpoint_prev_action == 0:
      return action, []
    
    midpoint_action = len(action) // 2
    additive_trgs = []
    additive_bal = []
    omitting_channels = []
    # trg, bals in action which are new

    for trg, bal in zip(action[:midpoint_action], action[midpoint_action:]):
      if (trg,bal) not in zip((prev_action[:midpoint_prev_action], prev_action[midpoint_prev_action:])):
        additive_trgs.append(trg)
        additive_bal.append(bal)
        
        if trg in prev_action[:midpoint_prev_action]:
          omitting_channels.append(trg)
          del self.network_dictionary[(self.src, trg)]
          del self.network_dictionary[(trg, self.src)]       

    # trgs in prev_action and not in action anymore      
    for trg in prev_action[:midpoint_prev_action]:
      if trg not in action[:midpoint_action]:
          omitting_channels.append(trg)
          del self.network_dictionary[(self.src, trg)]
          del self.network_dictionary[(trg, self.src)]
          
        
    #add budget in output if needed
    return additive_trgs + additive_bal, omitting_channels
    
  # Complementary
  def add_channels_to_network_dictionary(self, additive_channels):
    """
    Adds the provided additive channels to the network dictionary.
    
    Args:
        additive_channels (list): A list of tuples, where each tuple contains the target channel and the balance for that channel.
    
    Returns:
        int: The number of channels added to the network and active channels dictionaries.
    """

    if not additive_channels:
      return
    
    midpoint = len(additive_channels) // 2
    for trg, bal in zip(additive_channels[:midpoint], additive_channels[midpoint:]):
      # [balance, fee_rate, fee_base, capacity]
      self.network_dictionary[(self.src, trg)] = [bal, None, None, 2* bal]
      self.network_dictionary[(trg, self.src)] = [bal, None, None, 2* bal]
      


      
  def evolve_network_dict(self, src, trg, fee_base_src, fee_rate_src,fee_base_trg,fee_rate_trg, bal):
     self.network_dictionary[(src, trg)] = [2*bal, fee_base_src, fee_rate_src, bal]
     self.network_dictionary[(trg, src)] = [2*bal, fee_base_trg, fee_rate_trg, bal]    
      


  def get_local_graph(self, scale):
    # self.sync_network_dictionary()
    # graph = nx.DiGraph()

  #   for key in self.network_dictionary :
  #     val = self.network_dictionary[key]
  #     # val[0] represents balance key[0] src and key[1] target, val[1] is fee_rate and val[2] is fee_base
  #     graph.add_edge(key[0],key[1],weight = val[1]*scale + val[2])
    
  #   # assert self.src in self.network_dictionary
  #   return graph
    

  def update_network_data(self, path, transaction_amount):
      """
      Updates the network data by iterating through the provided path and updating the graphs and cumulative transaction amounts for each node in the path, excluding the source node.
      
      Args:
          path (list): A list of nodes representing the path to update.
          transaction_amount (int): The amount of the transaction to update the network data with.
      
      Returns:
          None
      """
            
      for i in range(len(path)-1) :
        src = path[i] 
        trg = path[i+1]
        self.update_graphs(src, trg, transaction_amount)
        if src != self.src and i > 0 :
          self.nodes_cumulative_trs_amounts[self.map_nodes_to_id[src]] += transaction_amount
      
  def is_active_channel(self, src, trg):
    return ((src,trg) in self.active_channels)
        
  def onchain_rebalancing(self, onchain_rebalancing_amount, src, trg, channel_id):
    if self.is_active_channel(src,trg) :
      self.active_channels[(src,trg)][0] += onchain_rebalancing_amount  
      self.active_channels[(src,trg)][3] += onchain_rebalancing_amount   
      self.active_channels[(trg,src)][3] += onchain_rebalancing_amount   
      self.update_graph(src, trg)
                  

  def get_path_value(self,nxpath,graph) :
    val = 0 
    for i in range(len(nxpath)-1):
      u,v = nxpath[i],nxpath[i+1]
      weight = graph.get_edge_data(u, v)['weight']
      val += weight
    return val
    

  def set_node_fee(self,src,trg,channel_id,action):
      alpha = action[0]
      beta = action[1]
      self.network_dictionary[(src,trg)][1] = alpha
      self.network_dictionary[(src,trg)][2] = beta
      if self.is_active_channel(src,trg) :
        self.active_channels[(src,trg)][1] = alpha
        self.active_channels[(src,trg)][2] = beta

  def set_channels_fees(self, fees, trgs) :
      """
      Sets the fees for the channels between the source node and the target nodes.
      
      Args:
          fees (list): A list containing the base fees and fee rates for the channels. The first half of the list are the base fees, and the second half are the fee rates.
          trgs (list): A list of target nodes to set the fees for.
      
      Returns:
          None
      """    
      
      n = len(trgs)

      betas = fees[:n]
      alphas = fees[n:]
 
      for i, trg in enumerate(trgs):
          self.network_dictionary[(self.src, trg)][1] = self.src_fee_rate
          self.network_dictionary[(self.src, trg)][2] = self.src_fee_base
          
          self.network_dictionary[(trg, self.src)][1] = alphas[i]
          self.network_dictionary[(trg, self.src)][2] = betas[i]
              


  def run_single_transaction(self,
                             transaction_id,
                             transaction_amount,
                             src,trg,
                             graph):
    
    result_bit = 0
    info = {''}
    try:
      path = nx.shortest_path(graph, source=src, target=trg, weight="weight", method='dijkstra')
      
    except nx.NetworkXNoPath:
      return None,-1,{'no path'}
    # val = self.get_path_value(path, graph)
    result_bit = 1
    info = {'successful'}
    return path, result_bit, info 
  
  def update_amount_graph(self, additive_channels, omitting_channels, fees):
    """
    Updates the amount graphs in the simulator by adding new channels and removing omitted channels.
    
    Args:
        additive_channels (list): A list of tuples, where each tuple contains the target node and the balance for a new channel to be added.
        omitting_channels (list): A list of nodes representing channels to be removed from the amount graphs.
        fees (list): A list of fee rates and base fees for the new channels, where the first half of the list are the base fees and the second half are the fee rates.
    
    Returns:
        None
    """

    midpoint = len(additive_channels) // 2
    additive_trg = additive_channels[:midpoint]
    additive_bal = additive_channels[midpoint:]
    
    #removing omitting channels from amount graphs
    for key in omitting_channels :
      for amount, graph in self.graphs_dict.items():
          try: 
            graph.remove_edge(self.src, key)
            self.graphs_dict[amount] = graph
          except:
             pass
          try:
            graph.remove_edge(key, self.src)
            self.graphs_dict[amount] = graph
          except:
             pass
          
    fee_rates = fees[ midpoint : ]
    base_fees = fees[ : midpoint ]
    #adding channels with weight to relevant amount graphs
    for i in range(midpoint):
      trg, bal = additive_trg[i], additive_bal[i]
      for amount, graph in self.graphs_dict.items():
        if bal >= amount:
          graph.add_edge(trg, self.src, weight = base_fees[i] * 1000 + fee_rates[i] * amount) # to turn fee base to mili msat multiply to 1000
          graph.add_edge(self.src, trg, weight = self.src_fee_base * 1000 + self.src_fee_rate * amount) # to turn fee base to mili msat multiply to 1000
          self.graphs_dict[amount] = graph

  def update_evolved_graph(self, fees, list_of_pairs):

    list_of_balances = self.get_list_of_balances(list_of_pairs)

    midpoint = len(fees) // 2
    fee_rates = fees[:midpoint]
    base_fees = fees[midpoint:]
    #adding channels with weight to relevant amount graphs
    for i in range(len(list_of_pairs)):
      src, trg = list_of_pairs[i][0], list_of_pairs[i][1]
      bal = list_of_balances[i]
      for amount, graph in self.graphs_dict.items():
        if bal >= amount:
          graph.add_edge(trg, src, weight = base_fees[2*i]*1000 + fee_rates[2*i]*amount) # to turn fee base to mili msat multiply to 1000
          graph.add_edge(src, trg, weight = base_fees[2*i + 1]*1000 + fee_rates[2*i + 1]*amount) # to turn fee base to mili msat multiply to 1000
          
          self.graphs_dict[amount] = graph
    return list_of_balances
  
  def get_list_of_balances(self, list_of_pairs): 
     """
     Calculates a list of balances for a given list of pairs.
     
     Args:
         list_of_pairs (list): A list of pairs, where each pair is a tuple of two strings.
     
     Returns:
         list: A list of balances, where each balance is a float representing the mean capacity for the corresponding source in the input list, divided by 2.
     """
     list_of_balances = []
     for i in range(len(list_of_pairs)):
        src = list_of_pairs[i][0]
        mean = self.calculate_mean_capacity_for_src(self.network_dictionary, src)/2
        list_of_balances.append(mean)
     return list_of_balances
  
  def calculate_mean_capacity_for_src(self, network_dict, src):
    """
    Calculates the mean capacity for a given source node in the network dictionary.
    
    Args:
        network_dict (dict): A dictionary representing the network, where the keys are tuples of (source, destination) nodes, and the values are lists containing various network metrics.
        src (str): The source node for which to calculate the mean capacity.
    
    Returns:
        float: The mean capacity for the given source node, or 0 if there are no values available.
    """
    filtered_values = [val[3] for key, val in network_dict.items() if key[0] == src]
    mean_value = sum(filtered_values) / len(filtered_values) if len(filtered_values) > 0 else 0
    return mean_value



  def run_simulation(self): 

    output_transactions_dict = dict()

    for (count, amount, epsilon) in self.transaction_types:
        trs = self.run_simulation_for_each_transaction_type(count, amount, epsilon)
        output_transactions_dict[amount] = trs
    
    return output_transactions_dict
   


  def run_simulation_for_each_transaction_type(self, count, amount, epsilon):  
      
      #Run Transactions
      if self.fixed_transactions : 
        transactions = self.transactions_dict[amount]
      else :
        transactions = generating_transactions.generate_transactions(self.src, amount, count, self.node_variables, epsilon, self.active_providers, verbose=False, exclude_src=True)
      transactions = transactions.assign(path=None)
      transactions['path'] = transactions['path'].astype('object')
      for index, transaction in transactions.iterrows():
        graph = self.graphs_dict[amount] 
        src, trg = transaction['src'], transaction['trg']
        if (not src in graph.nodes()) or (not trg in graph.nodes()):
          path, result_bit, info = [], -1, {'src and/or trg dont exist in the graph'}
        else : 
          path, result_bit, info = self.run_single_transaction(transaction['transaction_id'], amount,
                                                                transaction['src'], transaction['trg'], graph) 
          
        if result_bit == 1 : #successful transaction
            self.update_network_data(path, amount)
            transactions.at[index,"result_bit"] = 1
            transactions.at[index,"path"] = path

        elif result_bit == -1 : #failed transaction
            transactions.at[index,"result_bit"] = -1   
            transactions.at[index,"path"] = []
        
      # print("random transactions ended succussfully!")
      return transactions    #contains final result bits  #contains paths
  
  

  #getting the statistics

  
  def get_balance(self,src,trg,channel_id):
      return self.network_dictionary[(src,trg)][0]


  def get_capacity(self,src,trg,channel_id):
      return self.network_dictionary[(src,trg)][3]



  def get_k_and_tx(self, src, trg, transactions_dict):
    """
    Calculates the total number of transactions (k) and the total transaction amount (tx) between a given source (src) and target (trg) nodes across all transaction amounts in the transactions_dict.
    
    Args:
        src (str): The source node.
        trg (str): The target node.
        transactions_dict (dict): A dictionary mapping transaction amounts to DataFrames containing the transaction details.
    
    Returns:
        tuple: A tuple containing the total number of transactions (k) and the total transaction amount (tx) between the given source and target nodes.
    """ 

    k = 0
    tx = 0
    for amount, transactions in transactions_dict.items():
      temp_k = 0
      temp_tx = 0
      for index, row in transactions.iterrows():
          path = row["path"]  
          for i in range(len(path)-1) :
            if (path[i]==src) & (path[i+1]==trg) :
                temp_k += 1
      temp_tx = temp_k*amount
      k += temp_k
      tx += temp_tx
      
    return k,tx



  def get_total_fee(self,path) :
    alpha_bar = 0
    beta_bar = 0
    for i in range(len(path)-1):
      src = path[i]
      trg = path[i+1]
      src_trg = self.network_dictionary[(src,trg)]
      alpha_bar += src_trg[1]
      beta_bar += src_trg[2]
    return alpha_bar,beta_bar


  def get_excluded_total_fee(self,path, excluded_src, excluded_trg) :
    alpha_bar = 0
    beta_bar = 0
    for i in range(len(path)-1):
      src = path[i]
      trg = path[i+1]
      if (src!=excluded_src) or (trg!=excluded_trg) :
        src_trg = self.network_dictionary[(src,trg)]
        alpha_bar += src_trg[1]
        beta_bar += src_trg[2]
    return alpha_bar,beta_bar



  def find_rebalancing_cycle(self, rebalancing_type, src, trg, channel_id, rebalancing_amount):
      rebalancing_graph = self.generate_graph(rebalancing_amount)  
      cheapest_rebalancing_path = []
      
      alpha_bar = 0
      beta_bar = 0
      reult_bit = -1

      if rebalancing_type == -1 : #clockwise
          if (not src in rebalancing_graph.nodes()) or (not trg in rebalancing_graph.nodes()) or (not rebalancing_graph.has_edge(trg, src)):
            return -4,None,0,0
          if  rebalancing_graph.has_edge(src,trg):
            rebalancing_graph.remove_edge(src,trg)  
          cheapest_rebalancing_path,result_bit = self.run_single_transaction(-1,rebalancing_amount,src,trg,rebalancing_graph) 
          if result_bit == -1 :
            return -5,None,0,0
          elif result_bit == 1 :
            cheapest_rebalancing_path.append(src)
            #alpha_bar,beta_bar = self.get_total_fee(cheapest_rebalancing_path)
            alpha_bar,beta_bar = self.get_excluded_total_fee(cheapest_rebalancing_path,src,trg)
            

      elif rebalancing_type == -2 : #counter-clockwise
          if (not trg in rebalancing_graph.nodes()) or (not src in rebalancing_graph.nodes()) or (not rebalancing_graph.has_edge(src, trg)):
            return -6,None,0,0
          if  rebalancing_graph.has_edge(trg,src):
            rebalancing_graph.remove_edge(trg,src)  
          cheapest_rebalancing_path,result_bit = self.run_single_transaction(-2,rebalancing_amount,trg,src,rebalancing_graph) 
          if result_bit == -1 :
            return -7,None,0,0
          elif result_bit == 1 :
            cheapest_rebalancing_path.insert(0,src)
            #alpha_bar,beta_bar = self.get_total_fee(cheapest_rebalancing_path)
            alpha_bar,beta_bar = self.get_excluded_total_fee(cheapest_rebalancing_path,src,trg)
            
   
      
      return result_bit,cheapest_rebalancing_path,alpha_bar,beta_bar
      

  def get_coeffiecients(self,action,transactions,src,trg,channel_id, simulation_amount, onchain_transaction_fee):
        k = self.get_k(src,trg,channel_id,transactions)
        tx = simulation_amount*k
        rebalancing_fee, rebalancing_type  = self.operate_rebalancing(action[2],src,trg,channel_id,onchain_transaction_fee)
        return k,tx, rebalancing_fee, rebalancing_type


  def get_simulation_results(self, output_transactions_dict):
        """
        Retrieves the simulation results for the specified output transactions.
        
        Returns:
            tuple: A tuple containing the following elements:
                - channels_balances (list): A list of the current balances for each channel.
                - channels_ks (list): A list of the k values for each channel.
                - channels_txs (list): A list of the transaction amounts for each channel.
        """
                
        channels_balances = []
        channels_ks = []
        channels_txs = []
        src = self.src

        for i, trg in enumerate(self.trgs):
          k, tx = self.get_k_and_tx(src, trg, output_transactions_dict)
          balance = self.network_dictionary[(src,trg)][0]
          channels_ks.append(k)
          channels_txs.append(tx)
          channels_balances.append(balance)

        return channels_balances, channels_ks, channels_txs


  def operate_rebalancing(self,gamma,src,trg,channel_id,onchain_transaction_fee):
    if self.support_onchain_rebalancing==True :
        return self.operate_rebalancing_with_onchain(gamma,src,trg,channel_id,onchain_transaction_fee)
    else :
        return self.operate_rebalancing_without_onchain(gamma,src,trg,channel_id)



  def operate_rebalancing_with_onchain(self,gamma,src,trg,channel_id,onchain_transaction_fee):
    fee = 0
    if gamma == 0 :
      return 0,0  # no rebalancing
    elif gamma > 0 :
      rebalancing_type = -1 #clockwise
      result_bit, cheapest_rebalancing_path, alpha_bar, beta_bar = self.find_rebalancing_cycle(rebalancing_type, src, trg, channel_id, gamma)
      if result_bit == 1 :
        cost = alpha_bar*gamma + beta_bar
        if cost <= onchain_transaction_fee:
          self.update_network_data(cheapest_rebalancing_path, gamma)
          fee = cost
        else :
          self.onchain_rebalancing(gamma,src,trg,channel_id)
          fee = onchain_transaction_fee
          rebalancing_type = -3 #onchain
      
      else :
        self.onchain_rebalancing(gamma,src,trg,channel_id)
        fee = onchain_transaction_fee
        rebalancing_type = -3

      return fee, rebalancing_type

    else :
      rebalancing_type = -2 #counter-clockwise
      gamma = gamma*-1    
      result_bit,cheapest_rebalancing_path, alpha_bar, beta_bar = self.find_rebalancing_cycle(rebalancing_type, src, trg, channel_id, gamma)
      if result_bit == 1 :
        cost = alpha_bar*gamma + beta_bar
        if cost <= onchain_transaction_fee:
          self.update_network_data(cheapest_rebalancing_path, gamma)
          fee = cost
        else :
          self.onchain_rebalancing(gamma,trg,src,channel_id)
          fee = onchain_transaction_fee
          rebalancing_type = -3 #onchain
      
      elif result_bit == -1: 
        self.onchain_rebalancing(gamma,trg,src,channel_id)
        fee = onchain_transaction_fee
        rebalancing_type = -3

      return fee, rebalancing_type


  def operate_rebalancing_without_onchain(self,gamma,src,trg,channel_id):
    fee = 0
    if gamma == 0 :
      return 0,0  # no rebalancing


    elif gamma > 0 :
      rebalancing_type = -1 #clockwise
      result_bit, cheapest_rebalancing_path, alpha_bar, beta_bar = self.find_rebalancing_cycle(rebalancing_type, src, trg, channel_id, gamma)
      if result_bit == 1 :
          self.update_network_data(cheapest_rebalancing_path, gamma)
          fee = cost
      elif result_bit == -1 :
          fee, rebalancing_type = 0 , -10 # clockwise failed

    else :
      rebalancing_type = -2 #counter-clockwise
      gamma = gamma*-1    
      result_bit,cheapest_rebalancing_path, alpha_bar, beta_bar = self.find_rebalancing_cycle(rebalancing_type, src, trg, channel_id, gamma)
      if result_bit == 1 :
          self.update_network_data(cheapest_rebalancing_path, gamma)
          fee = cost
      elif result_bit == -1: 
          fee, rebalancing_type = 0 , -20 # counter-clockwise failed

    return fee, rebalancing_type
  
  
  def get_channel_fees(self, channels):
    """
    Calculates the channel fees for a list of channels.
    
    The approach taken is a general sense approach, and for the peer, the median approach is used.
    
    Args:
        channels (list): A list of channel IDs.
    
    Returns:
        list: A list containing the base fees and rates for the channels.
    """
        
    #NOTE: the approach taken is general sense approach, and for the peer, we use median approach
    bases = []
    rates = []
    midpoint = len(channels)//2
    channels = channels[:midpoint]
    for trg in channels:
      trg_base, trg_rate = self.fee_policy[trg]
      bases.append(trg_base)
      rates.append(trg_rate)
    return bases + rates
  
  
  def get_rates_and_bases(self, list_of_pairs):
      bases = []
      rates = []
      for i in range(0,len(list_of_pairs)):
          src = list_of_pairs[i][0]
          trg = list_of_pairs[i][1]
          base_neighbor, rate_neighbor = self.fee_policy[trg]
          base_src, rate_src = self.fee_policy[src]
          bases.extend([base_neighbor, base_src])
          rates.extend([rate_neighbor, rate_src])
      return rates + bases