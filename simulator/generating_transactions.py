import numpy as np
import pandas as pd


def sample_providers(src, K, node_variables, active_providers, exclude_src=True):
      provider_records = node_variables[node_variables["pub_key"].isin(active_providers)]
      if exclude_src :
        if src in set(node_variables['pub_key']):
           nodes = list(set(provider_records["pub_key"]) - set([src]))
        else:
           nodes = list(set(provider_records['pub_key']))

      else :
        nodes = list(provider_records["pub_key"])

      probas = list(provider_records["degree"] / provider_records["degree"].sum())
      np.random.seed()
      return np.random.choice(nodes, size=K, replace=True, p=probas)

def generate_transactions(src, amount_in_satoshi, K, node_variables, epsilon, active_providers,
                           verbose=False, exclude_src=True):
      """
      Generates a set of transactions between a source node and a set of target nodes, with an optional subset of the targets being selected from a set of active providers.
      
      Args:
          src (str): The public key of the source node.
          amount_in_satoshi (int): The amount of the transaction in satoshis.
          K (int): The number of transactions to generate.
          node_variables (pandas.DataFrame): A DataFrame containing information about the nodes in the network, including their public keys and degrees.
          epsilon (float): The proportion of target nodes that should be selected from the set of active providers.
          active_providers (list): A list of public keys of the active providers.
          verbose (bool, optional): If True, print additional information about the generated transactions.
          exclude_src (bool, optional): If True, exclude the source node from the set of target nodes.
      
      Returns:
          pandas.DataFrame: A DataFrame containing the generated transactions, with columns for the transaction ID, source node, target node, and transaction amount.
      """
            
      np.random.seed()
      if exclude_src :
        if src in set(node_variables['pub_key']):
           nodes = list(set(node_variables['pub_key']) - set([src]))
        else:
           nodes = list(set(node_variables['pub_key']))
      else :
        nodes = list(node_variables['pub_key'])      
      src_selected = np.random.choice(nodes, size=K, replace=True)
      if epsilon > 0 and len(active_providers) > 0 :
          n_prov = int(epsilon*K)
          trg_providers = sample_providers(src, n_prov, node_variables, active_providers, exclude_src=True)
          trg_rnd = np.random.choice(nodes, size=K-n_prov, replace=True)
          trg_selected = np.concatenate((trg_providers,trg_rnd))
          np.random.shuffle(trg_selected)
      else:
          trg_selected = np.random.choice(nodes, size=K, replace=True)

      transactions = pd.DataFrame(list(zip(src_selected, trg_selected)), columns=["src","trg"])
      transactions["amount_SAT"] = amount_in_satoshi
      transactions["transaction_id"] = transactions.index
      transactions = transactions[transactions["src"] != transactions["trg"]]
      if verbose:
          print("Number of loop transactions (removed):", K-len(transactions))
          print("Merchant target ratio:", len(transactions[transactions["target"].isin(active_providers)]) / len(transactions))
      return transactions[["transaction_id","src","trg","amount_SAT"]]
