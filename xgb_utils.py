import numpy as np
import pandas as pd

FEATURES = ['cofitness_cosub', 'cofitness_time_decay', 'num_appearance', 'num_cosub', 'coclick', 'cocart', 'coorder']
USER_FEATURES = ['num_sub', 'consistency', 'num_actions', 'pr']
ITEM_FEATURES = ['num_clicks', 'num_carts', 'num_orders', 'pr']
# INTERACTION_FEATURES = ['user_clicks', 'user_carts', 'user_orders']

INTERACTION_FEATURES = ['user_clicks', 'user_carts', 'user_orders', 'user_num_sub', 'user_time_decay', 'user_lts', 'user_fts']
feature_id_map = {
  'num_sub' : 0,
  'consistency': 1,
  'num_actions': 2,
  'num_clicks': 3,
  'num_carts': 4, 
  'num_orders': 5,
  'pr': 6,
}

def create_data(infer_data, infer = True):
  
  aggs = ['mean', 'var']
  
  agg_features_name = []
  quo_features_name = []
  prod_features_name = []
  
  for f in FEATURES:
    for agg in aggs: 
      agg_features_name.append(f + '_' + agg)
      quo_features_name.append('qou_' + f + '_' + agg)
      
  for i in range(len(FEATURES)):
    for j in range(i + 1, len(FEATURES)):
      prod_features_name.append('prod_' + FEATURES[i] + '_' + FEATURES[j]) 
  
  columns = [
              *FEATURES,
             *[f if f != 'pr' else 'item_pr' for f in ITEM_FEATURES],
             *quo_features_name,  
             *prod_features_name,
             ]
  
  

  rank_features_name = ['rank_' + f for f in columns]
  columns = ['user', 'item', 'type',*columns, 
             *rank_features_name,
             *agg_features_name,
             *[f if f != 'pr' else 'user_pr' for f in USER_FEATURES],]
  


  assert len(columns) == infer_data.shape[1], (len(columns), infer_data.shape[1])


  candidates = pd.DataFrame(data = infer_data, columns = columns, copy=False)
  candidates['clicks'] = np.where(candidates['type'] == 0, 1, 0)
  candidates['carts'] = np.where(candidates['type'] == 1, 1, 0)
  candidates['orders'] = np.where(candidates['type'] == 2, 1, 0)
  if infer:
    t =  candidates['type'].values
    del candidates['type']
    return candidates, t
  else:
    # candidates ['item'] = candidates['item'] - 100000000
    return candidates




def create_test_data(infer_data, infer = True):

  data =  {
      'user': infer_data[:,0].astype(np.int32),
      'item': infer_data[:, 1].astype(np.int32),
      'fitness': infer_data[:,2].astype(float),
  }


  for f in INTERACTION_FEATURES: 
    data[f] = infer_data[:, len(data.keys())]


  for f in USER_FEATURES: 
    f = f if f != 'pr' else 'user_pr'
    data[f] = infer_data[:, len(data.keys())]

  for f in ITEM_FEATURES: 
    f = f if f != 'pr' else 'item_pr'
    data[f] = infer_data[:, len(data.keys())]
  

  candidates = pd.DataFrame(data)
 
  if not infer:
    candidates ['item'] = candidates['item'] - 100000000
  return candidates


def get_len_group(idx, num_cands):
  groups = []
  i = 0
  n = len(idx)

  while i < n:
    g = num_cands[idx[i]]
   
    groups.append(g) 
    i += g
    
  assert np.sum(groups) == n, (np.sum(groups), n)
  return groups

