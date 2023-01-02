import numpy as np
import pandas as pd

FEATURES = ['cofitness_cosub', 'cofitness_time_decay', 'num_appearance', 'num_cosub', 'coclick', 'cocart', 'coorder']
USER_FEATURES = ['num_sub', 'consistency', 'num_actions', 'pr']


item_features = ['num_clicks', 'num_carts', 'num_orders', 'pr']
recent_features = []
for i in range(7,0,-1):
    for j in range(3):
        recent_features.append(f'recent_day{i}_type{j}')

ITEM_FEATURES = [*item_features, * recent_features]

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

recent_features_id_map = dict(zip(
  recent_features, [len(feature_id_map) + i for i in range(len(recent_features))]
))

feature_id_map = {**feature_id_map, **recent_features_id_map}

def create_data(infer_data, infer = True):
  
  aggs = ['mean', 'var']
  
  agg_features_name = []
  quo_features_name = []
  
  for f in FEATURES:
    for agg in aggs: 
      agg_features_name.append(f + '_' + agg)
      quo_features_name.append('qou_' + f + '_' + agg)
      
  columns = ['user', 'item', 'type',
             *FEATURES,
             *[f if f != 'pr' else 'item_pr' for f in ITEM_FEATURES],
             *quo_features_name,  
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


    
  columns = [
    'user', 'item', 'fiteness', 
    *INTERACTION_FEATURES,
    *[f if f != 'pr' else 'user_pr' for f in USER_FEATURES],
    *[f if f != 'pr' else 'item_pr' for f in ITEM_FEATURES],
  ]
  

  candidates = pd.DataFrame(data = infer_data, columns = columns, copy=False)
 
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

