import numpy as np
import pandas as pd
from utils import unshuffle
import xgboost as xgb
from itertools import permutations




#list all features
FEATURES = ['num_cosub',
            'coclick_sub_coef', 'cocart_sub_coef', 'coorder_sub_coef',
            'coclick_time_decay', 'cocart_time_decay', 'coorder_time_decay',
            'num_appearance']
USER_FEATURES = ['num_sub', 'consistency', 'num_actions', 'degree', 'pr', 'recent_degree', 'recent_pr']


item_features = ['num_clicks', 'num_carts', 'num_orders', 'degree', 'pr', 'recent_num_clicks', 'recent_num_carts', 'recent_num_orders', 'recent_degree', 'recent_pr']
glob_features = ['item_glob_' + f for f in[
    'last_action', 'first_action', 
    'time_decay_sum', 'time_decay_sum_click', 
    'time_decay_sum_cart', 'time_decay_sum_order', 'item_glob_durability'
]]

popular_features = ['popular_clicks', 'popular_carts', 'popular_orders', 'popular_num_appearance']


lincom_features_name = []

lincom_weights = list(permutations([1,3,6]))

for w1, w2, w3 in lincom_weights:
  lincom_features_name.extend([f'lincom_sub_coef_{w1}_{w2}_{w3}',
                                f'lincom_time_decay_{w1}_{w2}_{w3}',
                                ])

lincom_recent_features_name = []

recent_features = []

for i in range(7,0,-1):
  for j in range(3):
      recent_features.append(f'recent_day{i}_type{j}')
      
  for w1, w2, w3 in lincom_weights:
    lincom_recent_features_name.append(f'lincom_recent_day_{i}_{w1}_{w2}_{w3}')
          
        




ITEM_FEATURES = [*item_features, * recent_features, *glob_features]

INTERACTION_FEATURES = ['inter_clicks', 'inter_carts', 'inter_orders', 'inter_num_sub', 'inter_time_decay', 'inter_lts', 'inter_fts', 'inter_durability', 'inter_num_interacts']

RECENT_INTERACT_FEATURES = []

for i in range(7,0,-1):
    for j in range(3):
        RECENT_INTERACT_FEATURES.append(f'recent_inter_day{i}_type{j}')

shared_features = ['pr', 'recent_pr', 'degree', 'recent_degree']

feature_id_map = {
  'num_sub' : 0,
  'consistency': 1,
  'num_actions': 2,
  'num_clicks': 3,
  'num_carts': 4, 
  'num_orders': 5,
  'degree': 6,
  'pr': 7,
  'recent_num_clicks': 8,
  'recent_num_carts': 9, 
  'recent_num_orders': 10,
  'recent_degree': 11,
  'recent_pr': 12,
}



recent_features_id_map = dict(zip(
  recent_features, [len(feature_id_map) + i for i in range(len(recent_features))]
))

glob_features_id_map = dict(zip(
  glob_features, [len(feature_id_map) + len(recent_features_id_map) + i for i in range(len(glob_features))]
))


index_recent_features = (len(feature_id_map), len(feature_id_map) + len(recent_features_id_map))

feature_id_map = {**feature_id_map, **recent_features_id_map, **glob_features_id_map}



norm_features_name = []

for f in FEATURES + lincom_features_name:
  norm_features_name.append('qou_' + f + '_mean' )

qou_features_name = []
for  f in FEATURES:
  qou_features_name.append('qou_' + f + '_sqrt_num_cousers')

for  f in popular_features:
  qou_features_name.append('qou_' + f + '_sqrt_num_neighbourhood')


level2_columns = ['user', 'item', 
            *[f if f not in shared_features else 'item_' + f for f in ITEM_FEATURES],
            *popular_features,
            *norm_features_name,  
            *qou_features_name,
            *[f if f not in shared_features else 'user_' + f for f in USER_FEATURES],
            *lincom_recent_features_name,
            ]

level1_columns = [
    'item', 'fitness', 
    *INTERACTION_FEATURES,
    *RECENT_INTERACT_FEATURES,
    'is_level1',
  ]

def create_level2_data(infer_data):
  
  assert len(level2_columns) == infer_data.shape[1], (len(level2_columns), infer_data.shape[1])
  candidates = pd.DataFrame(data = infer_data, columns = level2_columns, copy=False)
  return candidates


def create_level1_data(infer_data):
  candidates = pd.DataFrame(data = infer_data, columns = level1_columns, copy=False)
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


def alpha_xgboost_infer(df_infer_data, models, perm):

  preds = np.zeros(df_infer_data.shape[0])
  dtest = xgb.DMatrix(data=df_infer_data.iloc[:, 2: ])
  for model in models:

      preds += model.predict(dtest, iteration_range=(0, model.best_ntree_limit)) / len(models)
      
  predictions = df_infer_data[['user','item']].copy()
  predictions['pred'] = preds

  predictions = predictions.sort_values(['user', 'pred'], ascending=[True, False]).reset_index(drop=True)
  predictions['n'] = predictions.groupby('user').item.cumcount()
  predictions = predictions.loc[predictions.n<20]

  sub = predictions.groupby('user').item.apply(list)
  sub = sub.to_frame().reset_index()
  items = sub.item.values.tolist()
  users = sub.user.values
  unshuffle([items, users], perm)
  return items, users
