import numpy as np
import pandas as pd
from utils import unshuffle
import xgboost as xgb

#need to careful distribution when 3 -> 4 weeks in train
#all sub, appearance, click, cart, order, degree lts fts



#list all features
FEATURES = ['cofitness_cosub', 'cofitness_time_decay', 'num_cosub', 'coclick', 'cocart', 'coorder', 'num_appearance']
USER_FEATURES = ['num_sub', 'consistency', 'num_actions', 'degree', 'pr', 'recent_degree', 'recent_pr']


item_features = ['num_clicks', 'num_carts', 'num_orders', 'degree', 'pr', 'recent_num_clicks', 'recent_num_carts', 'recent_num_orders', 'recent_degree', 'recent_pr']
recent_features = []
glob_features = ['item_glob_' + f for f in[
    'last_action', 'first_action', 
    'time_decay_sum', 'time_decay_sum_click', 
    'time_decay_sum_cart', 'time_decay_sum_order', 'item_glob_durability'
]]

popular_features = ['popular_clicks', 'popular_carts', 'popular_orders', 'popular_num_appearance']

for i in range(7,0,-1):
    for j in range(3):
        recent_features.append(f'recent_day{i}_type{j}')

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


feature_id_map = {**feature_id_map, **recent_features_id_map, **glob_features_id_map}



norm_features_name = []

for f in FEATURES:
  norm_features_name.append('qou_' + f + '_mean' )
  norm_features_name.append('standardized_' + f)

qou_features_name = []
for  f in FEATURES:
  qou_features_name.append('qou_' + f + '_sqrt_num_cousers')

for  f in popular_features:
  qou_features_name.append('qou_' + f + '_sqrt_num_neighbourhood')

columns = ['user', 'item', 'type',
            *[f if f not in shared_features else 'item_' + f for f in ITEM_FEATURES],
            *popular_features,
            *norm_features_name,  
            *qou_features_name,
            *[f if f not in shared_features else 'user_' + f for f in USER_FEATURES],
            "num_couser_edges", 'num_cousers'
            ]



test_columns = [
    'user', 'item', 'fitness', 
    *INTERACTION_FEATURES,
    *[f if f not in shared_features else 'user_' + f for f in USER_FEATURES],
    *[f if f not in shared_features else 'item_' + f for f in ITEM_FEATURES],
    *RECENT_INTERACT_FEATURES,
  ]

def create_data(infer_data, infer = True):
  

      

  
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
    return candidates




def create_test_data(infer_data, infer = True, max_session = None):
  candidates = pd.DataFrame(data = infer_data, columns = test_columns, copy=False)
 
  if not infer:
    candidates ['item'] = candidates['item'] - max_session
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


def xgboost_infer(infer_data, perm, models):
  df_infer_data, np_type = create_data(infer_data)
  preds = np.zeros(df_infer_data.shape[0])
  dtest = xgb.DMatrix(data=df_infer_data.iloc[:, 2: -3])

  for model in models:

      preds += model.predict(dtest, iteration_range=(0, model.best_ntree_limit)) / len(models)
      
  predictions = df_infer_data[['user','item']].copy()
  predictions['type'] = np_type
  predictions['pred'] = preds

  predictions = predictions.sort_values(['user', 'pred'], ascending=[True, False]).reset_index(drop=True)
  predictions['n'] = predictions.groupby('user').item.cumcount().astype('int8')
  predictions = predictions.loc[predictions.n<20]
  sub = predictions.groupby('user').item.apply(list)
  sub = sub.to_frame().reset_index()
  items = sub.item.values.tolist()
  users = sub.user.values
  unshuffle([items, users], perm)
  return items, users


def xgboost_test_infer(infer_data, perm, models):
  df_infer_data = create_test_data(infer_data)
  preds = np.zeros(df_infer_data.shape[0])
  dtest = xgb.DMatrix(data=df_infer_data.iloc[:, 2:])
  for c in ['clicks', 'carts', 'orders']:
    assert c not in df_infer_data.columns[2:]

  for model in models:

      preds += model.predict(dtest, iteration_range=(0, model.best_ntree_limit)) / len(models)
      
  predictions = df_infer_data[['user','item']].copy()
  predictions['pred'] = preds

  predictions = predictions.sort_values(['user', 'pred'], ascending=[True, False]).reset_index(drop=True)
  predictions['n'] = predictions.groupby('user').item.cumcount().astype('int8')
  predictions = predictions.loc[predictions.n<20]
  sub = predictions.groupby('user').item.apply(list)
  sub = sub.to_frame().reset_index()
  items = sub.item.values.tolist()
  users = sub.user.values
  unshuffle([items, users], perm)
  return items, users
    