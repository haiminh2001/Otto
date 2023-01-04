import numpy as np
import pandas as pd
from utils import unshuffle
import xgboost as xgb

#need to careful distribution when 3 -> 4 weeks in train
#all sub, appearance, click, cart, order, degree lts fts



#list all features
FEATURES = ['cofitness_cosub', 'cofitness_time_decay', 'num_appearance', 'num_cosub', 'coclick', 'cocart', 'coorder']
USER_FEATURES = ['num_sub', 'consistency', 'num_actions', 'pr', 'recent_pr', 'degree']


item_features = ['num_clicks', 'num_carts', 'num_orders', 'pr', 'recent_pr', 'degree']
recent_features = []
glob_features = ['item_glob_' + f for f in[
    'last_action', 'first_action', 
    'time_decay_sum', 'time_decay_sum_click', 
    'time_decay_sum_cart', 'time_decay_sum_order', 'item_glob_durability'
]]


for i in range(7,0,-1):
    for j in range(3):
        recent_features.append(f'recent_day{i}_type{j}')

ITEM_FEATURES = [*item_features, * recent_features, *glob_features]

INTERACTION_FEATURES = ['inter_clicks', 'inter_carts', 'inter_orders', 'inter_num_sub', 'inter_time_decay', 'inter_lts', 'inter_fts', 'inter_durability', 'inter_num_interacts']

shared_features = ['pr', 'recent_pr', 'degree']

feature_id_map = {
  'num_sub' : 0,
  'consistency': 1,
  'num_actions': 2,
  'num_clicks': 3,
  'num_carts': 4, 
  'num_orders': 5,
  'degree': 6,
  'pr': 7,
  'recent_pr': 8,
}

recent_features_id_map = dict(zip(
  recent_features, [len(feature_id_map) + i for i in range(len(recent_features))]
))

glob_features_id_map = dict(zip(
  glob_features, [len(feature_id_map) + len(recent_features_id_map) + i for i in range(len(glob_features))]
))


feature_id_map = {**feature_id_map, **recent_features_id_map, **glob_features_id_map}



aggs = ['mean', 'var']

agg_features_name = []
quo_features_name = []

for f in FEATURES:
  for agg in aggs: 
    agg_features_name.append(f + '_' + agg)
    quo_features_name.append('qou_' + f + '_' + agg)



columns = ['user', 'item', 'type',
            
            
            
            *FEATURES,
            *[f if f not in shared_features else 'item_' + f for f in ITEM_FEATURES],
            *quo_features_name,  
            *agg_features_name,
            *[f if f not in shared_features else 'user_' + f for f in USER_FEATURES],]



test_columns = [
    'user', 'item', 'fitness', 
    *INTERACTION_FEATURES,
    *[f if f not in shared_features else 'user_' + f for f in USER_FEATURES],
    *[f if f not in shared_features else 'item_' + f for f in ITEM_FEATURES],
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
  dtest = xgb.DMatrix(data=df_infer_data.iloc[:, 2:])

  for model in models:

      preds += model.predict(dtest) / len(models)
      
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

  for model in models:

      preds += model.predict(dtest) / len(models)
      
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
    