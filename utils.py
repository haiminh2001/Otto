import pandas as pd
import numpy as np
def get_score(df, session_list, split = False, n = 20):
    submission = df.copy()
    test_labels = pd.read_parquet('/home/anhphantq/otto/splitted_data/test_labels.parquet')
    test_labels = test_labels[(test_labels['session'] <= np.amax(session_list)) & (test_labels['session'] >= np.amin(session_list))]
    
    submission['session'] = submission.session_type.apply(lambda x: int(x.split('_')[0]))
    
    session = set(test_labels['session'].unique().tolist())
    for j in session_list:
      assert j in session


    submission['type'] = submission.session_type.apply(lambda x: x.split('_')[1])
    
    if split:
      submission.labels = submission.labels.apply(lambda x: [int(i) for i in x.split(' ')[:n]])
    else:
      submission.labels = submission.labels.apply(lambda x: x[:n])

      

    test_labels = test_labels.merge(submission, how='left', on=['session', 'type'])
    test_labels['hits'] = test_labels.apply(lambda x: len(set(x.ground_truth).intersection(set(x.labels))), axis=1).clip(0, 20)
    test_labels['gt_count'] = test_labels.ground_truth.str.len().clip(0,20)

    del submission

    recall_per_type = test_labels.groupby(['type'])['hits'].sum() / test_labels.groupby(['type'])['gt_count'].sum() 

    print (f"Score : {(recall_per_type * pd.Series({'clicks': 0.10, 'carts': 0.30, 'orders': 0.60})).sum()}")
    return recall_per_type

def get_recall20(df, session_list, split = False, n = 20):
    submission = df.copy()
    test_labels = pd.read_parquet('/home/anhphantq/otto/splitted_data/test_labels.parquet')
    test_labels = test_labels[(test_labels['session'] <= np.amax(session_list)) & (test_labels['session'] >= np.amin(session_list))]
    print(test_labels.columns)
    submission['session'] = submission.session_type.apply(lambda x: int(x.split('_')[0]))
    
    session = set(test_labels['session'].unique().tolist())
    for j in session_list:
      assert j in session


    submission['type'] = submission.session_type.apply(lambda x: x.split('_')[1])
    
    if split:
      submission.labels = submission.labels.apply(lambda x: [int(i) for i in x.split(' ')[:n]])
    else:
      submission.labels = submission.labels.apply(lambda x: x[:n])

      

    test_labels = test_labels.merge(submission, how='left', on=['session', 'type'])
    test_labels['hits'] = test_labels.apply(lambda x: len(set(x.ground_truth).intersection(set(x.labels))), axis=1).clip(0, 20)
    test_labels['gt_count'] = test_labels.ground_truth.str.len().clip(0,20)

    del submission
    test_labels['recall'] = (test_labels['hits'] / test_labels['gt_count'])
    recall_per_type = test_labels.groupby(['type'])['recall'].mean()

    # print (f"Score : {(recall_per_type * pd.Series({'clicks': 0.10, 'carts': 0.30, 'orders': 0.60})).sum()}")
    return recall_per_type, test_labels

def shuffle(l):  
    perm = np.random.permutation(len(l))  
    l[:] = [l[j] for j in perm]  
    return perm

def unshuffle(L, perm):  
    assert len(L[0]) == len(perm), (len(L[0]), len(perm))
    res = [[None] * len(l) for l in L]
    for i, j in enumerate(perm):
        for k, l in enumerate(L):
          res[k][j] = l[i]
          
    for i, l in enumerate(L):
      l[:] = res[i]
      

def create_submission(clicks, carts, orders, session_list, MAX_SESSION, JOIN):
    click_new_append = [' '.join([str(int(i) - MAX_SESSION) for i in a]) for a in clicks] if JOIN else [[int(j) - MAX_SESSION for j in i] for i in clicks]
    cart_new_append = [' '.join([str(int(i) - MAX_SESSION) for i in a]) for a in carts] if JOIN else [[int(j) - MAX_SESSION for j in i] for i in carts]
    order_new_append = [' '.join([str(int(i) - MAX_SESSION) for i in a]) for a in orders] if JOIN else [[int(j) - MAX_SESSION for j in i] for i in orders]  
    submission1 = pd.DataFrame({'session_type': session_list, 'labels': click_new_append})
    submission2 = pd.DataFrame({'session_type': session_list, 'labels': cart_new_append})
    submission3 = pd.DataFrame({'session_type': session_list, 'labels': order_new_append})
    submission1['session_type'] = submission1['session_type'].apply(lambda x: str(x) + '_clicks')
    submission2['session_type'] = submission2['session_type'].apply(lambda x: str(x) + '_carts')
    submission3['session_type'] = submission3['session_type'].apply(lambda x: str(x) + '_orders')
    final = pd.concat([submission1, submission2, submission3])
    del submission1, submission2, submission3
    return final