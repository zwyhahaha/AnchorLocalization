import pandas as pd

dic = dict()
dic['head_true'] = pd.read_csv('head_True.csv')
dic['head_false'] = pd.read_csv('head_False.csv')
dic['foot_true'] = pd.read_csv('foot_True.csv')
dic['foot_false'] = pd.read_csv('foot_False.csv')

mapping = {'no_anchor_ankle': 'foot_false', 'no_anchor_head': 'head_false',
           'anchor_ankle': 'foot_true', 'anchor_head': 'head_true'
           }
ids = ['0', '1','2','3','4','5','6','overall']

# Collect table 3
table3 = pd.DataFrame(columns=['target_id','init','no_anchor_ankle','no_anchor_head','anchor_ankle','anchor_head'])
table3['target_id'] = ids
table3['init'] = dic['foot_false']['init_dist']
for key, value in mapping.items():
    table3[key] = dic[value]['opt_dist']
table3.to_csv('table3.csv', index=False)

# Collect table 4
table4 = pd.DataFrame(columns=['target_id','multi_init','multi_no_anchor','multi_anchor','single_init','single_no_anchor','single_anchor'])
table4['target_id'] = ids
table4['multi_init'] = dic['head_true']['multi_init_dist'] # Same as head_false
table4['multi_no_anchor'] = dic['head_false']['multi_opt_dist'] 
table4['multi_anchor'] = dic['head_true']['multi_opt_dist']
table4['single_init'] = dic['head_true']['single_init_dist'] # Same as head_false
table4['single_no_anchor'] = dic['head_false']['single_opt_dist']
table4['single_anchor'] = dic['head_true']['single_opt_dist']
table4.to_csv('table4.csv', index=False)

# Imp ratio version table 3
table3_ = pd.DataFrame(columns=['target_id','no_anchor_ankle','no_anchor_head','anchor_ankle','anchor_head'])
table3_['target_id'] = ids
for key, value in mapping.items():
    table3_[key] = dic[value]['imp_ratio']
table3_.to_csv('table3_imp.csv', index=False)

# Imp ratio version table 4
table4_ = pd.DataFrame(columns = ['target_id','multi_no_anchor','multi_anchor','single_no_anchor','single_anchor'])
table4_['target_id'] = ids
table4_['multi_no_anchor'] = dic['head_false']['multi_imp_ratio']
table4_['multi_anchor'] = dic['head_true']['multi_imp_ratio']
table4_['single_no_anchor'] = dic['head_false']['single_imp_ratio']
table4_['single_anchor'] = dic['head_true']['single_imp_ratio']
table4_.to_csv('table4_imp.csv', index= False)