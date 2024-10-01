import pandas as pd
from collections import defaultdict

res = defaultdict(list)
res_ = defaultdict(list)
for batch_size in [1,3,5,6]:
    file_path = f"./anchor_True_{batch_size}.csv"
    file = pd.read_csv(file_path)
    res['batch_size'].append(batch_size)
    overall = file.iloc[-1]
    res['multi_cam_dis_mean'].append(overall['multi_batch(mean)'])
    res['multi_cam_dis_std'].append(overall['multi_batch(std)'])
    res['single_cam_dis_mean'].append(overall['single_batch(mean)'])
    res['single_cam_dis_std'].append(overall['single_batch(std)'])

    res_['batch_size'].append(batch_size)
    res_['multi_cam_imp'].append(overall['multi_batch/init_imp'])
    res_['single_cam_imp'].append(overall['single_batch/init_imp'])


# record result for the table 5
pd.DataFrame(res).to_csv('Table5.csv', index=False)

# record result for the imp version of table 5
pd.DataFrame(res_).to_csv('Table5_imp.csv', index=False)
