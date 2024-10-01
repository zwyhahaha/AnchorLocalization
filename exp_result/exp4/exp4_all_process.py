import os
import glob
import numpy as np
import pandas as pd
import pickle

params = ['Rx', 'Ry', 'T', 'D']
pattern_ = ['*_0.0_0.0_0.0_0.0_0.0.pkl', '0.0_*_0.0_0.0_0.0_0.0.pkl', '0.0_0.0_0.0_*_0.0_0.0.pkl','0.0_0.0_0.0_0.0_0.0_*.pkl']

for i in range(4):
    pattern = os.path.join(pattern_[i])
    pkl_files = glob.glob(pattern)
    file_names = [os.path.splitext(os.path.basename(file_path))[0] for file_path in pkl_files]
    max_na = 12

    records = []
    for file_path in pkl_files:
        with open(file_path, 'rb') as file:
            result = pickle.load(file)
        if 'no_anchor_gt_loc' in result.keys():
            name = os.path.splitext(os.path.basename(file_path))[0]
            def euclidean_distance(point1, point2):
                return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))
            result['init_dist'] = result.apply(lambda row: euclidean_distance(row['init_loc'], row['target_loc']), axis=1)
            for na in [4,6,8,10]:
                result[f'anchor_{na}_dist'] = result.apply(lambda row: euclidean_distance(row['target_loc'], row[f'frame_loc_{na}']), axis=1)
            result['no_anchor_dist'] = result.apply(lambda row: euclidean_distance(row['target_loc'], row['no_anchor_loc']), axis=1)
            result['no_anchor_gt_dist'] = result.apply(lambda row: euclidean_distance(row['target_loc'], row['no_anchor_gt_loc']), axis=1)
            single_data = result[result['nc']==1]
            multi_data = result[result['nc']>1]
            def get_metrics(df):
                init_dist_mean = df['init_dist'].mean()
                anchor_dists_mean = []
                for na in [4,6,8,10]:
                    anchor_dist_mean = df[f'anchor_{na}_dist'].mean()
                    anchor_dists_mean.append(anchor_dist_mean)
                no_anchor_dist_mean = df['no_anchor_dist'].mean()
                no_anchor_gt_dist_mean = df['no_anchor_gt_dist'].mean()

                anchor_dists_imp = []
                for na in [4,6,8,10]:
                    anchor_dist_imp = np.sum(df[f'anchor_{na}_dist']<=df['init_dist'])/len(df)
                    anchor_dists_imp.append(anchor_dist_imp)
                no_anchor_dist_imp = np.sum(df['no_anchor_dist']<=df['init_dist'])/len(df)
                no_anchor_gt_dist_imp = np.sum(df['no_anchor_gt_dist']<=df['init_dist'])/len(df)
                return [init_dist_mean, no_anchor_dist_mean, anchor_dists_mean, no_anchor_gt_dist_mean, no_anchor_dist_imp,anchor_dists_imp, no_anchor_gt_dist_imp]
            metric = get_metrics(result)
            single_metric = get_metrics(single_data)
            multi_metric = get_metrics(multi_data)
            records.append({
                "name": name,
                "init_dist_mean": metric[0],
                "no_anchor_dist_mean": metric[1],
                "anchor_dist_mean": metric[2],
                "no_anchor_gt_dist_mean": metric[3],
                "no_anchor_dist_imp": metric[4],
                "anchor_dist_imp": metric[5],
                "no_anchor_gt_dist_imp": metric[6],

                "multi_init_dist_mean": multi_metric[0],
                "multi_no_anchor_dist_mean": multi_metric[1],
                "multi_anchor_dist_mean": multi_metric[2],
                "multi_no_anchor_gt_dist_mean": multi_metric[3],
                "multi_no_anchor_dist_imp": multi_metric[4],
                "multi_anchor_dist_imp": multi_metric[5],
                "multi_no_anchor_gt_dist_imp": multi_metric[6],

                "single_init_dist_mean": single_metric[0],
                "single_no_anchor_dist_mean": single_metric[1],
                "single_anchor_dist_mean": single_metric[2],
                "single_no_anchor_gt_dist_mean": single_metric[3],
                "single_no_anchor_dist_imp": single_metric[4],
                "single_anchor_dist_imp": single_metric[5],
                "single_no_anchor_gt_dist_imp": single_metric[6],
            })
    record_df = pd.DataFrame(records)
    record_df.to_csv(f"./{params[i]}.csv")