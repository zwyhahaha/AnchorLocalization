import numpy as np
import pandas as pd
from utils.camera_real import realCamera
from angle_loc.angle_loc_batch import batch_localization_case
from utils.preprocess_real import obs_construction, compress_data
from utils.my_utils import real_loc_clip

def main(config, trace_data):
    cam = {}
    for id in config.cams:
        cam[id] = realCamera(id, config)
    obs_construction(cam, config, trace_data)
    n_target = 7
    record = []
    
    overall_init_dist = []
    overall_batch_dist = []
    overall_opt_dist = []
    overall_cam_num = []
    for i in range(n_target):
        selected_target_id = i
        obs_trace = trace_data[trace_data['gt_id'] == selected_target_id]
        
        compressed_data = compress_data(config.t1,config.t2,obs_trace,cam,config)
        init_loc = compressed_data["init_loc"].tolist()
        gt_loc = compressed_data["gt_loc"].tolist()
        opt_loc = compressed_data["opt_loc"].tolist()
        cam_num = compressed_data["nc"].tolist()
        overall_cam_num.extend(cam_num)
        cam_num = np.array(cam_num)
        batch_loc = []
        for t in range(len(compressed_data)-config.batch_size+1):
            batch_data = compressed_data[t:t+config.batch_size]
            batch_problem = batch_localization_case(config.batch_size,cam,batch_data['available_cam'].tolist(),
                        batch_data['opt_loc'].tolist(),batch_data['obs'].tolist())
            res = batch_problem.solve_batch_loc_real(config.batch_rho)
            batch_loc.append(real_loc_clip(res[0]))

        for t in range(config.batch_size-1):
            batch_loc.append(real_loc_clip(res[t+1]))

        init_dist = np.linalg.norm(np.array(init_loc)[:,0:3]-np.array(gt_loc)[:,0:3], axis=1)
        opt_dist = np.linalg.norm(np.array(opt_loc)[:,0:3]-np.array(gt_loc)[:,0:3], axis=1)
        batch_dist = np.linalg.norm(np.array(batch_loc)[:,0:3]-np.array(gt_loc)[:,0:3], axis=1)


        single_init_dist = init_dist[cam_num==1]
        single_opt_dist = opt_dist[cam_num==1]
        single_batch_dist = batch_dist[cam_num==1]

        multi_init_dist = init_dist[cam_num>1]
        multi_opt_dist = opt_dist[cam_num>1]
        multi_batch_dist = batch_dist[cam_num>1]

        overall_init_dist.extend(init_dist)
        overall_opt_dist.extend(opt_dist)
        overall_batch_dist.extend(batch_dist)

        # record each ID's result
        record.append({
            'target_id': selected_target_id,
            'single_batch(mean)': np.mean(single_batch_dist),
            'single_batch(std)': np.std(single_batch_dist),
            'single_opt/init_imp': len(np.where(single_opt_dist < single_init_dist)[0]) / len(single_init_dist),
            'single_batch/init_imp': len(np.where(single_batch_dist < single_init_dist)[0]) / len(single_init_dist),
            'multi_batch(mean)': np.mean(multi_batch_dist),
            'multi_batch(std)': np.std(multi_batch_dist),
            'multi_opt/init_imp': len(np.where(multi_opt_dist < multi_init_dist)[0]) / len(multi_init_dist),
            'multi_batch/init_imp': len(np.where(multi_batch_dist < multi_init_dist)[0]) / len(multi_init_dist)
        })

    # record overall result
    overall_batch_dist = np.array(overall_batch_dist)
    overall_opt_dist = np.array(overall_opt_dist)
    overall_init_dist = np.array(overall_init_dist)
    overall_cam_num = np.array(overall_cam_num)
    record.append({
        'target_id': 'overall',
        'single_batch(mean)': np.mean(overall_batch_dist[overall_cam_num==1]),
        'single_batch(std)': np.std(overall_batch_dist[overall_cam_num==1]),
        'single_batch/init_imp': len(np.where(overall_batch_dist[overall_cam_num==1] < overall_init_dist[overall_cam_num==1])[0]) / len(overall_init_dist[overall_cam_num==1]),
        'multi_batch(mean)': np.mean(overall_batch_dist[overall_cam_num>1]),
        'multi_batch(std)': np.std(overall_batch_dist[overall_cam_num>1]),
        'multi_batch/init_imp': len(np.where(overall_batch_dist[overall_cam_num>1] < overall_init_dist[overall_cam_num>1])[0]) / len(overall_init_dist[overall_cam_num>1])
    })

    pd.DataFrame(record).to_csv(f'exp_result/exp6/anchor_{config.use_anchor}_{config.batch_size}.csv')

    print('Experiment for Table 5')
    
