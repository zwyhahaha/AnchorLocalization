import numpy as np
import pandas as pd
from utils.camera_real import realCamera
from utils.preprocess_real import obs_construction, compress_data
from utils.my_utils import get_metric

def main(config, trace_data):
    cam = {}
    for id in config.cams:
        cam[id] = realCamera(id, config)
    obs_construction(cam, config, trace_data)
    n_target = 7
    record = []
    all_cam_num = []
    all_init_dist = []
    all_opt_dist = []

    for i in range(n_target):
        selected_target_id = i
        obs_trace = trace_data[trace_data['gt_id'] == selected_target_id]
        compressed_data = compress_data(config.t1,config.t2,obs_trace,cam,config)
        init_loc = compressed_data["init_loc"].tolist()
        gt_loc = compressed_data["gt_loc"].tolist()
        opt_loc = compressed_data["opt_loc"].tolist()
        cam_num = compressed_data["nc"].tolist()
        init_dist = np.linalg.norm(np.array(init_loc)[:,0:3]-np.array(gt_loc)[:,0:3], axis=1)
        opt_dist = np.linalg.norm(np.array(opt_loc)[:,0:3]-np.array(gt_loc)[:,0:3], axis=1)
        result = get_metric(opt_dist,init_dist,cam_num)
        record.append({
            'target_id': selected_target_id,
            'init_dist': result["full_data"][1],
            'opt_dist': result["full_data"][0],
            'imp_ratio': result["full_data"][2],
            'single_init_dist': result["single_camera"][1],
            'single_opt_dist': result["single_camera"][0],
            'single_imp_ratio': result["single_camera"][2],
            'multi_init_dist': result["multiple_cameras"][1],
            'multi_opt_dist': result["multiple_cameras"][0],
            'multi_imp_ratio': result["multiple_cameras"][2],
        })
        all_cam_num.extend(cam_num)
        all_init_dist.extend(init_dist)
        all_opt_dist.extend(opt_dist)
    
    result = get_metric(all_opt_dist,all_init_dist,all_cam_num)
    record.append({
        'target_id': 'overall',
        'init_dist': result["full_data"][1],
        'opt_dist': result["full_data"][0],
        'imp_ratio': result["full_data"][2],
        'single_init_dist': result["single_camera"][1],
        'single_opt_dist': result["single_camera"][0],
        'single_imp_ratio': result["single_camera"][2],
        'multi_init_dist': result["multiple_cameras"][1],
        'multi_opt_dist': result["multiple_cameras"][0],
        'multi_imp_ratio': result["multiple_cameras"][2]
    })

    pd.DataFrame(record).to_csv(f'exp_result/exp3/{config.det_type}_{config.use_anchor}.csv')
    print('finished')


    
