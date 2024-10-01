import pandas as pd
import pickle
from utils.camera_simu import Camera
from angle_loc.angle_loc_batch_without_obs_init import batch_localization_case
from utils.data_preprocess_for_all_na import obs_construction, compress_data

def main(config, trace_data):
    cam = {}
    for id in config.cams:
        cam[id] = Camera(id, config.max_na, config.camera_dir, config.use_distort, config)
    obs_construction(cam, config, trace_data)
    result_data = []
    ids = trace_data['target_id'].unique()
    for id in ids:
        target_data = trace_data[trace_data['target_id'] == id]
        
        compressed_data = compress_data(config.t1,config.t2,target_data,cam,config)
        cam_num = []
        for t in range(config.t1, config.t2-config.batch_size):
            frame_data = compressed_data[compressed_data['t'] == t]
            if len(frame_data) == 0:
                continue
            nc = frame_data['nc'].tolist()[0]
            cam_num.append(nc)
            frame_problem = batch_localization_case(1,cam,frame_data['available_cam'].tolist(),
                            frame_data['init_loc'].tolist(),frame_data['target_pix'].tolist())
            target_loc = frame_data['target_loc'].tolist()[0].squeeze()
            init_loc = frame_data['init_loc'].tolist()[0]
            frame_locs = []
            for na in [4,6,8,10]:
                frame_loc = frame_problem.solve_frame_loc(config.rho,na).squeeze()
                frame_locs.append(frame_loc)
            no_anchor_loc = frame_problem.solve_basic_loc(config.rho, 'prox').squeeze()
            no_anchor_gt_loc = frame_problem.solve_basic_loc(config.rho, 'gt').squeeze()
            result_data.append({
                "target_id": id,
                "findex": t,
                "nc": nc,
                "target_loc": target_loc,
                "init_loc": init_loc,
                "frame_loc_4": frame_locs[0],
                "frame_loc_6": frame_locs[1],
                "frame_loc_8": frame_locs[2],
                "frame_loc_10": frame_locs[3],
                "no_anchor_loc": no_anchor_loc,
                "no_anchor_gt_loc": no_anchor_gt_loc
            })
    result_df = pd.DataFrame(result_data)

    error_combine = str(config.rx_error) + '_' + str(config.ry_error) + '_' + str(config.rz_error) + '_' + str(config.T_error) + '_' + str(config.K_error) + '_' + str(config.D_error)

    with open(f'exp_result/exp4/{error_combine}.pkl', 'wb') as file:
        pickle.dump(result_df, file)
    
    print('Experiment for Fig.4')
