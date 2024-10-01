import json
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.camera_real import realCamera
import scipy.optimize as op


cam_dir = 'data/real/monitors_json'
camid_list = ['C142c79d90039ce7a', 'C78fcb895887a3935', 'Ccf4692cb9838d53c', 'C58ea83521a4169a3', 'C9f859b0853a4bfe0', 'Cf9ed46a5770558d3']

class Config():
    def __init__(self):
        self.camera_dir=cam_dir
        self.use_distort = True
        self.det_type = 'head'
        self.start_method = 'multi_camera_mean'
        self.h = 0.7
config = Config()

def get_pix_loc(cam, pix, *anchor_height):
    loc = cam.get_worldpt_from_pix(pix, *anchor_height)
    return loc

def get_pix_angle(cam, pix):
    angle,_ = cam.pix2angles(pix)
    return angle

def obs_construction(cam, config, trace_data: pd.DataFrame) -> pd.DataFrame:
    trace_data['loc'] = trace_data.apply(lambda row: get_pix_loc(cam[row['cam_id']], row['pix'], row["pts3d"][-1]), axis=1)
    trace_data['angle'] = trace_data.apply(lambda row: get_pix_angle(cam[row['cam_id']], row['pix']), axis=1)

def obs_loc(start,obs,cams,available_cam,config):
    def loss_function(z):
        res = 0
        if config.det_type == 'foot':
            z[2] = 0
        elif len(available_cam) == 1:
            z[2] = config.h
        for i,id in enumerate(available_cam):
            z = np.array(z).reshape(3, 1)
            z_pix = cams[id].world2pix(z)
            z_pix = np.array(z_pix)
            res += np.linalg.norm(z_pix-obs[i])
        return res
    solve_dict = op.minimize(loss_function, start, constraints=cons, method='SLSQP')
    x = np.array(solve_dict['x'])
    return x

def regression_loc(nc,polar_angles,camera_loc):
    A = np.zeros((3*nc, nc+3))
    b = np.zeros((3*nc))
    for j in range(nc):
        A[3*j:3*j+3, j] = np.cos(polar_angles[j, :])
        A[3*j, nc] = -1
        A[3*j+1, nc+1] = -1
        A[3*j+2, nc+2] = -1
        b[3*j:3*j+3] = -camera_loc[j,:]
    start = np.linalg.lstsq(A, b, rcond=10**(-8))[0][nc:]
    return start

def find_weights_pos(x, points, ord):
    def objective_function(w):
        return np.linalg.norm(x - np.dot(points.T, w), ord=ord)**2
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    constraints.append({'type': 'ineq', 'fun': lambda w: w})
    initial_weights = np.ones(points.shape[0]) / points.shape[0]
    result = op.minimize(objective_function, initial_weights, constraints=constraints)
    optimal_weights = result.x
    return optimal_weights

def compress_data(is_single, anchor_list, anchor_df, cams):
    max_nc = 6
    result = []
    for anchor_id in anchor_list:
        if anchor_id == 'c':
            continue
        data = anchor_df[anchor_df['anchor_id']==anchor_id]
        gt_loc = data['pts3d'].to_list()[0]
        available_cam = data['cam_id'].tolist()[0:max_nc]
        nc = len(available_cam)
        case1 = (is_single == 1 and nc == 1)
        case2 = (is_single == 0 and nc > 1)
        if case1 or case2:
            camera_loc = np.array([cams[id].cam_loc for id in available_cam])[0:max_nc]
            target_pix = np.array(data['pix'].to_list())[0:max_nc]
            target_angle = np.array(data['angle'].to_list())[0:max_nc]
            if config.start_method == 'multi_camera_mean':
                observed_locs = np.array(data['loc'].to_list())
                init_loc = observed_locs.mean(axis=0)
            elif config.start_method == 'regression':
                if len(available_cam) == 1:
                    init_loc = np.array(data['loc'].to_list()[0])
                else:
                    init_loc = regression_loc(nc,target_angle,camera_loc)
            obs = []
            pix = []
            for j,id in enumerate(available_cam):
                cam = cams[id]
                anchor_loc = cam.anchors
                w = find_weights_pos(init_loc,anchor_loc,ord=2)
                w = np.array(w).reshape(1,len(w))
                target_obs = target_pix[j]
                anchor_term = np.dot(w,(cam.prox_anchor_pix-cam.gt_anchor_pix)).squeeze()
                obs.append(target_obs + anchor_term)
                pix.append(target_obs)
            obs = np.array(obs)
            opt_loc = obs_loc(init_loc,obs,cams,available_cam,config)
            no_anchor_loc = obs_loc(init_loc,pix,cams,available_cam,config)
            result.append({
                        'anchor_id': anchor_id,
                        'nc': nc,
                        'gt_loc': gt_loc,
                        # 'available_cam': available_cam,
                        'init_loc': init_loc,
                        # 'target_pix': target_pix,
                        # 'obs': obs,
                        'opt_loc': opt_loc,
                        'no_anchor_loc': no_anchor_loc
                    })
    return result




anchor_data = []
for cam_id in camid_list:
    filename = '{}/{}.json'.format(cam_dir,cam_id)
    with open(filename, 'r') as f:
        cam_cfg = json.load(f)
    for i,anchor_id in enumerate(cam_cfg["anchors"]["id"]):
        anchor_data.append({
            'anchor_id': anchor_id,
            'pix': cam_cfg["anchors"]["pts2D"][i],
            'pts3d': cam_cfg["anchors"]["pts3D_measure"][i],
            'cam_id': cam_id
        })
    
anchor_df = pd.DataFrame(anchor_data)
# print(anchor_df)

anchor_list = anchor_df['anchor_id'].unique().tolist()

cams = {}
for id in camid_list:
    cams[id] = realCamera(id, config)

obs_construction(cams,config,anchor_df)

cons = [
        {'type': 'ineq', 'fun': lambda z: 11 - z[0]},
        {'type': 'ineq', 'fun': lambda z: z[0] + 15},
        {'type': 'ineq', 'fun': lambda z: 5 - z[1]},
        {'type': 'ineq', 'fun': lambda z: z[1] + 1.5},
        {'type': 'ineq', 'fun': lambda z: z[2] - 0},    # z[2] >= 1.4
        {'type': 'ineq', 'fun': lambda z: 4 - z[2]}       # z[2] <= 2
    ]

def main_run(is_single):    
    result = compress_data(is_single, anchor_list, anchor_df, cams)
    compressed_data = pd.DataFrame(result)
    init_loc = compressed_data["init_loc"].tolist()
    gt_loc = compressed_data["gt_loc"].tolist()
    opt_loc = compressed_data["opt_loc"].tolist()
    no_anchor_loc = compressed_data["no_anchor_loc"].tolist()
    cam_num = compressed_data["nc"].tolist()
    init_dist = np.linalg.norm(np.array(init_loc)[:,0:2]-np.array(gt_loc)[:,0:2], axis=1)
    init_dist_mean = np.mean(init_dist)
    opt_dist = np.linalg.norm(np.array(opt_loc)[:,0:2]-np.array(gt_loc)[:,0:2], axis=1)
    opt_dist_mean = np.mean(opt_dist)
    no_anchor_dist = np.linalg.norm(np.array(no_anchor_loc)[:,0:2]-np.array(gt_loc)[:,0:2], axis=1)
    no_anchor_dist_mean = np.mean(no_anchor_dist)
    name = 'single' if is_single else 'multi'
    print(f'{name} camera case:')
    print(init_dist_mean,no_anchor_dist_mean,opt_dist_mean)
    anchor_num = 16 if is_single else 19
    camera_type = 'Single' if is_single else 'Multi'
    with open('./exp_result/exp2/table2.txt', 'a') as f:
        f.write(f'\n{camera_type},{anchor_num},{init_dist_mean},{no_anchor_dist_mean},{opt_dist_mean}')

main_run(1)
main_run(0)