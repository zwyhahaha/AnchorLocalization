import pandas as pd
import ast
import numpy as np
import scipy.optimize as op
from utils.camera_real import calculate_angle
from utils.my_utils import real_loc_clip

def get_pix_loc(cam, det):
    loc, pix_head, pix_foot = cam.get_worldpt_from_det(det)
    return pd.Series([loc, pix_head, pix_foot])

def get_pix_vec(cam, pix_head, pix_foot):
    vec_head = cam.pix2vec_w(pix_head).reshape(-1)
    vec_foot = cam.pix2vec_w(pix_foot).reshape(-1)
    return pd.Series([vec_head, vec_foot])

def get_pix_angle(cam, pix_head, pix_foot):
    angle_head,anchor_angle_head = cam.pix2angles(pix_head)
    angle_foot,anchor_angle_foot = cam.pix2angles(pix_foot)
    return pd.Series([angle_head,anchor_angle_head,angle_foot,anchor_angle_foot])

def obs_construction(cam, config, trace_data: pd.DataFrame) -> pd.DataFrame:
    trace_data[['loc','pix_head','pix_foot']] = trace_data.apply(lambda row: get_pix_loc(cam[row['cam_id']], row['det']), axis=1)
    trace_data[['vec_head','vec_foot']] = trace_data.apply(lambda row: get_pix_vec(cam[row['cam_id']], row['pix_head'], row['pix_foot']), axis=1)
    trace_data[['angle_head','anchor_angle_head','angle_foot','anchor_angle_foot']] = trace_data.apply(lambda row: get_pix_angle(cam[row['cam_id']], row['pix_head'], row['pix_foot']), axis=1)
    if config.det_type == 'head':
        trace_data['pix'] = trace_data['pix_head']
        trace_data['vec'] = trace_data['vec_head']
        trace_data['angle'] = trace_data['angle_head']
        trace_data['anchor_angle'] = trace_data['anchor_angle_head']
    elif config.det_type == 'foot':
        trace_data['pix'] = trace_data['pix_foot']
        trace_data['vec'] = trace_data['vec_foot']
        trace_data['angle'] = trace_data['angle_foot']
        trace_data['anchor_angle'] = trace_data['anchor_angle_foot']
    else:
        raise NotImplementedError

def find_weights_pos(x, points):
    def objective_function(w):
        return np.linalg.norm(x - np.dot(points.T, w), ord=2)**2
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    constraints.append({'type': 'ineq', 'fun': lambda w: w})
    initial_weights = np.ones(points.shape[0]) / points.shape[0]
    result = op.minimize(objective_function, initial_weights, constraints=constraints)
    optimal_weights = result.x
    return optimal_weights

def find_weights_norm(x, points, lamda=10):
    def objective_function(w):
        return lamda*np.linalg.norm(w)**2 + np.linalg.norm(x - np.dot(points.T, w), ord = 2)**2
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    # constraints.append({'type': 'eq', 'fun': lambda w: x - np.dot(points.T, w)})
    initial_weights = np.ones(points.shape[0]) / points.shape[0]
    result = op.minimize(objective_function, initial_weights, constraints=constraints)
    hat_x = np.dot(points.T, result.x)
    eps = np.linalg.norm(hat_x - x)
    # if eps>0.5 and not relaxed:
    optimal_weights = result.x
    return optimal_weights

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

def regression_foot_loc(nc,polar_angles,camera_loc):
    A = np.zeros((3*nc, nc+2))
    b = np.zeros((3*nc))
    for j in range(nc):
        A[3*j:3*j+3, j] = np.cos(polar_angles[j, :])
        A[3*j, nc] = -1
        A[3*j+1, nc+1] = -1
        b[3*j:3*j+3] = -camera_loc[j,:]
    start = np.linalg.lstsq(A, b, rcond=10**(-8))[0][nc:]
    start = np.array([start[0],start[1],0])
    return start

cons = [
        {'type': 'ineq', 'fun': lambda z: 11.5 - z[0]},
        {'type': 'ineq', 'fun': lambda z: z[0] + 1},
        {'type': 'ineq', 'fun': lambda z: 4.5 - z[1]},
        {'type': 'ineq', 'fun': lambda z: z[1] + 1},
        {'type': 'ineq', 'fun': lambda z: z[2] - 0.7},    # z[2] >= 1.4
        {'type': 'ineq', 'fun': lambda z: 2 - z[2]}       # z[2] <= 2
    ]
# anchor location
# cons = [
#         {'type': 'ineq', 'fun': lambda z: 11 - z[0]},
#         {'type': 'ineq', 'fun': lambda z: z[0] + 15},
#         {'type': 'ineq', 'fun': lambda z: 5 - z[1]},
#         {'type': 'ineq', 'fun': lambda z: z[1] + 1.5},
#         {'type': 'ineq', 'fun': lambda z: z[2] - 0},    # z[2] >= 1.4
#         {'type': 'ineq', 'fun': lambda z: 4 - z[2]}       # z[2] <= 2
#     ]

def iterative_loc(start,target_pix,cams,available_cam,config):
    def loss_function(z):
        res = 0
        if config.det_type == 'foot':
            z[2] = 0
        elif len(available_cam) == 1:
            # z[2] = config.h
            res += 10*np.sum((start-z)**2)
        for i,id in enumerate(available_cam):
            z = np.array(z).reshape(3, 1)
            cam = cams[id]
            anchor_loc = cam.anchors
            w = find_weights_pos(z,anchor_loc,ord=2)
            w = np.array(w).reshape(1,len(w))
            target_obs = target_pix[i]
            anchor_term = np.dot(w,(cam.prox_anchor_pix-cam.gt_anchor_pix)).squeeze()
            obs = target_obs + anchor_term
            z_pix = cams[id].world2pix(z)
            z_pix = np.array(z_pix)
            res += np.linalg.norm(z_pix-obs)
        return res
    solve_dict = op.minimize(loss_function, start, method='SLSQP', constraints=cons, options={'maxiter': 10})
    x = np.array(solve_dict['x'])
    return x

def obs_loc(start,obs,cams,available_cam,config):
    def loss_function(z):
        res = 0
        if config.det_type == 'foot':
            z[2] = 0
        elif len(available_cam) == 1:
            # z[2] = config.h
            res += 10*np.sum((start-z)**2)
        for i,id in enumerate(available_cam):
            z = np.array(z).reshape(3, 1)
            z_pix = cams[id].world2pix(z)
            z_pix = np.array(z_pix)
            res += np.linalg.norm(z_pix-obs[i])
        return res
    solve_dict = op.minimize(loss_function, start, constraints=cons, method='SLSQP')
    x = np.array(solve_dict['x'])
    return x

def angle_loc(start,cams,available_cam,target_angle,target_anchor_angle,config):
    def loss_function(z):
        res = 0
        if config.det_type == 'foot':
            z[2] = 0
        elif len(available_cam) == 1:
            z[2] = config.h
        for i,id in enumerate(available_cam):
            z = np.array(z).reshape(3, 1)
            # z_vec = z - cams[id].cam_loc.reshape(3, 1)
            z_vec = cams[id].world2vec_w(z).reshape(3, 1)
            z_angle = np.arccos(z_vec/np.linalg.norm(z_vec)).reshape(-1)
            res += config.rho*np.linalg.norm(z_angle-target_angle[i,:])
            for j in range(int(cams[id].na)):
                anchor_vec = cams[id].gt_anchor_vec[j]
                z_anchor_angle = calculate_angle(z_vec, anchor_vec)
                res += np.linalg.norm(z_anchor_angle-target_anchor_angle[i][j])
        return 100*res 
    solve_dict = op.minimize(loss_function, start, constraints=cons, method='SLSQP')
    # solve_dict = op.minimize(loss_function, start, method='BFGS')
    return solve_dict['x']

def compress_data(t1, t2, trace_data, cams, config):
    data = []
    for i in range(t1, t2+1):
        obs_trace = trace_data[trace_data['findex'] == i]
        frame_dict = compress_frame_data(i,obs_trace, cams, config)
        if frame_dict:
            data.append(frame_dict)
    return pd.DataFrame(data)

def compress_frame_data(t,obs_trace, cams, config):
    if len(obs_trace) == 0:
        return {}
    gt_loc = obs_trace['pts_3d'].tolist()[0]
    gt_loc = np.array(gt_loc)
    available_cam = obs_trace['cam_id'].tolist()
    nc = len(available_cam)
    camera_loc = np.array([cams[id].cam_loc for id in available_cam])
    target_pix = np.array(obs_trace['pix'].to_list())
    target_vec = np.array(obs_trace['vec'].to_list())
    target_angle = np.array(obs_trace['angle'].to_list())
    # target_feature = np.array(obs_trace['feat'].to_list())
    target_anchor_angle = obs_trace['anchor_angle'].to_list()
    if config.start_method == 'multi_camera_mean':
        observed_locs = np.array(obs_trace['loc'].to_list())
        init_loc = observed_locs.mean(axis=0)
    elif config.start_method == 'regression':
        if len(available_cam) == 1:
            init_loc = np.array(obs_trace['loc'].to_list()[0])
        else:
            if config.det_type == 'head':
                init_loc = regression_loc(nc,target_angle,camera_loc)
            elif config.det_type == 'foot':
                init_loc = regression_foot_loc(nc,target_angle,camera_loc)
    else:
        raise NotImplementedError
    obs = []
    for j,id in enumerate(available_cam):
        cam = cams[id]
        anchor_loc = cam.anchors
        w = find_weights_norm(init_loc,anchor_loc,config.lamda)
        w = np.array(w).reshape(1,len(w))
        target_obs = target_pix[j]
        anchor_term = np.dot(w,(cam.prox_anchor_pix-cam.gt_anchor_pix)).squeeze()
        if config.use_anchor:
            obs.append(target_obs + anchor_term)
        else:
            obs.append(target_obs)
        obs = np.array(obs)
        opt_loc = obs_loc(init_loc,obs,cams,available_cam,config)
    frame_dict = {
        't': t,
        'nc': nc,
        'gt_loc': gt_loc,
        'available_cam': available_cam,
        'init_loc': init_loc,
        'target_pix': target_pix,
        # 'feature': target_feature,
        'obs': obs,
        'opt_loc': real_loc_clip(opt_loc),
        'is_multi': (nc > 1)
    }
    return frame_dict