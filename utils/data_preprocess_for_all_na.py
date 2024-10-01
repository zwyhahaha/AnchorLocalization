import pandas as pd
import ast
import numpy as np
import scipy.optimize as op
from utils.preprocess_real import regression_loc

def get_pix_loc(cam, det, H=1.7):
    return cam.pix2world(det, H)

def get_pix(cam, loc, seed=None, pix_perturb_level=None):
    result1, result2 = cam.world2pix(loc, pix_perturb_level, seed, my_type = "gt")
    return pd.Series([result1, result2])

def get_prox_pix(cam, loc, seed=None, pix_perturb_level=None):
    result1, result2 = cam.world2pix(loc, pix_perturb_level, seed, my_type = "prox")
    return pd.Series([result1, result2])

def get_pix_angle(cam, det):
    angle, dif = cam.pix2angles(det)
    return pd.Series([angle, dif])

def get_pix_vec(cam, det):
    return cam.pix2vec_w(det).reshape(-1)

def str_to_list(s):
    return ast.literal_eval(s)

def obs_construction(cam, config, trace_data: pd.DataFrame) -> pd.DataFrame:
    trace_data[['pix', 'perturb']] = trace_data.apply(lambda row: get_pix(cam[row['cam_id']], row['pts_3d'], row.name,config.pix_perturb_level), axis=1)
    # trace_data[['prox_pix', 'perturb']] = trace_data.apply(lambda row: get_prox_pix(cam[row['cam_id']], row['pts_3d'], row.name,config.pix_perturb_level), axis=1)
    trace_data[['angle', 'anchor_angle']] = trace_data.apply(lambda row: get_pix_angle(cam[row['cam_id']], row['pix']), axis=1)
    trace_data['loc'] = trace_data.apply(lambda row: get_pix_loc(cam[row['cam_id']], row['pix'], config.h), axis=1)
    trace_data['vec'] = trace_data.apply(lambda row: get_pix_vec(cam[row['cam_id']], row['pix']), axis=1)

def find_weights_pos(x, points, ord):
    def objective_function(w):
        return np.linalg.norm(x - np.dot(points.T, w), ord=ord)**2
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    constraints.append({'type': 'ineq', 'fun': lambda w: w})
    initial_weights = np.ones(points.shape[0]) / points.shape[0]
    result = op.minimize(objective_function, initial_weights, constraints=constraints)
    hat_x = np.dot(points.T, result.x)
    eps = np.linalg.norm(hat_x - x)
    # if eps>0.5 and not relaxed:
    optimal_weights = result.x
    return optimal_weights

def find_weights_norm(x, points, ord):
    def objective_function(w):
        return 10*np.linalg.norm(w)**2 + np.linalg.norm(x - np.dot(points.T, w), ord = ord)**2
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    # constraints.append({'type': 'eq', 'fun': lambda w: x - np.dot(points.T, w)})
    initial_weights = np.ones(points.shape[0]) / points.shape[0]
    result = op.minimize(objective_function, initial_weights, constraints=constraints)
    hat_x = np.dot(points.T, result.x)
    eps = np.linalg.norm(hat_x - x)
    # if eps>0.5 and not relaxed:
    optimal_weights = result.x
    return optimal_weights

def compress_data(t1, t2, trace_data, cams, config):
    data = []
    for i in range(t1, t2+1):
        obs_trace = trace_data[trace_data['findex'] == i]
        if len(obs_trace) < 1:
            continue
        target_loc = obs_trace['pts_3d'].tolist()[0]
        target_angle = np.array(obs_trace['angle'].to_list())
        target_loc = np.array(target_loc).reshape((3,1))
        available_cam = obs_trace['cam_id'].tolist()
        camera_loc = np.array([cams[id].cam_loc for id in available_cam])
        nc = len(available_cam)
        observed_locs = np.array(obs_trace['loc'].to_list())
        if config.start_method == 'multi_camera_mean':
            init_loc = observed_locs.mean(axis=0)
        elif config.start_method == 'regression':
            if len(available_cam) == 1:
                init_loc = np.array(obs_trace['loc'].to_list()[0])
            else:
                init_loc = regression_loc(nc,target_angle,camera_loc)
        target_pix = np.array(obs_trace['pix'].to_list())

        data.append({
            't': i,
            'nc': nc,
            'target_loc': target_loc,
            'available_cam': available_cam,
            'init_loc': init_loc,
            'target_pix': target_pix
        })
    return pd.DataFrame(data)