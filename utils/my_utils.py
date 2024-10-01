import numpy as np
import cv2
from utils.ea import Ea
from utils.Cam2World import Cam2World
import matplotlib.pyplot as plt

def real_loc_clip(loc):
    loc[0]=np.clip(loc[0],-1,11.5)
    loc[1]=np.clip(loc[1],-1,4.5)
    loc[2]=np.clip(loc[2],0.7,2)
    return loc

def get_metric(opt_dist, init_dist, cam_num):
    """
    Adjusted function to handle cases where single camera data or multiple camera data is empty,
    setting metrics to None for those scenarios.

    Parameters:
    - opt_dist: Numpy array or list of optimized distances.
    - init_dist: Numpy array or list of initial distances.
    - cam_num: Numpy array or list of camera numbers.

    Returns:
    A dictionary with metrics for full data, single camera, and multiple cameras.
    """
    # Convert inputs to numpy arrays to ensure compatibility with numpy operations
    opt_dist = np.array(opt_dist)
    init_dist = np.array(init_dist)
    cam_num = np.array(cam_num)

    # Full Data Metrics
    full_data_mean_opt = np.mean(opt_dist)
    full_data_mean_init = np.mean(init_dist)
    full_data_proportion = np.mean(opt_dist <= init_dist)
    
    # Single Camera Data Metrics (cam_num == 1)
    single_cam_filter = cam_num == 1
    if single_cam_filter.any():  # Check if the filter has any True values
        single_cam_mean_opt = np.mean(opt_dist[single_cam_filter])
        single_cam_mean_init = np.mean(init_dist[single_cam_filter])
        single_cam_proportion = np.mean(opt_dist[single_cam_filter] <= init_dist[single_cam_filter])
    else:  # Handle case where single camera data is empty
        single_cam_mean_opt = single_cam_mean_init = single_cam_proportion = None
    
    # Multiple Cameras Data Metrics (cam_num > 1)
    multiple_cam_filter = cam_num > 1
    if multiple_cam_filter.any():  # Check if the filter has any True values
        multiple_cam_mean_opt = np.mean(opt_dist[multiple_cam_filter])
        multiple_cam_mean_init = np.mean(init_dist[multiple_cam_filter])
        multiple_cam_proportion = np.mean(opt_dist[multiple_cam_filter] <= init_dist[multiple_cam_filter])
    else:  # Handle case where multiple camera data is empty
        multiple_cam_mean_opt = multiple_cam_mean_init = multiple_cam_proportion = None

    return {
        "full_data": (full_data_mean_opt, full_data_mean_init, full_data_proportion),
        "single_camera": (single_cam_mean_opt, single_cam_mean_init, single_cam_proportion),
        "multiple_cameras": (multiple_cam_mean_opt, multiple_cam_mean_init, multiple_cam_proportion)
    }

# This version of the function will now correctly handle scenarios where the filtered data for single or multiple camera data is empty.



def plot_loc(gt_loc,init_loc,opt_loc,id,result_dir):
    gt_loc = np.array(gt_loc)
    init_loc = np.array(init_loc)
    opt_loc = np.array(opt_loc)
    t = gt_loc.shape[0]
    gt_loc_2d = gt_loc[:, :2]
    init_loc_2d = init_loc[:, :2]
    opt_loc_2d = opt_loc[:, :2]

    plt.subplot(2, 1, 1)
    plt.scatter(gt_loc_2d[:, 0], gt_loc_2d[:, 1], label='gt_loc', marker='o')
    plt.scatter(init_loc_2d[:, 0], init_loc_2d[:, 1], label='init_loc', marker='x')
    for i in range(t):
        plt.plot([gt_loc_2d[i, 0], init_loc_2d[i, 0]], [gt_loc_2d[i, 1], init_loc_2d[i, 1]], color='gray', linestyle='--')

    plt.title('gt_loc vs init_loc')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.scatter(gt_loc_2d[:, 0], gt_loc_2d[:, 1], label='gt_loc', marker='o')
    plt.scatter(opt_loc_2d[:, 0], opt_loc_2d[:, 1], label='opt_loc', marker='^')
    for i in range(t):
         plt.plot([gt_loc_2d[i, 0], opt_loc_2d[i, 0]], [gt_loc_2d[i, 1], opt_loc_2d[i, 1]], color='gray', linestyle='--')
    plt.title('gt_loc vs anchor_loc')
    plt.legend()
    plt.grid(True)

    # 显示图形
    plt.tight_layout()
    # plt.show()
    plt.savefig('{}/loc/{}.png'.format(result_dir,id),dpi=300)

def plot_dist(cam_num,init_dist,opt_dist,id,result_dir):
    fig, ax1 = plt.subplots()
    ax1.plot(cam_num, label='cam_num', color='b')
    ax1.set_xlabel('frame')
    ax1.set_ylabel('cam_num', color='b')
    ax1.tick_params('y', colors='b')
    ax2 = ax1.twinx()
    ax2.plot(init_dist, label='init_distance', color='r')
    ax2.plot(opt_dist, label='opt_distance', color='orange',linestyle='--')
    ax2.set_ylabel('distance', color='r')
    ax2.tick_params('y', colors='r')
    fig.tight_layout()
    plt.legend()
    # plt.show()
    plt.savefig('{}/dist/{}.png'.format(result_dir,id),dpi=300)

def plot_loc_batch(gt_loc,init_loc,opt_loc,batch_loc,id,result_dir):
    gt_loc = np.array(gt_loc)
    init_loc = np.array(init_loc)
    opt_loc = np.array(opt_loc)
    batch_loc = np.array(batch_loc)
    t = gt_loc.shape[0]
    gt_loc_2d = gt_loc[:, :2]
    init_loc_2d = init_loc[:, :2]
    opt_loc_2d = opt_loc[:, :2]
    batch_loc_2d = batch_loc[:, :2]

    plt.subplot(3, 1, 1)
    plt.scatter(gt_loc_2d[:, 0], gt_loc_2d[:, 1], label='gt_loc', marker='o')
    plt.scatter(init_loc_2d[:, 0], init_loc_2d[:, 1], label='init_loc', marker='x')
    # for i in range(t):
    #     plt.plot([gt_loc_2d[i, 0], init_loc_2d[i, 0]], [gt_loc_2d[i, 1], init_loc_2d[i, 1]], color='gray', linestyle='--')

    plt.title('gt_loc vs init_loc')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.scatter(gt_loc_2d[:, 0], gt_loc_2d[:, 1], label='gt_loc', marker='o')
    plt.scatter(opt_loc_2d[:, 0], opt_loc_2d[:, 1], label='opt_loc', marker='^')
    # for i in range(t):
    #      plt.plot([gt_loc_2d[i, 0], opt_loc_2d[i, 0]], [gt_loc_2d[i, 1], opt_loc_2d[i, 1]], color='gray', linestyle='--')
    plt.title('gt_loc vs opt_loc')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.scatter(gt_loc_2d[:, 0], gt_loc_2d[:, 1], label='gt_loc', marker='o')
    plt.scatter(batch_loc_2d[:, 0], batch_loc_2d[:, 1], label='batch_loc', marker='*')
    # for i in range(t):
    #      plt.plot([gt_loc_2d[i, 0], batch_loc_2d[i, 0]], [gt_loc_2d[i, 1], batch_loc_2d[i, 1]], color='gray', linestyle='--')
    plt.title('gt_loc vs batch_loc')
    plt.legend()
    plt.grid(True)

    # 显示图形
    plt.tight_layout()
    # plt.show()
    plt.savefig('{}/batch_loc/{}.png'.format(result_dir,id),dpi=300)

def plot_dist_batch(cam_num,init_dist,opt_dist,batch_dist,id,result_dir):
    fig, ax1 = plt.subplots()
    ax1.plot(cam_num, label='cam_num', color='b')
    ax1.set_xlabel('frame')
    ax1.set_ylabel('cam_num', color='b')
    ax1.tick_params('y', colors='b')
    ax2 = ax1.twinx()
    ax2.plot(init_dist, label='init_distance', color='r')
    ax2.plot(opt_dist, label='opt_distance', color='orange',linestyle='--')
    ax2.plot(batch_dist, label='batch_distance', color='purple',linestyle='-.')
    ax2.set_ylabel('distance', color='r')
    ax2.tick_params('y', colors='r')
    fig.tight_layout()
    plt.legend()
    # plt.show()
    plt.savefig('{}/batch_dist/{}.png'.format(result_dir,id),dpi=300)

def normalize_mtx(mtx):
    return (mtx - np.min(mtx)) / (np.max(mtx) - np.min(mtx)+1e-8)

"""calculate the angle between vector x and y"""
def calculate_angle(x, y):
    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)
    assert nx != 0
    assert ny != 0
    cosine_value = np.sum(x*y)/nx/ny
    if cosine_value > 1.0:
        cosine_value = 1.0
    if cosine_value < -1.0:
        cosine_value = -1.0
    return np.arccos(cosine_value)

"""calculate the cosine value angle between vector x and y"""
def calculate_cos_angle(x,y):
    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)
    assert nx != 0
    assert ny != 0
    return np.sum(x*y)/nx/ny

"""transform cartesian coordinate into polar coordinate"""
def cartesian_to_polar(x, y, z):
    cartesian_coords = np.hstack([x, y, z])
    r = np.linalg.norm(cartesian_coords, axis=0)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r) - 0.5*np.pi
    return r, theta, phi


def cal_all_angles(target_loc,anchor_loc,cams_dict):
    nc = len(cams_dict)
    na = anchor_loc.shape[1]
    angles = np.zeros((nc, na))
    for i in range(nc):
        cam = list(cams_dict.keys())[i]
        cam_dict = cams_dict[str(cam)]
        for j in range(na):
            vx = simu_vec(target_loc,cam_dict)
            vy = simu_vec(anchor_loc[i][j,:],cam_dict)
            angles[i,j] = calculate_angle(vx, vy)
    return angles

def cal_single_angle(target_loc,anchor_loc,cam_dict):
    na = anchor_loc.shape[0]
    angle = np.zeros(na)
    for j in range(na):
        vx = simu_vec(target_loc,cam_dict)
        vy = simu_vec(anchor_loc[j,:],cam_dict)
        angle[j] = calculate_angle(vx, vy)
    return angle

def cal_world_coordinates(target_loc,cams_dict):
    nc = len(cams_dict)
    coordinate_angles = np.zeros((nc, 3))
    for i in range(nc):
        cam = list(cams_dict.keys())[i]
        cam_dict = cams_dict[str(cam)]
        vec = simu_vec(target_loc,cam_dict)
        coordinate_angles[i,:] = cam2world(vec,cam_dict)
    return coordinate_angles

def get_direction(target_loc, cams_dict):
    # simu_rel = cal_world_coordinates(target_loc,cams_dict)
    vec = simu_vec(target_loc,cams_dict)
    simu_rel = cam2world(vec,cams_dict)
    simu_an = np.arccos(simu_rel / np.linalg.norm(simu_rel, axis=1).reshape((-1, 1)))
    return simu_an

"""simulate the vector observed by camera"""
def simu_vec(target_loc,cam_dict):
    X_w = target_loc.reshape((3,1))
    R, T, dist, K, K_prox = cam_dict['R'], cam_dict['T'], cam_dict['dist'], cam_dict['K'],cam_dict['K_prox']
    T = np.array(T)
    R = np.array(R)
    K = np.array(K)
    K_prox = np.array(K_prox)
    T = T.reshape((3,1))
    X_cam = np.dot(R, X_w) + T
    scalar = X_cam[2]
    X_cam = X_cam / scalar
    X_pix = np.dot(K, X_cam)
    x_prime, y_prime = X_pix[0], X_pix[1]
    u, v = x_prime, y_prime
    # r2 = x_prime**2 + y_prime**2
    # k1, k2, p1, p2 = np.array(dist)/1000
    # u = x_prime * (1 + k1 * r2 + k2 * r2**2) + 2 * p1 * x_prime * y_prime + p2 * (r2 + 2 * x_prime**2)
    # v = y_prime * (1 + k1 * r2 + k2 * r2**2) + 2 * p2 * x_prime * y_prime + p1 * (r2 + 2 * y_prime**2)
    fx, fy = K_prox[0][0], K_prox[1][1]
    cx, cy = K_prox[0][2], K_prox[1][2]
    x = (u - cx) / fx
    y = (v - cy) / fy
    # xx, yy = undistort_point(x, y, k1, k2, p1, p2, fx, fy, cx, cy, iterations=10)
    vec = np.vstack([x,y,1])
    vec = vec * scalar
    Y_w = np.dot(R.T, vec - T)
    return vec.T

def cam2world(vec, cam_dict):
    R_prox,T_prox = cam_dict['R_prox'],cam_dict['T_prox']
    vec = vec.reshape((3,1))
    T_prox = np.array(T_prox)
    R_prox = np.array(R_prox)
    T_prox = T_prox.reshape((3,1))
    Y_w = np.dot(R_prox.T, vec - T_prox)
    Cam_w = np.dot(R_prox.T, np.vstack([0,0,0])-T_prox)
    return (Y_w-Cam_w).T


def res_report(method,result, initial_estimate, target, is_multi):
    if method == 'multi':
        is_multi = np.array(is_multi, dtype=bool)
        result = result[is_multi]
        initial_estimate = initial_estimate[is_multi]
        target = target[is_multi]
    elif method == 'single':
        is_single = np.array(is_multi) == 0
        result = result[is_single]
        initial_estimate = initial_estimate[is_single]
        target = target[is_single]
    if result.size:
        ratio = (np.linalg.norm(result - target, axis=1) / np.linalg.norm(initial_estimate - target, axis=1) - 1)
        imp_cases = np.sum(ratio <= 0)/ target.shape[0]
        imp_avg = -np.average(ratio)
        imp_med = -np.median(ratio)
        loss_distance = np.linalg.norm(result - target, axis=1).mean()
        init_distance = np.linalg.norm(initial_estimate - target, axis=1).mean()
        print(str(method),' improved cases',imp_cases)
        print('avg of improved ratio: ',imp_avg, 'median of improved ratio: ',imp_med)
        return [imp_cases, loss_distance, init_distance]
    else:
        return []
    
def res_report_per_target():
    return 
    
def small_invcos(x):
    y = np.cos(x)
    if abs(y) < 10**(-8):
        return np.sign(y) * 10**8
    else:
        return 1.0/y

invcos = np.vectorize(small_invcos)

def converting(i, j, N):
    # Let S be a (N, N) symmetrix matrix.
    # If S_{ij} = S_{ji} = 1, and all other elements of S are zero, 
    #   then vec(S) will have the 
    #       converting(i, j, N)
    #           -th element as its unique nonzero element. 
    # You should be VERY CAREFUL about the indices if using this function.
    if j < i: 
        i, j = j, i
    # return N + N-1 + ... + (N-i+1) + j-i
    return (2*N-i+1)*i//2 + j - i


def vec(S):
    # This is a general function dealing with semi-definite cones in scs.
    # See https://www.cvxgrp.org/scs/examples/python/mat_completion.html#py-mat-completion
    # Input: (n, n) symmetric matrix
    # Output: a padded (n(n+1)/2,) array s
    # e.g. [[1, 2, 3],
    #       [2, 5, 6],
    #       [3, 6, 9]]
    # is transformed into an array
    # [1, 2 \sqrt{2}, 3 \sqrt{2}, 5, 6 \sqrt{2}, 9]
    n = S.shape[0]
    # S = np.copy(S)
    S *= np.sqrt(2)
    S[range(n), range(n)] /= np.sqrt(2)
    return S[np.triu_indices(n)]

 
def mat(s):
    # This is a general function dealing with semi-definite cones in scs.
    # See https://www.cvxgrp.org/scs/examples/python/mat_completion.html#py-mat-completion
    # Input: (n(n+1)/2,) array s
    # Output: (n, n) symmetric matrix correspond to s
    # e.g. [1, 2 \sqrt{2}, 3 \sqrt{2}, 5, 6 \sqrt{2}, 9]
    # is transformed into a symmetric matrix
    #      [[1, 2, 3],
    #       [2, 5, 6],
    #       [3, 6, 9]]
    # We should note that 
    #   vec(mat(s)) = s, mat(vec(S)) = S.
    n = int((np.sqrt(8 * len(s) + 1) - 1) / 2)
    S = np.zeros((n, n))
    S[np.triu_indices(n)] = s / np.sqrt(2)
    S = S + S.T
    S[range(n), range(n)] /= np.sqrt(2)
    return S