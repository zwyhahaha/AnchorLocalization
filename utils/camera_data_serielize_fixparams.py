import sys, os
import cv2
import numpy as np
from utils.ea import Ea
from utils.cfg import Config
from utils.Cam2World import Cam2World
from utils.Map2Wrold import Map2DtoWorld
import math
import json
import math
import random

#uniform = RNG.uniform()
uniform = 1

def cartesian_to_polar(x, y, z):
    cartesian_coords = np.hstack([x, y, z])
    r = np.linalg.norm(cartesian_coords, axis=0)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r) - 0.5*np.pi
    return r, theta, phi

"""check if the target is in the visible field of camera"""
def is_available_target(target_loc, camera_loc,camera_angle,lowwer_angle,upper_angle,available_angle):
    relative_dist = target_loc - camera_loc
    r, theta,phi = cartesian_to_polar(relative_dist[0],relative_dist[1],relative_dist[2])
    cond_1 = True if lowwer_angle/180*np.pi <= phi <= upper_angle/180*np.pi else False
    angle_dif = abs(theta - camera_angle)
    cond_2 = True if abs(angle_dif) <= available_angle/180*np.pi else False
    return (cond_1 and cond_2)

class ErrorAdd():
    def __init__(self, config):
        self.config = config
    
    def _Rprox(self,cam, path):
        path = os.path.join(path, "rotations")
        data = json.load(open(path + "/" + cam + ".json", "r"))
        theta_old, beta_old, gama_old = data["angle"]
        if 'large' in path:
            theta_old, beta_old, gama_old = math.degrees(theta_old), math.degrees(beta_old), math.degrees(gama_old)  #弧度转角度

        R = data["R_est"]
        r_fu = np.array([[1, 0, 0], [0,0,-1], [0,1,0]])

        """
        rx, ry, rz 直接加载旋转角度上的误差
        """
        theta = theta_old + self.config.rx_error
        beta = beta_old + self.config.ry_error
        gama = gama_old + self.config.rz_error
        
        theta = math.radians(theta)
        beta = math.radians(beta)
        gama = math.radians(gama)

        rx = np.array([[1,0,0],
            [0, math.cos(theta), -math.sin(theta)],
            [0, math.sin(theta), math.cos(theta)]])
        ry = np.array([[math.cos(beta), 0, math.sin(beta)],
                    [0, 1, 0],
                    [-math.sin(beta), 0, math.cos(beta)]])
        rz = np.array([[math.cos(gama), -math.sin(gama), 0],
                    [math.sin(gama), math.cos(gama), 0],
                    [0, 0, 1]])
        
        rxz = np.dot(rz, rx)
        rxzy = np.dot(rxz, ry)
        r_ = np.dot(rxzy, r_fu)

        return R, r_, [theta_old, beta_old, gama_old], [math.degrees(theta),math.degrees(beta), math.degrees(gama)]
    def _Dprox(self, dist_):
        """
        加在畸变参数上, 0.5 x D_error
        """
        dist = dist_.copy()
        dist[0] *= (1 + self.config.D_error * uniform)
        dist[1] *= (1 + self.config.D_error * uniform)
        dist[2] *= (1 + self.config.D_error * uniform)
        dist[3] *= (1 + self.config.D_error * uniform)
        return dist
    
    def _Kprox(self, mtx): 
        """
        加在主点和焦距上, 0.5 x K_error
        """
        mtx_prox = mtx.copy() 
        mtx_prox[0][0] *= (1 + self.config.K_error * uniform)
        mtx_prox[1][1] *= (1 + self.config.K_error * uniform)
        mtx_prox[0][2] *= (1 + self.config.K_error * uniform)
        mtx_prox[1][2] *= (1 + self.config.K_error * uniform)
        return mtx_prox
    
    def _Tprox(self,tvec):
        if self.config.T_error == 0:
            return tvec
        while True:
            """
            T_error是误差的绝对值
            """
            x = random.uniform(tvec[0] - self.config.T_error, tvec[0] + self.config.T_error)
            y = random.uniform(tvec[1] - self.config.T_error, tvec[1] + self.config.T_error)
            
            cz = self.config.T_error**2 - (x - tvec[0]) ** 2 - (y - tvec[1]) **2
            
            if cz > 0 :
                z = math.sqrt(cz) + tvec[2] 
                if tvec[2] - 1 < z < tvec[2] + 1:
                    break
    
        return [x, y,z]
    
    def cartesian_to_polar(self, x, y, z):
        cartesian_coords = np.hstack([x, y, z])
        r = np.linalg.norm(cartesian_coords, axis=0)
        theta = np.arctan2(y, x)
        phi = np.arccos(z / r) - 0.5*np.pi
        return r, theta, phi

    def get_angles(self, rvec, tvec, mtx, dist, cam_name):
        cam2w =Cam2World.from_rvec_tvec(rvec, tvec, mtx, dist, name=cam_name)
        V_left = cam2w.direction(0, 360)
        V_right = cam2w.direction(1279, 360)
        V_up = cam2w.direction(640, 0)
        V_down = cam2w.direction(640, 719)
        theta = cam2w.calculate_angle(V_left, V_right)
        theta = np.rad2deg(theta)
        _,_,up = self.cartesian_to_polar(V_up[0],V_up[1],V_up[2])
        _,_,down = self.cartesian_to_polar(V_down[0],V_down[1],V_down[2])
        up = np.rad2deg(up)
        down = np.rad2deg(down)
        return theta,up,down

    def get_loc(self, rvec, tvec, mtx, dist, cam_name):
        #不添加误差
        c1 = Cam2World.from_rvec_tvec(rvec, tvec, mtx, dist, name=cam_name)
        look, at, up = c1.as_look_at_up()
        fov = c1.as_fov()
        return look, at, up, fov



def get_single_cam_info(test_path, cam, errorAdd):
    filename = os.path.join(test_path, "monitors_json/%s.json" % cam)
    with open(filename, 'r') as input_file:
        json_data = json.load(input_file)
    cam_cfg = Ea.load(filename)
    cam_name = cam_cfg.cameraid
    rvec = np.hstack(eval(cam_cfg.rvec))
    tvec = np.hstack(eval(cam_cfg.tvec))
    dist = eval(cam_cfg.dist)[0]
    mtx = eval(cam_cfg.mtx)

    #K
    K_prox = errorAdd._Kprox(mtx)
    
    #R
    R,R_prox, rotate_angle, rotate_angle_prox = errorAdd._Rprox(cam,test_path)
   
    #T
    tvec_prox = errorAdd._Tprox(tvec)
    
    #真实摄像头位置
    look, at, up, fov = errorAdd.get_loc(rvec, tvec, mtx, dist, cam)
   
    rvec_prox,_ = cv2.Rodrigues(R_prox)
    rvec_prox = rvec_prox.reshape(3,)
    look_prox, at_prox, up_prox, fov_prox = errorAdd.get_loc(rvec_prox, tvec_prox, K_prox, dist, cam)


    look = look - at
    _, camera_angle, _ = errorAdd.cartesian_to_polar(look[0],look[1],look[2])

    look_prox = look_prox - at_prox
    _, camera_angle_prox, _ = errorAdd.cartesian_to_polar(look_prox[0], look_prox[1], look_prox[2])


    theta,up,down = errorAdd.get_angles(rvec, tvec, mtx, dist, cam_name)
    theta_prox, up_prox, down_prox =  errorAdd.get_angles(rvec_prox, tvec_prox, K_prox, dist, cam_name)

    theta /= 2
    fov = np.hstack([up, down, theta])
    
    theta_prox /= 2
    fov_prox = np.hstack([up_prox, down_prox, theta_prox])


    dist_prox = errorAdd._Dprox(dist)

    """generate #20 anchors in the visible field of camera"""
    rng = np.random.default_rng(10)
    i=0
    if 'anchors' not in json_data:
        anchors = []
        while i<20:
            target_x = at[0] + rng.uniform(-5,5)
            target_y = at[1] + rng.uniform(-5,5)
            target_h = rng.uniform(0,1.5)
            target = np.hstack([target_x,target_y,target_h])
            ## check whether a target is visible to a camera, using ground truth param
            if is_available_target(target,at,camera_angle,fov[0],fov[1],fov[2]):
                i+=1
                anchors.append(target)
    else:
        anchors = json_data["anchors"]
    
    if 'anchors' not in json_data:
        dict = {'cameraid': cam,
                'loc': at,
                "loc_prox": at_prox,
                'angle': camera_angle,
                "angle_prox": camera_angle_prox,
                'fov': fov,
                "fov_prox": fov_prox,
                'R': R,
                'R_prox':R_prox,
                'rvec': rvec,
                'rvec_prox': rvec_prox,
                'tvec': tvec,
                'tvec_prox': tvec_prox,
                'dist': dist,
                'dist_prox': dist_prox,
                'mtx': mtx,
                'mtx_prox': K_prox,
                "rotate_angle": rotate_angle,
                "rotate_angle_prox": rotate_angle_prox,
                "anchors": anchors
                }
    else:
        dict = {'cameraid': cam,
                'angle': camera_angle_prox,
                'fov': fov_prox,
                'R': R_prox,
                'rvec': rvec_prox,
                'tvec': tvec_prox,
                'dist': dist_prox,
                'mtx': K_prox,
                "anchors": anchors
                }
    
    for key, value in dict.items():
         if isinstance(value, list):
             dict[key] = np.array(value)
    return dict

def convert(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {k: convert(v) for k, v in o.items()}
    if isinstance(o, list):
        return [convert(item) for item in o]
    return o

def do(config,test_path, save_dir):
    errorAdd = ErrorAdd(config)
    cams = config.cams

    cams_info = {}
    for cam in cams:
        cams_info[str(cam)] = get_single_cam_info(test_path, cam, errorAdd)
        filename = '{}/{}.json'.format(save_dir,cam)
        with open(filename, 'w') as f:
            json.dump(convert(get_single_cam_info(test_path, cam, errorAdd)),f, indent=4)
    return convert(cams_info)


