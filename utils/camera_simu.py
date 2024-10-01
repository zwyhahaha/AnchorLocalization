from utils.ea import Ea
import json
import numpy as np
import os
import cv2
import random
from utils.Cam2World import Cam2World

def calculate_angle(x, y):
    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)
    assert nx != 0
    assert ny != 0
    res = np.sum(x*y)/nx/ny
    res = np.clip(res,-1,1)
    return np.arccos(res)

def pix2vec_gw(pix, mtx, rvec, tvec):
    fx, fy = mtx[0][0], mtx[1][1]
    cx, cy = mtx[0][2], mtx[1][2]
    vec = np.vstack([(pix[0]-cx)/fx, (pix[1]-cy)/fy, 1])
    vec = vec.reshape((3,1))
    R, _ = cv2.Rodrigues(rvec)
    T = tvec.reshape(3,1)
    Y_w = np.dot(R.T, vec - T)
    Cam_w = np.dot(R.T, np.vstack([0,0,0])-T)
    rel_direction = (Y_w-Cam_w).T
    return np.array(rel_direction).reshape(1,3)

def pix2vec_gc(pix, mtx):
    fx, fy = mtx[0][0], mtx[1][1]
    cx, cy = mtx[0][2], mtx[1][2]
    vec = np.hstack([(pix[0]-cx)/fx, (pix[1]-cy)/fy, 1])
    return np.array(vec).reshape(1,3)

def generate_random_vectors(num_vectors, dim_range):
    rng = np.random.default_rng(10)
    vectors = []
    for _ in range(num_vectors):
        vector = [
            rng.uniform(dim_range[0], dim_range[1]) for _ in range(3)
        ]
        vectors.append(vector)
    return vectors

def anchor2pix(cam, loc, mytype = 'prox', perturb_level = 0, seed = None):
        """
        transform world coordinates into pixel coordinates--using ground truth parameters
        """
        X_w = np.array(loc).reshape((3,1))
        if mytype == 'gt':
            R, _ = cv2.Rodrigues(cam.rvec)
            T = cam.tvec.reshape(3,1)
            K = cam.mtx
        elif mytype == 'prox':
            R, _ = cv2.Rodrigues(cam.rvec_prox)
            T = cam.tvec_prox.reshape(3,1)
            K = cam.mtx_prox
        T = T.reshape((3,1))
        X_cam = np.dot(R, X_w) + T
        X_cam = X_cam / X_cam[2]
        X_pix = np.dot(K, X_cam)
        if seed: 
            rng = np.random.default_rng(8+seed)
        else:
            rng = np.random.default_rng(8)
        if perturb_level:
            a = rng.integers(-perturb_level, perturb_level)
            b = rng.integers(-perturb_level, perturb_level)
            pix = [X_pix[0,0]+a,X_pix[1,0]+b]
            # pix = [X_pix[0,0],X_pix[1,0]]
            # return pix, np.sqrt(a*a+b*b)
            return pix
        else:
            pix = [X_pix[0,0],X_pix[1,0]]
            # return pix, 0
            return pix
        
def anchor2vec(cam, pixs, mytype = 'prox'):
    if mytype == 'prox':
        anchor_vecs = [pix2vec_gw(pix, cam.mtx_prox, cam.rvec_prox, cam.tvec_prox) for pix in pixs]
    elif mytype == 'gt':
        anchor_vecs = [pix2vec_gw(pix, cam.mtx, cam.rvec, cam.tvec) for pix in pixs]
    return anchor_vecs

class Camera(object):
    def __init__(self, camera_name, na, camera_dir, use_distort=False, config=None):
        # filename = os.path.join(test_path, "monitors_json/%s.json" % camera_name)
        # filename = os.path.join("monitors_json/%s.json" % camera_name)
        filename = '{}/{}.json'.format(camera_dir,camera_name)
        with open(filename, 'r') as f:
            self.cam_cfg = json.load(f)
        self.use_distort = config.use_distort
        self.rvec, self.tvec, self.mtx, self.dist, self.cam_name = np.hstack(self.cam_cfg["rvec"]), np.hstack(self.cam_cfg["tvec"]), np.array(self.cam_cfg["mtx"]), np.array(self.cam_cfg["dist"]), self.cam_cfg["cameraid"]
        self.rvec_prox, self.tvec_prox, self.mtx_prox, self.dist_prox = np.hstack(self.cam_cfg["rvec_prox"]), np.hstack(self.cam_cfg["tvec_prox"]), np.array(self.cam_cfg["mtx_prox"]), np.array(self.cam_cfg["dist_prox"])
        self.trans = Cam2World(self.rvec_prox,self.tvec_prox, self.mtx_prox, self.dist_prox, "unknown")
        self.cam_loc = self.cam_cfg['loc_prox']
        self.R, _ = cv2.Rodrigues(self.rvec)
        self.R_prox, _ = cv2.Rodrigues(self.rvec_prox)

        self.A = np.dot(self.mtx, self.R)
        self.A_prox = np.dot(self.mtx_prox, self.R_prox)
        self.b = np.dot(self.mtx, self.tvec)
        self.b_prox = np.dot(self.mtx_prox, self.tvec_prox)
        self.na = na
        self.anchors = np.array([self.cam_cfg["anchors"][i] for i in range(na)])
        self.gt_anchor_pix = [self.world2pix(loc, my_type = 'gt')[0] for loc in self.anchors]
        self.prox_anchor_pix = [self.world2pix(loc, my_type = 'prox')[0] for loc in self.anchors]
        self.anchor_vec = [(anchor - self.cam_loc)/(np.dot(self.R_prox, anchor)+self.tvec_prox)[-1] for anchor in self.anchors]
        self.gt_anchor_vec = np.array([self.pix2vec_w(pix) for pix in self.gt_anchor_pix])
        pix_err = np.linalg.norm(np.array(self.gt_anchor_pix)-np.array(self.prox_anchor_pix), axis=1)
        self.pix_err = np.mean(pix_err)
        self.gt_anchor_vec = np.array([self.pix2vec_w(pix) for pix in self.gt_anchor_pix])
        self.prox_anchor_vec = np.array([self.pix2vec_w(pix) for pix in self.prox_anchor_pix])
        vec_err = [calculate_angle(self.gt_anchor_vec[i],self.prox_anchor_vec[i]) for i in range(self.na)]
        self.vec_err = np.mean(np.rad2deg(vec_err))

    def get_loc(self):
        """
        get camera location--from perturbated parameters
        """
        c1 = Cam2World.from_rvec_tvec(self.rvec_prox, self.tvec_prox, self.mtx_prox, self.dist_prox, name=self.cam_name)
        look, at, up = c1.as_look_at_up()
        fov = c1.as_fov()
        return at
    
    def clip(self,pix_c, w=1280, h=720):
        pix_c[0] = min(max(0, pix_c[0]), w-1)
        pix_c[1] = min(max(0, pix_c[1]), h-1)
        return pix_c
    
    def height_estimate(self,world_pt3_foot, world_pt3_head, world_pt3):
        foot_3d = np.array([world_pt3_foot[0], world_pt3_foot[1]])
        head_3d = np.array([world_pt3_head[0], world_pt3_head[1]])
        cam_3d = np.array([world_pt3[0], world_pt3[1]])
        cam_H = world_pt3[2]
        z = np.linalg.norm(foot_3d - cam_3d)
        z1 = np.linalg.norm(head_3d - cam_3d)
        
        ph = cam_H * (z1 - z) / z1
        return ph
    
    def world2pix(self, loc, perturb_level = 0, seed = None, my_type = "gt"):
        """
        transform world coordinates into pixel coordinates--using ground truth parameters
        """
        X_w = np.array(loc).reshape((3,1))
        R = self.R if my_type=='gt' else self.R_prox
        T = self.tvec.reshape(3,1) if my_type=='gt' else self.tvec_prox
        K = self.mtx if my_type=='gt' else self.mtx_prox
        T = T.reshape((3,1))
        X_cam = np.dot(R, X_w) + T
        X_cam = X_cam / X_cam[2]
        X_pix = np.dot(K, X_cam).squeeze()
        if self.use_distort:
            k1, k2, p1, p2, _ = self.dist if my_type=='gt' else self.dist_prox
            x_prime, y_prime = X_cam[0, 0], X_cam[1, 0]
            r2 = x_prime**2 + y_prime**2
            dx = x_prime * (1 + k1 * r2 + k2 * r2**2) + 2 * p1 * x_prime * y_prime + p2 * (r2 + 2 * x_prime**2)
            dy = y_prime * (1 + k1 * r2 + k2 * r2**2) + 2 * p2 * x_prime * y_prime + p1 * (r2 + 2 * y_prime**2)
            X_pix = np.dot(K, np.vstack([dx, dy, 1])).squeeze()
            # X_pix = self.clip(X_pix)
            # X_pix = np.array(self.trans.distort(X_pix))
        if seed: 
            rng = np.random.default_rng(8+seed)
        else:
            rng = np.random.default_rng(8)
        if perturb_level:
            a = rng.integers(-perturb_level, perturb_level)
            b = rng.integers(-perturb_level, perturb_level)
            pix = [X_pix[0]+a,X_pix[1]+b]
            return self.clip(pix), np.sqrt(a*a+b*b)
        else:
            pix = [X_pix[0],X_pix[1]]
            return self.clip(pix), 0

    def pix2world(self, det, H=1.7):
        """
        transform pixel coordinates into world coordinates--using perturbated parameters 'trans'
        """
        pix_head = det
        pix_head = self.clip(pix_head)
        # world_pt3_head = self.trans.img2world(pix_head[0], pix_head[1])
        world_pt3_head = self.trans.img2world(pix_head[0], pix_head[1], H)
        return world_pt3_head
    
    def pix2vec_w(self, pix, my_type = 'prox'):
        """
        transform pixel into relative vector in world coordinates--using perturbated parameters
        """
        ### get relative vector in world coord
        # pix = self.clip(pix)
        ### transform to world
        if my_type == 'prox':
            vec = self.pix2vec_c(pix)
            vec = vec.reshape(3, 1)
            rel_direction = np.dot(self.R_prox.T, vec)
            return np.array(rel_direction).reshape(1,3)
        elif my_type == 'gt':
            vec = self.pix2vec_c(pix)
            vec = vec.reshape(3, 1)
            rel_direction = np.dot(self.R.T, vec)
            return np.array(rel_direction).reshape(1,3)
    
    def pix2vec_c(self, pix, my_type = 'prox'):
        """
        transform pixel into relative vector in camera coordinates--using perturbated parameters
        """
        ### get relative vector in cam coord
        if self.use_distort:
            pix = self.clip(pix)
            pix = np.array(self.trans.undistort(pix))
        if my_type == 'prox':
            fx, fy = self.mtx_prox[0][0], self.mtx_prox[1][1]
            cx, cy = self.mtx_prox[0][2], self.mtx_prox[1][2]
        elif my_type == 'gt':
            fx, fy = self.mtx[0][0], self.mtx[1][1]
            cx, cy = self.mtx[0][2], self.mtx[1][2]
        vec = np.hstack([(pix[0]-cx)/fx, (pix[1]-cy)/fy, 1])
        return np.array(vec).reshape(1,3)

    def get_anchor_vec(self, my_type = 'prox'):
        anchor_pix = [(self.world2pix(anchor,0))[0] for anchor in self.anchors]
        if my_type == 'prox':
            anchor_vec_c = [self.pix2vec_c(pix) for pix in anchor_pix]
            self.anchor_vec_c = anchor_vec_c
        elif my_type == 'gt':
            anchor_vec_c = [self.pix2vec_c(pix, 'gt') for pix in anchor_pix]
        
        return anchor_vec_c
    
    def get_anchor_vec_from_pix(self):
        anchor_vec_c = [self.pix2vec_c(pix) for pix in self.anchors]
        self.anchor_vec_c = anchor_vec_c
        return anchor_vec_c
    
    def pix2angles(self, pix):
        pix = self.clip(pix)
        # given observed pix, only prox angle can be calculated
        target_vec_w = self.pix2vec_w(pix).reshape(1,3) 
        coordinate_angles = np.arccos(target_vec_w / np.linalg.norm(target_vec_w, axis=1).reshape((-1, 1))).reshape(-1)

        anchor_angles = []
        for anchor_vec_w in self.gt_anchor_vec:
            anchor_angles.append(calculate_angle(target_vec_w, anchor_vec_w))
        anchor_angles = np.array(anchor_angles)
        return coordinate_angles, anchor_angles
    
        # pix_head = det
        # target_vec_w = self.pix2vec_w(pix_head)
        
        # coordinate_angles = np.arccos(target_vec_w / np.linalg.norm(target_vec_w, axis=1).reshape((-1, 1))).reshape(1,3)
        # anchor_pix = [(self.world2pix(anchor,0))[0] for anchor in self.anchors]
        # anchor_vec = [self.pix2vec_w(pix) for pix in anchor_pix]
        # self.gt_anchor_vec = [self.pix2vec_w(pix, 'prox').reshape(-1) for pix in self.gt_anchor_pix]
        # self.prox_anchor_vec = [self.pix2vec_w(pix, 'prox').reshape(-1) for pix in self.prox_anchor_pix]
        # # self.anchor_vec = anchor_vec
        # anchor_angles = []
        # for anchor_vec_w in anchor_vec:
        #     anchor_angles.append(calculate_angle(target_vec_w, anchor_vec_w))
        # anchor_angles = np.array(anchor_angles).reshape(1,self.na)
        # res = np.hstack([coordinate_angles, anchor_angles])

        # target_vec_w_gt = self.pix2vec_w(pix_head, 'gt')
        # target_vec_c_gt = self.pix2vec_c(pix_head, 'gt')
        # coordinate_angles_gt = np.arccos(target_vec_w_gt / np.linalg.norm(target_vec_w_gt, axis=1).reshape((-1, 1))).reshape(1,3)
        # anchor_angles_gt = []
        # anchor_vec_gt = self.get_anchor_vec('gt')
        # for anchor_vec_c_gt in anchor_vec_gt:
        #     # anchor_vec = self.pix2vec(anchor)
        #     anchor_angles_gt.append(calculate_angle(target_vec_c_gt, anchor_vec_c_gt))
        # anchor_angles_gt = np.array(anchor_angles_gt).reshape(1,self.na)
        # res_gt = np.hstack([coordinate_angles_gt, anchor_angles_gt])

        # dif = np.linalg.norm(np.rad2deg(res)-np.rad2deg(res_gt))
        # return res, dif

