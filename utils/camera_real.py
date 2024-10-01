import json
import numpy as np
import cv2
from utils.Cam2World import Cam2World

def calculate_angle(x, y):
    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)
    assert nx != 0
    assert ny != 0
    res = np.sum(x*y)/nx/ny
    res = np.clip(res,-1,1)
    return np.arccos(res)

class realCamera(object):
    def __init__(self, camera_name, config):
        self.config = config
        filename = '{}/{}.json'.format(self.config.camera_dir,camera_name)
        with open(filename, 'r') as f:
            self.cam_cfg = json.load(f)
        self.rvec, self.tvec, self.mtx, self.dist, self.cam_name = np.hstack(self.cam_cfg["rvec"]), np.hstack(self.cam_cfg["tvec"]), np.array(self.cam_cfg["mtx"]), np.array(self.cam_cfg["dist"]), self.cam_cfg["cameraid"]
        self.trans = Cam2World(self.rvec,self.tvec, self.mtx, self.dist, "unknown")
        self.R, _ = cv2.Rodrigues(self.rvec)
        self.T = self.tvec.reshape(3,1)
        self.cam_loc = self.get_cam_loc() # shape(3,)
        self.na = len(self.cam_cfg["anchors"]["id"])
        self.anchors = np.array(self.cam_cfg["anchors"]["pts3D_measure"]) # better choice
        # self.anchors = np.array(self.cam_cfg["anchors"]["pts3D_restruction"])
        gt_anchor_pix = np.array(self.cam_cfg["anchors"]["pts2D"])
        prox_anchor_pix = np.array([self.world2pix(loc) for loc in self.anchors])
        pix_err = np.linalg.norm(gt_anchor_pix-prox_anchor_pix, axis=1)
        self.pix_err = np.mean(pix_err)
        sorted_indices = np.argsort(pix_err)[::-1]
        self.gt_anchor_pix = gt_anchor_pix[sorted_indices]
        self.prox_anchor_pix = prox_anchor_pix[sorted_indices]
        self.anchor_vec = [(anchor - self.cam_loc) for anchor in self.anchors]
        self.gt_anchor_vec = np.array([self.pix2vec_w(pix) for pix in self.gt_anchor_pix])
        self.prox_anchor_vec = np.array([self.pix2vec_w(pix) for pix in self.prox_anchor_pix])
        vec_err = [calculate_angle(self.gt_anchor_vec[i],self.prox_anchor_vec[i]) for i in range(self.na)]
        self.vec_err = np.mean(np.rad2deg(vec_err))

    def get_cam_loc(self):
        c1 = Cam2World.from_rvec_tvec(self.rvec, self.tvec, self.mtx, self.dist, name=self.cam_name)
        _, at, _ = c1.as_look_at_up()
        return at
    
    def world2vec_w(self, loc):
        """
        transform world coordinates into relative vec--using calibrated parameters
        """
        pix = self.world2pix(loc)
        vec = self.pix2vec_w(pix)
        return vec
    
    def world2pix(self, loc):
        """
        transform world coordinates into pixel coordinates--using calibrated parameters
        """
        X_w = np.array(loc).reshape((3,1))
        X_cam = np.dot(self.R, X_w) + self.T
        X_cam = X_cam / X_cam[2]
        X_pix = np.dot(self.mtx, X_cam).squeeze()
        if self.config.use_distort:
            k1, k2, p1, p2 = self.dist
            x_prime, y_prime = X_cam[0, 0], X_cam[1, 0]
            r2 = x_prime**2 + y_prime**2
            dx = x_prime * (1 + k1 * r2 + k2 * r2**2) + 2 * p1 * x_prime * y_prime + p2 * (r2 + 2 * x_prime**2)
            dy = y_prime * (1 + k1 * r2 + k2 * r2**2) + 2 * p2 * x_prime * y_prime + p1 * (r2 + 2 * y_prime**2)
            X_pix = np.dot(self.mtx, np.vstack([dx, dy, 1])).squeeze()
        pix = [X_pix[0],X_pix[1]]
        return self.clip(pix)

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
    
    def get_worldpt_from_det(self, det):
        """
        transform pixel coordinates into world coordinates--using calibrated parameters 'trans'
        used for getting initial estimation
        """
        pix_head = [(det[0]+det[2])/2, det[1]]
        pix_head = self.clip(pix_head)
        pix_foot = [(det[0]+det[2])/2, det[3]]
        pix_foot = self.clip(pix_foot)
        # if self.config.det_type == 'head':
        #     world_pt3_head = self.trans.img2world(pix_head[0], pix_head[1], self.config.h)
        #     return world_pt3_head, pix_head, pix_foot
        # elif self.config.det_type == 'foot':
        #     world_pt3_head = self.trans.img2world(pix_head[0], pix_head[1])
        #     world_pt3_foot = self.trans.img2world(pix_foot[0], pix_foot[1])
        #     height = self.height_estimate(world_pt3_foot, world_pt3_head, self.cam_loc)
        #     world_pt3_foot[2] = height
        #     return world_pt3_foot, pix_head, pix_foot
        
        
        world_pt3_head = self.trans.img2world(pix_head[0], pix_head[1])
        world_pt3_foot = self.trans.img2world(pix_foot[0], pix_foot[1])
        height = self.height_estimate(world_pt3_foot, world_pt3_head, self.cam_loc)
        world_pt3_foot[2] = height
        return world_pt3_foot, pix_head, pix_foot
        

    def get_worldpt_from_pix(self, pix_head, *anchor_height):
        """
        transform pixel coordinates into world coordinates--using calibrated parameters 'trans'
        used for getting initial estimation
        """
        pix_head = self.clip(pix_head)
        if not anchor_height:
            world_pt3_head = self.trans.img2world(pix_head[0], pix_head[1], self.config.h)
        else:
            world_pt3_head = self.trans.img2world(pix_head[0], pix_head[1], anchor_height[0])
        return world_pt3_head
        
    def pix2vec_w(self, pix):
        """
        transform pixel into relative vector in world coordinates--using calibrated parameters
        """
        vec = self.pix2vec_c(pix)
        vec = vec.reshape(3, 1)
        rel_direction = np.dot(self.R.T, vec)
        return np.array(rel_direction).reshape(-1)
    
    def pix2vec_c(self, pix):
        """
        transform pixel into relative vector in camera coordinates--using perturbated parameters
        """
        if self.config.use_distort:
            pix = self.clip(pix)
            pix = np.array(self.trans.undistort(pix))
        fx, fy = self.mtx[0][0], self.mtx[1][1]
        cx, cy = self.mtx[0][2], self.mtx[1][2]
        vec = np.hstack([(pix[0]-cx)/fx, (pix[1]-cy)/fy, 1])
        return np.array(vec).reshape(1,3)
    
    def pix2angles(self, pix):
        pix = self.clip(pix)
        target_vec_w = self.pix2vec_w(pix).reshape(1,3)
        coordinate_angles = np.arccos(target_vec_w / np.linalg.norm(target_vec_w, axis=1).reshape((-1, 1))).reshape(-1)
        anchor_angles = []
        for anchor_vec_w in self.gt_anchor_vec:
            anchor_angles.append(calculate_angle(target_vec_w, anchor_vec_w))
        anchor_angles = np.array(anchor_angles)
        # res = np.hstack([coordinate_angles, anchor_angles]).reshape(-1)
        return coordinate_angles, anchor_angles