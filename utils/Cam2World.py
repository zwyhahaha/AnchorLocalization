#! /Users/sijiu/Env/o3d14/bin/python
import os
import sys
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as super_R
import math

import matplotlib.pyplot as plt


current_work_dir = os.path.dirname(__file__)
src_path, _ = os.path.split(current_work_dir)
sys.path.append(src_path)

from utils.ea import Ea

np.set_printoptions(precision=5, floatmode='maxprec', suppress=True)

def _arr(l):
    if type(l) == list:
        return np.array(l)
    elif l is None:
        return np.array([])
    else:
        return l

def _normalize(v):
    return v / np.linalg.norm(v)

class Cam2World(Ea):
    @staticmethod
    def _inv_rvec_tvec(rvec, tvec):
        sR = super_R.from_rotvec(rvec)
        tvec = _arr(tvec)
        inv_rvec = sR.inv().as_rotvec()
        inv_tvec = - tvec @ sR.as_matrix()
        return inv_rvec, inv_tvec

    @staticmethod
    def _transform(p3ds, rvec, tvec):
        sR = super_R.from_rotvec(rvec)
        tvec = _arr(tvec)
        out_p3ds = p3ds @ sR.inv().as_matrix() + tvec
        return out_p3ds

    @staticmethod
    def from_rvec_tvec(rvec, tvec, mtx=None, dist=None, name=""):
        return Cam2World(rvec, tvec, mtx, dist, name)

    @staticmethod
    def from_look_at_up(look, at, up, mtx=None, dist=None, name=""):
        z = _normalize(_arr(look) - _arr(at))
        x = _normalize(np.cross(z, _arr(up) - _arr(at)))
        y = _normalize(np.cross(z, x))

        sR = super_R.from_matrix(_arr([x, y, z]))
        rvec = sR.as_rotvec()
        tvec = - _arr(at) @ sR.inv().as_matrix()
        return Cam2World(rvec, tvec, mtx, dist, name)

    @staticmethod
    def mtx_from_fov(fov, width, height):
        f = 0.5 * max(width, height) * 1 / np.tan(0.5 * fov / 180.0 * np.pi)
        cx = (width) / 2.0
        cy = (height) / 2.0
        mtx = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1],
        ], np.float32)
        # K_inv = np.linalg.inv(K)
        return mtx

    def __init__(self, rvec, tvec, mtx, dist, cam_name):
        Ea.__init__(self)
        self.name = cam_name
        self.wh = (1280,720)
        # 相机外参 这是 world to camera 的变换
        self.rvec = _arr(rvec)
        self.tvec = _arr(tvec)
        # 相机内参
        self.mtx = _arr(mtx)   # 投影矩阵
        self.dist = _arr(dist) # 畸变系数
        # 坐标畸变反畸变算法用数据初始化
        self.new_mtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, self.wh, 1, self.wh)
        self.distort_map = cv2.initUndistortRectifyMap(self.mtx, self.dist, None, self.new_mtx, self.wh, cv2.CV_32FC1)
        self.undistort_map = cv2.initInverseRectificationMap(self.mtx, self.dist, None, self.new_mtx, self.wh,
                                                             cv2.CV_32FC1)

        # print(self.new_mtx)

    # 无畸变坐标， 到有畸变坐标
    def distort(self, xy):
        mapx, mapy = self.distort_map
        return (mapx[round(xy[1]), round(xy[0])], mapy[round(xy[1]), round(xy[0])])

    # 有畸变坐标， 到无畸变坐标
    def undistort(self, xy):
        mapx, mapy = self.undistort_map
        #print("undistort", xy)
        return (mapx[round(xy[1]), round(xy[0])], mapy[round(xy[1]), round(xy[0])])

    @ staticmethod
    def calculate_angle(x, y):
        nx = np.linalg.norm(x)
        ny = np.linalg.norm(y)
        assert nx != 0
        assert ny != 0
        return np.arccos(np.sum(x * y) / nx / ny)

    # 获得相机某个像素，在世界坐标系下的朝向
    def direction(self,x, y, r=1.0):
        # 先做反畸变
        px, py = self.undistort([x, y])
        p1 = self.c2w([[0, 0, 0]])
        p2 = self.c2w([[px - self.new_mtx[0][2], (py - self.new_mtx[1][2]) * (self.new_mtx[0][0] / self.new_mtx[1][1]),
             self.new_mtx[0][0]]])
        p2 = p1 + (p2-p1) / np.linalg.norm(p2-p1)
        return p2[0] - p1[0]

    # 照片上某个像素，投影到世界坐标系（求解光线于H水平面的交点）
    def img2world(self, x, y, H = 0.0):
        # 对照片像素照片做反畸变处理
        #print(x, y)
        px, py = self.undistort([int(x), int(y)])
        px, py = x, y
       # print(px, py)
        c = np.array([0, 0, 0, 1], dtype=np.float64)
        # p1 = np.dot(self.m4_cam2world, c)
        p1 = self.c2w([[0, 0, 0]])

        # p2 = np.dot(self.m4_cam2world, np.array(
        #     [px - self.new_mtx[0][2], (py - self.new_mtx[1][2]) * (self.new_mtx[0][0] / self.new_mtx[1][1]),
        #      self.new_mtx[0][0], 1]))
        # p2 = self.c2w([[px - self.new_mtx[0][2], (py - self.new_mtx[1][2]) * (self.new_mtx[0][0] / self.new_mtx[1][1]),
        #      self.new_mtx[0][0]]])

        p2 = self.c2w([[px - self.mtx[0][2], (py - self.mtx[1][2]) * (self.mtx[0][0] / self.mtx[1][1]),
              self.mtx[0][0]]])
        
        #print(p1)

        #x1, y1, z1, _ = p1.tolist()[0]
        x1, y1, z1 = p1.tolist()[0]
        # print("p1:", x1, y1, z1)
        p1 = np.array([x1, y1, z1], dtype=np.float64)

        #x2, y2, z2, _ = p2.tolist()[0]
        x2, y2, z2 = p2.tolist()[0]
        # print("p2:", x2, y2, z2)
        p2 = np.array([x2, y2, z2], dtype=np.float64)

        # The floor is XY plane, and the plane abcd is: 0, 0, 1, 0
        # H = 0.0
        a, b, c, d = 0.0, 0.0, 1.0, -H
        m = abs((a * x1 + b * y1 + c * z1 + d) / (a * (x2 - x1) + b * (y2 - y1) + c * (z2 - z1)))
        p = p1 + m * (p2 - p1)
        # 注意，p 是按照p1->p2的方向求的交点。 如果p1->p2方向，与地面H没有交点，则 p[2] 远不等于 H。用来判断此点无交点。
        return p

    # 世界坐标系上的点，投影到照片上
    def world2img(self, p3d_arr):
        # p2d_arr, _ = cv2.projectPoints(p3d_arr,
        #                               self.rvec, self.tvec,
        #                               self.mtx, self.dist)
        #print(self.dist)
        p2d_arr, _ = cv2.projectPoints(p3d_arr, self.rvec, self.tvec, self.mtx, np.array([[0.0, 0.0, 0.0, 0.0]]))

        return p2d_arr

    def inv(self):
        rvec, tvec = self._inv_rvec_tvec(self.rvec, self.tvec)
        return Cam2World(rvec, tvec, self.mtx, self.dist, self.name)

    def w2c(self, w_p3ds):
        return self._transform(w_p3ds, self.rvec, self.tvec)

    def c2w(self, c_p3ds):
        rvec, tvec = self._inv_rvec_tvec(self.rvec, self.tvec)
        return self._transform(c_p3ds, rvec, tvec)

    def as_q4(self):
        return super_R.from_rotvec(self.rvec).as_quat()
    def as_euler(self, seq):
        r1 = super_R.from_rotvec(self.rvec)
        return r1.as_euler(seq, degrees=True)
    def as_matrix(self):
        r1 = super_R.from_rotvec(self.rvec)
        return r1.as_matrix()
    def as_look_at_up(self, r=1.0):
        look = [0, 0, r]
        at = [0, 0, 0]
        up = [0, -r, 0]
        
        return self.c2w([look, at, up])
    def as_distort_fov(self):
        if self.new_mtx is not None:
            rv = _arr([self.new_mtx[0][2], self.new_mtx[1][2]]) / _arr([self.new_mtx[0][0], self.new_mtx[1][1]])
            fov_wh = np.rad2deg(np.arctan(rv)) * 2
            return fov_wh.tolist()
        else:
            return 0,0
    def as_fov(self):
        if self.mtx is not None:
            rv = _arr([self.mtx[0][2], self.mtx[1][2]]) / _arr([self.mtx[0][0], self.mtx[1][1]])
            fov_wh = np.rad2deg(np.arctan(rv)) * 2
            return fov_wh.tolist()
        else:
            return 0,0

    def rotate_cam_euler(self, seq, euler, degree=True):
        r1 = super_R.from_rotvec(self.rvec)
        r2 = super_R.from_euler(seq, euler, degrees=degree)
        self.rvec = super_R.from_matrix(r2.as_matrix() @ r1.as_matrix()).as_rotvec()
        self.tvec = self.tvec @ r2.inv().as_matrix()

    def crop_pts(self, pcd):
        p3d_arr = np.array(pcd.points)
        color_arr = np.array(pcd.colors) * 255
        color_arr = color_arr.astype(int)

        p3d_arr = p3d_arr @ super_R.from_rotvec(self.rvec).as_matrix().T + self.tvec
        color_arr = color_arr[p3d_arr[:, 0] > -5]
        p3d_arr = p3d_arr[p3d_arr[:, 0] > -5]
        color_arr = color_arr[p3d_arr[:, 0] < 5]
        p3d_arr = p3d_arr[p3d_arr[:, 0] < 5]
        color_arr = color_arr[p3d_arr[:, 1] > -5]
        p3d_arr = p3d_arr[p3d_arr[:, 1] > -5]
        color_arr = color_arr[p3d_arr[:, 1] < 5]
        p3d_arr = p3d_arr[p3d_arr[:, 1] < 5]
        color_arr = color_arr[p3d_arr[:, 1] > -15]
        p3d_arr = p3d_arr[p3d_arr[:, 2] > -15]
        color_arr = color_arr[p3d_arr[:, 1] < 15]
        p3d_arr = p3d_arr[p3d_arr[:, 2] < 15]

        p3d_arr = p3d_arr @ super_R.from_rotvec(self.rvec).as_matrix() - super_R.from_rotvec(
            self.rvec).as_matrix().T @ self.tvec
        return p3d_arr, color_arr



    # 创建在相机坐标系中，像素成像平面点云
    def camView2XYZ(self, width, height):
        K = self.mtx
        K_inv = np.linalg.inv(K)
        x = np.arange(width)
        y = np.arange(height)
        x, y = np.meshgrid(x, y)
        z = np.ones_like(x)
        xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
        # xyz = xyz @ K_inv.T
        xyz = xyz.dot(K_inv.T)
        return xyz

def get_rvec(theta, beta, gama):
    rx = np.array([[1,0,0],
        [0, math.cos(theta), -math.sin(theta)],
        [0, math.sin(theta), math.cos(theta)]])
    ry = np.array([[math.cos(beta), 0, math.sin(beta)],
                [0, 1, 0],
                [-math.sin(beta), 0, math.cos(beta)]])
    rz = np.array([[math.cos(gama), -math.sin(gama), 0],
                [math.sin(gama), math.cos(gama), 0],
                [0, 0, 1]])
    rxy = np.dot(rx, ry)
    rxyz = np.dot(rxy, rz)
    
    rvec, _ = cv2.Rodrigues(rxyz)
    rvec = rvec.reshape(3,)
    return rvec



def demo():
    cam_name = "C9b4609dc75246b72"
    mtx = [[902.0713925717712, 0.0, 640.0], [0.0, 897.4015646364949, 360.0], [0.0, 0.0, 1.0]]
    dist = [[-0.4159924837186414, 0.17610949740239404, 0.0015247394441061525, 0.0010117850167096368]]

    theta = 2.2
    beta = 144.52
    gama = 111.51

    

    rvec = get_rvec(theta, beta, gama)
    print("rvec", rvec)

    tvec = [
			0.7126370824070432,
			0.34290128774347186,
			5.413570164458235
		]


    #rvec = [1.353567642288163, 1.8203107366102882, -1.0420136092610854]
    #tvec = [-0.8341879619556444, 4.273592142857085, -1.1351029202540512]
 
    trans = Cam2World(rvec, tvec, mtx, dist, cam_name)
    pts = trans.img2world(720.0,640.0)
    print("1", pts)
    
    # print(pts)
    # rvec1 = get_rvec(theta,beta,gama+1)
    
    # trans1 = Cam2World(rvec1, tvec, mtx, dist, cam_name)

    pix1 = trans.world2img(pts)
    print(pix1)
    


    #print("result", pts)

    # print("=== Check Look At Up ===")
    # c1 = Cam2World.from_rvec_tvec(rvec, tvec, mtx, dist, name=cam_name)
    # look, at, up = c1.as_look_at_up(5)
    # fov = c1.as_fov()
    # print(look, at, up, fov)
    # # c2 = Cam2World.from_look_at_up(look, at, up, name=cam_name)
    # # look, at, up = c2.as_look_at_up(5)
    # # print(look, at, up)

    

    # print("=== Check Rotate cam euler ===")
    # print(c1.as_euler('xyz'))

    # print(c1.as_look_at_up(5)[1])
    # c1.rotate_cam_euler('xyz', [1,2,3])
    # print(c1.as_look_at_up(5)[1])

    # print(c1.as_euler('xyz'))

if __name__ == '__main__':
    demo()
