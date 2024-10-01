import os, sys
import numpy as np
import math
import itertools

current_work_dir = os.path.dirname(__file__)
src_path, _ = os.path.split(current_work_dir)
sys.path.append(src_path)

np.set_printoptions(precision=5, floatmode='maxprec', suppress=True)

# 2D平面坐标系 和 3D 点云坐标系之间的坐标转换
class Map2DtoWorld:
    @staticmethod
    def create(world_pts, map2d_pts):
        rt, s = Map2DtoWorld.get_reasonable_rt_2d(world_pts, map2d_pts)
        if rt is not None:
            return Map2DtoWorld(np.mat(rt), s)
        else:
            return None

    def __init__(self, rt, scale=1.0):
        self.rt = rt
        self.scale = scale

    def img2world(self, pts):
        img_pts = np.array(pts, dtype=np.float64) / self.scale
        img_pts = np.hstack((img_pts, np.ones(shape=(len(img_pts), 1))))
        word_pts = np.dot(self.rt.I, img_pts.T).T
        word_pts[:, 2] = 0.0
        return word_pts.A

    def world2img(self, pts):
        world_pts = np.array(pts, dtype=np.float64)
        # world_pts 是 2d 世界坐标的齐次坐标，这里第三纬复制1，是为了做2D图的旋转平移变换
        if world_pts.shape[1] == 2:
            world_pts = np.hstack((world_pts, np.ones(shape=(len(world_pts), 1))))
        elif world_pts.shape[1] == 3:
            # print("===")
            world_pts[:, 2] = 1
        else:
            return []
        img_pts = self.scale * np.dot(self.rt, world_pts.T).T
        return img_pts[:, :2].astype(int).tolist()

    @staticmethod
    def RT_2d(world_pts, map2d_pts):

        def s(world_pts, map2d_pts):
            np.seterr(divide='ignore', invalid='ignore')
            shuffle_ic = [1, 2, 0]
            note_qc_len = np.linalg.norm(world_pts[shuffle_ic] - world_pts, ord=2, axis=1, keepdims=False)
            note_pixel_len = np.linalg.norm(map2d_pts[shuffle_ic] - map2d_pts, ord=2, axis=1, keepdims=False)
            S = note_pixel_len / note_qc_len
            S = np.nanmean(S)
            return S

        S = s(world_pts, map2d_pts)
        note_pixel = map2d_pts / S
        A_a, A_b = [], []
        B_a, B_b = [], []
        for qc, pixel in zip(world_pts[0:3], note_pixel[0:3]):
            A_a.append([qc[0], qc[1], 1])
            A_b.append([pixel[0]])
            B_a.append([qc[0], qc[1], 1])
            B_b.append([pixel[1]])
        so = np.linalg.solve(np.mat(A_a), np.mat(A_b))
        so_b = np.linalg.solve(np.mat(B_a), np.mat(B_b))
        RT = np.vstack([so.reshape(1, 3), so_b.reshape(1, 3), np.array([[0, 0, 1]])])
        RT = RT.A  # .A 将矩阵化为数组
        if math.isclose(RT[0][0], -RT[1][1], abs_tol=1e-2) and math.isclose(RT[0][1], RT[1][0], abs_tol=1e-2):
            return RT, S
        return None, 0

    @staticmethod
    def get_reasonable_rt_2d(world_pts, map2d_pts, try_count=1000, distance_pixel=20, distance_rate=0.5):
        distance = 100
        RT = None
        S = 0
        if len(world_pts) != len(map2d_pts) or len(world_pts) < 3:
            return RT, S
        world_pts = np.array(world_pts)
        map2d_pts = np.array(map2d_pts)
        list_comb = [i for i in itertools.combinations(range(len(world_pts)), 3)]
        resonable_try_count = min(len(list_comb), try_count)
        for i in range(resonable_try_count):
            index = list(list_comb[i])
            RT_i, S_i = Map2DtoWorld.RT_2d(world_pts[index], map2d_pts[index])
            if RT_i is None:
                continue
            note_qc_copy = np.c_[world_pts, np.array([1] * world_pts.shape[0])]
            note_pixel_r = S_i * np.dot(RT_i, note_qc_copy.T)
            distance_np = np.linalg.norm(note_pixel_r[[0, 1]].T - map2d_pts, ord=2, axis=1, keepdims=False)
            distance_np = np.where(distance_np < distance_pixel, distance_np, np.nan)
            if np.count_nonzero(np.isnan(distance_np)) / world_pts.shape[0] > distance_rate:
                return None, 0
            distance_i = np.nanmean(distance_np)
            if distance_i < distance:
                distance = distance_i
                RT = RT_i
                S = S_i
        return RT, S

def demo():
    #map2d_word_points = [[-61.319, 57.908], [-62.357, 11.147], [16.448, 96.690], [27.737, -23.181]]
    #map2d_pix_points = [[802.988610, 2304.874715], [758.378132, 4401.567198], [4290.041002, 565.066059], [4780.756264, 5925.758542]]
    
    # map2d_pix_points = [[80.0,1122.0],[80.0,80.0],[1175.0,80.0]]
    # map2d_word_points =[[-5.4568,-5.2805],[-5.456838607788086,
	# 					5.147112846374512],[5.502,5.1471]]
    map2d_pix_points= [
					[
						40.0,
						910.0
					],
					[
						1440.0,
						910.0
					],
					[
						40.0,
						10.0
					]
				]
    map2d_word_points= [
					[
						0.0,
						0.0
					],
					[
						14.0,
						0.0
					],
					[
						0.0,
						9.0
					]
				]
    
    map2world = Map2DtoWorld.create(map2d_word_points, map2d_pix_points)
    pts1 = map2world.img2world([[0,0]])
    pts2 = map2world.img2world([[1256,1203]])
    print(pts1, pts2)

if __name__ == '__main__':
    demo()