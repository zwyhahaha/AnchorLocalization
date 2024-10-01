import numpy as np
import scipy.optimize as op

cons = [
        {'type': 'ineq', 'fun': lambda z: 25 - z[0]},
        {'type': 'ineq', 'fun': lambda z: z[0] + 25},
        {'type': 'ineq', 'fun': lambda z: 25 - z[1]},
        {'type': 'ineq', 'fun': lambda z: z[1] + 25},
        {'type': 'ineq', 'fun': lambda z: z[2] - 1.4},    # z[2] >= 1.4
        {'type': 'ineq', 'fun': lambda z: 2 - z[2]}       # z[2] <= 2
    ]

class batch_localization_case():
    def __init__(self, batch_size, cams, cams_id, init_loc, obs, target_pix=None):
        self.batch_size = batch_size
        self.start = init_loc
        self.obs = obs
        self.target_pix = target_pix
        self.cams_id = cams_id
        self.cams = cams
        self.loc_history = []

    def solve_batch_loc(self,batch_rho):
        def loss_function(z):
            res = 0
            z = np.array(z).reshape(self.batch_size,3)
            start = np.array(self.start).reshape(self.batch_size,3)
            # if len(self.loc_history):
            #     res += rho * np.linalg.norm(z[0,:]-self.loc_history[-1])
            for t in range(self.batch_size):
                if t > 0:
                    res += batch_rho * np.linalg.norm(z[t,:]-z[t-1,:])
                for i,id in enumerate(self.cams_id[t]):
                    zt = np.array(z[t,:]).reshape(3, 1)
                    z_pix,_ = self.cams[id].world2pix(zt)
                    z_pix = np.array(z_pix)
                    res += np.linalg.norm(z_pix-self.obs[t][i])
                if len(self.cams_id[t]) == 1:
                    res += batch_rho * np.linalg.norm(z[t,:]-start[t,:]) / 4
            return res
        def cons_function(z):
            n = z.shape[0]
            cons = []
            for i in range(n):
                if i%3 == 0:
                    cons.append(25 - z[i])
                    cons.append(z[i] + 25)
                elif i%3 == 1:
                    cons.append(25 - z[i])
                    cons.append(z[i] + 25)
                elif i%3 == 2:
                    cons.append(2 - z[i])
                    cons.append(z[i] - 1)
            return cons
        solve_dict = op.minimize(loss_function, self.start, constraints={'type': 'ineq', 'fun': cons_function}, method='SLSQP')
        x = np.array(solve_dict['x'])
        loc = x.reshape(self.batch_size,3)
        self.loc_history.append(loc)
        return loc
    
    def solve_batch_loc_real(self,batch_rho):
        def loss_function(z):
            res = 0
            z = np.array(z).reshape(self.batch_size,3)
            start = np.array(self.start).reshape(self.batch_size,3)
            # if len(self.loc_history):
            #     res += rho * np.linalg.norm(z[0,:]-self.loc_history[-1])
            for t in range(self.batch_size):
                if t > 0:
                    res += batch_rho * np.linalg.norm(z[t,:]-z[t-1,:])
                for i,id in enumerate(self.cams_id[t]):
                    zt = np.array(z[t,:]).reshape(3, 1)
                    z_pix = self.cams[id].world2pix(zt)
                    z_pix = np.array(z_pix)
                    res += np.linalg.norm(z_pix-self.obs[t][i])
                if len(self.cams_id[t]) == 1:
                    res += batch_rho * np.linalg.norm(z[t,:]-start[t,:]) / 4
            return res
        def cons_function(z):
            n = z.shape[0]
            cons = []
            for i in range(n):
                if i%3 == 0:
                    cons.append(11.5 - z[i])
                    cons.append(z[i] + 1)
                elif i%3 == 1:
                    cons.append(4.5 - z[i])
                    cons.append(z[i] + 1)
                elif i%3 == 2:
                    cons.append(2 - z[i])
                    cons.append(z[i] - 0.7)
            return cons
        solve_dict = op.minimize(loss_function, self.start[0], constraints={'type': 'ineq', 'fun': cons_function}, method='SLSQP')
        x = np.array(solve_dict['x'])
        loc = x.reshape(self.batch_size,3)
        self.loc_history.append(loc)
        return loc
    
    def solve_frame_loc(self,rho):
        def loss_function(z):
            res = 0
            z = z = np.array(z).reshape(3,1)
            start = np.array(self.start).reshape(3,1)
            if len(self.cams_id[0]) == 1:
                res += rho * np.linalg.norm(z-start)
            for i,id in enumerate(self.cams_id[0]):
                z_pix,_ = self.cams[id].world2pix(z, my_type = 'prox')
                z_pix = np.array(z_pix)
                res += np.linalg.norm(z_pix-self.obs[0][i])
            return res
        solve_dict = op.minimize(loss_function, self.start, constraints=cons, method='SLSQP')
        x = np.array(solve_dict['x'])
        loc = x.reshape(1,3)
        self.loc_history.append(loc)
        return loc
    
    def solve_basic_loc(self,rho,my_type='prox'):
        def loss_function(z):
            res = 0
            z = z = np.array(z).reshape(3,1)
            start = np.array(self.start).reshape(3,1)
            if len(self.cams_id[0]) == 1:
                res += rho * np.linalg.norm(z-start)
            for i,id in enumerate(self.cams_id[0]):
                z_pix,_ = self.cams[id].world2pix(z, my_type=my_type)
                z_pix = np.array(z_pix)
                res += np.linalg.norm(z_pix-self.target_pix[0][i])
            return res
        solve_dict = op.minimize(loss_function, self.start, constraints=cons, method='SLSQP')
        x = np.array(solve_dict['x'])
        loc = x.reshape(1,3)
        self.loc_history.append(loc)
        return loc
        