import numpy as np
from scipy.optimize import root
# import matplotlib.pyplot as plt
import CONST as c 
import time
# plt.rcParams['text.latex.preamble'] = ''.join([r'\usepackage{siunitx}', r'\usepackage{amsmath}'])
# np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

# from fishing_rod_12_joints._inertia_matrix import _inertia_matrix  
# from fishing_rod_12_joints._coriolis_matrix import _coriolis_matrix 
# from fishing_rod_12_joints._gravity_matrix import _gravity_matrix 
# from fishing_rod_12_joints._stiffness_torque import _stiffness_torque 
# from fishing_rod_12_joints._p_tip import _p_tip
# from fishing_rod_12_joints._damping_torque import _damping_torque
# from fishing_rod_12_joints._jacobian_diff import _jacobian_diff

from fishing_rod_12_joints_np._inertia_matrix_np import _inertia_matrix  
from fishing_rod_12_joints_np._coriolis_matrix_np import _coriolis_matrix 
from fishing_rod_12_joints_np._gravity_matrix_np import _gravity_matrix 
from fishing_rod_12_joints_np._stiffness_torque_np import _stiffness_torque 
from fishing_rod_12_joints_np._p_tip_np import _p_tip
from fishing_rod_12_joints_np._damping_torque_np import _damping_torque
from fishing_rod_12_joints_np._jacobian_diff_np import _jacobian_diff


class DataRobot():
    def __init__(self, 
                q0,
                q0_dot,
                q0_ddot,
                action_eq,
                underactuation_matrix=None, 
                C_y=None):
        """
        Class handling the robot dynamics.

        Args:
            q0 (_type_): initial condition configuration
            q0_dot (_type_): initial condition velocities
            q0_ddot (_type_): initial condition acceleration
            action_eq (_type_): initial guess control or just the first jest 
            underactuation_matrix (_type_, optional): Matrix that maps the input into the structure
            C_y (_type_, optional): Output function matrix to select a LINEAR combination of states
        """
        self.ampli = 1e10 # 5e0
        self.x0 = np.concatenate([q0, q0_dot], dtype=np.float64)
        self.q0 = np.asanyarray(q0, dtype=np.float64)
        self.q0_dot = np.asanyarray(q0_dot, dtype=np.float64)
        self.q = np.asanyarray(q0, dtype=np.float64)
        self.q_dot = np.asanyarray(q0_dot, dtype=np.float64)
        self.q_ddot = np.asanyarray(q0_ddot, dtype=np.float64)
        self.action = np.asanyarray(action_eq, dtype=np.float64)
        self.m = len(action_eq) # number of actuators 
        self.x = np.concatenate([self.q, self.q_dot], dtype=np.float64)
        self.x_dot = np.concatenate([self.q_dot, self.q_ddot], dtype=np.float64)
        self.n = len(self.q) # state dimension 
        henkel = self.henkel_matrix()
        self.scale = 1

        if C_y is None:
            self.C_y = np.eye(self.n)
        else:
            self.C_y = C_y                  
        
        self.M = np.asanyarray(_inertia_matrix(self.q, c.L, c.m, c.I_zz), dtype=np.float64).reshape((self.n,self.n))
        self.C = np.asanyarray(_coriolis_matrix(self.q, self.q_dot, c.L, c.m, c.I_zz), dtype=np.float64).reshape((self.n,self.n))
        self.G = np.asanyarray(_gravity_matrix(self.q, c.L, c.m, c.I_zz, c.g), dtype=np.float64).reshape((-1,))
        # self.K = self.scale * np.diag(np.asanyarray(_stiffness_torque(self.q, c.k), dtype=np.float64).reshape((-1,)))
        # self.D = self.scale * np.diag(np.asanyarray(_damping_torque(self.q_dot, c.d), dtype=np.float64).reshape((-1,)))
        # self.D = np.diag(c.d) * self.henkel_matrix()
        # self.K = np.diag(c.k) * self.henkel_matrix() 
        self.D = np.diag(c.d) + np.diag(c.d[:-1] / self.ampli, k=-1) + np.diag(c.d[:-1] / self.ampli, k=1) # + np.diag(c.d[:-2] / (self.ampli / 1), k=-2) + np.diag(c.d[:-2] / (self.ampli / 1), k=2)
        self.K = np.diag(c.k) + np.diag(c.k[:-1] / self.ampli, k=-1) + np.diag(c.k[:-1] / self.ampli, k=1) # + np.diag(c.k[:-2] / (self.ampli / 1), k=-2) + np.diag(c.k[:-2] / (self.ampli / 1), k=2)
        self.pos = np.asanyarray(_p_tip(self.q, c.L), dtype=np.float64).reshape((-1,))  
        
        self.iM = np.linalg.pinv(self.M)    
        if underactuation_matrix is None:
            assert 'Insert the Underactuation Matrix matherfucker!'
        else:
            self.underactuation_matrix = underactuation_matrix  
        
        self.A = self.underactuation_matrix
        
        self.q_ddot = np.asanyarray(-np.dot(self.iM, np.dot(self.C, self.q_dot) + self.G + np.dot(self.D, self.q_dot) + np.dot(self.K, self.q)) \
                    + np.dot(self.iM, self.A * self.action), dtype=np.float64).reshape((-1,))

        self.y = np.dot(self.C_y, self.q)
        self.y_dot = np.dot(self.C_y, self.q_dot)
        self.y_ddot = np.dot(self.C_y, self.q_ddot)
    
    def henkel_matrix(self):
        henkel = [[1/(i + j + 1) for j in range(self.n)] for i in range(self.n)]
        return henkel
    
    @staticmethod
    def write_inputs_to_python(inputs, file_path):
        with open(file_path, 'w') as file:
            file.write('inputs = {\n')
            for key, value in inputs.items():
                file.write(f'    \'{key}\': {value},\n')
            file.write('}\n')
                    
    def spinOnes(self, q, q_dot, action):
        
        self.q = np.asanyarray(q, dtype=np.float64)
        self.q_dot = np.asanyarray(q_dot, dtype=np.float64)
        self.action = np.asanyarray(action, dtype=np.float64)
        self.y = np.dot(self.C_y, self.q)
        self.y_dot = np.dot(self.C_y, self.q_dot)
        self.x_dot = np.concatenate([self.q_dot, self.q_ddot], dtype=np.float64)
        self.x = np.concatenate([self.q, self.q_dot],  dtype=np.float64)
        self.M = np.asanyarray(_inertia_matrix(self.q, c.L, c.m, c.I_zz), dtype=np.float64).reshape((self.n,self.n))
        self.C = np.asanyarray(_coriolis_matrix(self.q, self.q_dot, c.L, c.m, c.I_zz), dtype=np.float64).reshape((self.n,self.n))
        self.G = np.asanyarray(_gravity_matrix(self.q, c.L, c.m, c.I_zz, c.g), dtype=np.float64).reshape((-1,))
        # self.K = self.scale * np.diag(np.asanyarray(_stiffness_torque(self.q, c.k), dtype=np.float64).reshape((-1,)))
        # self.D = self.scale * np.diag(np.asanyarray(_damping_torque(self.q_dot, c.d), dtype=np.float64).reshape((-1,)))
        # self.D = np.diag(c.d) * self.henkel_matrix()
        # self.K = np.diag(c.k) * self.henkel_matrix() 
        self.D = np.diag(c.d) + np.diag(c.d[:-1] / self.ampli, k=-1) + np.diag(c.d[:-1] / self.ampli, k=1) # + np.diag(c.d[:-2] / (self.ampli / 1), k=-2) + np.diag(c.d[:-2] / (self.ampli / 1), k=2)
        self.K = np.diag(c.k) + np.diag(c.k[:-1] / self.ampli, k=-1) + np.diag(c.k[:-1] / self.ampli, k=1) # + np.diag(c.k[:-2] / (self.ampli / 1), k=-2) + np.diag(c.k[:-2] / (self.ampli / 1), k=2)
        self.pos = np.asanyarray(_p_tip(self.q, c.L), dtype=np.float64).reshape((-1,))   
        
        time_here = time.time()
        self.iM = np.linalg.pinv(self.M)
        self.q_ddot = np.asanyarray(-np.dot(self.iM, np.dot(self.C, self.q_dot) + self.G + np.dot(self.D, self.q_dot) + np.dot(self.K, self.q)) \
                    + np.dot(self.iM, (self.A * self.action)), dtype=np.float64).reshape((-1,))
        time_2_dyn = time.time() - time_here
        c.conta_all = c.conta_all + 1
        if c.conta == c.TIME2PRINT:
            # print('[INFO]:\t\tTime to compute q_ddot: {} [s]'.format(np.around(time_2_dyn, decimals=6)))
            c.conta = 0 
        else: 
            c.conta = c.conta + 1 
            
        c.append_time_dyn.append(time_2_dyn)
        if c.conta_all > c.max_step - 1:
            mean_time_dyn = np.mean(c.append_time_dyn) 
            print('\n=============================================================================================================================================')
            print('[INFO]:\t\tMEAN Time compute q_ddot: {} [s]'.format(np.around(mean_time_dyn, decimals=6)))
            print('=============================================================================================================================================\n')
        else:
            pass
        
        self.y = np.dot(self.C_y, self.q)
        self.y_dot = np.dot(self.C_y, self.q_dot)
        self.y_ddot = np.dot(self.C_y, self.q_ddot)


    def eulerStep(self, x, u, dt):
        x_new = x + np.multiply(self.getNewStateExplicit(x, u), dt)
        self.x = np.asanyarray(x_new, dtype=np.float64)
        self.q = x_new[:self.n]
        self.q_dot = x_new[-self.n:]
        return x_new
    
    def adaptiveEulerStep(self, x, u, dt):
        # x = self.solveSingularityStatic(x)
        h = dt  # Initial step size
        t = 0.0  # Current time
        tol = 1e-3
        while True:
            ## Try taking a step with step size h
            x1 = x + self.getNewStateExplicit(x, u) * h
            x2 = x + self.getNewStateExplicit(x1, u) * h
            err = np.linalg.norm((x2 - x1) / h)
            if err <= tol:
                self.x = np.asanyarray(x2, dtype=np.float64)
                self.q = x2[:self.n]
                self.q_dot = x2[-self.n:]
                return x2
            else:
                h = h / 2.0
                t += h
                if t >= dt or h < 1e-10:
                    raise Exception("Adaptive Euler step failed: maximum number of steps exceeded.")
    
    def rk4Step(self, x, u, dt):
        k1 = self.getNewStateExplicit(x, u)
        k2 = self.getNewStateExplicit(x + np.multiply(k1, dt / 2), u) 
        k3 = self.getNewStateExplicit(x + np.multiply(k2, dt / 2), u) 
        k4 = self.getNewStateExplicit(x + np.multiply(k3, dt), u) 
        x_new = x + np.multiply(k1 + np.multiply(k2 + k3, 2) + k4, dt / 6)
        self.x = np.asanyarray(x_new, dtype=np.float64)
        self.q = x_new[:self.n]
        self.q_dot = x_new[-self.n:]
        return x_new
    
    def rkfStep(self, x, u, dt):
        ## RKF coefficients
        a = np.array([[0, 0, 0, 0, 0, 0],
                    [1/4, 0, 0, 0, 0, 0],
                    [3/32, 9/32, 0, 0, 0, 0],
                    [1932/2197, -7200/2197, 7296/2197, 0, 0, 0],
                    [439/216, -8, 3680/513, -845/4104, 0, 0],
                    [-8/27, 2, -3544/2565, 1859/4104, -11/40, 0]])
        b = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
        b_star = np.array([25/216, 0, 1408/2565, 2197/4104, -1/5, 0])

        t = 0
        # x = self.solveSingularityStatic(x)
        k = np.zeros((6, len(x)))
        k[0] = self.getNewStateExplicit(x, u)
        err_toll = 1e-2 # 1e-10
        while t < dt:
            dt_left = dt - t
            if dt_left < err_toll:
                break
            dt_next = min(dt_left, 1e-3)
            for i in range(1, 6):
                k[i] = self.getNewStateExplicit(x + dt_next * np.dot(a[i,:i], k[:i]), u)
            x_err = dt_next * np.dot((b - b_star), k)
            err_norm = np.linalg.norm(x_err)
            if err_norm < err_toll:
                x_new = x + x_err
                t += dt_next
                k[0] = self.getNewStateExplicit(x, u)
            else:
                dt_next = 0.8 * dt_left * (1 / err_norm) ** 0.25
        # x = self.solveSingularityStatic(x_new)
        self.x = np.asanyarray(x, dtype=np.float64)
        self.q = x[:self.n]
        self.q_dot = x[-self.n:]
        return x
    
    def trapzOptiStep(self, x_n, x, u, u_n, dt):
        f = self.getNewStateExplicit(x, u)
        f_n = self.getNewStateExplicit(x_n, u_n)
        x_new = x + np.multiply(f + f_n, dt / 2) - x_n
        self.x = np.asanyarray(x_new, dtype=np.float64)
        self.q = x_new[:self.n]
        self.q_dot = x_new[-self.n:]
        return x_new

    def trapzStep(self, x, u, u_n, dt):
        x_0 = self.eulerStep(x, u, dt)
        x_n = root(self.trapzOptiStep, x_0, (x, u, u_n, dt))
        x_new = x_n.x
        self.x = x_new
        self.q = x_new[:self.n]
        self.q_dot = x_new[-self.n:]
        return x_new

    def getNewStateExplicit(self, x, action):
        self.action = action.reshape((-1,))
        self.q = np.asanyarray(x[:self.n], dtype=np.float64)
        self.q_dot = np.asanyarray(x[-self.n:], dtype=np.float64)
        self.M = np.asanyarray(_inertia_matrix(self.q, c.L, c.m, c.I_zz), dtype=np.float64).reshape((self.n,self.n))
        self.C = np.asanyarray(_coriolis_matrix(self.q, self.q_dot, c.L, c.m, c.I_zz), dtype=np.float64).reshape((self.n,self.n))
        self.G = np.asanyarray(_gravity_matrix(self.q, c.L, c.m, c.I_zz, c.g), dtype=np.float64).reshape((-1,))
        # self.K = self.scale * np.diag(np.asanyarray(_stiffness_torque(self.q, c.k), dtype=np.float64).reshape((-1,)))
        # self.D = self.scale * np.diag(np.asanyarray(_damping_torque(self.q_dot, c.d), dtype=np.float64).reshape((-1,)))
        # self.D = np.diag(c.d) * self.henkel_matrix()
        # self.K = np.diag(c.k) * self.henkel_matrix() 
        self.D = np.diag(c.d) + np.diag(c.d[:-1] / self.ampli, k=-1) + np.diag(c.d[:-1] / self.ampli, k=1) # + np.diag(c.d[:-2] / (self.ampli / 1), k=-2) + np.diag(c.d[:-2] / (self.ampli / 1), k=2)
        self.K = np.diag(c.k) + np.diag(c.k[:-1] / self.ampli, k=-1) + np.diag(c.k[:-1] / self.ampli, k=1) # + np.diag(c.k[:-2] / (self.ampli / 1), k=-2) + np.diag(c.k[:-2] / (self.ampli / 1), k=2)
        self.pos = np.asanyarray(_p_tip(self.q, c.L), dtype=np.float64).reshape((-1,))   
        
        self.q_ddot = np.asanyarray(-np.dot(self.iM, np.dot(self.C, self.q_dot) + self.G + np.dot(self.D, self.q_dot) + np.dot(self.K, self.q)) \
                    + np.dot(self.iM, (self.A * self.action)), dtype=np.float64).reshape((-1,))
        
        self.y = np.dot(self.C_y, self.q)
        self.y_dot = np.dot(self.C_y, self.q_dot)
        self.y_ddot = np.dot(self.C_y, self.q_ddot)
        
        x_dot = np.concatenate([self.q_dot, self.q_ddot], dtype=np.float64)                             
        self.x_dot = x_dot
        self.x = np.concatenate([self.q, self.q_dot], dtype=np.float64)
        return x_dot
        
    # def getControl(self, y_des, y_dot_des, y_ddot_des, r, y_dddot_des=None):
    #     '''This code only works for a SISO system and it computes the control action. It implements both a feddback I/O lineratization algorithm in the case of relative degree equal to 2 or 3 '''
    #     K_P = c.K_P
    #     K_V = c.K_V
    #     K_A = c.K_A
    #     if r < 3: 
    #         v = y_ddot_des + K_P * (y_des - self.y) + K_V * (y_dot_des - self.y_dot) 
    #         u = v
    #     else:
    #         v = y_dddot_des + K_P * (y_des - self.y) + K_V * (y_dot_des - self.y_dot) + K_A * (y_ddot_des - self.y_ddot)
    #         u = v
    #     return u
 