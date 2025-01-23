from casadi import *
import numpy as np
import matplotlib.pyplot as plt

import torch 
from scipy.interpolate import interp1d

from omniisaacgymenvs.tasks.fishing_rod_lumped_param.fishing_rod_12_joints_np_casadi._coriolis_matrix_np import _coriolis_matrix
from omniisaacgymenvs.tasks.fishing_rod_lumped_param.fishing_rod_12_joints_np_casadi._inertia_matrix_np import _inertia_matrix
from omniisaacgymenvs.tasks.fishing_rod_lumped_param.fishing_rod_12_joints_np_casadi._gravity_matrix_np import _gravity_matrix
from omniisaacgymenvs.tasks.fishing_rod_lumped_param.fishing_rod_12_joints_np_casadi._p_tip_np import _p_tip
from omniisaacgymenvs.tasks.fishing_rod_lumped_param.fishing_rod_12_joints_np_casadi._jacobian_diff_np import _jacobian_diff

########################################################################################################################
## Model Equations Fishing Rod
## I am using a simplified model with error. This is an initial guess
########################################################################################################################

class ConstFishSiply:
    n = 12 
    ampli = 1
    g = -9.81 * 0
    m = ampli * vertcat(0.04, 0.04, 0.01, 0.008, 0.006, 0.005, 0.004, 0.004, 0.003, 0.0028, 0.0026, 0.0022)
    k = 1e0 * vertcat(0, 34.61, 17.20, 11.89, 12.61, 8.88, 3.65, 3.05, 6.4, 2.63, 3.02, 1.37)
    d = 5e-1 * vertcat(0.19, 0.16, 0.08, 0.06, 0.06, 0.04, 0.02, 0.15, 0.3, 0.1, 0.15, 0.06)
    L = vertcat(0.7, 0.35, 0.35, 0.20, 0.20, 0.15, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1)
    
    alp_vel = 0.25e1
    alp_pos_X = 1e0
    alph_pos_Y = 1e15

    
    max_iter_ipopt = 1e4
    u_bound = 10
    ipopt_tol = 1e-4
    linear_solver = 'ma27'
    warn_initial_bounds = 1
    

def compute_dyn(x, action):
    ampli = ConstFishSiply.ampli
    g = ConstFishSiply.g
    m = ConstFishSiply.m
    k = ConstFishSiply.k
    d = ConstFishSiply.d
    L = ConstFishSiply.L
    I_zz = 1 * m * L**2  

    n = ConstFishSiply.n # len(x) // 2
    q = x[:n]
    q_dot = x[-n:]
    q = reshape(q, n, 1)
    q_dot = reshape(q_dot, n, 1)

    M = _inertia_matrix(q, L, m, I_zz)
    iM = pinv(M[0][0])
    G = _gravity_matrix(q, L, m, I_zz, g)
    G = G[0][0]
    # D = diag(d) + diag(d[:-1] / ampli, k=-1) + diag(d[:-1] / ampli, k=1) 
    # K = diag(k) + diag(k[:-1] / ampli, k=-1) + diag(k[:-1] / ampli, k=1)
    d2 = d[:-1] 
    k2 = k[:-1]
        
    D = DM.zeros(n, n)
    K = DM.zeros(n, n)

    # Populate main diagonal
    for i in range(n):
        D[i, i] = d[i]
        K[i, i] = k[i]

    for i in range(n - 1):
        D[i + 1, i] = d2[i] / ampli  
        D[i, i + 1] = d2[i] / ampli 

        K[i + 1, i] = k2[i] / ampli  
        K[i, i + 1] = k2[i] / ampli  
    
    A = vertcat(1, DM.zeros(n-1, 1))
    q_ddot = -mtimes(iM, MX(G) + mtimes(D, q_dot) + mtimes(K, q)) + mtimes(iM, A * action)  #  mtimes(C, q_dot)
    
    x_dot = vertcat(q_dot, q_ddot)
    return x_dot

def get_pos_err(x):
    n = ConstFishSiply.n # len(x) // 2
    q = x[:n]
    L = ConstFishSiply.L
    pos = _p_tip(q, L)
    pos = pos
    # print(f"pos: {pos} \n")
    return pos

def get_vel_err(x):
    n = ConstFishSiply.n # len(x) // 2
    q = x[:n]
    q_dot = x[-n:]
    L = ConstFishSiply.L 
    J = _jacobian_diff(q, L)
    J = J[0]
    vel = norm_2(mtimes(J, q_dot))
    return vel

def get_vel_X(x):
    n = ConstFishSiply.n # len(x) // 2
    q = x[:n]
    q_dot = x[-n:]
    L = ConstFishSiply.L 
    J = _jacobian_diff(q, L)
    J = J[0]
    vel = mtimes(J, q_dot)
    return vel[0]

########################################################################################################################
## Starting with the Optmial Plannning anf Control Problem
########################################################################################################################

def main_fun_optmial_casadi(tracking_Z_bool=True,
                            pos_d=[2.58, 0.15], 
                            _max_episode_length_s=100,
                            vel_des=10.0, 
                            wanna_plot=True):
    ''' pos_des [Z_coordinate, Y_coordinate] [m], vel_des Module of the velocity [m/s] '''
    print('\n\n')
    
    if isinstance(pos_d, torch.Tensor):
        pos_d = pos_d.cpu().numpy() if pos_d.is_cuda else pos_d.numpy()
    
    if isinstance(vel_des, torch.Tensor):
        vel_des = vel_des.cpu().numpy() if vel_des.is_cuda else vel_des.numpy()
    
    # Direct multiple shooting optimization
    N_states = ConstFishSiply.n * 2
    # N_joints = int(N_states / 2)
    N_controls = 1
    
    if tracking_Z_bool:
        n_tracking = 0
    else:
        n_tracking = 1
    
    # Simulation time
    T = 1.0
    # Horizon length (number of optimization steps)
    N = 200
    # Discretization step (time between two optimization steps)
    DT = T / N
    # Number of integrations between two steps
    n_int = 1
    # Integration step
    h = DT / n_int
    u_lb, u_ub = -ConstFishSiply.u_bound , ConstFishSiply.u_bound 
    x0 = [0.001] * N_states 
    u0 = [0.0] 
    pos_des = vertcat(pos_d) # desired position final   

    u = MX.sym("u", N_controls)            # control
    x = MX.sym("x", N_states)              # states

    x_dot = compute_dyn(x, u)
    # pos = get_pos_err(x)
    # vel_err = get_vel_err(x)

    L = 0.5 * u ** 2  # control cost

    dynamics_fishing = Function('compute_dyn', [x, u], [x_dot, L])
    # pos_fun = Function('get_pos_err', [x], [pos[0], pos[1]])
    # vel_fun = Function('get_vel_err', [x], [vel_err])
    # vel_X = get_vel_X(x)

    # Integrator Contact H-F Phase
    X0 = MX.sym('X0', N_states)
    U = MX.sym('U', N_controls)
    X = X0
    Q = 0
    # print('\tComputing Dynamic ... \n')
    for j in range(n_int):
        k1, k1_q = dynamics_fishing(X, U)
        X = X + h * k1 ## stacking all my dynamics
        Q = Q + h * k1_q
    ## dynamic function to be integrated
    F_dynamics = Function('F_dynamics', [X0, U], [X, Q], ['x0', 'u0'], ['xf', 'qf'])

    # Start with an empty NLP
    w, w0= [], [] # intial guess
    lbw, ubw = [], []
    J = 0
    G = [] # constarint set 
    lbg, ubg = [], [] # bound set 

    # "Lift" initial conditions
    Xk = MX.sym('X0', N_states)
    w += [Xk]   # It will create a structure like [X0, U_0, X1, U_1 ...]
    lbw += x0   # bound only for states
    ubw += x0

    # Initial dynamics propagation with constant input
    x0_k = x0
    u0_k = u0
    w0 += x0_k
    w0 += u0_k

    for k in range(N):

        x0_k = F_dynamics(x0=x0_k, u0=u0_k)  # return a DM type structure
        x0_k = x0_k['xf']

        x0t = np.transpose(x0_k.__array__())
        x0l = x0t.tolist()
        x0_k = x0l[0]

        w0 += x0_k
        
        if k != N-1:
            w0 += u0_k

    w_last = w0

    # Formulate the NLP
    for k in range(N):
        # New NLP variable for the control
        Uk = MX.sym('U_' + str(k), N_controls)
        w += [Uk]
        lbw += [u_lb]
        ubw += [u_ub]
        # Integrate till the end of the interval
        Fk = F_dynamics(x0=Xk, u0=Uk)  # return a DM type structure

        Xk_end = Fk['xf']
        J = J + Fk['qf']
        # New NLP variable for state at end of interval
        Xk = MX.sym('X_' + str(k+1), N_states)
        w += [Xk]
        lbw += [-inf] * N_states  # bound only for states
        ubw += [inf] * N_states
        
        # Add continuity constraint
        G += [Xk_end - Xk]
        lbg += [0.0] * N_states
        ubg += [0.0] * N_states
        
        # # Velocities constraints
        # vel_k = get_vel_X(Xk)
        # G += [vel_k]
        # lbg += [-inf] # velocity must be less than 0, so no lower bound (or -inf)
        # ubg += [0.0]  
        
        #print(f"\tAdding constraints {lbg} and {ubg} for step {k} \n")
        # final step
        if k == N-1:
            # Xk_target = Xk_end
            pos_k = _p_tip(Xk, ConstFishSiply.L)
            vel_k = get_vel_err(Xk)
            
            # G += [pos_k[0] - pos_des[0]]
            # lbg += [-0.01]
            # ubg += [0.01]
                        
            G += [pos_k[n_tracking] - pos_d[n_tracking]]    
            lbg += [-0.001]
            ubg += [0.001]
            # G += [vel_k - vel_des]
            # lbg += [-5]
            # ubg += [5]

    pos_k = _p_tip(Xk_end, ConstFishSiply.L)
    # vel_k = get_vel_err(Xk_end)
    vel_k = get_vel_X(Xk_end)
    
    J = J + ConstFishSiply.alp_vel * (vel_k - vel_des)**2 + ConstFishSiply.alp_pos_X * (pos_k[0] - pos_des[0])**2 + ConstFishSiply.alp_pos_X * (pos_k[1] - pos_des[1])**2
    
    ## Create an NLP solver  
    prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*G)}
    # opts = {'ipopt.max_iter': ConstFishSiply.max_iter_ipopt, 'warn_initial_bounds': ConstFishSiply.warn_initial_bounds,'ipopt.linear_solver': ConstFishSiply.linear_solver, 'ipopt.hessian_approximation': 'limited-memory', 'ipopt.tol': 1e-1}
    opts = {'ipopt.max_iter': ConstFishSiply.max_iter_ipopt, 'warn_initial_bounds': ConstFishSiply.warn_initial_bounds, 'ipopt.tol': ConstFishSiply.ipopt_tol}
    solver = nlpsol('solver', 'ipopt', prob, opts) 
    sol = solver(x0=w_last, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    
    stats = solver.stats()
    w_opt = sol['x'].full().flatten()
    
    if stats['return_status'] == 'Solve_Succeeded':
        print("Optimization successful!")
        
        # q_opt, u1_opt = [], []
        # for i in range(N):
        #     start_index = i * (N_states + N_controls)
        #     q_step_opt = w_opt[start_index : start_index + N_states] # Extract states for this time step
        #     q_opt.append(q_step_opt) # Append the state vector for this time step
        #     u1_opt += [w_opt[start_index + N_states]] # control is after states
        # q_opt_array = np.array(q_opt)
        # # qq1_opt1 = q_opt_array[:, 1]
        
        # Optimal joint trajectories 
        q1_opt, q2_opt, q3_opt, q4_opt = [], [], [], []
        q5_opt, q6_opt, q7_opt, q8_opt = [], [], [], []
        q9_opt, q10_opt, q11_opt, q12_opt = [], [], [], []
        # Optimal controls
        u1_opt = []
        
        for i in range(N):
            q1_opt += [w_opt[i*(N_states+N_controls)]]
            q2_opt += [w_opt[i*(N_states+N_controls) + 1]]
            q3_opt += [w_opt[i * (N_states + N_controls) + 2]]
            q4_opt += [w_opt[i * (N_states + N_controls) + 3]]
            q5_opt += [w_opt[i * (N_states + N_controls) + 4]]
            q6_opt += [w_opt[i * (N_states + N_controls) + 5]]
            q7_opt += [w_opt[i * (N_states + N_controls) + 6]]
            q8_opt += [w_opt[i * (N_states + N_controls) + 7]]
            q9_opt += [w_opt[i * (N_states + N_controls) + 8]]
            q10_opt += [w_opt[i * (N_states + N_controls) + 9]]
            q11_opt += [w_opt[i * (N_states + N_controls) + 10]]
            q12_opt += [w_opt[i * (N_states + N_controls) + 11]]

            u1_opt += [w_opt[i*(N_states+N_controls) + 24]]
            
        if wanna_plot:
            tgrid = np.linspace(0, T, N)
            plt.figure(1)
            plt.clf()
            plt.plot(tgrid, q1_opt, '-')
            plt.plot(tgrid, q2_opt, '-')
            plt.plot(tgrid, q3_opt, '-')
            plt.plot(tgrid, q4_opt, '-')
            plt.plot(tgrid, q5_opt, '-')
            plt.plot(tgrid, q6_opt, '-')
            plt.plot(tgrid, q7_opt, '-')
            plt.plot(tgrid, q8_opt, '-')
            plt.plot(tgrid, q9_opt, '-')
            plt.plot(tgrid, q10_opt, '-')
            plt.plot(tgrid, q11_opt, '-')
            plt.plot(tgrid, q12_opt, '-')
            plt.ylabel('q')
            plt.xlabel('t')
            plt.grid()
            plt.legend(['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11', 'q12'])
            plt.savefig('/home/michele/Desktop/q.png')
            # plt.show()

            plt.figure(2)
            plt.clf()
            plt.plot(tgrid, u1_opt, '-')
            plt.ylabel('u')
            plt.xlabel('t')
            plt.grid()
            plt.savefig('/home/michele/Desktop/u.png')
            # plt.show()


        # EE position
        print("\n")
        end_opt = [q1_opt[N-1], q2_opt[N-1], q3_opt[N-1], q4_opt[N-1], q5_opt[N-1], q6_opt[N-1], q7_opt[N-1], q8_opt[N-1], q9_opt[N-1], q10_opt[N-1], q11_opt[N-1], q12_opt[N-1]]
        
        x_pos = _p_tip(end_opt, ConstFishSiply.L)
        v_end = get_vel_err(end_opt)
        
        print(f"End effector pos     :  {np.array([np.round(float(value), 2) for value in x_pos])}")
        print(f"pose des             :  {np.round(pos_d, 2)}\n")
        print(f"End effector vel     :  {v_end}")
        print(f"vel des              :  {vel_des}")
        
        x_original = np.linspace(0, 1, len(u1_opt)) 
        x_target = np.linspace(0, 1, int(_max_episode_length_s))  
        interpolation_func = interp1d(x_original, u1_opt, kind='linear')
        
        # x_original_q1 = np.linspace(0, 1, len(q1_opt)) 
        # x_target_q1 = np.linspace(0, 1, int(_max_episode_length_s))  
        # interpolation_func_q1 = interp1d(x_original_q1, q1_opt, kind='linear')
        # q1_opt = interpolation_func_q1(x_target_q1) 

        u1_opt = interpolation_func(x_target)
    
        # return u1_opt, q1_opt
        return u1_opt
        
    else:
        print(f"Optimization failed: {stats['return_status']}")
        u1_opt = main_fun_optmial_casadi()
        return u1_opt


if __name__ == "__main__":
    u_opt = main_fun_optmial_casadi()
    print('\n')
    u_opt = np.array(u_opt)
    print(f"Control len   : {len(u_opt)}")
    print(f"Control shape : {np.shape(u_opt)}")
    print(f"Control type  : {type(u_opt)}")
    

