from casadi import *
# import matplotlib.pyplot as plt
import numpy as np

from fishing_rod_12_joints_np_casadi._coriolis_matrix_np import _coriolis_matrix
from fishing_rod_12_joints_np_casadi._inertia_matrix_np import _inertia_matrix
from fishing_rod_12_joints_np_casadi._gravity_matrix_np import _gravity_matrix
from fishing_rod_12_joints_np_casadi._p_tip_np import _p_tip
from fishing_rod_12_joints_np_casadi._jacobian_diff_np import _jacobian_diff

########################################################################################################################
## Model Equations Fishing Rod
########################################################################################################################

def compute_dyn(x, action):
    ampli = 1
    g = -9.81 * 0
    m = 1 * vertcat(0.04, 0.04, 0.01, 0.008, 0.006, 0.005, 0.004, 0.004, 0.003, 0.0028, 0.0026, 0.0022)
    k = 1e0 * vertcat(0, 34.61, 17.20, 11.89, 12.61, 8.88, 3.65, 3.05, 6.4, 2.63, 3.02, 1.37)
    d = 5e-1 * vertcat(0.19, 0.16, 0.08, 0.06, 0.06, 0.04, 0.02, 0.15, 0.3, 0.1, 0.15, 0.06)
    L = vertcat(0.7, 0.35, 0.35, 0.20, 0.20, 0.15, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1)
    I_zz = 1 * m * L**2  

    n = 12 # len(x) // 2
    q = x[:n]
    q_dot = x[-n:]
    q = reshape(q, n, 1)
    q_dot = reshape(q_dot, n, 1)

    M = _inertia_matrix(q, L, m, I_zz)
    iM = pinv(M[0][0])
    # C = _coriolis_matrix(q, q_dot, L, m, I_zz)
    # C = C[0][0]
    G = _gravity_matrix(q, L, m, I_zz, g)
    G = G[0][0]
    # D = diag(d) + diag(d[:-1] / ampli, k=-1) + diag(d[:-1] / ampli, k=1) 
    # K = diag(k) + diag(k[:-1] / ampli, k=-1) + diag(k[:-1] / ampli, k=1)
    d2 = d[:-1]  # first off-diagonal
    k2 = k[:-1]  # first off-diagonal
    
    # D = Sparsity_diag([d, d2, d2], [-1, 0, 1], n, n)
    # K = Sparsity_diag([k, k2, k2], [-1, 0, 1], n, n)
    
    # Create sparsity patterns
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
    q_ddot = -mtimes(iM, MX(G) + mtimes(D, q_dot) + mtimes(K, q)) + mtimes(iM, A * action)    #  mtimes(C, q_dot)
    
    # M = _inertia_matrix(q, L, m, I_zz)
    # # C = _coriolis_matrix(q, q_dot, L, m, I_zz)
    # G = _gravity_matrix(q, L, m, I_zz, g)
    # q_ddot = -pinv(M[0][0]) @ (G[0][0] + diag(d) @ q_dot + diag(k) @ q) + ampli * action
    # # q_ddot = -pinv(M[0][0]) @ (C[0][0] @ q_dot + G[0][0] + diag(d) @ q_dot + diag(k) @ q) + ampli * action
    
    x_dot = vertcat(q_dot, q_ddot)
    # x_dot = reshape(x_dot, n * 2, 1)
    return x_dot

def get_pos_err(x):
    n = 12 # len(x) // 2
    q = x[:n]
    L = vertcat(0.7, 0.35, 0.35, 0.20, 0.20, 0.15, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1)
    pos = _p_tip(q, L)
    pos = pos
    # print(f"pos: {pos} \n")
    return pos

def get_vel_err(x):
    n = 12 # len(x) // 2
    q = x[:n]
    q_dot = x[-n:]
    L = vertcat(0.7, 0.35, 0.35, 0.20, 0.20, 0.15, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1) 
    J = _jacobian_diff(q, L)
    J = J[0]
    vel = norm_2(mtimes(J, q_dot))
    return vel

########################################################################################################################
## Starting with the Optmial Plannning anf Control Problem
########################################################################################################################

print('\n\n')
# Direct multiple shooting optimization
N_states = 24
N_joints = int(N_states/2)
N_controls = 1
# Simulation time
T = 1.0
# Horizon length (number of optimization steps)
N = 10
# Discretization step (time between two optimization steps)
DT = T/N
# Number of integrations between two steps
n_int = 3
# Integration step
h = DT/n_int
# print(f"\tint_step = {h}")
u_lb = -10
u_ub = 10 
# Initial Condition
x0 = [0.001] * N_states 
u0 = [0.0] 
pos_d = [2.58, 0.15]
pos_des = vertcat(pos_d) # desired position final   
vel_des = 10.0  # desired final velocity               

# State and control variables
u = MX.sym("u", N_controls)            # control
x = MX.sym("x", N_states)              # states

x_dot = compute_dyn(x, u)
pos= get_pos_err(x)
vel_err = get_vel_err(x)

# Objective function term
L = 0.5 * u**2  # control cost

print(pos)
# print(pos.shape)
print('\n')
# print(vel_err)
# print(vel_err.shape)
# # input()

dynamics_fishing = Function('compute_dyn', [x, u], [x_dot, L])
# dynamics_fishing = Function('compute_dyn', [x, u], [compute_dyn(x, u)])
pos_fun = Function('get_pos_err', [x], [pos[0], pos[1]])
vel_fun = Function('get_vel_err', [x], [vel_err])

# Integrator Contact H-F Phase
X0 = MX.sym('X0', N_states)
U = MX.sym('U', N_controls)
X = X0
Q = 0
print('\tComputing Dynamic ... \n')
for j in range(n_int):
    k1, k1_q = dynamics_fishing(X, U)
    X = X + h * k1 ## tutta la dinamica esplosa simbolic
    Q = Q + h*k1_q
## definisco la funzione integrante
F_dynamics = Function('F_dynamics', [X0, U], [X, Q], ['x0', 'u0'], ['xf', 'qf'])

########################################################################################################################
# Start with an empty NLP
w = []
w0 = []    # Initial guess
lbw = []
ubw = []
J = 0
G = []
lbg = []
ubg = []

# "Lift" initial conditions
Xk = MX.sym('X0', N_states)
w += [Xk]   # It will create a structure like [X0, U_0, X1, U_1 ...]
lbw += x0  # bound only for states
ubw += x0

# Initial dynamics propagation with constant input
x0_k = x0
u0_k = u0
w0 += x0_k
w0 += u0_k

terrain = []
for k in range(N):

    x0_k = F_dynamics(x0=x0_k, u0=u0_k)  # return a DM type structure
    pos_0 = pos_fun(x0)
    vel_0 = vel_fun(x0)
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
    J = J+Fk['qf']
    # New NLP variable for state at end of interval
    Xk = MX.sym('X_' + str(k+1), N_states)
    w += [Xk]
    lbw += [-inf] * N_states  # bound only for states
    ubw += [inf] * N_states

    # pos_k = pos_fun(Xk)
    # print(pos_k[0].value)
    # vel_k = vel_fun(Xk)
    
    # Add continuity constraint
    G += [Xk_end-Xk]
    lbg += [0.0] * N_states
    ubg += [0.0] * N_states
    #print(f"\tAdding constraints {lbg} and {ubg} for step {k} \n")
    # final step
    if k == N-1:
        #   Xk_target = Xk_end
        L_links = vertcat(0.7, 0.35, 0.35, 0.20, 0.20, 0.15, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1)
        pos_k = _p_tip(Xk, L_links)
        print(f"pos_k: {pos_k} \n")
        # vel_k = get_vel_err(Xk)
        # G += [pos_k[0] - pos_des[0]]
        # lbg += [-0.01]
        # ubg += [0.01]
        G += [pos_k[1] - pos_d[1]]
        lbg += [-0.001]
        ubg += [0.001]
        # G += [vel_k - vel_des]
        # lbg += [-5]
        # ubg += [5]

# Add final state cost term
# alpha = 1e8
# beta = 1e-1
# J = J + alpha * ((Xk_end[2]-x_des[2])**2) + beta * ((Xk_end[7]-x_des[7])**2 + (Xk_end[8]-x_des[8])**2 + (Xk_end[9]-x_des[9])**2)
L_links = vertcat(0.7, 0.35, 0.35, 0.20, 0.20, 0.15, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1)
pos_k = _p_tip(Xk_end, L_links)
vel_k = get_vel_err(Xk_end)
print(f"pos_k: {pos_k} \n")
J = J + 0.25e1 * (vel_k - vel_des)**2 + 1e0 * (pos_k[0] - pos_des[0])**2 + 1e15 * (pos_k[1] - pos_des[1])**2
linear_solver = 'ma27'
# Create an NLP solver
prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*G)}
# NLP solver options
# opts = {'ipopt.max_iter': 1e4, 'warn_initial_bounds': 1,'ipopt.linear_solver': linear_solver, 'ipopt.hessian_approximation': 'limited-memory', 'ipopt.tol': 1e-1}
opts = {'ipopt.max_iter': 1e4, 'warn_initial_bounds': 1, 'ipopt.tol': 1e-4}#,'ipopt.linear_solver': linear_solver, 'ipopt.tol': 1e-8}
solver = nlpsol('solver', 'ipopt', prob, opts) 

# print(f'Lower bounds are {lbw} \n')
# Solve the NLP
sol = solver(x0=w_last, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

w_opt = sol['x'].full().flatten()

# Extract the solution
# print(f"Solution: {w_opt}")

# Optimal joint trajectories 
q1_opt = []
q2_opt = []
q3_opt = []
q4_opt = []
q5_opt = []
q6_opt = []
q7_opt = []
q8_opt = []
q9_opt = []
q10_opt = []
q11_opt = []
q12_opt = []
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


## Plot the results

# tgrid = np.linspace(0, T, N)
# plt.figure(1)
# plt.clf()
# plt.plot(tgrid, q1_opt, '-')
# plt.plot(tgrid, q2_opt, '-')
# plt.plot(tgrid, q3_opt, '-')
# plt.plot(tgrid, q4_opt, '-')
# plt.plot(tgrid, q5_opt, '-')
# plt.plot(tgrid, q6_opt, '-')
# plt.plot(tgrid, q7_opt, '-')
# plt.plot(tgrid, q8_opt, '-')
# plt.plot(tgrid, q9_opt, '-')
# plt.plot(tgrid, q10_opt, '-')
# plt.plot(tgrid, q11_opt, '-')
# plt.plot(tgrid, q12_opt, '-')
# plt.ylabel('q')
# plt.xlabel('t')
# plt.grid()
# plt.legend(['q1_opt', 'q2_opt', 'q3_opt', 'q4_opt', 'q5_opt', 'q6_opt', 'q7_opt', 'q8_opt', 'q9_opt', 'q10_opt', 'q11_opt', 'q12_opt'])
# plt.show()

# EE position
print("\n")
end_opt = [q1_opt[N-1], q2_opt[N-1], q3_opt[N-1], q4_opt[N-1], q5_opt[N-1], q6_opt[N-1], q7_opt[N-1], q8_opt[N-1], q9_opt[N-1], q10_opt[N-1], q11_opt[N-1], q12_opt[N-1]]
x_pos = _p_tip(end_opt, L_links)
print(f"End effector position: {x_pos}\n")
print(f"pose des: {pos_d} \n")
v_end = get_vel_err(end_opt)
print(f"End effector velocity: {v_end}\n")
# print(f"End effector position error: {pos_d - x_pos}\n")

