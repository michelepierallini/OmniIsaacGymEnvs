from casadi import *
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import rc
import os
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
plt.rcParams['text.usetex'] = True

from termcolor import colored

from omniisaacgymenvs.tasks.fishing_rod_lumped_param.fishing_rod_12_joints_np_casadi._coriolis_matrix_np import _coriolis_matrix
from omniisaacgymenvs.tasks.fishing_rod_lumped_param.fishing_rod_12_joints_np_casadi._inertia_matrix_np import _inertia_matrix
from omniisaacgymenvs.tasks.fishing_rod_lumped_param.fishing_rod_12_joints_np_casadi._gravity_matrix_np import _gravity_matrix
from omniisaacgymenvs.tasks.fishing_rod_lumped_param.fishing_rod_12_joints_np_casadi._p_tip_np import _p_tip
from omniisaacgymenvs.tasks.fishing_rod_lumped_param.fishing_rod_12_joints_np_casadi._jacobian_diff_np import _jacobian_diff

class ConstFishSiply:
    n = 12
    ampli = 1
    g = -9.81 * 0
    m = ampli * vertcat(0.04, 0.04, 0.01, 0.008, 0.006, 0.005, 0.004, 0.004, 0.003, 0.0028, 0.0026, 0.0022)
    k = 1e0 * vertcat(0, 34.61, 17.20, 11.89, 12.61, 8.88, 3.65, 3.05, 6.4, 2.63, 3.02, 1.37)
    d = 5e-1 * vertcat(0.19, 0.16, 0.08, 0.06, 0.06, 0.04, 0.02, 0.15, 0.3, 0.1, 0.15, 0.06)
    L = vertcat(0.7, 0.35, 0.35, 0.20, 0.20, 0.15, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1)
    toll_init = 0.001
    N_control = 1

    alp_vel = 0.25e1
    alp_pos_X = 1e0
    alp_pos_Y = 1e15
    T = 1.0 # Total simulation time (can be longer for MPC simulation)
    N = 200 # Optimization horizon steps (MPC horizon)
    DT = T / N
    n_int = 1
    inf = 1e3

    max_iter_ipopt = 1e4
    u_bound = 10
    ipopt_tol = 1e-4
    linear_solver = 'ma27'
    warn_initial_bounds = 1

    linewidth = 3
    fontsize = 15


def compute_dyn(x, action):
    ampli = ConstFishSiply.ampli
    g = ConstFishSiply.g
    m = ConstFishSiply.m
    k = ConstFishSiply.k
    d = ConstFishSiply.d
    L = ConstFishSiply.L
    I_zz = 1 * m * L**2

    n = ConstFishSiply.n 
    q = x[:n]
    q_dot = x[-n:]
    q = reshape(q, n, 1)
    q_dot = reshape(q_dot, n, 1)

    M = _inertia_matrix(q, L, m, I_zz)
    iM = pinv(M[0][0])
    G = _gravity_matrix(q, L, m, I_zz, g)
    G = G[0][0]
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
    n = ConstFishSiply.n 
    q = x[:n]
    L = ConstFishSiply.L
    pos = _p_tip(q, L)
    pos = pos
    return pos

def get_vel_err(x):
    n = ConstFishSiply.n 
    q = x[:n]
    q_dot = x[-n:]
    L = ConstFishSiply.L
    J = _jacobian_diff(q, L)
    J = J[0]
    vel = norm_2(mtimes(J, q_dot))
    return vel

def get_vel_X(x):
    n = ConstFishSiply.n
    q = x[:n]
    q_dot = x[-n:]
    L = ConstFishSiply.L
    J = _jacobian_diff(q, L)
    J = J[0]
    vel = mtimes(J, q_dot)
    ## [Z, X]
    return vel[1]


def solve_mpc_nlp(current_state, pos_d, vel_des, N_mpc, DT, n_int, wanna_info=False):
    """Solves the NLP for MPC given the current state and target."""

    N_states = ConstFishSiply.n * 2
    N_controls = ConstFishSiply.N_control
    u_lb, u_ub = -ConstFishSiply.u_bound, ConstFishSiply.u_bound
    inf = ConstFishSiply.inf
    h = DT / n_int

    u = MX.sym("u", N_controls)            # control
    x = MX.sym("x", N_states)              # states
    x_dot = compute_dyn(x, u)
    dyn_func = Function('dyn_func', [x, u], [x_dot], ['x', 'u'], ['x_dot'])

    ## Integrator RK4 - Casadi function
    X0 = MX.sym('X0', N_states)
    U = MX.sym('U', N_controls)
    X = X0
    for _ in range(n_int):
        k1 = dyn_func(X, U)
        k2 = dyn_func(X + h / 2 * k1, U)
        k3 = dyn_func(X + h / 2 * k2, U)
        k4 = dyn_func(X + h * k3, U)
        X = X + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    F_dynamics = Function('F_dynamics', [X0, U], [X], ['x0', 'u0'], ['xf'])

    ## Start with an empty NLP
    w, w0 = [], []    # variables and initial guess
    lbw, ubw = [], [] # lower and upper bounds for w
    J = 0             # cost function
    g = []            # constraints
    lbg, ubg = [], [] # lower and upper bounds for constraints

    ## "Lift" initial conditions
    Xk = MX.sym('X0', N_states)
    w.append(Xk)
    current_state_evaluated = DM(Function("eval", [], [current_state])()["o0"])
    w0.extend(current_state_evaluated.full().flatten().tolist())
    lbw.extend(current_state_evaluated.full().flatten().tolist())
    ubw.extend(current_state_evaluated.full().flatten().tolist())

    ## Formulate the NLP
    for k in range(N_mpc):
        Uk = MX.sym('U_' + str(k), N_controls)
        w.append(Uk)
        w0.extend([0] * N_controls)  # Initial guess
        lbw.extend([u_lb] * N_controls)
        ubw.extend([u_ub] * N_controls) 
        
        ## Integrate dynamics
        X_next = F_dynamics(x0=Xk, u0=Uk)['xf']
        Xk = MX.sym(f'X_{k + 1}', N_states)
        g.append(X_next - Xk)  # Dynamic constraints
        lbg.extend([0] * N_states)
        ubg.extend([0] * N_states)

        ## Add state variables
        w.append(Xk)
        w0.extend([0] * N_states)  # Initial guess
        lbw.extend([-inf] * N_states)
        ubw.extend([inf] * N_states)
        
        if wanna_info:
            print(f"[INFO - w0 init]: Length of w0 after step {k}: {len(w0)}")
            print(f"[INFO]: Length of w  : {len(w)}")
            print(f"[INFO]: Length of w0 : {len(w0)}")

        ## Cost function
        pos_k = _p_tip(Xk, ConstFishSiply.L)
        # vel_k = get_vel_X(Xk)  
        J += 0.5 * Uk**2  # Control cost
        if k == N_mpc - 1:  # Only track position at the final step of MPC horizon
            # J += ConstFishSiply.alp_vel * (vel_k - vel_des)**2
            J += ConstFishSiply.alp_pos_X * (pos_k[0] - pos_d[0])**2
            J += ConstFishSiply.alp_pos_Y * (pos_k[1] - pos_d[1])**2
            
    if wanna_info:
        print(f"[INFO - Solver Input]: Length of w  : {len(w)}")  
        print(f"[INFO - Solver Input]: Length of w0 : {len(w0)}") 

    ## Create an NLP solver
    w = vertcat(*w)
    g = vertcat(*g)
    prob = {'f': J, 'x': w, 'g': g}
    # opts = {'ipopt.max_iter': ConstFishSiply.max_iter_ipopt, 'warn_initial_bounds': ConstFishSiply.warn_initial_bounds, 'ipopt.tol': ConstFishSiply.ipopt_tol}
    # solver = nlpsol('solver', 'ipopt', prob, opts)
    solver = nlpsol('solver', 'ipopt', prob)

    ## Solve the NLP
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    stats = solver.stats()

    if stats['return_status'] == 'Solve_Succeeded':
        w_opt = sol['x'].full().flatten()
        u_optimal = w_opt[N_states] # The first control action in w_opt after initial state
        return u_optimal, w_opt, stats
    else:
        print(colored(f"[INFO]: Optimization failed in MPC: {stats['return_status']}", 'yellow'))
        return np.array([0.0]), None, stats
    
    
def run_mpc_control(pos_d=[2.58, 0.15], 
                    vel_des=-10.0, 
                    simulation_time=10.0, 
                    initial_state=None, 
                    # tracking_Z_bool=True, 
                    wanna_plot=True, 
                    wanna_info=False):
    
    """Runs the MPC control loop."""

    ## MPC Parameters
    N_mpc = ConstFishSiply.N
    DT_mpc = ConstFishSiply.DT
    n_int_mpc = ConstFishSiply.n_int

    # n_tracking_mpc = 0 if tracking_Z_bool else 1 

    ## Initialize state
    if initial_state is None:
        current_state = DM([ConstFishSiply.toll_init] * (ConstFishSiply.n * 2))
    else:
        current_state = DM(initial_state)

    state_history = []
    control_history = []
    time_history = [0.0]
    current_time = 0.0

    ## Simulation loop
    num_steps = int(simulation_time / DT_mpc)

    for step in range(num_steps):
        print(colored(f"\n[INFO]: MPC Step {step + 1}/{num_steps}, Time: {current_time:.2f}", 'cyan'))  

        ## Solve NLP for MPC
        optimal_control, _, stats = solve_mpc_nlp(
            current_state=current_state,
            pos_d=pos_d,
            vel_des=vel_des,
            N_mpc=N_mpc,
            DT=DT_mpc,
            n_int=n_int_mpc)
        
        if wanna_info:
            print(f"[INFO]: Optimal control: {optimal_control},\n{dir(optimal_control)},\n{type(optimal_control)}")

        ## Apply the first optimal control action
        if isinstance(optimal_control, (DM, MX, np.float64, float)):
            applied_control = float(optimal_control)
        else:
            try:
                applied_control = float(optimal_control[0])  
            except TypeError as e:
                raise ValueError(colored(f"Unsupported optimal_control type: {type(optimal_control)}", 'red')) from e

        control_history.append(applied_control)

        ## Simulate system one step forward using the dynamics and applied control
        if isinstance(current_state, DM):
            # next_state_casadi_integrated = current_state  
            next_state_casadi_integrated = DM(Function("eval", [], [current_state])()["o0"])
        else:
            next_state_casadi_integrated = current_state
    
        h_mpc = DT_mpc / n_int_mpc
        for _ in range(n_int_mpc):
            k1 = compute_dyn(next_state_casadi_integrated, applied_control)
            next_state_casadi_integrated = next_state_casadi_integrated + h_mpc * k1

        current_state = next_state_casadi_integrated

        if wanna_info:
            print(f"[INFO]: Current state  : {current_state}, type: {type(current_state)}")
            print(f"[INFO]: State hystory  : {state_history}, type: {type(state_history)}")
            
        evaluate_mx = Function("evaluate_mx", [], [current_state])
        
        result = evaluate_mx()
        if wanna_info:
            print("Type of result   : ", type(result))
            print("Result contents  : ", result)

        if isinstance(result, dict):
            current_state_evaluated = result['o0'].full()
        else:
            current_state_evaluated = result.full()
        state_history.append(current_state_evaluated.T)  # Transpose if necessary

        current_time += DT_mpc
        time_history.append(current_time)

        if stats['return_status'] != 'Solve_Succeeded':
            print(colored("[INFO]: Solver failed, stopping MPC loop.", 'red'))
            break

    print(colored("\n[INFO]: MPC Control Loop Finished.", 'green'))

    if wanna_plot:
        state_history_np = np.array(state_history)
        q_opt = state_history_np[:, :ConstFishSiply.n]

        p_tip_opt_X, p_tip_opt_Z, p_tip_opt_app = [], [], []
        for i in range(q_opt.shape[0]):
            q_current = q_opt[i, :]

            p_tip_current = _p_tip(q_current, ConstFishSiply.L)
            p_tip_opt_app.append(p_tip_current)
            p_tip_opt_X.append(float(p_tip_current[1].full()))
            p_tip_opt_Z.append(float(p_tip_current[0].full()))

        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        tgrid = np.array(time_history[:-1])

        plt.figure(1)
        plt.clf()
        for j in range(ConstFishSiply.n):
            plt.plot(tgrid, q_opt[:-1, j], '-', linewidth=ConstFishSiply.linewidth, label=f'$q_{j+1}$')
        plt.ylabel(r'$ q\,\, [rad]$', fontsize=ConstFishSiply.fontsize)
        plt.xlabel(r'$t\,\, [sec]$', fontsize=ConstFishSiply.fontsize)
        plt.grid()
        plt.legend(fontsize=ConstFishSiply.fontsize, ncol=3)
        plt.savefig(os.path.join(desktop_path, 'mpc_q.png'))

        plt.figure(2)
        plt.clf()
        plt.plot(tgrid, control_history[:-1], '-', linewidth=ConstFishSiply.linewidth)
        plt.ylabel('u [Nm]', fontsize=ConstFishSiply.fontsize)
        plt.xlabel(r'$t\,\, [sec]$', fontsize=ConstFishSiply.fontsize)
        plt.grid()
        plt.savefig(os.path.join(desktop_path, 'mpc_u.png') )

        plt.figure(3)
        plt.clf()
        plt.plot(tgrid, p_tip_opt_X[:-1], '-', linewidth=ConstFishSiply.linewidth)
        plt.plot(tgrid, p_tip_opt_Z[:-1], '-', linewidth=ConstFishSiply.linewidth)
        plt.plot(tgrid, pos_d[0] * np.ones(len(p_tip_opt_X[:-1])), '--', color='r', linewidth=ConstFishSiply.linewidth)
        plt.plot(tgrid, pos_d[1] * np.ones(len(p_tip_opt_X[:-1])), '--', color='r', linewidth=ConstFishSiply.linewidth)
        plt.ylabel(r'$ pos\,\, tip\,\, [m]$', fontsize=ConstFishSiply.fontsize)
        plt.xlabel(r'$t\,\, [sec]$', fontsize=ConstFishSiply.fontsize)
        plt.grid()
        plt.legend([r'$X$', r'$Z$', r'$Z_{des}$', r'$X_{des}$'], fontsize=ConstFishSiply.fontsize)
        plt.savefig(os.path.join(desktop_path, 'mpc_pos_tip.png') )

    return control_history, state_history

if __name__ == "__main__":
    
    pos_des = [2.58, 0.15]
    vel_des = -10.0
    simulation_time = 1.0 # Reduced simulation time for quicker testing
    control_history, state_history = run_mpc_control(pos_d=pos_des, vel_des=vel_des, simulation_time=simulation_time, wanna_plot=True)

    print('\n')
    control_history_np = np.array(control_history)
    print(f"Control history len   : {len(control_history_np)}")
    print(f"Control history shape : {np.shape(control_history_np)}")
    print(f"Control history type  : {type(control_history_np)}")