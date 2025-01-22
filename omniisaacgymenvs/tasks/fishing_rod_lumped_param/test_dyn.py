import numpy as np

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

if __name__ == '__main__':
    n = 12
    toll = 1e-2
    q = toll * np.ones(n) # this should be the inital position from my test
    q_dot = np.zeros(n)
    
    k = 1e2 * np.array([5.2228, 6.8533, 8.179, 6.4178,
                      4.9576, 3.729, 2.7075, 1.9117, 1.6321,
                      1.0708, 0.6685, 0.3919])

    d = 1e-1 * np.array([5.2228, 6.8533, 8.179, 6.4178,
                      4.9576, 3.729, 2.7075, 1.9117, 1.6321,
                      1.0708, 0.6685, 0.3919])
        
    g = 9.81
    L = np.array([1.0, 0.5, 0.3, 0.2, 0.15, 0.15, 0.1, 0.1, 0.1, 0.05, 0.03, 0.02])
    m = np.array([0.08, 0.06, 0.04, 0.03, 0.015, 0.01, 0.007, 0.007, 0.007, 0.003, 0.002, 0.001]) 
    I_xx = m * L**2
    
    G = _gravity_matrix(q, L, m, I_xx, g)
    K = _stiffness_torque(q, k)
    D = _damping_torque(q_dot, d)
    M = _inertia_matrix(q, L, I_xx, m)
    C = _coriolis_matrix(q, q_dot, L, I_xx, m)
    A = np.zeros(n)
    A[0] = 1
    pos = _p_tip(q, L)
    J = _jacobian_diff(q, L)
    print('=============================================================================================================')
    # print('Tendon Matrix Lin:\n',Alin, np.shape(Alin))
    print('Tendon Matrix:\n', A, np.shape(A))
    print('Inertial:\n', M, np.shape(M))
    print('Coriolis:\n', C, np.shape(C))
    print('Damping:\n', D, np.shape(D))
    print('Stifnness:\n', K, np.shape(K))
    print('Gravity:\n', G, np.shape(G))
    print('Jacobian Pos:\n', J, np.shape(J))
    print('=============================================================================================================')
    print('=============================================================================================================')
    print('Forward Kine:\n', pos, np.shape(pos))