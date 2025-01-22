import numpy as np
def _damping_torque(_Dummy_52, _Dummy_53):
    [q_dot_0, q_dot_1, q_dot_2, q_dot_3, q_dot_4, q_dot_5, q_dot_6, q_dot_7, q_dot_8, q_dot_9, q_dot_10, q_dot_11] = _Dummy_52
    [d_0, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, d_9, d_10, d_11] = _Dummy_53
    return np.array([[d_0], [d_1], [d_2], [d_3], [d_4], [d_5], [d_6], [d_7], [d_8], [d_9], [d_10], [d_11]])
