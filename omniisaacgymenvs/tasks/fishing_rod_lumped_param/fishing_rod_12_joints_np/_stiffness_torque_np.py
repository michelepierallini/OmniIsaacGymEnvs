import numpy as np
def _stiffness_torque(_Dummy_54, _Dummy_55):
    [q_0, q_1, q_2, q_3, q_4, q_5, q_6, q_7, q_8, q_9, q_10, q_11] = _Dummy_54
    [k_0, k_1, k_2, k_3, k_4, k_5, k_6, k_7, k_8, k_9, k_10, k_11] = _Dummy_55
    return np.array([[k_0], [k_1], [k_2], [k_3], [k_4], [k_5], [k_6], [k_7], [k_8], [k_9], [k_10], [k_11]])
