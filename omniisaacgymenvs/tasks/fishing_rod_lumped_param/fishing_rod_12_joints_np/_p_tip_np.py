import numpy as np
def _p_tip(_Dummy_38, _Dummy_39):
    [q_0, q_1, q_2, q_3, q_4, q_5, q_6, q_7, q_8, q_9, q_10, q_11] = _Dummy_38
    [L_0, L_1, L_2, L_3, L_4, L_5, L_6, L_7, L_8, L_9, L_10, L_11] = _Dummy_39
    return np.array([[L_0*np.cos(q_0) 
                      + L_1*np.cos(q_0 + q_1) 
                      + L_10*np.cos(q_0 + q_1 + q_10 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9) 
                      + L_11*np.cos(q_0 + q_1 + q_10 + q_11 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9)
                      + L_2*np.cos(q_0 + q_1 + q_2) + L_3*np.cos(q_0 + q_1 + q_2 + q_3) 
                      + L_4*np.cos(q_0 + q_1 + q_2 + q_3 + q_4) 
                      + L_5*np.cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5) 
                      + L_6*np.cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6) 
                      + L_7*np.cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7) 
                      + L_8*np.cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8) 
                      + L_9*np.cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9)], 
                    [L_0*np.sin(q_0) 
                     + L_1*np.sin(q_0 + q_1) 
                     + L_10*np.sin(q_0 + q_1 + q_10 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9) 
                     + L_11*np.sin(q_0 + q_1 + q_10 + q_11 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9) 
                     + L_2*np.sin(q_0 + q_1 + q_2) 
                     + L_3*np.sin(q_0 + q_1 + q_2 + q_3) 
                     + L_4*np.sin(q_0 + q_1 + q_2 + q_3 + q_4) 
                     + L_5*np.sin(q_0 + q_1 + q_2 + q_3 + q_4 + q_5) 
                     + L_6*np.sin(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6) 
                     + L_7*np.sin(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7) 
                     + L_8*np.sin(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8) 
                     + L_9*np.sin(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9)],
                    [0]])
