from casadi import *
def _p_tip(_Dummy_38, _Dummy_39):
    [q_0, q_1, q_2, q_3, q_4, q_5, q_6, q_7, q_8, q_9, q_10, q_11] = _Dummy_38[0], _Dummy_38[1], _Dummy_38[2],_Dummy_38[3], _Dummy_38[4], _Dummy_38[5], _Dummy_38[6], _Dummy_38[7], _Dummy_38[8], _Dummy_38[9], _Dummy_38[10], _Dummy_38[11]
    [L_0, L_1, L_2, L_3, L_4, L_5, L_6, L_7, L_8, L_9, L_10, L_11] = _Dummy_39[0], _Dummy_39[1], _Dummy_39[2],_Dummy_39[3], _Dummy_39[4], _Dummy_39[5], _Dummy_39[6], _Dummy_39[7], _Dummy_39[8], _Dummy_39[9], _Dummy_39[10], _Dummy_39[11]
    return [L_0*cos(q_0) 
            + L_1*cos(q_0 + q_1) 
            + L_10*cos(q_0 + q_1 + q_10 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9) 
            + L_11*cos(q_0 + q_1 + q_10 + q_11 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9)
            + L_2*cos(q_0 + q_1 + q_2) 
            + L_3*cos(q_0 + q_1 + q_2 + q_3) 
            + L_4*cos(q_0 + q_1 + q_2 + q_3 + q_4) 
            + L_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5) 
            + L_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6) 
            + L_7*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7) 
            + L_8*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8) 
            + L_9*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9), 
            L_0*sin(q_0) 
            + L_1*sin(q_0 + q_1) 
            + L_10*sin(q_0 + q_1 + q_10 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9) 
            + L_11*sin(q_0 + q_1 + q_10 + q_11 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9) 
            + L_2*sin(q_0 + q_1 + q_2) 
            + L_3*sin(q_0 + q_1 + q_2 + q_3) 
            + L_4*sin(q_0 + q_1 + q_2 + q_3 + q_4) 
            + L_5*sin(q_0 + q_1 + q_2 + q_3 + q_4 + q_5) 
            + L_6*sin(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6) 
            + L_7*sin(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7) 
            + L_8*sin(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8) 
            + L_9*sin(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9)]
