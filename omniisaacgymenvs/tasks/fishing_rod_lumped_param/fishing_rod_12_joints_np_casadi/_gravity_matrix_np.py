from casadi import *
def _gravity_matrix(_Dummy_48, _Dummy_49, _Dummy_50, _Dummy_51, g0):
    
    q_0, q_1, q_2, q_3, q_4, q_5, q_6, q_7, q_8, q_9, q_10, q_11 = _Dummy_48[0], _Dummy_48[1], _Dummy_48[2],_Dummy_48[3], _Dummy_48[4], _Dummy_48[5], _Dummy_48[6], _Dummy_48[7], _Dummy_48[8], _Dummy_48[9], _Dummy_48[10], _Dummy_48[11]
    L_0, L_1, L_2, L_3, L_4, L_5, L_6, L_7, L_8, L_9, L_10, L_11 = _Dummy_49[0], _Dummy_49[1], _Dummy_49[2],_Dummy_49[3], _Dummy_49[4], _Dummy_49[5], _Dummy_49[6], _Dummy_49[7], _Dummy_49[8], _Dummy_49[9], _Dummy_49[10], _Dummy_49[11]
    m_0, m_1, m_2, m_3, m_4, m_5, m_6, m_7, m_8, m_9, m_10, m_11 = _Dummy_50[0], _Dummy_50[1], _Dummy_50[2],_Dummy_50[3], _Dummy_50[4], _Dummy_50[5], _Dummy_50[6], _Dummy_50[7], _Dummy_50[8], _Dummy_50[9], _Dummy_50[10], _Dummy_50[11]
    I_zz_0, I_zz_1, I_zz_2, I_zz_3, I_zz_4, I_zz_5, I_zz_6, I_zz_7, I_zz_8, I_zz_9, I_zz_10, I_zz_11 = _Dummy_51[0], _Dummy_51[1], _Dummy_51[2],_Dummy_51[3], _Dummy_51[4], _Dummy_51[5], _Dummy_51[6], _Dummy_51[7], _Dummy_51[8], _Dummy_51[9], _Dummy_51[10], _Dummy_51[11]
    
    return [[-1/2*L_0*g0*m_0*cos(q_0) - 1/2*g0*m_1*(L_0*cos(q_0) + L_1*cos(q_0 + q_1)) - 1/2*g0*m_10*(L_0*cos(q_0) + L_1*cos(q_0 + q_1) + L_10*cos(q_0 + q_1 + q_10 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9) + L_2*cos(q_0 + q_1 + q_2) + L_3*cos(q_0 + q_1 + q_2 + q_3) + L_4*cos(q_0 + q_1 + q_2 + q_3 + q_4) + L_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5) + L_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6) + L_7*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7) + L_8*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8) + L_9*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9)) - 1/2*g0*m_11*(L_0*cos(q_0) + L_1*cos(q_0 + q_1) + L_10*cos(q_0 + q_1 + q_10 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9) + L_11*cos(q_0 + q_1 + q_10 + q_11 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9) + L_2*cos(q_0 + q_1 + q_2) + L_3*cos(q_0 + q_1 + q_2 + q_3) + L_4*cos(q_0 + q_1 + q_2 + q_3 + q_4) + L_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5) + L_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6) + L_7*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7) + L_8*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8) + L_9*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9)) - 1/2*g0*m_2*(L_0*cos(q_0) + L_1*cos(q_0 + q_1) + L_2*cos(q_0 + q_1 + q_2)) - 1/2*g0*m_3*(L_0*cos(q_0) + L_1*cos(q_0 + q_1) + L_2*cos(q_0 + q_1 + q_2) + L_3*cos(q_0 + q_1 + q_2 + q_3)) - 1/2*g0*m_4*(L_0*cos(q_0) + L_1*cos(q_0 + q_1) + L_2*cos(q_0 + q_1 + q_2) + L_3*cos(q_0 + q_1 + q_2 + q_3) + L_4*cos(q_0 + q_1 + q_2 + q_3 + q_4)) - 1/2*g0*m_5*(L_0*cos(q_0) + L_1*cos(q_0 + q_1) + L_2*cos(q_0 + q_1 + q_2) + L_3*cos(q_0 + q_1 + q_2 + q_3) + L_4*cos(q_0 + q_1 + q_2 + q_3 + q_4) + L_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5)) - 1/2*g0*m_6*(L_0*cos(q_0) + L_1*cos(q_0 + q_1) + L_2*cos(q_0 + q_1 + q_2) + L_3*cos(q_0 + q_1 + q_2 + q_3) + L_4*cos(q_0 + q_1 + q_2 + q_3 + q_4) + L_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5) + L_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6)) - 1/2*g0*m_7*(L_0*cos(q_0) + L_1*cos(q_0 + q_1) + L_2*cos(q_0 + q_1 + q_2) + L_3*cos(q_0 + q_1 + q_2 + q_3) + L_4*cos(q_0 + q_1 + q_2 + q_3 + q_4) + L_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5) + L_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6) + L_7*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7)) - 1/2*g0*m_8*(L_0*cos(q_0) + L_1*cos(q_0 + q_1) + L_2*cos(q_0 + q_1 + q_2) + L_3*cos(q_0 + q_1 + q_2 + q_3) + L_4*cos(q_0 + q_1 + q_2 + q_3 + q_4) + L_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5) + L_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6) + L_7*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7) + L_8*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8)) - 1/2*g0*m_9*(L_0*cos(q_0) + L_1*cos(q_0 + q_1) + L_2*cos(q_0 + q_1 + q_2) + L_3*cos(q_0 + q_1 + q_2 + q_3) + L_4*cos(q_0 + q_1 + q_2 + q_3 + q_4) + L_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5) + L_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6) + L_7*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7) + L_8*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8) + L_9*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9))], [-1/2*L_1*g0*m_1*cos(q_0 + q_1) - 1/2*g0*m_10*(L_1*cos(q_0 + q_1) + L_10*cos(q_0 + q_1 + q_10 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9) + L_2*cos(q_0 + q_1 + q_2) + L_3*cos(q_0 + q_1 + q_2 + q_3) + L_4*cos(q_0 + q_1 + q_2 + q_3 + q_4) + L_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5) + L_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6) + L_7*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7) + L_8*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8) + L_9*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9)) - 1/2*g0*m_11*(L_1*cos(q_0 + q_1) + L_10*cos(q_0 + q_1 + q_10 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9) + L_11*cos(q_0 + q_1 + q_10 + q_11 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9) + L_2*cos(q_0 + q_1 + q_2) + L_3*cos(q_0 + q_1 + q_2 + q_3) + L_4*cos(q_0 + q_1 + q_2 + q_3 + q_4) + L_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5) + L_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6) + L_7*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7) + L_8*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8) + L_9*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9)) - 1/2*g0*m_2*(L_1*cos(q_0 + q_1) + L_2*cos(q_0 + q_1 + q_2)) - 1/2*g0*m_3*(L_1*cos(q_0 + q_1) + L_2*cos(q_0 + q_1 + q_2) + L_3*cos(q_0 + q_1 + q_2 + q_3)) - 1/2*g0*m_4*(L_1*cos(q_0 + q_1) + L_2*cos(q_0 + q_1 + q_2) + L_3*cos(q_0 + q_1 + q_2 + q_3) + L_4*cos(q_0 + q_1 + q_2 + q_3 + q_4)) - 1/2*g0*m_5*(L_1*cos(q_0 + q_1) + L_2*cos(q_0 + q_1 + q_2) + L_3*cos(q_0 + q_1 + q_2 + q_3) + L_4*cos(q_0 + q_1 + q_2 + q_3 + q_4) + L_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5)) - 1/2*g0*m_6*(L_1*cos(q_0 + q_1) + L_2*cos(q_0 + q_1 + q_2) + L_3*cos(q_0 + q_1 + q_2 + q_3) + L_4*cos(q_0 + q_1 + q_2 + q_3 + q_4) + L_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5) + L_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6)) - 1/2*g0*m_7*(L_1*cos(q_0 + q_1) + L_2*cos(q_0 + q_1 + q_2) + L_3*cos(q_0 + q_1 + q_2 + q_3) + L_4*cos(q_0 + q_1 + q_2 + q_3 + q_4) + L_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5) + L_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6) + L_7*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7)) - 1/2*g0*m_8*(L_1*cos(q_0 + q_1) + L_2*cos(q_0 + q_1 + q_2) + L_3*cos(q_0 + q_1 + q_2 + q_3) + L_4*cos(q_0 + q_1 + q_2 + q_3 + q_4) + L_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5) + L_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6) + L_7*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7) + L_8*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8)) - 1/2*g0*m_9*(L_1*cos(q_0 + q_1) + L_2*cos(q_0 + q_1 + q_2) + L_3*cos(q_0 + q_1 + q_2 + q_3) + L_4*cos(q_0 + q_1 + q_2 + q_3 + q_4) + L_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5) + L_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6) + L_7*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7) + L_8*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8) + L_9*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9))], [-1/2*L_2*g0*m_2*cos(q_0 + q_1 + q_2) - 1/2*g0*m_10*(L_10*cos(q_0 + q_1 + q_10 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9) + L_2*cos(q_0 + q_1 + q_2) + L_3*cos(q_0 + q_1 + q_2 + q_3) + L_4*cos(q_0 + q_1 + q_2 + q_3 + q_4) + L_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5) + L_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6) + L_7*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7) + L_8*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8) + L_9*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9)) - 1/2*g0*m_11*(L_10*cos(q_0 + q_1 + q_10 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9) + L_11*cos(q_0 + q_1 + q_10 + q_11 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9) + L_2*cos(q_0 + q_1 + q_2) + L_3*cos(q_0 + q_1 + q_2 + q_3) + L_4*cos(q_0 + q_1 + q_2 + q_3 + q_4) + L_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5) + L_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6) + L_7*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7) + L_8*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8) + L_9*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9)) - 1/2*g0*m_3*(L_2*cos(q_0 + q_1 + q_2) + L_3*cos(q_0 + q_1 + q_2 + q_3)) - 1/2*g0*m_4*(L_2*cos(q_0 + q_1 + q_2) + L_3*cos(q_0 + q_1 + q_2 + q_3) + L_4*cos(q_0 + q_1 + q_2 + q_3 + q_4)) - 1/2*g0*m_5*(L_2*cos(q_0 + q_1 + q_2) + L_3*cos(q_0 + q_1 + q_2 + q_3) + L_4*cos(q_0 + q_1 + q_2 + q_3 + q_4) + L_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5)) - 1/2*g0*m_6*(L_2*cos(q_0 + q_1 + q_2) + L_3*cos(q_0 + q_1 + q_2 + q_3) + L_4*cos(q_0 + q_1 + q_2 + q_3 + q_4) + L_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5) + L_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6)) - 1/2*g0*m_7*(L_2*cos(q_0 + q_1 + q_2) + L_3*cos(q_0 + q_1 + q_2 + q_3) + L_4*cos(q_0 + q_1 + q_2 + q_3 + q_4) + L_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5) + L_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6) + L_7*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7)) - 1/2*g0*m_8*(L_2*cos(q_0 + q_1 + q_2) + L_3*cos(q_0 + q_1 + q_2 + q_3) + L_4*cos(q_0 + q_1 + q_2 + q_3 + q_4) + L_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5) + L_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6) + L_7*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7) + L_8*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8)) - 1/2*g0*m_9*(L_2*cos(q_0 + q_1 + q_2) + L_3*cos(q_0 + q_1 + q_2 + q_3) + L_4*cos(q_0 + q_1 + q_2 + q_3 + q_4) + L_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5) + L_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6) + L_7*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7) + L_8*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8) + L_9*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9))], [-1/2*L_3*g0*m_3*cos(q_0 + q_1 + q_2 + q_3) - 1/2*g0*m_10*(L_10*cos(q_0 + q_1 + q_10 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9) + L_3*cos(q_0 + q_1 + q_2 + q_3) + L_4*cos(q_0 + q_1 + q_2 + q_3 + q_4) + L_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5) + L_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6) + L_7*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7) + L_8*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8) + L_9*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9)) - 1/2*g0*m_11*(L_10*cos(q_0 + q_1 + q_10 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9) + L_11*cos(q_0 + q_1 + q_10 + q_11 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9) + L_3*cos(q_0 + q_1 + q_2 + q_3) + L_4*cos(q_0 + q_1 + q_2 + q_3 + q_4) + L_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5) + L_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6) + L_7*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7) + L_8*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8) + L_9*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9)) - 1/2*g0*m_4*(L_3*cos(q_0 + q_1 + q_2 + q_3) + L_4*cos(q_0 + q_1 + q_2 + q_3 + q_4)) - 1/2*g0*m_5*(L_3*cos(q_0 + q_1 + q_2 + q_3) + L_4*cos(q_0 + q_1 + q_2 + q_3 + q_4) + L_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5)) - 1/2*g0*m_6*(L_3*cos(q_0 + q_1 + q_2 + q_3) + L_4*cos(q_0 + q_1 + q_2 + q_3 + q_4) + L_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5) + L_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6)) - 1/2*g0*m_7*(L_3*cos(q_0 + q_1 + q_2 + q_3) + L_4*cos(q_0 + q_1 + q_2 + q_3 + q_4) + L_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5) + L_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6) + L_7*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7)) - 1/2*g0*m_8*(L_3*cos(q_0 + q_1 + q_2 + q_3) + L_4*cos(q_0 + q_1 + q_2 + q_3 + q_4) + L_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5) + L_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6) + L_7*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7) + L_8*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8)) - 1/2*g0*m_9*(L_3*cos(q_0 + q_1 + q_2 + q_3) + L_4*cos(q_0 + q_1 + q_2 + q_3 + q_4) + L_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5) + L_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6) + L_7*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7) + L_8*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8) + L_9*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9))], [-1/2*L_4*g0*m_4*cos(q_0 + q_1 + q_2 + q_3 + q_4) - 1/2*g0*m_10*(L_10*cos(q_0 + q_1 + q_10 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9) + L_4*cos(q_0 + q_1 + q_2 + q_3 + q_4) + L_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5) + L_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6) + L_7*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7) + L_8*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8) + L_9*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9)) - 1/2*g0*m_11*(L_10*cos(q_0 + q_1 + q_10 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9) + L_11*cos(q_0 + q_1 + q_10 + q_11 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9) + L_4*cos(q_0 + q_1 + q_2 + q_3 + q_4) + L_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5) + L_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6) + L_7*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7) + L_8*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8) + L_9*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9)) - 1/2*g0*m_5*(L_4*cos(q_0 + q_1 + q_2 + q_3 + q_4) + L_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5)) - 1/2*g0*m_6*(L_4*cos(q_0 + q_1 + q_2 + q_3 + q_4) + L_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5) + L_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6)) - 1/2*g0*m_7*(L_4*cos(q_0 + q_1 + q_2 + q_3 + q_4) + L_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5) + L_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6) + L_7*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7)) - 1/2*g0*m_8*(L_4*cos(q_0 + q_1 + q_2 + q_3 + q_4) + L_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5) + L_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6) + L_7*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7) + L_8*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8)) - 1/2*g0*m_9*(L_4*cos(q_0 + q_1 + q_2 + q_3 + q_4) + L_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5) + L_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6) + L_7*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7) + L_8*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8) + L_9*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9))], [-1/2*L_5*g0*m_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5) - 1/2*g0*m_10*(L_10*cos(q_0 + q_1 + q_10 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9) + L_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5) + L_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6) + L_7*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7) + L_8*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8) + L_9*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9)) - 1/2*g0*m_11*(L_10*cos(q_0 + q_1 + q_10 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9) + L_11*cos(q_0 + q_1 + q_10 + q_11 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9) + L_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5) + L_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6) + L_7*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7) + L_8*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8) + L_9*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9)) - 1/2*g0*m_6*(L_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5) + L_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6)) - 1/2*g0*m_7*(L_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5) + L_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6) + L_7*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7)) - 1/2*g0*m_8*(L_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5) + L_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6) + L_7*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7) + L_8*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8)) - 1/2*g0*m_9*(L_5*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5) + L_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6) + L_7*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7) + L_8*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8) + L_9*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9))], [-1/2*L_6*g0*m_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6) - 1/2*g0*m_10*(L_10*cos(q_0 + q_1 + q_10 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9) + L_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6) + L_7*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7) + L_8*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8) + L_9*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9)) - 1/2*g0*m_11*(L_10*cos(q_0 + q_1 + q_10 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9) + L_11*cos(q_0 + q_1 + q_10 + q_11 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9) + L_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6) + L_7*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7) + L_8*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8) + L_9*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9)) - 1/2*g0*m_7*(L_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6) + L_7*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7)) - 1/2*g0*m_8*(L_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6) + L_7*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7) + L_8*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8)) - 1/2*g0*m_9*(L_6*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6) + L_7*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7) + L_8*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8) + L_9*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9))], [-1/2*L_7*g0*m_7*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7) - 1/2*g0*m_10*(L_10*cos(q_0 + q_1 + q_10 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9) + L_7*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7) + L_8*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8) + L_9*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9)) - 1/2*g0*m_11*(L_10*cos(q_0 + q_1 + q_10 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9) + L_11*cos(q_0 + q_1 + q_10 + q_11 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9) + L_7*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7) + L_8*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8) + L_9*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9)) - 1/2*g0*m_8*(L_7*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7) + L_8*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8)) - 1/2*g0*m_9*(L_7*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7) + L_8*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8) + L_9*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9))], [-1/2*L_8*g0*m_8*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8) - 1/2*g0*m_10*(L_10*cos(q_0 + q_1 + q_10 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9) + L_8*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8) + L_9*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9)) - 1/2*g0*m_11*(L_10*cos(q_0 + q_1 + q_10 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9) + L_11*cos(q_0 + q_1 + q_10 + q_11 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9) + L_8*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8) + L_9*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9)) - 1/2*g0*m_9*(L_8*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8) + L_9*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9))], [-1/2*L_9*g0*m_9*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9) - 1/2*g0*m_10*(L_10*cos(q_0 + q_1 + q_10 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9) + L_9*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9)) - 1/2*g0*m_11*(L_10*cos(q_0 + q_1 + q_10 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9) + L_11*cos(q_0 + q_1 + q_10 + q_11 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9) + L_9*cos(q_0 + q_1 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9))], [-1/2*L_10*g0*m_10*cos(q_0 + q_1 + q_10 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9) - 1/2*g0*m_11*(L_10*cos(q_0 + q_1 + q_10 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9) + L_11*cos(q_0 + q_1 + q_10 + q_11 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9))], [-1/2*L_11*g0*m_11*cos(q_0 + q_1 + q_10 + q_11 + q_2 + q_3 + q_4 + q_5 + q_6 + q_7 + q_8 + q_9)]]