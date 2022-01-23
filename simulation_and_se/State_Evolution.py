import numpy as np
# import matplotlib.pyplot as plt


def pa_average(L, P):
    pa = np.repeat(P/L, L)
    return pa


def g_in_se(L, B, v_pri, Pl, N_test):
    N = L * B
    rt_n_Pl = np.sqrt(N * Pl).reshape((1, -1))
    rt_n_Pl = np.repeat(rt_n_Pl, B * N_test, axis=0).reshape((B, N_test, L))
    x1_pri = np.random.normal(0, 1, (B, N_test, L))
    x2_pri = np.zeros(B)
    x2_pri[0] = 1
    x2_pri = np.repeat(x2_pri, L * N_test).reshape((B, N_test, L))
    x_pri = rt_n_Pl / np.sqrt(v_pri) * (x1_pri + rt_n_Pl / np.sqrt(v_pri) * x2_pri)
    x_pri_max = x_pri.max(axis=0).repeat(B, axis=1).reshape((N_test, L, B)).transpose((2, 0, 1))
    exps = np.exp(x_pri - x_pri_max)
    top = exps[0]
    bottom = np.sum(exps, axis=0)
    v_post = 1 - np.sum(np.array(Pl) * np.mean(top / bottom, axis=0))
    return float(v_post)


def se_row(Pl, Ite_Max, N_test, L, B, M, var_noise, v_A_pri_se):
    mse_row_se = np.zeros(Ite_Max)
    N = L * B

    for it in range(Ite_Max):
        v_B_pri_se = (N - M) / M * v_A_pri_se + N / M * var_noise
        v_B_post_se = g_in_se(L, B, v_B_pri_se, Pl, N_test)
        v_A_pri_se = 1 / (1 / v_B_post_se - 1 / v_B_pri_se)
        mse_row_se[it] = max(1e-6, v_B_post_se)

    return mse_row_se


def se_gauss(Pl, Ite_Max, N_test, L, B, M, var_noise, v_x_se):
    mse_gauss_se = np.zeros(Ite_Max)
    N = L * B

    for it in range(Ite_Max):
        v_p_se = v_x_se
        v_z_se = (v_p_se * var_noise) / (v_p_se + var_noise)
        v_s_se = (v_p_se - v_z_se) / v_p_se ** 2
        v_r_se = 1 / ((M / N) * v_s_se)
        v_x_se = g_in_se(L, B, v_r_se, Pl, N_test)
        mse_gauss_se[it] = max(1e-6, v_x_se)

    return mse_gauss_se
