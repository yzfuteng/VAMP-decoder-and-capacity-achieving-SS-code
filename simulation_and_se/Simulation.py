import numpy as np
from scipy import fftpack
from scipy import io


def sparc_transforms_dct(L, B, M):
    N = L * B
    index_dct = np.random.choice(N, size=(M, ), replace=False)
    phase = np.random.choice([-1, 1], size=(N, ))

    def Ab(b):
        b = b.reshape(-1)
        z = fftpack.dct(b, norm='ortho')
        z = phase * z
        z = fftpack.idct(z, norm='ortho')
        return z[index_dct].reshape(-1, 1)

    def Az(z):
        z = z.reshape(-1)
        z_f = np.zeros((N, ))
        z_f[index_dct] = z
        b = fftpack.dct(z_f, n=N, norm='ortho')
        b = np.conj(phase) * b
        b = fftpack.idct(b, norm='ortho')
        return b.reshape(-1, 1)

    return Ab, Az


def g_in(L, B, beta_PRI, Var_PRI):
    N = L * B
    rt_n_Pl = np.sqrt(B).repeat(N).reshape(-1, 1)
    u = beta_PRI * rt_n_Pl / Var_PRI
    max_u = u.reshape(L, B).max(axis=1).repeat(B).reshape(-1, 1)
    exps = np.exp(u - max_u)
    sums = exps.reshape(L, B).sum(axis=1).repeat(B).reshape(-1, 1)
    beta_POST = (rt_n_Pl * exps / sums).reshape(-1, 1)
    Var_POST = (rt_n_Pl ** 2 * (exps / sums) * (1 - exps / sums)).reshape(-1, 1)
    Var_POST_res = float(np.mean(Var_POST))
    return beta_POST, Var_POST_res


def experiments(L, B, R):
    # Initialization
    # L, B = 2 ** 12, 16
    # SNRdB_channel = 10.0
    # SNR_channel = np.power(10, SNRdB_channel / 10)
    SNR_channel = 15.0
    P = 1.0
    var_noise = P / SNR_channel
    # C = 0.5 * np.log2(1 + SNR_Channel)
    # Pl = pa_average(L, P)
    # R = 1.4
    N = L * B
    M = int(L * np.log2(B) / R)

    NSIM = 500      # The number of trials
    Ite_Max = 200   # The number of iterations per trial

    SER_VAMP = np.zeros(Ite_Max)
    MSE_VAMP_aid = np.zeros(Ite_Max)
    MSE_VAMP_cal = np.zeros(Ite_Max)
    MSE_VAMP_se = np.zeros(Ite_Max)

    # ------------------------------Simulation------------------------------

    for nsim in range(NSIM):
        # Generate random message in [0..B)^L
        tx_message = np.random.randint(0, B, L)
        tx_message.tolist()

        # Generate our transmitted signal X
        x_0 = np.zeros((N, 1))
        for l in range(L):
            x_0[l * B + tx_message[l], 0] = np.sqrt(B)  # (1/N) * |x_0|^2 = 1

        # Generate the SPARC transform functions A.beta and A'.z
        [Ab, Az] = sparc_transforms_dct(L, B, M)
        # [Ab, Az] = sparc_transforms_row(L, B, M)

        # Generate random channel noise and then received signal y
        noise = np.sqrt(var_noise) * np.random.randn(M, 1)
        y = Ab(x_0) + noise
        # y = A @ x_0 + noise

        # Initialization
        v_A_pri = 1
        x_A_pri = np.zeros((N, 1))

        for it in range(Ite_Max):
            print("B = %d, R = %.2f: nsim = %d of %d, it = %d of %d" %
                  (B, R, nsim + 1, NSIM, it + 1, Ite_Max))
            # log_file.write("B = %d, R = %.2f: nsim = %d of %d, it = %d of %d\n" % 
            #                (B, R, nsim + 1, NSIM, it + 1, Ite_Max))

            x_A_post = x_A_pri + v_A_pri / (v_A_pri + var_noise) * Az(y - Ab(x_A_pri))
            alpha_post = 1 - (M / N) * (v_A_pri / (v_A_pri + var_noise))
            v_A_post = v_A_pri * alpha_post
            # v_A_post_GA = np.sum((x_A_post - x_0) ** 2) / N

            v_B_pri = 1 / (1 / v_A_post - 1 / v_A_pri)
            x_B_pri = v_B_pri * (x_A_post / v_A_post - x_A_pri / v_A_pri)
            # v_B_pri_GA = np.sum((x_B_pri - x_0) ** 2) / N

            x_B_post, v_B_post = g_in(L, B, x_B_pri, v_B_pri)
            # v_B_post_GA = np.sum((x_B_post - x_0) ** 2) / N

            v_A_pri = 1 / (1 / v_B_post - 1 / v_B_pri)
            x_A_pri = v_A_pri * (x_B_post / v_B_post - x_B_pri / v_B_pri)
            # v_A_pri_GA = np.sum((x_A_pri - x_0) ** 2) / N

            rx_message = []
            for l in range(L):
                idx = np.argmax(x_B_post[l * B: (l + 1) * B])
                rx_message.append(idx)
            SER_VAMP[it] += 1 - np.sum(np.array(rx_message) == np.array(tx_message)) / L

            MSE_VAMP_aid[it] += max(1e-6, np.sum((x_B_post - x_0)**2) / N)
            MSE_VAMP_cal[it] += max(1e-6, v_B_post)

            pass

    SER_VAMP /= NSIM
    MSE_VAMP_aid /= NSIM
    MSE_VAMP_cal /= NSIM

    print("B = %d, R = %.2f: aid = %.3e, cal = %.3e, se = %.3e, ser = %.3e" %
          (B, R, MSE_VAMP_aid[-1], MSE_VAMP_cal[-1], MSE_VAMP_se[-1], SER_VAMP[-1]))

    res = {'L': L, 'B': B, 'M': M, 'N': N, 'var_noise': var_noise, 'NSIM': NSIM, 'Ite_Max': Ite_Max,
           'MSE_VAMP_aid': MSE_VAMP_aid, 'MSE_VAMP_cal': MSE_VAMP_cal, 'MSE_VAMP_se': MSE_VAMP_se,
           'SER_VAMP': SER_VAMP}

    io.savemat('./results/data_mat/L%d/B%d/VAMP_dct_B%d_R%.2f.mat' % (int(np.log2(L)), B, B, R), res)
