import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
# import functools
# import random
# import matplotlib
# matplotlib.use('Agg')


B = 4
N = 200000  # sample times
snr = 15  # 1/sigma^2


def R_gauss(x, alpha):    # R-transform of Gauss matrix
    return alpha/(1+x)


def R_row(x, alpha):     # R-transform of row-orthogonal matrix
    temp = math.sqrt((1+x)**2-4*alpha*x)
    return -(1-x+temp)/(2*x)+1/x


def R_counter(x, alpha):  # R-transform of Three-Dirac spectrum
    z = -x
    p = [4*z, -8*z-4, 3*z+8-4*alpha, 3*(alpha-1)]
    ans = min(np.roots(p))
    return ans-1/z


def I0_gauss_1(E, alpha):
    return -0.5*B*snr*(1-E)*R_gauss(snr*E, alpha)


def I0_row_1(E, alpha):
    return -0.5*B*snr*(1-E)*R_row(snr*E, alpha)


def I0_counter_1(E, alpha):
    return -0.5*B*snr*(1-E)*R_counter(snr*E, alpha)


def integral_R_gauss(E, alpha):
    integral = integrate.quad(R_gauss, 0, snr*E, args=(alpha,))
    return -0.5*B*integral[0]


def integral_R_row(E, alpha):
    integral = integrate.quad(R_row, 0, snr*E, args=(alpha,))
    return -0.5*B*integral[0]


def integral_R_counter(E, alpha):
    integral = integrate.quad(R_counter, 0, snr*E, args=(alpha,))
    return -0.5*B*integral[0]


def constant(E, alpha):
    return 1/2*alpha*B*math.log(snr)-math.log(B)


def I0_gauss_2_sample(E, alpha):
    qx_hat = B*snr*R_gauss(snr*E, alpha)
    result = 0
    for k in range(10):
        temp = 0
        sample = np.random.randn(N, B)
        for i in range(N):
            temp1 = math.exp(0.5*qx_hat+math.sqrt(qx_hat)*sample[i][0])
            temp2 = 0
            for j in range(1, B):
                temp2 = temp2+math.exp(-0.5*qx_hat+math.sqrt(qx_hat)*sample[i][j])
            temp = temp+math.log(temp1+temp2)
        result = result+temp/N
    return result/10   


def I0_row_2_sample(E, alpha):
    qx_hat = B*snr*R_row(snr*E, alpha)
    result = 0
    for k in range(10):
        temp = 0
        sample = np.random.randn(N, B)
        for i in range(N):
            temp1 = math.exp(0.5*qx_hat+math.sqrt(qx_hat)*sample[i][0])
            temp2 = 0
            for j in range(1, B):
                temp2 = temp2+math.exp(-0.5*qx_hat+math.sqrt(qx_hat)*sample[i][j])
            temp = temp+math.log(temp1+temp2)
        result = result+temp/N
    return result/10   


def I0_counter_2_sample(E, alpha):
    qx_hat = B*snr*R_counter(snr*E, alpha)
    result = 0
    for k in range(10):
        temp = 0
        sample = np.random.randn(N, B)
        for i in range(N):
            temp1 = math.exp(0.5*qx_hat+math.sqrt(qx_hat)*sample[i][0])
            temp2 = 0
            for j in range(1, B):
                temp2 = temp2+math.exp(-0.5*qx_hat+math.sqrt(qx_hat)*sample[i][j])
            temp = temp+math.log(temp1+temp2)
        result = result+temp/N
    return result/10   


def Phi_gauss_sample(E, alpha):
    return I0_gauss_1(E, alpha)+integral_R_gauss(E, alpha)+I0_gauss_2_sample(E, alpha)+constant(E, alpha)


def Phi_row_sample(E, alpha):
    return I0_row_1(E, alpha)+integral_R_row(E, alpha)+I0_row_2_sample(E, alpha)+constant(E, alpha)


def Phi_counter_sample(E, alpha):
    return I0_counter_1(E, alpha)+integral_R_counter(E, alpha)+I0_counter_2_sample(E, alpha)+constant(E, alpha)


for i in range(50):
    R_i = 1.35+0.01*(i+1)
    alpha_i = math.log(B)/(math.log(2)*R_i*B)
    MSE_sample = [0 for j in range(151)]
    Energy_gauss_sample = [0 for j in range(151)]
    Energy_row_sample = [0 for j in range(151)]
    Energy_counter_sample = [0 for j in range(151)]
    for j in range(151):
        MSE_sample[j] = 0.006*(j+1)
        Energy_gauss_sample[j] = Phi_gauss_sample(MSE_sample[j], alpha_i)
        Energy_row_sample[j] = Phi_row_sample(MSE_sample[j], alpha_i)
        Energy_counter_sample[j] = Phi_counter_sample(MSE_sample[j], alpha_i)
    plt.figure()
    plt.axes(xscale="log")
    plt.plot(MSE_sample, Energy_gauss_sample)    # Free Energy curve of Gaussian matrix
    plt.plot(MSE_sample, Energy_row_sample)      # Free Energy curve of row-Orthogonal matrix
    plt.plot(MSE_sample, Energy_counter_sample)  # Free Energy curve of Three-Dirac matrix
    plt.legend(['counter'])
    plt.xlabel('MSE')
    plt.ylabel('Free Energy')
    plt.title("B=%d,snr=%d,R=%.3f" % (B, snr, R_i))
    plt.savefig('B=%d,snr=%d,R=%.3f.jpg' % (B, snr, R_i))
    plt.close()
    print(i)
