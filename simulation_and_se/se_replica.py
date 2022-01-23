import math
import numpy as np

B = 4
N = 100000  # sample times
snr = 15.0  # 1/sigma^2

R = 1.48
alpha0 = 0.5 / R


def R_gauss(x, alpha):    # R-transform of Gauss matrix .In fact, it's R(-x)
    return alpha/(1+x)


def R_row(x, alpha):     # R-transform of row-orthogonal matrix .In fact, it's R(-x)
    temp = math.sqrt((1+x)**2-4*alpha*x)
    return -(1-x+temp)/(2*x)+1/x


def R_counter(x, alpha):  # R-transform of Three-Dirac spectrum
    z = -x
    p = [4*z, -8*z-4, 3*z+8-4*alpha, 3*(alpha-1)]
    ans = min(np.roots(p))
    return ans - 1/z


def I_gauss(E, alpha):
    qx_hat = B*snr*R_gauss(snr*E, alpha)
    result = 0
    for k in range(10):
        sample = np.random.randn(N, B)
        temp = 0
        for i in range(N):
            temp1 = math.exp(0.5*qx_hat+math.sqrt(qx_hat)*sample[i][0])
            
            temp2 = 0
            for j in range(1, B):
                temp2 = temp2+math.exp(-0.5*qx_hat+math.sqrt(qx_hat)*sample[i][j])
            temp3 = math.exp(0.5*qx_hat+math.sqrt(qx_hat)*sample[i][0])*(-1-sample[i][0]/math.sqrt(qx_hat))
            temp4 = (B-1)*math.exp(-0.5*qx_hat+math.sqrt(qx_hat)*sample[i][1])*(1-sample[i][1]/math.sqrt(qx_hat))
            temp = temp+(temp3+temp4)/(temp1+temp2)
        result = result+temp/N
    return result/10


def I_row(E, alpha):
    qx_hat = B*snr*R_row(snr*E, alpha)
    result = 0
    for k in range(10):
        sample = np.random.randn(N, B)
        temp = 0
        for i in range(N):
            temp1 = math.exp(0.5*qx_hat+math.sqrt(qx_hat)*sample[i][0])
            temp2 = 0
            for j in range(1, B):
                temp2 = temp2+math.exp(-0.5*qx_hat+math.sqrt(qx_hat)*sample[i][j])
            temp3 = math.exp(0.5*qx_hat+math.sqrt(qx_hat)*sample[i][0])*(-1-sample[i][0]/math.sqrt(qx_hat))
            temp4 = (B-1)*math.exp(-0.5*qx_hat+math.sqrt(qx_hat)*sample[i][1])*(1-sample[i][1]/math.sqrt(qx_hat))
            temp = temp+(temp3+temp4)/(temp1+temp2)
        result = result+temp/N
    return result/10


def I_counter(E, alpha):
    qx_hat = B*snr*R_counter(snr*E, alpha)
    result = 0
    for k in range(10):
        sample = np.random.randn(N, B)
        temp = 0
        for i in range(N):
            temp1 = math.exp(0.5*qx_hat+math.sqrt(qx_hat)*sample[i][0])
            temp2 = 0
            for j in range(1, B):
                temp2 = temp2+math.exp(-0.5*qx_hat+math.sqrt(qx_hat)*sample[i][j])
            temp3 = math.exp(0.5*qx_hat+math.sqrt(qx_hat)*sample[i][0])*(-1-sample[i][0]/math.sqrt(qx_hat))
            temp4 = (B-1)*math.exp(-0.5*qx_hat+math.sqrt(qx_hat)*sample[i][1])*(1-sample[i][1]/math.sqrt(qx_hat))
            temp = temp+(temp3+temp4)/(temp1+temp2)
        result = result+temp/N
    return result/10


E_it_gauss = 0.99
E_it_row = 0.99
E_it_counter = 0.99

for _ in range(50):
    E_it_gauss = 1 + I_gauss(E_it_gauss, alpha0)  # replica SE of gauss matrix
    E_it_row = 1 + I_row(E_it_row, alpha0)     # replica SE of row-orthogonal matrix
    E_it_counter = 1 + I_counter(E_it_counter, alpha0)  # replica SE of Three-Dirac spectrum


print(str(E_it_gauss))
print(str(E_it_row))
print(str(E_it_counter))
