import numpy as np
from _run_analytic_paper_final import arctan_k_tan_2
import _ell3_paper as icels


# def Hz_zk_case214(r, phi_bar_j, theta_M, z_bar_k):
#     t = np.sqrt(r**2 + z_bar_k**2)
#     def Pi(sign):
#         return icels.el3_angle_vectorized(phi_bar_j/2.0,
#             2.0 * r / (r + sign * t), -4.0 * r**2 / z_bar_k**2)
#     def Pi_coef(sign):
#         return np.cos(theta_M) * np.sign(z_bar_k)
#     return Pi_coef(1) * Pi(1) + Pi_coef(-1) * Pi(-1)

# N = 100
# x1, x2, x3, x4 = (np.random.rand(4,N)-.5)*15
# A = Hz_zk_case214(abs(x1),x2,x3,x4)

from time import perf_counter as pf

T0 = pf()
for _ in range(100000):
    x = 1.*2*3*4*5*6*7*8*9*8*7*6*5*4*2*3*4*5*6*7*8*9*8*7*6*5*4*2*3*4*5*6*7*8*9*8*7*6*5*4
T1 = pf()
print(T1-T0)

T0 = pf()
for _ in range(100000):
    x = 1.*2.*3.*4.*5.*6.*7.*8.*9.*8.*7.*6.*5.*4.*2.*3.*4.*5.*6.*7.*8.*9.*8.*7.*6.*5.*4.*2.*3.*4.*5.*6.*7.*8.*9.*8.*7.*6.*5.*4.
T1 = pf()
print(T1-T0)

T0 = pf()
for _ in range(100000):
    x = 1.*2*3*4*5*6*7*8*9*8*7*6*5*4*2*3*4*5*6*7*8*9*8*7*6*5*4*2*3*4*5*6*7*8*9*8*7*6*5*4
T1 = pf()
print(T1-T0)
