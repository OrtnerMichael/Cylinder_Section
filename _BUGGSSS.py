import numpy as np
from florian_run_analytic_paper_final import H_total_final, arctan_k_tan_2
import florian_ell3_paper as icels


# Hr_zk_case213 is always zero
if False:
    from florian_run_analytic_paper_final import Hr_zk_case213
    N = 10000
    x1, x2, x3, x4 = (np.random.rand(4,N)-.5)*15
    print(np.allclose(Hr_zk_case213(abs(x1),x2,x3,x4), np.zeros(N)))


# RARE crash case 235
if False:
    dim=np.array([[ 0.37326364 ,-0.05258784 , 6.06777749 , 8.92271999 ,-2.07775455 ,-4.54108339]])
    mag=np.array([[1.30684832, 0.59188274 ,2.37654756]])
    obs_pos = np.array([ [-1.18400611,  1.92119189, -2.07793907]])
    H_total_final(obs_pos, dim, mag)


# something wrong with Pi_coef() !!! unused argument - probably a bug ?
if False:
    def Hz_zk_case214(r, phi_bar_j, theta_M, z_bar_k):
        t = np.sqrt(r**2 + z_bar_k**2)
        def Pi(sign):
            return icels.el3_angle_vectorized(phi_bar_j/2.0,
                2.0 * r / (r + sign * t), -4.0 * r**2 / z_bar_k**2)
        def Pi_coef(sign): #<<<<<<<<<<<<<<<<
            return np.cos(theta_M) * np.sign(z_bar_k)
        return Pi_coef(1) * Pi(1) + Pi_coef(-1) * Pi(-1)

# same for case 234
if False:
    def Hz_zk_case234(r, phi_bar_j, theta_M, z_bar_k):
        t = np.sqrt(r**2 + z_bar_k**2)
        def Pi(sign):
            return icels.el3_angle_vectorized(phi_bar_j/2.0, 2.0 * r / (r + sign * t), -4.0 * r**2 / z_bar_k**2)
        def Pi_coef(sign): # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            return np.cos(theta_M) * np.sign(z_bar_k)
        return Pi_coef(1) * Pi(1) + Pi_coef(-1) * Pi(-1)