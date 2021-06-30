"""
Test numerical integrations vs analytics
"""

import numpy as np
from cylinder_tile import integrals_quad
from cylinder_tile import magpy_getH

import warnings
warnings.filterwarnings("error")

r = [0.0, 5.0]
phi = [0.8]
z = [3.0]
r_i12 = np.array([[2.0, 6.0], [0.0, 6.0], [0.0, 3.0], [7.0, 8.0]])
phi_j12 = [np.array([0.6, 0.9]), np.array([0.6, 0.7]), np.array([0.6, 0.8 + 4.0 * np.pi]), np.array([0.6, 0.8 + 5.0 * np.pi])]
z_k12 = [np.array([1.0, 8.0]), np.array([3.0, 8.0]), np.array([1.0, 2.0]), np.array([7.0, 8.0])]
M = np.random.random() * 20.0
phi_M = np.random.random() * 2.0 * np.pi
theta_M = np.random.random() * np.pi


for r_var in r:
    for phi_var in phi:
        for z_var in z:
            for r_i_var in r_i12:
                for phi_j_var in phi_j12:
                    for z_k_var in z_k12:
                        try:
                            res_analytic = magpy_getH.getH_cy_section(np.array([[r_var, phi_var, z_var]]), np.array([np.concatenate((r_i_var, phi_j_var, z_k_var))]), np.array([[M, phi_M, theta_M]]))
                            res_quad = np.array([[integrals_quad.H_r_ri(r_var, phi_var, z_var, r_i_var, phi_j_var, z_k_var, phi_M, theta_M), integrals_quad.H_r_phij(r_var, phi_var, z_var, r_i_var, phi_j_var, z_k_var, phi_M, theta_M), integrals_quad.H_r_zk(r_var, phi_var, z_var, r_i_var, phi_j_var, z_k_var, phi_M, theta_M)], [integrals_quad.H_phi_ri(r_var, phi_var, z_var, r_i_var, phi_j_var, z_k_var, phi_M, theta_M), integrals_quad.H_phi_phij(r_var, phi_var, z_var, r_i_var, phi_j_var, z_k_var, phi_M, theta_M), integrals_quad.H_phi_zk(r_var, phi_var, z_var, r_i_var, phi_j_var, z_k_var, phi_M, theta_M)], [integrals_quad.H_z_ri(r_var, phi_var, z_var, r_i_var, phi_j_var, z_k_var, phi_M, theta_M), integrals_quad.H_z_phij(r_var, phi_var, z_var, r_i_var, phi_j_var, z_k_var, phi_M, theta_M), integrals_quad.H_z_zk(r_var, phi_var, z_var, r_i_var, phi_j_var, z_k_var, phi_M, theta_M)]]) * M / (4.0 * np.pi)
                            print(np.max(np.abs(np.sum(res_quad, axis = -1)-res_analytic)))
                        except:
                            print(f'none - {res_analytic}')

r_var = np.random.random() * 20.0
phi_var = np.random.random() * 20.0 - 10.0
z_var = np.random.random() * 20.0 - 10.0
r_i_var = np.sort(np.random.random(2) * 20.0 - 10.0)
phi_j_var = np.sort(np.random.random(2) * 20.0 - 10.0)
z_k_var = np.sort(np.random.random(2) * 20.0 - 10.0)
res_quad = np.array([[integrals_quad.H_r_ri(r_var, phi_var, z_var, r_i_var, phi_j_var, z_k_var, phi_M, theta_M), integrals_quad.H_r_phij(r_var, phi_var, z_var, r_i_var, phi_j_var, z_k_var, phi_M, theta_M), integrals_quad.H_r_zk(r_var, phi_var, z_var, r_i_var, phi_j_var, z_k_var, phi_M, theta_M)], [integrals_quad.H_phi_ri(r_var, phi_var, z_var, r_i_var, phi_j_var, z_k_var, phi_M, theta_M), integrals_quad.H_phi_phij(r_var, phi_var, z_var, r_i_var, phi_j_var, z_k_var, phi_M, theta_M), integrals_quad.H_phi_zk(r_var, phi_var, z_var, r_i_var, phi_j_var, z_k_var, phi_M, theta_M)], [integrals_quad.H_z_ri(r_var, phi_var, z_var, r_i_var, phi_j_var, z_k_var, phi_M, theta_M), integrals_quad.H_z_phij(r_var, phi_var, z_var, r_i_var, phi_j_var, z_k_var, phi_M, theta_M), integrals_quad.H_z_zk(r_var, phi_var, z_var, r_i_var, phi_j_var, z_k_var, phi_M, theta_M)]]) * M / (4.0 * np.pi)
res_analytic = magpy_getH.getH_cy_section(np.array([[r_var, phi_var, z_var]]), np.array([np.concatenate((r_i_var, phi_j_var, z_k_var))]), np.array([[M, phi_M, theta_M]]))
print(np.max(np.abs(np.sum(res_quad, axis = -1)-res_analytic)))
