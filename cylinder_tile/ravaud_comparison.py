import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import mu_0

from cylinder_tile import magpy_getH
from cylinder_tile import integrals_quad

#comparison to the results in
#Ravaud et al. Progress In Electromagnetics Research B, Vol. 24, 17–32, 2010
#to verify the correctness of field components

#Fig. 6
datapoints = 100
r = np.linspace(0.005, 0.02, datapoints)
phi = np.ones(datapoints) * np.pi / 8.0
z = np.ones(datapoints) * 0.0015
obs_pos = np.column_stack((r, phi, z))

r_i12 = np.array([0.01, 0.015])
phi_j12 = np.array([0.0, np.pi / 4.0])
z_k12 = np.array([0.0, 0.003])
dim = np.tile(np.concatenate((r_i12, phi_j12, z_k12)), (datapoints, 1))

M = 1.0 / mu_0
phi_M = np.pi + np.pi / 8.0
theta_M = np.pi / 2.0
mag = np.tile((M, phi_M, theta_M), (datapoints, 1))

results = magpy_getH.getH_cy_section(obs_pos, dim, mag)
###########
# quad
res_quad = np.zeros(datapoints)
for i in range(datapoints):
    res_quad[i] = (integrals_quad.H_r_ri(r[i], phi[i], z[i], r_i12, phi_j12, z_k12, phi_M, theta_M) + integrals_quad.H_r_phij(r[i], phi[i], z[i], r_i12, phi_j12, z_k12, phi_M, theta_M) + integrals_quad.H_r_zk(r[i], phi[i], z[i], r_i12, phi_j12, z_k12, phi_M, theta_M)) * M / (4.0 * np.pi)

###########


plt.plot(r, results[:,0], 'k', linewidth=4)
plt.plot(r, res_quad, 'y--', linewidth=2)

plt.grid()
plt.xlabel('r [m]')
plt.ylabel('Hr [A/m]')
plt.legend(('analytic','quad'))
plt.savefig("r_Hr.png")
plt.close()


#Fig. 7
datapoints = 100
r = np.ones(datapoints) * 0.022
phi = np.linspace(-np.pi / 4.0, np.pi / 2.0, datapoints)
z = np.ones(datapoints) * 0.001
obs_pos = np.column_stack((r, phi, z))

r_i12 = np.array([0.025, 0.03])
phi_j12 = np.array([0.0, np.pi / 4.0])
z_k12 = np.array([0.0, 0.003])
dim = np.tile(np.concatenate((r_i12, phi_j12, z_k12)), (datapoints, 1))

M = 1.0 / mu_0
phi_M = np.pi + np.pi / 8.0
theta_M = np.pi / 2.0
mag = np.tile((M, phi_M, theta_M), (datapoints, 1))

results = magpy_getH.getH_cy_section(obs_pos, dim, mag)

###########
# quad
res_quad = np.zeros(datapoints)
for i in range(datapoints):
    res_quad[i] = (integrals_quad.H_phi_ri(r[i], phi[i], z[i], r_i12, phi_j12, z_k12, phi_M, theta_M) + integrals_quad.H_phi_phij(r[i], phi[i], z[i], r_i12, phi_j12, z_k12, phi_M, theta_M) + integrals_quad.H_phi_zk(r[i], phi[i], z[i], r_i12, phi_j12, z_k12, phi_M, theta_M)) * M / (4.0 * np.pi)

###########

plt.plot(phi, results[:,1], 'k', linewidth=4)
plt.plot(phi, res_quad, 'y--', linewidth=2)
plt.grid()
plt.xlabel('phi [rad]')
plt.ylabel('Hphi [A/m]')
plt.legend(('analytic','quad'))
plt.savefig("phi_Hphi.png")
plt.close()

#Fig. 8
datapoints = 100
r = np.ones(datapoints) * 0.0249
phi = np.ones(datapoints) * np.pi / 8.0
z = np.linspace(-0.004, 0.008, datapoints)
obs_pos = np.column_stack((r, phi, z))

r_i12 = np.array([0.025, 0.03])
phi_j12 = np.array([0.0, np.pi / 4.0])
z_k12 = np.array([0.0, 0.003])
dim = np.tile(np.concatenate((r_i12, phi_j12, z_k12)), (datapoints, 1))

M = 1.0 / mu_0
phi_M = np.pi + np.pi / 8.0
theta_M = np.pi / 2.0
mag = np.tile((M, phi_M, theta_M), (datapoints, 1))

results = magpy_getH.getH_cy_section(obs_pos, dim, mag)

###########
# quad
# res_quad = np.zeros(datapoints)
# for i in range(datapoints):
#     res_quad[i] = (integrals_quad.H_z_ri(r[i], phi[i], z[i], r_i12, phi_j12, z_k12, phi_M, theta_M) + integrals_quad.H_z_phij(r[i], phi[i], z[i], r_i12, phi_j12, z_k12, phi_M, theta_M) + integrals_quad.H_z_zk(r[i], phi[i], z[i], r_i12, phi_j12, z_k12, phi_M, theta_M)) * M / (4.0 * np.pi)

###########

plt.plot(z, results[:,2], 'k', linewidth=4)
# plt.plot(z, res_quad, 'y--', linewidth=2)
plt.grid()
plt.xlabel('z [m]')
plt.ylabel('Hz [A/m]')
plt.legend(('analytic','quad'))
plt.show()
# plt.savefig("z_Hz.png")
# plt.close()
