import numpy as np
from florian_run_analytic_paper_final import H_total_final
from magpy_getH import getH_cy_section

N = 1000
null = np.zeros(N)
r = np.random.rand(N)*10
ri,ra = np.random.rand(2,N)*5
ra = ri+ra
phi,phi1,phi2 = (np.random.rand(3,N)-.5)*10
phi2 = phi1+phi2
z,z1,z2 = (np.random.rand(3,N)-.5)*10
z2 = z1+z2
mag = np.random.rand(N,3)

# case 112 - 212 - 132 - 232
if False:
    z1=z
    phi1 = phi
    r = null
    obs_pos = np.array([r, phi, z]).T
    dim = np.array([ri,ra, phi1,phi2, z1,z2]).T

    H1 = getH_cy_section(obs_pos, dim, mag)
    H2 = H_total_final(obs_pos, dim, mag)
    assert np.allclose(H1,H2)

# case 113
if False:
    z1=z
    phi1 = phi
    ri = null
    obs_pos = np.array([r, phi, z]).T
    dim = np.array([ri,ra, phi1,phi2, z1,z2]).T
    H1 = getH_cy_section(obs_pos, dim, mag)
    H2 = H_total_final(obs_pos, dim, mag)
    assert np.allclose(H1,H2)
