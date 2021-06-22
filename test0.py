"""
Test computation of Cylinder implementation
"""

import numpy as np
from florian_run_analytic_paper_final import H_total_final
from magpy_getH import getH_cy_section

#original test from florian

# observer positions (cylinder CS) r,phi,z, units: [m], [rad]
obs_pos = np.array([(0,.6,3), (1,np.pi,4), (2,2*np.pi,5), (2,1,1)])
# cylinder dimensions (cylinder CS) r1,r2,phi1,phi2,z1,z2, units: [m], [rad]
dim = np.array([
    (0, 2, .6, np.pi, 3, 5),
    (1, 3, .1, 4.5, 4, 6),
    (3, 5, 0, 2*np.pi, 6, 10),
    (0, .1, .1, .2, 1, 2)])
# magnetization vectors (spherical CS) [A/m], [rad]
mag = np.array([
    (.7, .7, .3),
    (.8, .8, .4),
    (.9, .9, .5),
    (1, 1, .6)])
H1 = H_total_final(obs_pos, dim, mag)
H2 = getH_cy_section(obs_pos, dim, mag)
assert np.allclose(np.nan_to_num(H1), np.nan_to_num(H2))

# # old code VS new code testing
# N = 500
# def rpos():
#     return (np.random.rand(N)-0.5)*5
# def rang():
#     return np.random.rand(N)*np.pi*2

# obs_pos = np.array([rpos(), rang(), rpos()]).T
# ri = rpos()
# ra = ri + rpos()
# phi1 = rang()
# phi2 = phi1+rang()
# z1 = rpos()
# z2 = z1+rpos()
# dim = np.array([ri, ra, phi1, phi2, z1, z2]).T
# mag = np.array([rang(), rang(), rang()]).T

# H1 = field_H_cylinder_section(obs_pos, dim, mag)
# H2 = H_total_final(obs_pos, dim, mag)
# assert np.allclose(np.nan_to_num(H1), np.nan_to_num(H2))

# # RARE 235 BUG
# #dim=np.array([[ 0.37326364 ,-0.05258784 , 6.06777749 , 8.92271999 ,-2.07775455 ,-4.54108339]])
# #mag=np.array([[1.30684832, 0.59188274 ,2.37654756]])
# #obs_pos = np.array([ [-1.18400611,  1.92119189, -2.07793907]])
# #H_total_final(obs_pos, dim, mag)


# # magpylib test
# obs_pos = np.array([(0,0,2)])
# dim = np.array([(0,1,0,2*np.pi,-1,1)])
# mag = np.array([(1,0,np.pi/2)])
# H1 = field_H_cylinder_section(obs_pos, dim, mag)

# import magpylib as mag3
# cyl = mag3.magnet.Cylinder((1,0,0), (2,2))
# H2 = cyl.getB(0,0,2)
# assert np.allclose(H1,H2)


