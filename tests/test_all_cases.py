"""
Testing all cases against a large set of pre-computed values
"""

import numpy as np
from cylinder_tile.magpy_getH import field_H_cylinder_tile


# creating test data

# from florian_run_analytic_paper_final import H_total_final
# N=1111
# null = np.zeros(N)
# R = np.random.rand(N)*10
# R1,R2 = np.random.rand(2,N)*5
# R2 = R1+R2
# PHI,PHI1,PHI2 = (np.random.rand(3,N)-.5)*10
# PHI2 = PHI1+PHI2
# Z,Z1,Z2 = (np.random.rand(3,N)-.5)*10
# Z2 = Z1+Z2
# mag = np.random.rand(N,3)

# DATA = []

# # cases [112, 212, 132, 232]
# r, r1, r2, phi, phi1, phi2, z, z1, z2 = R, R1, R2, PHI, PHI1, PHI2, Z, Z1, Z2
# z1=z
# phi1 = phi
# r = null
# obs_pos = np.array([r, phi, z]).T
# dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
# H1 = field_H_cylinder_tile(obs_pos, dim, mag)
# H2 = H_total_final(obs_pos, dim, mag)
# assert np.allclose(H1,H2)

# DATA += [H2]

# # cases [122, 222, 132, 232]
# r, r1, r2, phi, phi1, phi2, z, z1, z2 = R, R1, R2, PHI, PHI1, PHI2, Z, Z1, Z2
# z1=z
# phi1 = phi+np.pi
# r = null
# obs_pos = np.array([r, phi, z]).T
# dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
# H1 = field_H_cylinder_tile(obs_pos, dim, mag)
# H2 = H_total_final(obs_pos, dim, mag)
# assert np.allclose(H1,H2)

# DATA += [H2]

# # cases [113, 213, 133, 233, 115, 215, 135, 235]
# r, r1, r2, phi, phi1, phi2, z, z1, z2 = R, R1, R2, PHI, PHI1, PHI2, Z, Z1, Z2
# z1 = z
# phi1 = phi
# r1 = null
# obs_pos = np.array([r, phi, z]).T
# dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
# H1 = field_H_cylinder_tile(obs_pos, dim, mag)
# H2 = H_total_final(obs_pos, dim, mag)
# assert np.allclose(np.nan_to_num(H1), np.nan_to_num(H2))

# DATA += [H2]


# # cases [123, 223, 133, 233, 125, 225, 135, 235]
# r, r1, r2, phi, phi1, phi2, z, z1, z2 = R, R1, R2, PHI, PHI1, PHI2, Z, Z1, Z2
# z1 = z
# phi1 = phi+np.pi
# r1 = null
# obs_pos = np.array([r, phi, z]).T
# dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
# H1 = field_H_cylinder_tile(obs_pos, dim, mag)
# H2 = H_total_final(obs_pos, dim, mag)
# assert np.allclose(np.nan_to_num(H1), np.nan_to_num(H2))

# DATA += [H2]

# # cases [125, 225, 135, 235, 124, 224, 134, 234]
# r, r1, r2, phi, phi1, phi2, z, z1, z2 = R, R1, R2, PHI, PHI1, PHI2, Z, Z1, Z2
# z1 = z
# phi1 = phi+np.pi
# r = r2
# obs_pos = np.array([r, phi, z]).T
# dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
# H1 = field_H_cylinder_tile(obs_pos, dim, mag)
# H2 = H_total_final(obs_pos, dim, mag)
# assert np.allclose(np.nan_to_num(H1), np.nan_to_num(H2))

# DATA += [H2]

# # cases [211, 221, 212, 222]
# r, r1, r2, phi, phi1, phi2, z, z1, z2 = R, R1, R2, PHI, PHI1, PHI2, Z, Z1, Z2
# phi1 = phi
# phi2 = phi+np.pi
# r = null
# r1 = null
# obs_pos = np.array([r, phi, z]).T
# dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
# H1 = field_H_cylinder_tile(obs_pos, dim, mag)
# H2 = H_total_final(obs_pos, dim, mag)
# assert np.allclose(np.nan_to_num(H1), np.nan_to_num(H2))

# DATA += [H2]

# # cases [214, 224, 215, 225]
# r, r1, r2, phi, phi1, phi2, z, z1, z2 = R, R1, R2, PHI, PHI1, PHI2, Z, Z1, Z2
# phi1 = phi
# phi2 = phi+np.pi
# r = r1
# obs_pos = np.array([r, phi, z]).T
# dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
# H1 = field_H_cylinder_tile(obs_pos, dim, mag)
# H2 = H_total_final(obs_pos, dim, mag)
# assert np.allclose(np.nan_to_num(H1), np.nan_to_num(H2))

# DATA += [H2]

# # cases [111, 211, 121, 221, 112, 212, 122, 222]
# r, r1, r2, phi, phi1, phi2, z, z1, z2 = R, R1, R2, PHI, PHI1, PHI2, Z, Z1, Z2
# z = z1
# phi1 = phi
# phi2 = phi+np.pi
# r = null
# r1 = null
# obs_pos = np.array([r, phi, z]).T
# dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
# H1 = field_H_cylinder_tile(obs_pos, dim, mag)
# H2 = H_total_final(obs_pos, dim, mag)
# assert np.allclose(np.nan_to_num(H1), np.nan_to_num(H2))

# DATA += [H2]

# # cases [111, 211, 131, 231, 112, 212, 132, 232]
# r, r1, r2, phi, phi1, phi2, z, z1, z2 = R, R1, R2, PHI, PHI1, PHI2, Z, Z1, Z2
# z = z1
# phi1 = phi
# r = null
# r1 = null
# obs_pos = np.array([r, phi, z]).T
# dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
# H1 = field_H_cylinder_tile(obs_pos, dim, mag)
# H2 = H_total_final(obs_pos, dim, mag)
# assert np.allclose(np.nan_to_num(H1), np.nan_to_num(H2))

# DATA += [H2]

# # cases [115, 215, 135, 235, 114, 214, 134, 234]
# r, r1, r2, phi, phi1, phi2, z, z1, z2 = R, R1, R2, PHI, PHI1, PHI2, Z, Z1, Z2
# z = z1
# phi1 = phi
# r = r2
# obs_pos = np.array([r, phi, z]).T
# dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
# H1 = field_H_cylinder_tile(obs_pos, dim, mag)
# H2 = H_total_final(obs_pos, dim, mag)
# assert np.allclose(np.nan_to_num(H1), np.nan_to_num(H2))

# DATA += [H2]

# DATA = np.array(DATA)
# np.save('data_test_cy_cases', DATA)

# DATA_INPUT = np.array([R, R1, R2, PHI, PHI1, PHI2, Z, Z1, Z2])
# np.save('data_test_cy_cases_inp', DATA_INPUT)
# np.save('data_test_cy_cases_inp2', mag)


DATA_INPUT = np.load('tests/data_test_cy_cases_inp.npy')
mag = np.load('tests/data_test_cy_cases_inp2.npy')
DATA = np.load('tests/data_test_cy_cases.npy')
null = np.zeros(1111)


def test_cases0():
    """ cases [112, 212, 132, 232]
    """
    r, r1, r2, phi, phi1, phi2, z, z1, z2 = DATA_INPUT
    z1=z
    phi1 = phi
    r = null
    obs_pos = np.array([r, phi, z]).T
    dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
    H = field_H_cylinder_tile(obs_pos, dim, mag)
    H0 = DATA[0]
    assert np.allclose(H, H0)


def test_cases1():
    """ cases [122, 222, 132, 232]
    """
    r, r1, r2, phi, phi1, phi2, z, z1, z2 = DATA_INPUT
    z1=z
    phi1 = phi+np.pi
    r = null
    obs_pos = np.array([r, phi, z]).T
    dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
    H = field_H_cylinder_tile(obs_pos, dim, mag)
    H0 = DATA[1]
    assert np.allclose(H, H0)


def test_cases2():
    """cases [113, 213, 133, 233, 115, 215, 135, 235]
    """
    r, r1, r2, phi, phi1, phi2, z, z1, z2 = DATA_INPUT
    z1 = z
    phi1 = phi
    r1 = null
    obs_pos = np.array([r, phi, z]).T
    dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
    H = field_H_cylinder_tile(obs_pos, dim, mag)
    H0 = DATA[2]
    assert np.allclose(np.nan_to_num(H), np.nan_to_num(H0))


def test_cases3():
    """ cases [123, 223, 133, 233, 125, 225, 135, 235]
    """
    r, r1, r2, phi, phi1, phi2, z, z1, z2 = DATA_INPUT
    z1 = z
    phi1 = phi+np.pi
    r1 = null
    obs_pos = np.array([r, phi, z]).T
    dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
    H = field_H_cylinder_tile(obs_pos, dim, mag)
    H0 = DATA[3]
    assert np.allclose(np.nan_to_num(H), np.nan_to_num(H0))

def test_cases4():
    """ cases [125, 225, 135, 235, 124, 224, 134, 234]
    """ 
    r, r1, r2, phi, phi1, phi2, z, z1, z2 = DATA_INPUT
    z1 = z
    phi1 = phi+np.pi
    r = r2
    obs_pos = np.array([r, phi, z]).T
    dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
    H = field_H_cylinder_tile(obs_pos, dim, mag)
    H0 = DATA[4]
    assert np.allclose(np.nan_to_num(H), np.nan_to_num(H0))

def test_cases5():
    """ cases [211, 221, 212, 222]
    """
    r, r1, r2, phi, phi1, phi2, z, z1, z2 = DATA_INPUT
    phi1 = phi
    phi2 = phi+np.pi
    r = null
    r1 = null
    obs_pos = np.array([r, phi, z]).T
    dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
    H = field_H_cylinder_tile(obs_pos, dim, mag)
    H0 = DATA[5]
    assert np.allclose(np.nan_to_num(H), np.nan_to_num(H0))


def test_cases6():
    """ cases [214, 224, 215, 225]
    """
    r, r1, r2, phi, phi1, phi2, z, z1, z2 = DATA_INPUT
    phi1 = phi
    phi2 = phi+np.pi
    r = r1
    obs_pos = np.array([r, phi, z]).T
    dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
    H = field_H_cylinder_tile(obs_pos, dim, mag)
    H0 = DATA[6]
    assert np.allclose(np.nan_to_num(H), np.nan_to_num(H0))


def test_cases7():
    """ cases [111, 211, 121, 221, 112, 212, 122, 222]
    """
    r, r1, r2, phi, phi1, phi2, z, z1, z2 = DATA_INPUT
    z = z1
    phi1 = phi
    phi2 = phi+np.pi
    r = null
    r1 = null
    obs_pos = np.array([r, phi, z]).T
    dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
    H = field_H_cylinder_tile(obs_pos, dim, mag)
    H0 = DATA[7]
    assert np.allclose(np.nan_to_num(H), np.nan_to_num(H0))


def test_cases8():
    """ cases [111, 211, 131, 231, 112, 212, 132, 232]
    """
    r, r1, r2, phi, phi1, phi2, z, z1, z2 = DATA_INPUT
    z = z1
    phi1 = phi
    r = null
    r1 = null
    obs_pos = np.array([r, phi, z]).T
    dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
    H = field_H_cylinder_tile(obs_pos, dim, mag)
    H0 = DATA[8]
    assert np.allclose(np.nan_to_num(H), np.nan_to_num(H0))


def test_cases9():
    """ cases [115, 215, 135, 235, 114, 214, 134, 234]
    """
    r, r1, r2, phi, phi1, phi2, z, z1, z2 = DATA_INPUT
    z = z1
    phi1 = phi
    r = r2
    obs_pos = np.array([r, phi, z]).T
    dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
    H = field_H_cylinder_tile(obs_pos, dim, mag)
    H0 = DATA[9]
    assert np.allclose(np.nan_to_num(H), np.nan_to_num(H0))
