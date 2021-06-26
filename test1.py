"""
Testing all cases of new VS original
"""

# pylint: disable=using-constant-test
import numpy as np
from florian_run_analytic_paper_final import H_total_final
from magpy_getH import getH_cy_section

N=333
null = np.zeros(N)
R = np.random.rand(N)*10
R1,R2 = np.random.rand(2,N)*5
R2 = R1+R2
PHI,PHI1,PHI2 = (np.random.rand(3,N)-.5)*10
PHI2 = PHI1+PHI2
Z,Z1,Z2 = (np.random.rand(3,N)-.5)*10
Z2 = Z1+Z2
mag = np.random.rand(N,3)
cases = []


if True:
    cases += [112, 212, 132, 232]
    r, r1, r2, phi, phi1, phi2, z, z1, z2 = R, R1, R2, PHI, PHI1, PHI2, Z, Z1, Z2
    z1=z
    phi1 = phi
    r = null
    obs_pos = np.array([r, phi, z]).T
    dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
    H1 = getH_cy_section(obs_pos, dim, mag)
    H2 = H_total_final(obs_pos, dim, mag)
    assert np.allclose(H1,H2)

if True:
    cases += [122, 222, 132, 232]
    r, r1, r2, phi, phi1, phi2, z, z1, z2 = R, R1, R2, PHI, PHI1, PHI2, Z, Z1, Z2
    z1=z
    phi1 = phi+np.pi
    r = null
    obs_pos = np.array([r, phi, z]).T
    dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
    H1 = getH_cy_section(obs_pos, dim, mag)
    H2 = H_total_final(obs_pos, dim, mag)
    assert np.allclose(H1,H2)


if True:
    cases += [113, 213, 133, 233, 115, 215, 135, 235]
    r, r1, r2, phi, phi1, phi2, z, z1, z2 = R, R1, R2, PHI, PHI1, PHI2, Z, Z1, Z2
    z1 = z
    phi1 = phi
    r1 = null
    obs_pos = np.array([r, phi, z]).T
    dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
    H1 = getH_cy_section(obs_pos, dim, mag)
    H2 = H_total_final(obs_pos, dim, mag)
    assert np.allclose(np.nan_to_num(H1), np.nan_to_num(H2))


if True:
    cases += [123, 223, 133, 233, 125, 225, 135, 235]
    r, r1, r2, phi, phi1, phi2, z, z1, z2 = R, R1, R2, PHI, PHI1, PHI2, Z, Z1, Z2
    z1 = z
    phi1 = phi+np.pi
    r1 = null
    obs_pos = np.array([r, phi, z]).T
    dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
    H1 = getH_cy_section(obs_pos, dim, mag)
    H2 = H_total_final(obs_pos, dim, mag)
    assert np.allclose(np.nan_to_num(H1), np.nan_to_num(H2))


if True:
    cases += [125, 225, 135, 235, 124, 224, 134, 234]
    r, r1, r2, phi, phi1, phi2, z, z1, z2 = R, R1, R2, PHI, PHI1, PHI2, Z, Z1, Z2
    z1 = z
    phi1 = phi+np.pi
    r = r2
    obs_pos = np.array([r, phi, z]).T
    dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
    H1 = getH_cy_section(obs_pos, dim, mag)
    H2 = H_total_final(obs_pos, dim, mag)
    assert np.allclose(np.nan_to_num(H1), np.nan_to_num(H2))


if True:
    cases += [211, 221, 212, 222]
    r, r1, r2, phi, phi1, phi2, z, z1, z2 = R, R1, R2, PHI, PHI1, PHI2, Z, Z1, Z2
    phi1 = phi
    phi2 = phi+np.pi
    r = null
    r1 = null
    obs_pos = np.array([r, phi, z]).T
    dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
    H1 = getH_cy_section(obs_pos, dim, mag)
    H2 = H_total_final(obs_pos, dim, mag)
    assert np.allclose(np.nan_to_num(H1), np.nan_to_num(H2))


if True:
    cases += [214, 224, 215, 225]
    r, r1, r2, phi, phi1, phi2, z, z1, z2 = R, R1, R2, PHI, PHI1, PHI2, Z, Z1, Z2
    phi1 = phi
    phi2 = phi+np.pi
    r = r1
    obs_pos = np.array([r, phi, z]).T
    dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
    H1 = getH_cy_section(obs_pos, dim, mag)
    H2 = H_total_final(obs_pos, dim, mag)
    assert np.allclose(np.nan_to_num(H1), np.nan_to_num(H2))

if True:
    cases += [111, 211, 121, 221, 112, 212, 122, 222]
    r, r1, r2, phi, phi1, phi2, z, z1, z2 = R, R1, R2, PHI, PHI1, PHI2, Z, Z1, Z2
    z = z1
    phi1 = phi
    phi2 = phi+np.pi
    r = null
    r1 = null
    obs_pos = np.array([r, phi, z]).T
    dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
    H1 = getH_cy_section(obs_pos, dim, mag)
    H2 = H_total_final(obs_pos, dim, mag)
    assert np.allclose(np.nan_to_num(H1), np.nan_to_num(H2))


if True:
    cases += [111, 211, 131, 231, 112, 212, 132, 232]
    r, r1, r2, phi, phi1, phi2, z, z1, z2 = R, R1, R2, PHI, PHI1, PHI2, Z, Z1, Z2
    z = z1
    phi1 = phi
    r = null
    r1 = null
    obs_pos = np.array([r, phi, z]).T
    dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
    H1 = getH_cy_section(obs_pos, dim, mag)
    H2 = H_total_final(obs_pos, dim, mag)
    assert np.allclose(np.nan_to_num(H1), np.nan_to_num(H2))


if True:
    cases += [115, 215, 135, 235, 114, 214, 134, 234]
    r, r1, r2, phi, phi1, phi2, z, z1, z2 = R, R1, R2, PHI, PHI1, PHI2, Z, Z1, Z2
    z = z1
    phi1 = phi
    r = r2
    obs_pos = np.array([r, phi, z]).T
    dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
    H1 = getH_cy_section(obs_pos, dim, mag)
    H2 = H_total_final(obs_pos, dim, mag)
    assert np.allclose(np.nan_to_num(H1), np.nan_to_num(H2))

# check if all cases have been sampled
if True:
    cases = list(set(cases))
    cases.sort()
    case_id = [111, 112, 113, 114, 115, 121, 122, 123, 124, 125, 131, 132, 133,
        134, 135, 211, 212, 213, 214, 215, 221,
        222, 223, 224, 225, 231, 232, 233, 234, 235]
    case_id.sort()
    assert np.allclose(np.array(cases), np.array(case_id))
