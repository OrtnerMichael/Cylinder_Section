# pylint: disable=no-member
# pylint: disable=too-many-lines
# pylint: disable=line-too-long
# pylint: disable=missing-function-docstring
import numpy as np
import scipy.special
from cylinder_tile.special_el3 import el3_angle

def close(arg1, arg2):
    """
    determine if arg1 and arg2 lie close to each other
    input: ndarray, shape (n,) or numpy-interpretable scalar
    output: ndarray, dtype=bool
    """
    EDGESIZE = 1e-10
    return np.isclose(arg1, arg2, rtol=0, atol=EDGESIZE)


def arctan_k_tan_2(k, phi):
    """
    help function for periodic continuation

    what is this function doing exactly ? what are the arguement types, ranges, ...

    can be replaced by non-masked version ?
    """

    full_periods = np.round(phi / (2.0 * np.pi))
    phi_red = phi - full_periods * 2.0 * np.pi

    result = full_periods * np.pi

    return np.where(np.abs(phi_red) < np.pi, result + np.arctan(k * np.tan(phi_red / 2.0)), result + phi_red / 2.0)


def determine_cases(r, phi, z, r1, phi1, z1):
    """
    Determine case of input parameter set.
        r, phi, z: observer positions
        r1, phi1, z1: boundary values

    All inputs must be ndarrays, shape (n,)

    Returns: case numbers, ndarray, shape (n,), dtype=int

    The case number is a three digits integer, where the digits can be the following values
      1st digit: 1:z=z1,  2:general
      2nd digit: 1:phi-phi1= 2n*pi,  2:phi-phi1=(2n+1)*pi,  3:general
      3rd digit: 1:r=r1=0,  2:r=0,  3:r1=0,  4:r=r1>0,  5:general
    """
    n = len(r)           # input length

    # allocate result
    result = np.ones((3,n))

    # identify z-case
    mask_z = close(z, z1)
    result[0] = 200
    result[0,mask_z] = 100

    # identify phi-case
    mod_2pi = np.abs(phi-phi1)%(2*np.pi)
    mask_phi1 = np.logical_or(close(mod_2pi,0), close(mod_2pi,2*np.pi))
    mod_pi = np.abs(phi-phi1)%np.pi
    mask_phi2 = np.logical_or(close(mod_pi,0), close(mod_pi,np.pi))
    result[1] = 30
    result[1,mask_phi2] = 20
    result[1,mask_phi1] = 10

    # identify r-case
    mask_r2 = close(r,0)
    mask_r3 = close(r1,0)
    mask_r4 = close(r,r1)
    mask_r1 = mask_r2 * mask_r3
    result[2] = 5
    result[2,mask_r4] = 4
    result[2,mask_r3] = 3
    result[2,mask_r2] = 2
    result[2,mask_r1] = 1

    return np.array(np.sum(result, axis=0), dtype=int)


# Implementation of all non-zero field components in every special case
# e.g. Hphi_zk stands for field component in phi-direction originating
# from the cylinder tile face at zk

# 112 ##############

def Hphi_zk_case112(r_i, theta_M):
    return np.cos(theta_M) * np.log(r_i)

def Hz_ri_case112(phi_bar_M, theta_M):
    return -np.sin(theta_M) * np.sin(phi_bar_M)

def Hz_phij_case112(r_i, phi_bar_M, theta_M):
    return np.sin(theta_M) * np.sin(phi_bar_M) * np.log(r_i)

# 113 ##############

def Hphi_zk_case113(r, theta_M):
    return -np.cos(theta_M) * np.log(r)

def Hz_phij_case113(r, phi_bar_M, theta_M):
    return -np.sin(theta_M) * np.sin(phi_bar_M) * np.log(r)

# 115 ##############

def Hr_zk_case115(r, r_i, r_bar_i, phi_bar_j, theta_M):
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r * r_i / r_bar_i**2)
    E_coef = np.cos(theta_M) * np.abs(r_bar_i) / r
    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r * r_i / r_bar_i**2)
    F_coef = -np.cos(theta_M) * (r**2 + r_i**2) / (r * np.abs(r_bar_i))
    return E_coef * E + F_coef * F

def Hphi_zk_case115(r, r_i, r_bar_i, theta_M):
    t1 = r_i / r
    t1_coef = -np.cos(theta_M) * np.sign(r_bar_i)
    t2 = np.log(np.abs(r_bar_i)) * np.sign(r_bar_i)
    t2_coef = -np.cos(theta_M)
    return t1_coef * t1 + t2_coef * t2

def Hz_ri_case115(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M):
    t1 = np.abs(r_bar_i) / r
    t1_coef = np.sin(theta_M) * np.sin(phi_bar_M)
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r * r_i / r_bar_i**2)
    E_coef = np.sin(theta_M) * np.cos(phi_bar_M) * np.abs(r_bar_i) / r
    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r * r_i / r_bar_i**2)
    F_coef = -np.sin(theta_M) * np.cos(phi_bar_M) * (r**2 + r_i**2) / (r * np.abs(r_bar_i))
    return t1_coef * t1 + E_coef * E + F_coef * F

def Hz_phij_case115(r_bar_i, phi_bar_M, theta_M):
    t1 = np.log(np.abs(r_bar_i)) * np.sign(r_bar_i)
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_M)
    return t1_coef * t1

# 122 ##############

def Hphi_zk_case122(r_i, theta_M):
    return -np.cos(theta_M) * np.log(r_i)

def Hz_ri_case122(phi_bar_M, theta_M):
    return np.sin(theta_M) * np.sin(phi_bar_M)

def Hz_phij_case122(r_i, phi_bar_M, theta_M):
    return -np.sin(theta_M) * np.sin(phi_bar_M) * np.log(r_i)

# 123 ##############

def Hphi_zk_case123(r, theta_M):
    return -np.cos(theta_M) * np.log(r)

def Hz_phij_case123(r, phi_bar_M, theta_M):
    return -np.sin(theta_M) * np.sin(phi_bar_M) * np.log(r)

# 124 ##############

def Hphi_zk_case124(r, theta_M):
    return np.cos(theta_M) * (1.0 - np.log(2.0 * r))

def Hz_ri_case124(phi_bar_M, theta_M):
    return 2.0 * np.sin(theta_M) * np.sin(phi_bar_M)

def Hz_phij_case124(r, phi_bar_M, theta_M):
    return -np.sin(theta_M) * np.sin(phi_bar_M) * np.log(2.0 * r)

# 125 ##############

def Hr_zk_case125(r, r_i, r_bar_i, phi_bar_j, theta_M):
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r * r_i / r_bar_i**2)
    E_coef = np.cos(theta_M) * np.abs(r_bar_i) / r
    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r * r_i / r_bar_i**2)
    F_coef = -np.cos(theta_M) * (r**2 + r_i**2) / (r * np.abs(r_bar_i))
    return E_coef * E + F_coef * F

def Hphi_zk_case125(r, r_i, theta_M):
    return np.cos(theta_M) / r * ( r_i - r * np.log(r + r_i) )

def Hz_ri_case125(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M):
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r * r_i / r_bar_i**2)
    E_coef = np.sin(theta_M) * np.cos(phi_bar_M) * np.abs(r_bar_i) / r
    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r * r_i / r_bar_i**2)
    F_coef = -np.sin(theta_M) * np.cos(phi_bar_M) * (r**2 + r_i**2) / (r * np.abs(r_bar_i))
    return np.sin(theta_M) * np.sin(phi_bar_M) * (r + r_i) / r + E_coef * E + F_coef * F

def Hz_phij_case125(r, r_i, phi_bar_M, theta_M):
    return -np.sin(theta_M) * np.sin(phi_bar_M) * np.log(r + r_i)

# 132 ##############

def Hr_zk_case132(r_i, phi_bar_j, theta_M):
    return np.cos(theta_M) * np.sin(phi_bar_j) * np.log(r_i)

def Hphi_zk_case132(r_i, phi_bar_j, theta_M):
    return np.cos(theta_M) * np.cos(phi_bar_j) * np.log(r_i)

def Hz_ri_case132(phi_bar_Mj, theta_M):
    return -np.sin(theta_M) * np.sin(phi_bar_Mj)

def Hz_phij_case132(r_i, phi_bar_Mj, theta_M):
    return np.sin(theta_M) * np.sin(phi_bar_Mj) * np.log(r_i)

# 133 ##############

def Hr_zk_case133(r, phi_bar_j, theta_M):
    return -np.cos(theta_M) * np.sin(phi_bar_j) + np.cos(theta_M) * np.sin(phi_bar_j) * np.log(r * (1.0 - np.cos(phi_bar_j)))

def Hphi_zk_case133(phi_bar_j, theta_M):
    return np.cos(theta_M) - np.cos(theta_M) * np.cos(phi_bar_j) * np.arctanh(np.cos(phi_bar_j))

def Hz_phij_case133(phi_bar_j, phi_bar_Mj, theta_M):
    return -np.sin(theta_M) * np.sin(phi_bar_Mj) * np.arctanh(np.cos(phi_bar_j))

# 134 ##############

def Hr_zk_case134(r, phi_bar_j, theta_M):
    t1 = np.sin(phi_bar_j)
    t1_coef = -np.cos(theta_M)
    t2 = np.sin(phi_bar_j) / np.sqrt(1.0 - np.cos(phi_bar_j))
    t2_coef = -np.sqrt(2.0) * np.cos(theta_M)
    t3 = np.log( r * (1.0 - np.cos(phi_bar_j) + np.sqrt(2.0) * np.sqrt(1.0 - np.cos(phi_bar_j))) )
    t3_coef = np.cos(theta_M) * np.sin(phi_bar_j)
    t4 = np.arctanh(np.sin(phi_bar_j) / ( np.sqrt(2.0) * np.sqrt(1.0 - np.cos(phi_bar_j)) ))
    t4_coef = np.cos(theta_M)
    return t1_coef * t1 + t2_coef * t2 + t3_coef * t3 + t4_coef * t4

def Hphi_zk_case134(phi_bar_j, theta_M):
    return np.sqrt(2) * np.cos(theta_M) * np.sqrt(1-np.cos(phi_bar_j)) + np.cos(theta_M) * np.cos(phi_bar_j) * np.arctanh(np.sqrt((1-np.cos(phi_bar_j))/2))

def Hz_ri_case134(phi_bar_j, phi_bar_M, theta_M):
    t1 = np.sqrt(1.0 - np.cos(phi_bar_j))
    t1_coef = np.sqrt(2.0) * np.sin(theta_M) * np.sin(phi_bar_M)
    t2 = np.sin(phi_bar_j) / t1
    t2_coef = -np.sqrt(2.0) * np.sin(theta_M) * np.cos(phi_bar_M)
    t3 = np.arctanh(t2 / np.sqrt(2.0))
    t3_coef = np.sin(theta_M) * np.cos(phi_bar_M)
    return t1_coef * t1 + t2_coef * t2 + t3_coef * t3

def Hz_phij_case134(phi_bar_j, phi_bar_Mj, theta_M):
    return np.sin(theta_M) * np.sin(phi_bar_Mj) * np.arctanh(np.sqrt((1.0-np.cos(phi_bar_j))/2.0))

# 135 ##############

def Hr_zk_case135(r, r_i, r_bar_i, phi_bar_j, theta_M):
    t1 = np.sin(phi_bar_j)
    t1_coef = -np.cos(theta_M)
    t2 = np.log( r_i - r*np.cos(phi_bar_j) + np.sqrt(r_i**2 + r**2 - 2*r_i*r*np.cos(phi_bar_j)) )
    t2_coef = np.cos(theta_M) * np.sin(phi_bar_j)
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r * r_i / r_bar_i**2)
    E_coef = np.cos(theta_M) * np.abs(r_bar_i) / r
    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r * r_i / r_bar_i**2)
    F_coef = -np.cos(theta_M) * (r**2 + r_i**2) / (r * np.abs(r_bar_i))
    return t1_coef * t1 + t2_coef * t2 + E_coef * E + F_coef * F

def Hphi_zk_case135(r, r_i, phi_bar_j, theta_M):
    t1 = np.sqrt(r**2 + r_i**2 - 2.0 * r * r_i * np.cos(phi_bar_j))
    t1_coef = np.cos(theta_M) / r
    t2 = np.arctanh((r * np.cos(phi_bar_j) - r_i) / t1)
    t2_coef = -np.cos(theta_M) * np.cos(phi_bar_j)
    return t1_coef * t1 + t2_coef * t2

def Hz_ri_case135(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M):
    t = r_bar_i**2
    t1 = np.sqrt(r**2 + r_i**2 - 2.0 * r * r_i * np.cos(phi_bar_j)) / r
    t1_coef = np.sin(theta_M) * np.sin(phi_bar_M)
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r * r_i / t)
    E_coef = np.sin(theta_M) * np.cos(phi_bar_M) * np.sqrt(t) / r
    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r * r_i / t)
    F_coef = -np.sin(theta_M) * np.cos(phi_bar_M) * (r**2 + r_i**2) / (r * np.sqrt(t))
    return t1_coef * t1 + E_coef * E + F_coef * F

def Hz_phij_case135(r, r_i, phi_bar_j, phi_bar_Mj, theta_M):
    t1 = np.arctanh( (r*np.cos(phi_bar_j)-r_i) / np.sqrt(r**2 + r_i**2 - 2*r*r_i*np.cos(phi_bar_j)) )
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_Mj)
    return t1_coef * t1

# 211 ##############

def Hr_phij_case211(phi_bar_M, theta_M, z_bar_k):
    return -np.sin(theta_M) * np.sin(phi_bar_M) * np.sign(z_bar_k) * np.log(np.abs(z_bar_k))

def Hz_zk_case211(phi_j, theta_M, z_bar_k):
    return -np.cos(theta_M) * np.sign(z_bar_k) * phi_j

# 212 ##############

def Hr_ri_case212(r_i, phi_j, phi_bar_M, theta_M, z_bar_k):
    t1 = np.sin(theta_M) * z_bar_k / np.sqrt(r_i**2 + z_bar_k**2)
    t2 = 1.0/2.0 * phi_j * np.cos(phi_bar_M)
    t3 = 1.0/4.0 * np.sin(phi_bar_M)
    return t1 * (t2 - t3)

def Hr_phij_case212(r_i, phi_bar_M, theta_M, z_bar_k):
    t1 = np.arctanh(z_bar_k / np.sqrt(r_i**2 + z_bar_k**2))
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_M)
    return t1_coef * t1

def Hphi_ri_case212(r_i, phi_j, phi_bar_M, theta_M, z_bar_k):
    t1 = np.sin(theta_M) * z_bar_k / np.sqrt(r_i**2 + z_bar_k**2)
    t2 = 1.0/4.0 * np.cos(phi_bar_M)
    t3 = 1.0/2.0 * phi_j * np.sin(phi_bar_M)
    return t1 * (-t2 + t3)

def Hphi_zk_case212(r_i, theta_M, z_bar_k):
    t1 = r_i / np.sqrt(r_i**2 + z_bar_k**2)
    t1_coef = -np.cos(theta_M)
    t2 = np.arctanh(t1)
    t2_coef = np.cos(theta_M)
    return t1_coef * t1 + t2_coef * t2

def Hz_ri_case212(r_i, phi_bar_M, theta_M, z_bar_k):
    t1 = r_i / np.sqrt(r_i**2 + z_bar_k**2)
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_M)
    return t1_coef * t1

def Hz_phij_case212(r_i, phi_bar_M, theta_M, z_bar_k):
    return np.sin(theta_M) * np.sin(phi_bar_M) * np.arctanh(r_i / np.sqrt(r_i**2 + z_bar_k**2))

def Hz_zk_case212(r_i, phi_j, theta_M, z_bar_k):
    t1 = phi_j / np.sqrt(r_i**2 + z_bar_k**2)
    t1_coef = -np.cos(theta_M) * z_bar_k
    return t1_coef * t1

# 213 ##############

def Hr_phij_case213(r, phi_bar_M, theta_M, z_bar_k):
    t1 = np.arctanh(z_bar_k / np.sqrt(r**2 + z_bar_k**2))
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_M)
    return t1_coef * t1

def Hphi_zk_case213(r, theta_M, z_bar_k):
    t1 = np.sqrt(r**2 + z_bar_k**2)
    t1_coef = np.cos(theta_M) / r
    t2 = np.arctanh(r / t1)
    t2_coef = -np.cos(theta_M)
    return t1_coef * t1 + t2_coef * t2

def Hz_phij_case213(r, phi_bar_M, theta_M, z_bar_k):
    t1 = np.arctanh(r / np.sqrt(r**2 + z_bar_k**2))
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_M)
    return t1_coef * t1

def Hz_zk_case213(phi_bar_j, theta_M, z_bar_k):
    t1 = np.sign(z_bar_k)
    t1_coef = np.cos(theta_M) * phi_bar_j
    return t1_coef * t1

# 214 ##############

def Hr_ri_case214(r, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r**2 / z_bar_k**2)
    E_coef = -np.sin(theta_M) * np.cos(phi_bar_M) * z_bar_k**2 * np.sign(z_bar_k) / ( 2.0 * r**2 )
    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r**2 / z_bar_k**2)
    F_coef = (np.sin(theta_M) * np.cos(phi_bar_M) * np.sign(z_bar_k) * (2.0 * r**2 + z_bar_k**2) / ( 2.0 * r**2 ))
    return -np.sin(theta_M) * np.sin(phi_bar_M) * np.sign(z_bar_k) * z_bar_k**2 / ( 2.0 * r**2) + E_coef * E + F_coef * F

def Hr_phij_case214(phi_bar_M, theta_M, z_bar_k):
    return -np.sin(theta_M) * np.sin(phi_bar_M) * np.sign(z_bar_k) * np.log(np.abs(z_bar_k))

def Hr_zk_case214(r, phi_bar_j, theta_M, z_bar_k):
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r**2 / z_bar_k**2)
    E_coef = np.cos(theta_M) * np.abs(z_bar_k) / r
    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r**2 / z_bar_k**2)
    F_coef = -np.cos(theta_M) * (2.0 * r**2 + z_bar_k**2) / ( r * np.abs(z_bar_k) )
    t = np.sqrt(r**2 + z_bar_k**2)
    def Pi1(sign):
        return el3_angle(phi_bar_j/2, 2*r/(r+sign*t), -4*r**2/z_bar_k**2)
    def Pi1_coef(sign):
        return -np.cos(theta_M) / ( r * np.sqrt((r**2 +  z_bar_k**2) * z_bar_k**2) ) * (t - sign * r) * (r + sign * t)**2
    def Pi2(sign):
        return el3_angle(phi_bar_j / 2.0, 1.0 - z_bar_k**4 / ( (4.0 * r**2 + z_bar_k**2) * (r + sign * t)**2 ), 4.0 * r**2 / ( 4.0 * r**2 + z_bar_k**2 ))
    def Pi2_coef(sign):
        return sign * np.cos(theta_M) * z_bar_k**4 / ( r * np.sqrt((r**2 + z_bar_k**2) * (4.0*r**2 + z_bar_k**2)) * (r + sign * t) )
    return E_coef * E + F_coef * F + Pi1_coef(1) * Pi1(1) + Pi1_coef(-1) * Pi1(-1) + Pi2_coef(1) * Pi2(1) + Pi2_coef(-1) * Pi2(-1)

def Hphi_ri_case214(r, phi_j, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t1 = -np.sin(theta_M) * np.cos(phi_bar_M) * np.sign(z_bar_k) / 2.0
    t2 = phi_j
    t2_coef = np.sin(theta_M) * np.sin(phi_bar_M) * np.sign(z_bar_k) / 2.0
    t3 = np.sign(z_bar_k) * z_bar_k**2 / (2.0 * r**2)
    t3_coef = -np.sin(theta_M) * np.cos(phi_bar_M)
    t4 = np.log(np.abs(z_bar_k) / (np.sqrt(2.0) * r))
    t4_coef = -np.sin(theta_M) * np.cos(phi_bar_M) * np.sign(z_bar_k)
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r**2 / z_bar_k**2)
    E_coef = np.sin(theta_M) * np.sin(phi_bar_M) * z_bar_k**2 * np.sign(z_bar_k) / ( 2.0 * r**2 )
    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r**2 / z_bar_k**2)
    F_coef = (-np.sin(theta_M) * np.sin(phi_bar_M) * np.sign(z_bar_k) * (4.0*r**2 + z_bar_k**2) / (2.0*r**2 ))
    return t1 + t2_coef * t2 + t3_coef * t3 + t4_coef * t4 + E_coef * E + F_coef * F

def Hphi_zk_case214(r, theta_M, z_bar_k):
    t1 = np.abs(z_bar_k)
    t1_coef = np.cos(theta_M) / r
    return t1_coef * t1

def Hz_ri_case214(r, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r**2 / z_bar_k**2)
    E_coef = np.sin(theta_M) * np.cos(phi_bar_M) * np.abs(z_bar_k) / r
    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r**2 / z_bar_k**2)
    F_coef = -np.sin(theta_M) * np.cos(phi_bar_M) * (2.0*r**2 + z_bar_k**2) / (r*np.abs(z_bar_k))
    return np.sin(theta_M) * np.sin(phi_bar_M) * np.abs(z_bar_k) / r + E_coef * E + F_coef * F

def Hz_zk_case214(r, phi_bar_j, theta_M, z_bar_k):
    t = np.sqrt(r**2 + z_bar_k**2)
    def Pi(sign):
        return el3_angle(phi_bar_j/2, 2*r/(r+sign*t), -4*r**2/z_bar_k**2)
    Pi_coef = np.cos(theta_M) * np.sign(z_bar_k)
    return Pi_coef * Pi(1) + Pi_coef * Pi(-1)

# 215 ##############

def Hr_ri_case215(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t2 = np.arctanh(z_bar_k / np.sqrt(r_bar_i**2 + z_bar_k**2))
    t2_coef = np.sin(theta_M) * np.sin(phi_bar_M) / 2.0 * (1.0 - r_i**2 / r**2)
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    E_coef = -np.sin(theta_M) * np.cos(phi_bar_M) * z_bar_k * np.sqrt(r_bar_i**2 + z_bar_k**2) / (2*r**2)
    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    F_coef = np.sin(theta_M) * np.cos(phi_bar_M) * z_bar_k * (2.0 * r_i**2 + z_bar_k**2) / (2*r**2 * np.sqrt(r_bar_i**2 + z_bar_k**2) )
    Pi = el3_angle(phi_bar_j/2.0, -4.0 * r * r_i / r_bar_i**2, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    Pi_coef = np.sin(theta_M) * np.cos(phi_bar_M) * z_bar_k * (r**2 + r_i**2) * (r + r_i) / (2.0 * r**2 * r_bar_i * np.sqrt(r_bar_i**2 + z_bar_k**2) )
    return -np.sin(theta_M) * np.sin(phi_bar_M) * z_bar_k * np.sqrt(r_bar_i**2 + z_bar_k**2) / (2.0 * r**2) + t2_coef * t2 + E_coef * E + F_coef * F + Pi_coef * Pi

def Hr_phij_case215(r_bar_i, phi_bar_M, theta_M, z_bar_k):
    t1 = np.arctanh(z_bar_k / np.sqrt(r_bar_i**2 + z_bar_k**2))
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_M)
    return t1_coef * t1

def Hr_zk_case215(r, r_i, r_bar_i, phi_bar_j, theta_M, z_bar_k):
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    E_coef = np.cos(theta_M) * np.sqrt(r_bar_i**2 + z_bar_k**2) / r
    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    F_coef = -np.cos(theta_M) * (r**2 + r_i**2 + z_bar_k**2) / ( r * np.sqrt(r_bar_i**2 + z_bar_k**2) )
    t = np.sqrt(r**2 + z_bar_k**2)
    def Pi1(sign):
        return el3_angle(phi_bar_j/2.0, 2.0 * r / (r + sign * t), -4*r*r_i / (r_bar_i**2 + z_bar_k**2))
    def Pi1_coef(sign):
        return -np.cos(theta_M) / ( r * np.sqrt((r**2+z_bar_k**2) * (r_bar_i**2+z_bar_k**2)) ) * (t - sign * r) * (r_i + sign * t)**2
    def Pi2(sign):
        return el3_angle(phi_bar_j / 2.0, 1.0 - z_bar_k**2 * (r_bar_i**2 + z_bar_k**2) / (((r + r_i)**2 + z_bar_k**2) * (r + sign * t)**2 ), 4*r*r_i / ((r + r_i)**2 + z_bar_k**2 ))
    def Pi2_coef(sign):
        return sign * np.cos(theta_M) * z_bar_k**2 * (r_bar_i**2 + z_bar_k**2) / ( r * np.sqrt((r**2 + z_bar_k**2) * ((r + r_i)**2 + z_bar_k**2)) * (r + sign * t) )
    return E_coef*E + F_coef*F + Pi1_coef(1)*Pi1(1) + Pi1_coef(-1)*Pi1(-1) + Pi2_coef(1)*Pi2(1) + Pi2_coef(-1)*Pi2(-1)


def Hphi_ri_case215(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t1 = np.sqrt(r_bar_i**2 + z_bar_k**2) * z_bar_k / (2.0 * r**2)
    t1_coef = -np.sin(theta_M) * np.cos(phi_bar_M)
    t2 = np.arctanh(z_bar_k / np.sqrt(r_bar_i**2 + z_bar_k**2))
    t2_coef = -np.sin(theta_M) * np.cos(phi_bar_M) * (r**2 + r_i**2) / ( 2.0 * r**2 )
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    E_coef = np.sin(theta_M) * np.sin(phi_bar_M) * z_bar_k * np.sqrt(r_bar_i**2+z_bar_k**2) / (2*r**2)
    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    F_coef = -np.sin(theta_M) * np.sin(phi_bar_M) * z_bar_k  * (2.0 * r**2 + 2.0 * r_i**2 + z_bar_k**2) / ( 2.0 * r**2 * np.sqrt(r_bar_i**2 + z_bar_k**2) )
    Pi = el3_angle(phi_bar_j/2.0, -4.0 * r * r_i / r_bar_i**2, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    Pi_coef = np.sin(theta_M) * np.sin(phi_bar_M) * z_bar_k * (r + r_i)**2 / ( 2.0 * r**2 * np.sqrt(r_bar_i**2 + z_bar_k**2) )
    return t1_coef * t1 + t2_coef * t2 + E_coef * E + F_coef * F + Pi_coef * Pi

def Hphi_zk_case215(r, r_bar_i, theta_M, z_bar_k):
    t1 = np.sqrt(r_bar_i**2 + z_bar_k**2)
    t1_coef = np.cos(theta_M) / r
    t2 = np.arctanh(r_bar_i / t1)
    t2_coef = -np.cos(theta_M)
    return t1_coef * t1 + t2_coef * t2

def Hz_ri_case215(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t = r_bar_i**2 + z_bar_k**2
    t1 = np.sqrt(r_bar_i**2 + z_bar_k**2) / r
    t1_coef = np.sin(theta_M) * np.sin(phi_bar_M)
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r * r_i / t)
    E_coef = np.sin(theta_M) * np.cos(phi_bar_M) * np.sqrt(t) / r
    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r * r_i / t)
    F_coef = -np.sin(theta_M) * np.cos(phi_bar_M) * (r**2 + r_i**2 + z_bar_k**2) / (r * np.sqrt(t))
    return t1_coef * t1 + E_coef * E + F_coef * F

def Hz_phij_case215(r_bar_i, phi_bar_M, theta_M, z_bar_k):
    t1 = np.arctanh(r_bar_i / np.sqrt(r_bar_i**2 + z_bar_k**2))
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_M)
    return t1_coef * t1

def Hz_zk_case215(r, r_i, r_bar_i, phi_bar_j, theta_M, z_bar_k):
    t = np.sqrt(r**2 + z_bar_k**2)
    def Pi(sign):
        return el3_angle(phi_bar_j/2.0, 2.0 * r / (r + sign * t), -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    def Pi_coef(sign):
        return np.cos(theta_M) * z_bar_k * (r_i + sign * t) / ( np.sqrt(r_bar_i**2 + z_bar_k**2) * (r + sign * t) )
    return Pi_coef(1) * Pi(1) + Pi_coef(-1) * Pi(-1)

# 221 ##############

def Hr_phij_case221(phi_bar_M, theta_M, z_bar_k):
    return -np.sin(theta_M) * np.sin(phi_bar_M) * np.sign(z_bar_k) * np.log(np.abs(z_bar_k))

def Hz_zk_case221(phi_j, theta_M, z_bar_k):
    return -np.cos(theta_M) * np.sign(z_bar_k) * phi_j

# 222 ##############

def Hr_ri_case222(r_i, phi_j, phi_bar_M, theta_M, z_bar_k):
    t1 = np.sin(theta_M) * z_bar_k / np.sqrt(r_i**2 + z_bar_k**2)
    t2 = 1.0/2.0 * phi_j * np.cos(phi_bar_M)
    t3 = 1.0/4.0 * np.sin(phi_bar_M)
    return t1 * (t2 - t3)

def Hr_phij_case222(r_i, phi_bar_M, theta_M, z_bar_k):
    t1 = np.arctanh(z_bar_k / np.sqrt(r_i**2 + z_bar_k**2))
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_M)
    return t1_coef * t1

def Hphi_ri_case222(r_i, phi_j, phi_bar_M, theta_M, z_bar_k):
    t1 = np.sin(theta_M) * z_bar_k / np.sqrt(r_i**2 + z_bar_k**2)
    t2 = 1.0/4.0 * np.cos(phi_bar_M)
    t3 = 1.0/2.0 * phi_j * np.sin(phi_bar_M)
    return t1 * (-t2 + t3)

def Hphi_zk_case222(r_i, theta_M, z_bar_k):
    t1 = r_i / np.sqrt(r_i**2 + z_bar_k**2)
    t1_coef = np.cos(theta_M)
    t2 = np.arctanh(t1)
    t2_coef = -np.cos(theta_M)
    return t1_coef * t1 + t2_coef * t2

def Hz_ri_case222(r_i, phi_bar_M, theta_M, z_bar_k):
    t1 = r_i / np.sqrt(r_i**2 + z_bar_k**2)
    t1_coef = np.sin(theta_M) * np.sin(phi_bar_M)
    return t1_coef * t1

def Hz_phij_case222(r_i, phi_bar_M, theta_M, z_bar_k):
    t1 = np.arctanh(r_i / np.sqrt(r_i**2 + z_bar_k**2))
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_M)
    return t1_coef * t1

def Hz_zk_case222(r_i, phi_j, theta_M, z_bar_k):
    t1 = z_bar_k / np.sqrt(r_i**2 + z_bar_k**2)
    t1_coef = -np.cos(theta_M) * phi_j
    return t1_coef * t1

# 223 ##############

def Hr_phij_case223(r, phi_bar_M, theta_M, z_bar_k):
    t1 = np.arctanh(z_bar_k / np.sqrt(r**2 + z_bar_k**2))
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_M)
    return t1_coef * t1

def Hphi_zk_case223(r, theta_M, z_bar_k):
    t1 = np.sqrt(r**2 + z_bar_k**2)
    t1_coef = np.cos(theta_M) / r
    t2 = np.arctanh(r / t1)
    t2_coef = -np.cos(theta_M)
    return t1_coef * t1 + t2_coef * t2

def Hz_phij_case223(r, phi_bar_M, theta_M, z_bar_k):
    t1 = np.arctanh(r / np.sqrt(r**2 + z_bar_k**2))
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_M)
    return t1_coef * t1

def Hz_zk_case223(r, phi_bar_j, theta_M, z_bar_k):
    t1 = arctan_k_tan_2(np.sqrt(r**2 + z_bar_k**2) / np.abs(z_bar_k), 2.0 * phi_bar_j)
    t1_coef = np.cos(theta_M) * np.sign(z_bar_k)
    return t1_coef * t1

# 224 ##############

def Hr_ri_case224(r, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t1 = np.sqrt(4.0 * r**2 + z_bar_k**2) * z_bar_k / (2.0 * r**2)
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_M)
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r**2 / z_bar_k**2)
    E_coef = -np.sin(theta_M) * np.cos(phi_bar_M) * z_bar_k**2 * np.sign(z_bar_k) / ( 2.0 * r**2 )
    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r**2 / z_bar_k**2)
    F_coef = np.sin(theta_M) * np.cos(phi_bar_M) * np.sign(z_bar_k) * (2.0 * r**2 + z_bar_k**2) / ( 2.0 * r**2 )
    return t1_coef * t1 + E_coef * E + F_coef * F

def Hr_phij_case224(r, phi_bar_M, theta_M, z_bar_k):
    t1 = np.arctanh(z_bar_k / np.sqrt(4.0 * r**2 + z_bar_k**2))
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_M)
    return t1_coef * t1

def Hr_zk_case224(r, phi_bar_j, theta_M, z_bar_k):
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r**2 / z_bar_k**2)
    E_coef = np.cos(theta_M) * np.abs(z_bar_k) / r
    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r**2 / z_bar_k**2)
    F_coef = -np.cos(theta_M) * (2.0 * r**2 + z_bar_k**2) / ( r * np.abs(z_bar_k) )
    t = np.sqrt(r**2 + z_bar_k**2)
    def Pi1(sign):
        return el3_angle(phi_bar_j/2.0, 2.0 * r / (r + sign * t), -4.0 * r**2 / z_bar_k**2)
    def Pi1_coef(sign):
        return -np.cos(theta_M) / ( r * np.sqrt((r**2 + z_bar_k**2) * z_bar_k**2) ) * (t - sign * r) * (r + sign * t)**2
    def Pi2(sign):
        return el3_angle(phi_bar_j / 2.0, 1.0 - z_bar_k**4 / ( (4.0 * r**2 + z_bar_k**2) * (r + sign * t)**2 ), 4.0 * r**2 / ( 4.0 * r**2 + z_bar_k**2 ))
    def Pi2_coef(sign):
        return sign * np.cos(theta_M) * z_bar_k**4 / ( r * np.sqrt((r**2 + z_bar_k**2) * (4.0 * r**2 + z_bar_k**2)) * (r + sign * t) )
    return E_coef * E + F_coef * F + Pi1_coef(1) * Pi1(1) + Pi1_coef(-1) * Pi1(-1) + Pi2_coef(1) * Pi2(1) + Pi2_coef(-1) * Pi2(-1)

def Hphi_ri_case224(r, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t1 = np.sqrt(4.0 * r**2 + z_bar_k**2) * z_bar_k / (2.0 * r**2)
    t1_coef = -np.sin(theta_M) * np.cos(phi_bar_M)
    t2 = np.arctanh(z_bar_k / np.sqrt(4.0 * r**2 + z_bar_k**2))
    t2_coef = -np.sin(theta_M) * np.cos(phi_bar_M)
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r**2 / z_bar_k**2)
    E_coef = np.sin(theta_M) * np.sin(phi_bar_M) * z_bar_k**2 * np.sign(z_bar_k) / ( 2.0 * r**2 )
    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r**2 / z_bar_k**2)
    F_coef = -np.sin(theta_M) * np.sin(phi_bar_M) * np.sign(z_bar_k) * (4.0 * r**2 + z_bar_k**2) / ( 2.0 * r**2 )
    return t1_coef * t1 + t2_coef * t2 + E_coef * E + F_coef * F

def Hphi_zk_case224(r, theta_M, z_bar_k):
    t1 = np.sqrt(4.0 * r**2 + z_bar_k**2)
    t1_coef = np.cos(theta_M) / r
    t2 = np.arctanh(2.0 * r / t1)
    t2_coef = -np.cos(theta_M)
    return t1_coef * t1 + t2_coef * t2

def Hz_ri_case224(r, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t1 = np.sqrt(4.0 * r**2 + z_bar_k**2) / r
    t1_coef = np.sin(theta_M) * np.sin(phi_bar_M)
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r**2 / z_bar_k**2)
    E_coef = np.sin(theta_M) * np.cos(phi_bar_M) * np.abs(z_bar_k) / r
    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r**2 / z_bar_k**2)
    F_coef = -np.sin(theta_M) * np.cos(phi_bar_M) * (2.0 * r**2 + z_bar_k**2) / (r * np.abs(z_bar_k))
    return t1_coef * t1 + E_coef * E + F_coef * F

def Hz_phij_case224(r, phi_bar_M, theta_M, z_bar_k):
    t1 = np.arctanh(2.0 * r / np.sqrt(4.0 * r**2 + z_bar_k**2))
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_M)
    return t1_coef * t1

def Hz_zk_case224(r, phi_bar_j, theta_M, z_bar_k):
    t = np.sqrt(r**2 + z_bar_k**2)
    def Pi(sign):
        return el3_angle(phi_bar_j/2.0, 2.0 * r / (r + sign * t), -4.0 * r**2 / z_bar_k**2)
    Pi_coef = np.cos(theta_M) * np.sign(z_bar_k)
    return Pi_coef * Pi(1) + Pi_coef * Pi(-1)

# 225 ##############

def Hr_ri_case225(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t1 = np.sqrt((r + r_i)**2 + z_bar_k**2) * z_bar_k / (2.0 * r**2)
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_M)
    t2 = np.arctanh(z_bar_k / np.sqrt((r + r_i)**2 + z_bar_k**2))
    t2_coef = np.sin(theta_M) * np.sin(phi_bar_M) / 2.0 * (1.0 - r_i**2 / r**2)
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    E_coef = -np.sin(theta_M) * np.cos(phi_bar_M) * z_bar_k * np.sqrt(r_bar_i**2 + z_bar_k**2) / ( 2.0 * r**2 )
    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    F_coef = np.sin(theta_M) * np.cos(phi_bar_M) * z_bar_k * (2.0 * r_i**2 + z_bar_k**2) / ( 2.0 * r**2 * np.sqrt(r_bar_i**2 + z_bar_k**2) )
    Pi = el3_angle(phi_bar_j/2.0, -4.0 * r * r_i / r_bar_i**2, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    Pi_coef = np.sin(theta_M) * np.cos(phi_bar_M) * z_bar_k * (r**2 + r_i**2) * (r + r_i) / ( 2.0 * r**2 * r_bar_i * np.sqrt(r_bar_i**2 + z_bar_k**2) )
    return t1_coef * t1 + t2_coef * t2 + E_coef * E + F_coef * F + Pi_coef * Pi

def Hr_phij_case225(r, r_i, phi_bar_M, theta_M, z_bar_k):
    t1 = np.arctanh(z_bar_k / np.sqrt((r + r_i)**2 + z_bar_k**2))
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_M)
    return t1_coef * t1

def Hr_zk_case225(r, r_i, r_bar_i, phi_bar_j, theta_M, z_bar_k):
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    E_coef = np.cos(theta_M) * np.sqrt(r_bar_i**2 + z_bar_k**2) / r
    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    F_coef = -np.cos(theta_M) * (r**2 + r_i**2 + z_bar_k**2) / ( r * np.sqrt(r_bar_i**2 + z_bar_k**2) )
    t = np.sqrt(r**2 + z_bar_k**2)
    def Pi1(sign):
        return el3_angle(phi_bar_j/2.0, 2.0 * r / (r + sign * t), -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    def Pi1_coef(sign):
        return -np.cos(theta_M) / ( r * np.sqrt((r**2 +  z_bar_k**2) * (r_bar_i**2 + z_bar_k**2)) ) * (t - sign * r) * (r_i + sign * t)**2
    def Pi2(sign):
        return el3_angle(phi_bar_j / 2.0, 1.0 - z_bar_k**2 * (r_bar_i**2 + z_bar_k**2) / ( ((r + r_i)**2 + z_bar_k**2) * (r + sign * t)**2 ), 4.0 * r * r_i / ( (r + r_i)**2 + z_bar_k**2 ))
    def Pi2_coef(sign):
        return sign * np.cos(theta_M) * z_bar_k**2 * (r_bar_i**2 + z_bar_k**2) / ( r * np.sqrt((r**2 + z_bar_k**2) * ((r + r_i)**2 + z_bar_k**2)) * (r + sign * t) )
    return E_coef * E + F_coef * F + Pi1_coef(1) * Pi1(1) + Pi1_coef(-1) * Pi1(-1) + Pi2_coef(1) * Pi2(1) + Pi2_coef(-1) * Pi2(-1)

def Hphi_ri_case225(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t1 = np.sqrt((r + r_i)**2 + z_bar_k**2) * z_bar_k / (2.0 * r**2)
    t1_coef = -np.sin(theta_M) * np.cos(phi_bar_M)
    t2 = np.arctanh(z_bar_k / np.sqrt((r + r_i)**2 + z_bar_k**2))
    t2_coef = -np.sin(theta_M) * np.cos(phi_bar_M) * (r**2 + r_i**2)  / ( 2.0 * r**2 )
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    E_coef = np.sin(theta_M) * np.sin(phi_bar_M) * z_bar_k * np.sqrt(r_bar_i**2 + z_bar_k**2) / ( 2.0 * r**2 )
    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    F_coef = -np.sin(theta_M) * np.sin(phi_bar_M) * z_bar_k  * (2.0 * r**2 + 2.0 * r_i**2 + z_bar_k**2) / ( 2.0 * r**2 * np.sqrt(r_bar_i**2 + z_bar_k**2) )
    Pi = el3_angle(phi_bar_j/2.0, -4.0 * r * r_i / r_bar_i**2, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    Pi_coef = np.sin(theta_M) * np.sin(phi_bar_M) * z_bar_k * (r + r_i)**2 / ( 2.0 * r**2 * np.sqrt(r_bar_i**2 + z_bar_k**2) )
    return t1_coef * t1 + t2_coef * t2 + E_coef * E + F_coef * F + Pi_coef * Pi

def Hphi_zk_case225(r, r_i, theta_M, z_bar_k):
    t1 = np.sqrt((r + r_i)**2 + z_bar_k**2)
    t1_coef = np.cos(theta_M) / r
    t2 = np.arctanh((r + r_i) / t1)
    t2_coef = -np.cos(theta_M)
    return t1_coef * t1 + t2_coef * t2

def Hz_ri_case225(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t = r_bar_i**2 + z_bar_k**2
    t1 = np.sqrt((r + r_i)**2 + z_bar_k**2) / r
    t1_coef = np.sin(theta_M) * np.sin(phi_bar_M)
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r * r_i / t)
    E_coef = np.sin(theta_M) * np.cos(phi_bar_M) * np.sqrt(t) / r
    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r * r_i / t)
    F_coef = -np.sin(theta_M) * np.cos(phi_bar_M) * (r**2 + r_i**2 + z_bar_k**2) / (r * np.sqrt(t))
    return t1_coef * t1 + E_coef * E + F_coef * F

def Hz_phij_case225(r, r_i, phi_bar_M, theta_M, z_bar_k):
    t1 = np.arctanh((r + r_i) / np.sqrt((r + r_i)**2 + z_bar_k**2))
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_M)
    return t1_coef * t1

def Hz_zk_case225(r, r_i, r_bar_i, phi_bar_j, theta_M, z_bar_k):
    t = np.sqrt(r**2 + z_bar_k**2)
    def Pi(sign):
        return el3_angle(phi_bar_j/2.0, 2.0 * r / (r + sign * t), -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    def Pi_coef(sign):
        return np.cos(theta_M) * z_bar_k * (r_i + sign * t) / ( np.sqrt(r_bar_i**2 + z_bar_k**2) * (r + sign * t) )
    return Pi_coef(1) * Pi(1) + Pi_coef(-1) * Pi(-1)

# 231 ##############

def Hr_phij_case231(phi_bar_j, phi_bar_Mj, theta_M, z_bar_k):
    return -np.sin(theta_M) * np.sin(phi_bar_Mj) * np.cos(phi_bar_j) * np.sign(z_bar_k) * np.log(np.abs(z_bar_k))

def Hphi_phij_case231(phi_bar_j, phi_bar_Mj, theta_M, z_bar_k):
    t1 = np.log(np.abs(z_bar_k))
    t1_coef = np.sin(theta_M) * np.sin(phi_bar_Mj) * np.sin(phi_bar_j) * np.sign(z_bar_k)
    return t1_coef * t1

def Hz_zk_case231(phi_j, theta_M, z_bar_k):
    t1 = phi_j * np.sign(z_bar_k)
    t1_coef = -np.cos(theta_M)
    return t1_coef * t1

# 232 ##############

def Hr_ri_case232(r_i, phi_j, phi_bar_j, phi_bar_M, phi_bar_Mj, theta_M, z_bar_k):
    t1 = np.sin(theta_M) * z_bar_k / np.sqrt(r_i**2 + z_bar_k**2)
    t2 = 1.0/2.0 * phi_j * np.cos(phi_bar_M)
    t3 = 1.0/4.0 * np.sin(phi_bar_Mj + phi_bar_j)
    return t1 * (t2 - t3)

def Hr_phij_case232(r_i, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k):
    t1 = np.arctanh(z_bar_k / np.sqrt(r_i**2 + z_bar_k**2))
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_Mj) * np.cos(phi_bar_j)
    return t1_coef * t1

def Hr_zk_case232(r_i, phi_bar_j, theta_M, z_bar_k):
    t1 = r_i / np.sqrt(r_i**2 + z_bar_k**2)
    t1_coef = -np.cos(theta_M) * np.sin(phi_bar_j)
    t2 = np.arctanh(t1)
    t2_coef = np.cos(theta_M) * np.sin(phi_bar_j)
    return t1_coef * t1 + t2_coef * t2

def Hphi_ri_case232(r_i, phi_j, phi_bar_j, phi_bar_M, phi_bar_Mj, theta_M, z_bar_k):
    t1 = np.sin(theta_M) * z_bar_k / np.sqrt(r_i**2 + z_bar_k**2)
    t2 = 1.0/4.0 * np.cos(phi_bar_Mj + phi_bar_j)
    t3 = 1.0/2.0 * phi_j * np.sin(phi_bar_M)
    return t1 * (-t2 + t3)

def Hphi_phij_case232(r_i, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k):
    t1 = np.arctanh(z_bar_k / np.sqrt(r_i**2 + z_bar_k**2))
    t1_coef = np.sin(theta_M) * np.sin(phi_bar_Mj) * np.sin(phi_bar_j)
    return t1_coef * t1

def Hphi_zk_case232(r_i, phi_bar_j, theta_M, z_bar_k):
    t1 = r_i / np.sqrt(r_i**2 + z_bar_k**2)
    t1_coef = -np.cos(theta_M) * np.cos(phi_bar_j)
    t2 = np.arctanh(t1)
    t2_coef = np.cos(theta_M) * np.cos(phi_bar_j)
    return t1_coef * t1 + t2_coef * t2

def Hz_ri_case232(r_i, phi_bar_Mj, theta_M, z_bar_k):
    t1 = r_i / np.sqrt(r_i**2 + z_bar_k**2)
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_Mj)
    return t1_coef * t1

def Hz_phij_case232(r_i, phi_bar_Mj, theta_M, z_bar_k):
    t1 = np.arctanh(r_i / np.sqrt(r_i**2 + z_bar_k**2))
    t1_coef = np.sin(theta_M) * np.sin(phi_bar_Mj)
    return t1_coef * t1

def Hz_zk_case232(r_i, phi_j, theta_M, z_bar_k):
    t1 = z_bar_k / np.sqrt(r_i**2 + z_bar_k**2)
    t1_coef = -np.cos(theta_M) * phi_j
    return t1_coef * t1

# 233 ##############

def Hr_phij_case233(r, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k):
    t1 = np.arctanh(z_bar_k / np.sqrt(r**2 + z_bar_k**2))
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_Mj) * np.cos(phi_bar_j)
    t2 = np.arctan(z_bar_k * np.cos(phi_bar_j) / np.sin(phi_bar_j) / np.sqrt(r**2 + z_bar_k**2))
    t2_coef = np.sin(theta_M) * np.sin(phi_bar_Mj) * np.sin(phi_bar_j)
    return t1_coef * t1 + t2_coef * t2

def Hr_zk_case233(r, phi_bar_j, theta_M, z_bar_k):
    t = np.sqrt(r**2 + z_bar_k**2)
    t1 = np.sin(phi_bar_j)
    t1_coef = -np.cos(theta_M)
    t2 = np.log(-r * np.cos(phi_bar_j) + t)
    t2_coef = np.cos(theta_M) * np.sin(phi_bar_j)
    t3 = np.arctan(r * np.sin(phi_bar_j) / z_bar_k)
    t3_coef = np.cos(theta_M) * z_bar_k / r
    t4 = arctan_k_tan_2(t / np.abs(z_bar_k), 2.0 * phi_bar_j)
    t4_coef = -t3_coef
    def t5(sign):
        return arctan_k_tan_2(np.abs(z_bar_k) / np.abs(r + sign * t), phi_bar_j)
    t5_coef = t3_coef
    return t1_coef * t1 + t2_coef * t2 + t3_coef * t3 + t4_coef * t4 + t5_coef * t5(1) + t5_coef * t5(-1)

def Hphi_phij_case233(r, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k):
    t = np.sqrt(r**2 + z_bar_k**2)
    t1 = np.arctan(z_bar_k * np.cos(phi_bar_j) / ( np.sin(phi_bar_j) * t ))
    t1_coef = np.sin(theta_M) * np.sin(phi_bar_Mj) * np.cos(phi_bar_j)
    t2 = np.arctanh(z_bar_k / t)
    t2_coef = np.sin(theta_M) * np.sin(phi_bar_Mj) * np.sin(phi_bar_j)
    return t1_coef * t1 + t2_coef * t2

def Hphi_zk_case233(r, phi_bar_j, theta_M, z_bar_k):
    t1 = np.sqrt(r**2 + z_bar_k**2)
    t1_coef = np.cos(theta_M) / r
    t2 = np.arctanh(r * np.cos(phi_bar_j) / t1)
    t2_coef = -np.cos(theta_M) * np.cos(phi_bar_j)
    return t1_coef * t1 + t2_coef * t2

def Hz_phij_case233(r, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k):
    t1 = np.arctanh(r * np.cos(phi_bar_j) / np.sqrt(r**2 + z_bar_k**2))
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_Mj)
    return t1_coef * t1

def Hz_zk_case233(r, phi_bar_j, theta_M, z_bar_k):
    t1 = arctan_k_tan_2(np.sqrt(r**2 + z_bar_k**2) / np.abs(z_bar_k), 2.0 * phi_bar_j)
    t1_coef = np.cos(theta_M) * np.sign(z_bar_k)
    return t1_coef * t1

# 234 ##############

def Hr_ri_case234(r, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t1 = np.sqrt(2.0 * r**2 * (1.0 - np.cos(phi_bar_j)) + z_bar_k**2)
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_M) * z_bar_k / ( 2.0 * r**2 )
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r**2 / z_bar_k**2)
    E_coef = -np.sin(theta_M) * np.cos(phi_bar_M) * z_bar_k**2 * np.sign(z_bar_k) / ( 2.0 * r**2 )
    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r**2 / z_bar_k**2)
    F_coef = np.sin(theta_M) * np.cos(phi_bar_M) * np.sign(z_bar_k) * (2.0 * r**2 + z_bar_k**2) / ( 2.0 * r**2 )
    return t1_coef * t1 + E_coef * E + F_coef * F

def Hr_phij_case234(r, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k):
    t1 = np.arctanh(z_bar_k / np.sqrt(2.0 * r**2 * (1.0 - np.cos(phi_bar_j)) + z_bar_k**2))
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_Mj) * np.cos(phi_bar_j)
    t2 = np.arctan(z_bar_k * (1.0 - np.cos(phi_bar_j)) / ( np.sin(phi_bar_j) * np.sqrt(2.0 * r**2 * (1.0 - np.cos(phi_bar_j)) + z_bar_k**2) ))
    t2_coef = -np.sin(theta_M) * np.sin(phi_bar_Mj) * np.sin(phi_bar_j)
    return t1_coef * t1 + t2_coef * t2

def Hr_zk_case234(r, phi_bar_j, theta_M, z_bar_k):
    t1 = np.sin(phi_bar_j)
    t1_coef = -np.cos(theta_M)
    t2 = np.log( r * (1.0 - np.cos(phi_bar_j)) + np.sqrt(2.0 * r**2 * (1.0 - np.cos(phi_bar_j)) + z_bar_k**2) )
    t2_coef = np.cos(theta_M) * np.sin(phi_bar_j)
    t3 = np.arctan(r * np.sin(phi_bar_j) / z_bar_k)
    t3_coef = np.cos(theta_M) * z_bar_k / r
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r**2 / z_bar_k**2)
    E_coef = np.cos(theta_M) * np.abs(z_bar_k) / r
    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r**2/ z_bar_k**2)
    F_coef = -np.cos(theta_M) * (2.0 * r**2 + z_bar_k**2) / ( r * np.abs(z_bar_k) )
    t = np.sqrt(r**2 + z_bar_k**2)
    def Pi1(sign):
        return el3_angle(phi_bar_j/2.0, 2.0 * r / (r + sign * t), -4.0 * r**2 / z_bar_k**2)
    def Pi1_coef(sign):
        return -np.cos(theta_M) / ( r * np.sqrt((r**2 + z_bar_k**2) * z_bar_k**2) ) * (t - sign * r) * (r + sign * t)**2
    def Pi2(sign):
        return el3_angle(arctan_k_tan_2(np.sqrt((4.0 * r**2 + z_bar_k**2)/z_bar_k**2),phi_bar_j), 1.0 - z_bar_k**4 / ( (4.0 * r**2 + z_bar_k**2) * (r + sign * t)**2 ), 4.0 * r**2 / ( 4.0 * r**2 + z_bar_k**2 ))
    def Pi2_coef(sign):
        return sign * np.cos(theta_M) * z_bar_k**4 / ( r * np.sqrt((r**2 + z_bar_k**2) * (4.0 * r**2 + z_bar_k**2)) * (r + sign * t) )
    return t1_coef * t1 + t2_coef * t2 + t3_coef * t3 + E_coef * E + F_coef * F + Pi1_coef(1) * Pi1(1) + Pi1_coef(-1) * Pi1(-1) + Pi2_coef(1) * Pi2(1) + Pi2_coef(-1) * Pi2(-1)

def Hphi_ri_case234(r, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t1 = np.sqrt(2.0 * r**2 * (1.0 - np.cos(phi_bar_j)) + z_bar_k**2)
    t1_coef = -np.sin(theta_M) * np.cos(phi_bar_M) * z_bar_k / ( 2.0 * r**2 )
    t2 = np.arctanh(z_bar_k / np.sqrt(2.0 * r**2 * (1.0 - np.cos(phi_bar_j)) + z_bar_k**2))
    t2_coef = -np.sin(theta_M) * np.cos(phi_bar_M)
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r**2 / z_bar_k**2)
    E_coef = np.sin(theta_M) * np.sin(phi_bar_M) * z_bar_k**2 * np.sign(z_bar_k) / ( 2.0 * r**2 )
    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r**2 / z_bar_k**2)
    F_coef = -np.sin(theta_M) * np.sin(phi_bar_M) * np.sign(z_bar_k) * (4.0 * r**2 + z_bar_k**2) / ( 2.0 * r**2 )
    return t1_coef * t1 + t2_coef * t2 + E_coef * E + F_coef * F

def Hphi_phij_case234(r, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k):
    t = np.sqrt(2.0 * r**2 * (1.0 - np.cos(phi_bar_j)) + z_bar_k**2)
    t1 = np.arctan(z_bar_k * (1.0 - np.cos(phi_bar_j)) / ( np.sin(phi_bar_j) * t ))
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_Mj) * np.cos(phi_bar_j)
    t2 = np.arctanh(z_bar_k / t)
    t2_coef = np.sin(theta_M) * np.sin(phi_bar_Mj) * np.sin(phi_bar_j)
    return t1_coef * t1 + t2_coef * t2

def Hphi_zk_case234(r, phi_bar_j, theta_M, z_bar_k):
    t1 = np.sqrt(2.0 * r**2 * (1.0 - np.cos(phi_bar_j)) + z_bar_k**2)
    t1_coef = np.cos(theta_M) / r
    t2 = np.arctanh(r * (1.0 - np.cos(phi_bar_j)) / t1)
    t2_coef = np.cos(theta_M) * np.cos(phi_bar_j)
    return t1_coef * t1 + t2_coef * t2

def Hz_ri_case234(r, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t1 = np.sqrt(2.0 * r**2 * (1.0 - np.cos(phi_bar_j)) + z_bar_k**2)
    t1_coef = np.sin(theta_M) * np.sin(phi_bar_M) / r
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r**2 / z_bar_k**2)
    E_coef = np.sin(theta_M) * np.cos(phi_bar_M) * np.abs(z_bar_k) / r
    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r**2 / z_bar_k**2)
    F_coef = -np.sin(theta_M) * np.cos(phi_bar_M) * (2.0 * r**2 + z_bar_k**2) / (r * np.abs(z_bar_k))
    return t1_coef * t1 + E_coef * E + F_coef * F

def Hz_phij_case234(r, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k):
    t1 = np.arctanh(r * (1.0 - np.cos(phi_bar_j)) / np.sqrt(2.0 * r**2 * (1.0 - np.cos(phi_bar_j)) + z_bar_k**2))
    t1_coef = np.sin(theta_M) * np.sin(phi_bar_Mj)
    return t1_coef * t1

def Hz_zk_case234(r, phi_bar_j, theta_M, z_bar_k):
    t = np.sqrt(r**2 + z_bar_k**2)
    def Pi(sign):
        return el3_angle(phi_bar_j/2.0, 2.0 * r / (r + sign * t), -4.0 * r**2 / z_bar_k**2)
    Pi_coef = np.cos(theta_M) * np.sign(z_bar_k)
    return Pi_coef * Pi(1) + Pi_coef * Pi(-1)

# 235 ##############

def Hr_ri_case235(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t1 = np.sqrt(r**2 + r_i**2 - 2.0 * r * r_i * np.cos(phi_bar_j) + z_bar_k**2)
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_M) * z_bar_k / (2.0 * r**2)
    t2 = np.arctanh(z_bar_k / np.sqrt(r**2 + r_i**2 - 2.0 * r * r_i * np.cos(phi_bar_j) + z_bar_k**2))
    t2_coef = np.sin(theta_M) * np.sin(phi_bar_M) / 2.0 * (1.0 - r_i**2 / r**2)
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    E_coef = -np.sin(theta_M) * np.cos(phi_bar_M) * z_bar_k * np.sqrt(r_bar_i**2 + z_bar_k**2) / ( 2.0 * r**2 )
    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    F_coef = np.sin(theta_M) * np.cos(phi_bar_M) * z_bar_k * (2.0 * r_i**2 + z_bar_k**2) / ( 2.0 * r**2 * np.sqrt(r_bar_i**2 + z_bar_k**2) )
    Pi = el3_angle(phi_bar_j/2.0, -4.0 * r * r_i / r_bar_i**2, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    Pi_coef = np.sin(theta_M) * np.cos(phi_bar_M) * z_bar_k * (r**2 + r_i**2) * (r + r_i) / ( 2.0 * r**2 * r_bar_i * np.sqrt(r_bar_i**2 + z_bar_k**2) )
    return t1_coef * t1 + t2_coef * t2 + E_coef * E + F_coef * F + Pi_coef * Pi

def Hr_phij_case235(r, r_i, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k):
    t1 = np.arctanh(z_bar_k / np.sqrt(r**2 + r_i**2 - 2.0 * r * r_i * np.cos(phi_bar_j) + z_bar_k**2))
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_Mj) * np.cos(phi_bar_j)
    t2 = np.arctan(z_bar_k * (r * np.cos(phi_bar_j) - r_i) / ( r * np.sin(phi_bar_j) * np.sqrt(r**2 + r_i**2 - 2.0 * r * r_i * np.cos(phi_bar_j) + z_bar_k**2) ))
    t2_coef = np.sin(theta_M) * np.sin(phi_bar_Mj) * np.sin(phi_bar_j)
    return t1_coef * t1 + t2_coef * t2

def Hr_zk_case235(r, r_i, r_bar_i, phi_bar_j, theta_M, z_bar_k):
    t1 = np.sin(phi_bar_j)
    t1_coef = -np.cos(theta_M)
    t2 = np.log( r_i - r * np.cos(phi_bar_j) + np.sqrt(r_i**2 + r**2 - 2.0 * r_i * r * np.cos(phi_bar_j) + z_bar_k**2) )
    t2_coef = np.cos(theta_M) * np.sin(phi_bar_j)
    t3 = np.arctan(r * np.sin(phi_bar_j) / z_bar_k)
    t3_coef = np.cos(theta_M) * z_bar_k / r
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    E_coef = np.cos(theta_M) * np.sqrt(r_bar_i**2 + z_bar_k**2) / r
    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    F_coef = -np.cos(theta_M) * (r**2 + r_i**2 + z_bar_k**2) / ( r * np.sqrt(r_bar_i**2 + z_bar_k**2) )
    t = np.sqrt(r**2 + z_bar_k**2)
    def Pi1(sign):
        return el3_angle(phi_bar_j/2.0, 2.0 * r / (r + sign * t), -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    def Pi1_coef(sign):
        return -np.cos(theta_M) / ( r * np.sqrt((r**2 +  z_bar_k**2) * (r_bar_i**2 + z_bar_k**2)) ) * (t - sign * r) * (r_i + sign * t)**2
    def Pi2(sign):
        return el3_angle(arctan_k_tan_2(np.sqrt(((r_i + r)**2 + z_bar_k**2)/(r_bar_i**2 + z_bar_k**2)),phi_bar_j), 1.0 - z_bar_k**2 * (r_bar_i**2 + z_bar_k**2) / ( ((r + r_i)**2 + z_bar_k**2) * (r + sign * t)**2 ), 4.0 * r * r_i / ( (r + r_i)**2 + z_bar_k**2 ))
    def Pi2_coef(sign):
        return sign * np.cos(theta_M) * z_bar_k**2 * (r_bar_i**2 + z_bar_k**2) / ( r * np.sqrt((r**2 + z_bar_k**2) * ((r + r_i)**2 + z_bar_k**2)) * (r + sign * t) )
    return t1_coef * t1 + t2_coef * t2 + t3_coef * t3 + E_coef * E + F_coef * F + Pi1_coef(1) * Pi1(1) + Pi1_coef(-1) * Pi1(-1) + Pi2_coef(1) * Pi2(1) + Pi2_coef(-1) * Pi2(-1)

def Hphi_ri_case235(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t1 = np.sqrt(r**2 + r_i**2 - 2.0 * r * r_i * np.cos(phi_bar_j) + z_bar_k**2)
    t1_coef = -np.sin(theta_M) * np.cos(phi_bar_M) * z_bar_k / (2.0 * r**2)
    t2 = np.arctanh(z_bar_k / np.sqrt(r**2 + r_i**2 - 2.0 * r * r_i * np.cos(phi_bar_j) + z_bar_k**2))
    t2_coef = -np.sin(theta_M) * np.cos(phi_bar_M) * (r**2 + r_i**2) / ( 2.0 * r**2 )
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    E_coef = np.sin(theta_M) * np.sin(phi_bar_M) * z_bar_k * np.sqrt(r_bar_i**2 + z_bar_k**2) / ( 2.0 * r**2 )
    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    F_coef = -np.sin(theta_M) * np.sin(phi_bar_M) * z_bar_k  * (2.0 * r**2 + 2.0 * r_i**2 + z_bar_k**2) / ( 2.0 * r**2 * np.sqrt(r_bar_i**2 + z_bar_k**2) )
    Pi = el3_angle(phi_bar_j/2.0, -4.0 * r * r_i / r_bar_i**2, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    Pi_coef = np.sin(theta_M) * np.sin(phi_bar_M) * z_bar_k * (r + r_i)**2 / ( 2.0 * r**2 * np.sqrt(r_bar_i**2 + z_bar_k**2) )
    return t1_coef * t1 + t2_coef * t2 + E_coef * E + F_coef * F + Pi_coef * Pi

def Hphi_phij_case235(r, r_i, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k):
    t = np.sqrt(r**2 + r_i**2 - 2.0 * r * r_i * np.cos(phi_bar_j) + z_bar_k**2)
    t1 = np.arctan(z_bar_k * (r * np.cos(phi_bar_j) - r_i) / ( r * np.sin(phi_bar_j) * t ))
    t1_coef = np.sin(theta_M) * np.sin(phi_bar_Mj) * np.cos(phi_bar_j)
    t2 = np.arctanh(z_bar_k / t)
    t2_coef = np.sin(theta_M) * np.sin(phi_bar_Mj) * np.sin(phi_bar_j)
    return t1_coef * t1 + t2_coef * t2

def Hphi_zk_case235(r, r_i, phi_bar_j, theta_M, z_bar_k):
    t1 = np.sqrt(r**2 + r_i**2 - 2.0 * r * r_i * np.cos(phi_bar_j) + z_bar_k**2)
    t1_coef = np.cos(theta_M) / r
    t2 = np.arctanh((r * np.cos(phi_bar_j) - r_i) / t1)
    t2_coef = -np.cos(theta_M) * np.cos(phi_bar_j)
    return t1_coef * t1 + t2_coef * t2

def Hz_ri_case235(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t = r_bar_i**2 + z_bar_k**2
    t1 = np.sqrt(r**2 + r_i**2 - 2.0 * r * r_i * np.cos(phi_bar_j) + z_bar_k**2)
    t1_coef = np.sin(theta_M) * np.sin(phi_bar_M) / r
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r * r_i / t)
    E_coef = np.sin(theta_M) * np.cos(phi_bar_M) * np.sqrt(t) / r
    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r * r_i / t)
    F_coef = -np.sin(theta_M) * np.cos(phi_bar_M) * (r**2 + r_i**2 + z_bar_k**2) / (r * np.sqrt(t))
    return t1_coef * t1 + E_coef * E + F_coef * F

def Hz_phij_case235(r, r_i, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k):
    t1 = np.arctanh((r * np.cos(phi_bar_j) - r_i) / np.sqrt(r**2 + r_i**2 - 2.0 * r * r_i * np.cos(phi_bar_j) + z_bar_k**2))
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_Mj)
    return t1_coef * t1

def Hz_zk_case235(r, r_i, r_bar_i, phi_bar_j, theta_M, z_bar_k):
    t = np.sqrt(r**2 + z_bar_k**2)
    def Pi(sign):
        return el3_angle(phi_bar_j/2.0, 2.0 * r / (r + sign * t), -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    def Pi_coef(sign):
        return np.cos(theta_M) * z_bar_k * (r_i + sign * t) / ( np.sqrt(r_bar_i**2 + z_bar_k**2) * (r + sign * t) )
    return Pi_coef(1) * Pi(1) + Pi_coef(-1) * Pi(-1)


####################
####################
####################
#calculation of all field components for each case
#especially these function show, which inputs are needed for the calculation
#full vectorization for all cases could be implemented here
# input: ndarray, shape (n,)
# out: ndarray, shape (n,3,3) # (n)vector, (3)r_phi_z, (3)face

def case112(r_i, phi_bar_M, theta_M):
    results = np.zeros((len(r_i), 3, 3))
    results[:,1,2] = Hphi_zk_case112(r_i, theta_M)
    results[:,2,0] = Hz_ri_case112(phi_bar_M, theta_M)
    results[:,2,1] = Hz_phij_case112(r_i, phi_bar_M, theta_M)
    return results

def case113(r, phi_bar_M, theta_M):
    results = np.zeros((len(r), 3, 3))
    results[:,1,2] = Hphi_zk_case113(r, theta_M)
    results[:,2,1] = Hz_phij_case113(r, phi_bar_M, theta_M)
    return results

def case115(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M):
    results = np.zeros((len(r), 3, 3))
    results[:,0,2] = Hr_zk_case115(r, r_i, r_bar_i, phi_bar_j, theta_M)
    results[:,1,2] = Hphi_zk_case115(r, r_i, r_bar_i, theta_M)
    results[:,2,0] = Hz_ri_case115(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M)
    results[:,2,1] = Hz_phij_case115(r_bar_i, phi_bar_M, theta_M)
    return results

def case122(r_i, phi_bar_M, theta_M):
    results = np.zeros((len(r_i), 3, 3))
    results[:,1,2] = Hphi_zk_case122(r_i, theta_M)
    results[:,2,0] = Hz_ri_case122(phi_bar_M, theta_M)
    results[:,2,1] = Hz_phij_case122(r_i, phi_bar_M, theta_M)
    return results

def case123(r, phi_bar_M, theta_M):
    results = np.zeros((len(r), 3, 3))
    results[:,1,2] = Hphi_zk_case123(r, theta_M)
    results[:,2,1] = Hz_phij_case123(r, phi_bar_M, theta_M)
    return results

def case124(r, phi_bar_M, theta_M):
    results = np.zeros((len(r), 3, 3))
    results[:,1,2] = Hphi_zk_case124(r, theta_M)
    results[:,2,0] = Hz_ri_case124(phi_bar_M, theta_M)
    results[:,2,1] = Hz_phij_case124(r, phi_bar_M, theta_M)
    return results

def case125(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M):
    results = np.zeros((len(r), 3, 3))
    results[:,0,2] = Hr_zk_case125(r, r_i, r_bar_i, phi_bar_j, theta_M)
    results[:,1,2] = Hphi_zk_case125(r, r_i, theta_M)
    results[:,2,0] = Hz_ri_case125(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M)
    results[:,2,1] = Hz_phij_case125(r, r_i, phi_bar_M, theta_M)
    return results

def case132(r, r_i, phi_bar_j, phi_bar_Mj, theta_M):
    results = np.zeros((len(r), 3, 3))
    results[:,0,2] = Hr_zk_case132(r_i, phi_bar_j, theta_M)
    results[:,1,2] = Hphi_zk_case132(r_i, phi_bar_j, theta_M)
    results[:,2,0] = Hz_ri_case132(phi_bar_Mj, theta_M)
    results[:,2,1] = Hz_phij_case132(r_i, phi_bar_Mj, theta_M)
    return results

def case133(r, phi_bar_j, phi_bar_Mj, theta_M):
    results = np.zeros((len(r), 3, 3))
    results[:,0,2] = Hr_zk_case133(r, phi_bar_j, theta_M)
    results[:,1,2] = Hphi_zk_case133(phi_bar_j, theta_M)
    results[:,2,1] = Hz_phij_case133(phi_bar_j, phi_bar_Mj, theta_M)
    return results

def case134(r, phi_bar_j, phi_bar_M, phi_bar_Mj, theta_M):
    results = np.zeros((len(r), 3, 3))
    results[:,0,2] = Hr_zk_case134(r, phi_bar_j, theta_M)
    results[:,1,2] = Hphi_zk_case134(phi_bar_j, theta_M)
    results[:,2,0] = Hz_ri_case134(phi_bar_j, phi_bar_M, theta_M)
    results[:,2,1] = Hz_phij_case134(phi_bar_j, phi_bar_Mj, theta_M)
    return results

def case135(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, phi_bar_Mj, theta_M):
    results = np.zeros((len(r), 3, 3))
    results[:,0,2] = Hr_zk_case135(r, r_i, r_bar_i, phi_bar_j, theta_M)
    results[:,1,2] = Hphi_zk_case135(r, r_i, phi_bar_j, theta_M)
    results[:,2,0] = Hz_ri_case135(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M)
    results[:,2,1] = Hz_phij_case135(r, r_i, phi_bar_j, phi_bar_Mj, theta_M)
    return results

def case211(phi_j, phi_bar_M, theta_M, z_bar_k):
    results = np.zeros((len(phi_j), 3, 3))
    results[:,0,1] = Hr_phij_case211(phi_bar_M, theta_M, z_bar_k)
    results[:,2,2] = Hz_zk_case211(phi_j, theta_M, z_bar_k)
    return results

def case212(r_i, phi_j, phi_bar_M, theta_M, z_bar_k):
    results = np.zeros((len(r_i), 3, 3))
    results[:,0,0] = Hr_ri_case212(r_i, phi_j, phi_bar_M, theta_M, z_bar_k)
    results[:,0,1] = Hr_phij_case212(r_i, phi_bar_M, theta_M, z_bar_k)
    results[:,1,0] = Hphi_ri_case212(r_i, phi_j, phi_bar_M, theta_M, z_bar_k)
    results[:,1,2] = Hphi_zk_case212(r_i, theta_M, z_bar_k)
    results[:,2,0] = Hz_ri_case212(r_i, phi_bar_M, theta_M, z_bar_k)
    results[:,2,1] = Hz_phij_case212(r_i, phi_bar_M, theta_M, z_bar_k)
    results[:,2,2] = Hz_zk_case212(r_i, phi_j, theta_M, z_bar_k)
    return results

def case213(r, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    results = np.zeros((len(r), 3, 3))
    results[:,0,1] = Hr_phij_case213(r, phi_bar_M, theta_M, z_bar_k)
    results[:,1,2] = Hphi_zk_case213(r, theta_M, z_bar_k)
    results[:,2,1] = Hz_phij_case213(r, phi_bar_M, theta_M, z_bar_k)
    results[:,2,2] = Hz_zk_case213(phi_bar_j, theta_M, z_bar_k)
    return results

def case214(r, phi_j, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    results = np.zeros((len(r), 3, 3))
    results[:,0,0] = Hr_ri_case214(r, phi_bar_j, phi_bar_M, theta_M, z_bar_k)
    results[:,0,1] = Hr_phij_case214(phi_bar_M, theta_M, z_bar_k)
    results[:,0,2] = Hr_zk_case214(r, phi_bar_j, theta_M, z_bar_k)
    results[:,1,0] = Hphi_ri_case214(r, phi_j, phi_bar_j, phi_bar_M, theta_M, z_bar_k)
    results[:,1,2] = Hphi_zk_case214(r, theta_M, z_bar_k)
    results[:,2,0] = Hz_ri_case214(r, phi_bar_j, phi_bar_M, theta_M, z_bar_k)
    results[:,2,2] = Hz_zk_case214(r, phi_bar_j, theta_M, z_bar_k)
    return results

def case215(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    results = np.zeros((len(r), 3, 3))
    results[:,0,0] = Hr_ri_case215(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k)
    results[:,0,1] = Hr_phij_case215(r_bar_i, phi_bar_M, theta_M, z_bar_k)
    results[:,0,2] = Hr_zk_case215(r, r_i, r_bar_i, phi_bar_j, theta_M, z_bar_k)
    results[:,1,0] = Hphi_ri_case215(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k)
    results[:,1,2] = Hphi_zk_case215(r, r_bar_i, theta_M, z_bar_k)
    results[:,2,0] = Hz_ri_case215(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k)
    results[:,2,1] = Hz_phij_case215(r_bar_i, phi_bar_M, theta_M, z_bar_k)
    results[:,2,2] = Hz_zk_case215(r, r_i, r_bar_i, phi_bar_j, theta_M, z_bar_k)
    return results

def case221(phi_j, phi_bar_M, theta_M, z_bar_k):
    results = np.zeros((len(phi_j), 3, 3))
    results[:,0,1] = Hr_phij_case221(phi_bar_M, theta_M, z_bar_k)
    results[:,2,2] = Hz_zk_case221(phi_j, theta_M, z_bar_k)
    return results

def case222(r_i, phi_j, phi_bar_M, theta_M, z_bar_k):
    results = np.zeros((len(r_i), 3, 3))
    results[:,0,0] = Hr_ri_case222(r_i, phi_j, phi_bar_M, theta_M, z_bar_k)
    results[:,0,1] = Hr_phij_case222(r_i, phi_bar_M, theta_M, z_bar_k)
    results[:,1,0] = Hphi_ri_case222(r_i, phi_j, phi_bar_M, theta_M, z_bar_k)
    results[:,1,2] = Hphi_zk_case222(r_i, theta_M, z_bar_k)
    results[:,2,0] = Hz_ri_case222(r_i, phi_bar_M, theta_M, z_bar_k)
    results[:,2,1] = Hz_phij_case222(r_i, phi_bar_M, theta_M, z_bar_k)
    results[:,2,2] = Hz_zk_case222(r_i, phi_j, theta_M, z_bar_k)
    return results

def case223(r, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    results = np.zeros((len(r), 3, 3))
    results[:,0,1] = Hr_phij_case223(r, phi_bar_M, theta_M, z_bar_k)
    results[:,1,2] = Hphi_zk_case223(r, theta_M, z_bar_k)
    results[:,2,1] = Hz_phij_case223(r, phi_bar_M, theta_M, z_bar_k)
    results[:,2,2] = Hz_zk_case223(r, phi_bar_j, theta_M, z_bar_k)
    return results

def case224(r, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    results = np.zeros((len(r), 3, 3))
    results[:,0,0] = Hr_ri_case224(r, phi_bar_j, phi_bar_M, theta_M, z_bar_k)
    results[:,0,1] = Hr_phij_case224(r, phi_bar_M, theta_M, z_bar_k)
    results[:,0,2] = Hr_zk_case224(r, phi_bar_j, theta_M, z_bar_k)
    results[:,1,0] = Hphi_ri_case224(r, phi_bar_j, phi_bar_M, theta_M, z_bar_k)
    results[:,1,2] = Hphi_zk_case224(r, theta_M, z_bar_k)
    results[:,2,0] = Hz_ri_case224(r, phi_bar_j, phi_bar_M, theta_M, z_bar_k)
    results[:,2,1] = Hz_phij_case224(r, phi_bar_M, theta_M, z_bar_k)
    results[:,2,2] = Hz_zk_case224(r, phi_bar_j, theta_M, z_bar_k)
    return results

def case225(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    results = np.zeros((len(r), 3, 3))
    results[:,0,0] = Hr_ri_case225(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k)
    results[:,0,1] = Hr_phij_case225(r, r_i, phi_bar_M, theta_M, z_bar_k)
    results[:,0,2] = Hr_zk_case225(r, r_i, r_bar_i, phi_bar_j, theta_M, z_bar_k)
    results[:,1,0] = Hphi_ri_case225(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k)
    results[:,1,2] = Hphi_zk_case225(r, r_i, theta_M, z_bar_k)
    results[:,2,0] = Hz_ri_case225(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k)
    results[:,2,1] = Hz_phij_case225(r, r_i, phi_bar_M, theta_M, z_bar_k)
    results[:,2,2] = Hz_zk_case225(r, r_i, r_bar_i, phi_bar_j, theta_M, z_bar_k)
    return results

def case231(phi_j, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k):
    results = np.zeros((len(phi_j), 3, 3))
    results[:,0,1] = Hr_phij_case231(phi_bar_j, phi_bar_Mj, theta_M, z_bar_k)
    results[:,1,1] = Hphi_phij_case231(phi_bar_j, phi_bar_Mj, theta_M, z_bar_k)
    results[:,2,2] = Hz_zk_case231(phi_j, theta_M, z_bar_k)
    return results

def case232(r_i, phi_j, phi_bar_j, phi_bar_M, phi_bar_Mj, theta_M, z_bar_k):
    results = np.zeros((len(r_i), 3, 3))
    results[:,0,0] = Hr_ri_case232(r_i, phi_j, phi_bar_j, phi_bar_M, phi_bar_Mj, theta_M, z_bar_k)
    results[:,0,1] = Hr_phij_case232(r_i, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k)
    results[:,0,2] = Hr_zk_case232(r_i, phi_bar_j, theta_M, z_bar_k)
    results[:,1,0] = Hphi_ri_case232(r_i, phi_j, phi_bar_j, phi_bar_M, phi_bar_Mj, theta_M, z_bar_k)
    results[:,1,1] = Hphi_phij_case232(r_i, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k)
    results[:,1,2] = Hphi_zk_case232(r_i, phi_bar_j, theta_M, z_bar_k)
    results[:,2,0] = Hz_ri_case232(r_i, phi_bar_Mj, theta_M, z_bar_k)
    results[:,2,1] = Hz_phij_case232(r_i, phi_bar_Mj, theta_M, z_bar_k)
    results[:,2,2] = Hz_zk_case232(r_i, phi_j, theta_M, z_bar_k)
    return results

def case233(r, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k):
    results = np.zeros((len(r), 3, 3))
    results[:,0,1] = Hr_phij_case233(r, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k)
    results[:,0,2] = Hr_zk_case233(r, phi_bar_j, theta_M, z_bar_k)
    results[:,1,1] = Hphi_phij_case233(r, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k)
    results[:,1,2] = Hphi_zk_case233(r, phi_bar_j, theta_M, z_bar_k)
    results[:,2,1] = Hz_phij_case233(r, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k)
    results[:,2,2] = Hz_zk_case233(r, phi_bar_j, theta_M, z_bar_k)
    return results

def case234(r, phi_bar_j, phi_bar_M, phi_bar_Mj, theta_M, z_bar_k):
    results = np.zeros((len(r), 3, 3))
    results[:,0,0] = Hr_ri_case234(r, phi_bar_j, phi_bar_M, theta_M, z_bar_k)
    results[:,0,1] = Hr_phij_case234(r, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k)
    results[:,0,2] = Hr_zk_case234(r, phi_bar_j, theta_M, z_bar_k)
    results[:,1,0] = Hphi_ri_case234(r, phi_bar_j, phi_bar_M, theta_M, z_bar_k)
    results[:,1,1] = Hphi_phij_case234(r, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k)
    results[:,1,2] = Hphi_zk_case234(r, phi_bar_j, theta_M, z_bar_k)
    results[:,2,0] = Hz_ri_case234(r, phi_bar_j, phi_bar_M, theta_M, z_bar_k)
    results[:,2,1] = Hz_phij_case234(r, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k)
    results[:,2,2] = Hz_zk_case234(r, phi_bar_j, theta_M, z_bar_k)
    return results

def case235(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, phi_bar_Mj, theta_M, z_bar_k):
    results = np.zeros((len(r), 3, 3))
    results[:,0,0] = Hr_ri_case235(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k)
    results[:,0,1] = Hr_phij_case235(r, r_i, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k)
    results[:,0,2] = Hr_zk_case235(r, r_i, r_bar_i, phi_bar_j, theta_M, z_bar_k)
    results[:,1,0] = Hphi_ri_case235(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k)
    results[:,1,1] = Hphi_phij_case235(r, r_i, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k)
    results[:,1,2] = Hphi_zk_case235(r, r_i, phi_bar_j, theta_M, z_bar_k)
    results[:,2,0] = Hz_ri_case235(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k)
    results[:,2,1] = Hz_phij_case235(r, r_i, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k)
    results[:,2,2] = Hz_zk_case235(r, r_i, r_bar_i, phi_bar_j, theta_M, z_bar_k)
    return results


def field_H_cylinder_tile(obs_pos: np.ndarray, dim: np.ndarray, mag:np.ndarray) ->np.ndarray:
    """
    Core computation of the Cylinder tile field based on Slanovc 2021

    obs_pos : ndarray, shape (N,3)
        observer positions (r,phi,z) in cy CS, units: [mm] [rad]
    dim: ndarray, shape (N,6)
        section dimensions (r1,r2,phi1,phi2,z1,z2) in cy CS , units: [mm] [rad]
    mag: ndarray, shape (N,3)
        magnetization vector (|M|, phi, th) in spherical CS, units: [mT] [rad]
    """

    # tile inputs into 8-stacks (boundary cases)
    r, phi, z = np.repeat(obs_pos, 8, axis=0).T
    r_i = np.repeat(dim[:,:2], 4)
    phi_j = np.repeat(np.tile(dim[:,2:4],2), 2)
    z_k = np.ravel(np.tile(dim[:,4:6], 4))
    _, phi_M, theta_M = np.repeat(mag, 8, axis=0).T

    #initialize results array with nan
    result = np.empty((len(r), 3, 3))
    result[:] = np.nan

    # cases to evaluate
    cases = determine_cases(r, phi, z, r_i, phi_j, z_k)

    # list of all possible cases - excluding the nan-cases 111, 114, 121, 131
    case_id = np.array([112, 113, 115, 122, 123, 124, 125, 132, 133, 134,
        135, 211, 212, 213, 214, 215, 221, 222, 223, 224, 225, 231, 232, 233, 234, 235])

    # corresponding case evaluation functions
    case_fkt = [case112, case113, case115, case122, case123, case124, case125, case132,
        case133, case134, case135, case211, case212, case213, case214, case215, case221,
        case222, case223, case224, case225, case231, case232, case233, case234, case235]

    # required case function arguments
    r_bar_i = r - r_i
    phi_bar_j = phi - phi_j
    phi_bar_M = phi_M - phi
    phi_bar_Mj = phi_M - phi_j
    z_bar_k = z - z_k
    #          0   1      2         3          4          5          6        7       8
    allargs = [r, r_i, r_bar_i, phi_bar_j, phi_bar_M, phi_bar_Mj, theta_M, z_bar_k, phi_j]
    case_args = [(1,4,6), (0,4,6), (0,1,2,3,4,6), (1,4,6), (0,4,6), (0,4,6), (0,1,2,3,4,6),
    (0,1,3,5,6), (0,3,5,6), (0,3,4,5,6), (0,1,2,3,4,5,6), (8,4,6,7), (1,8,4,6,7), (0,3,4,6,7),
    (0,8,3,4,6,7), (0,1,2,3,4,6,7), (8,4,6,7), (1,8,4,6,7), (0,3,4,6,7), (0,3,4,6,7),
    (0,1,2,3,4,6,7), (8,3,5,6,7), (1,8,3,4,5,6,7), (0,3,5,6,7), (0,3,4,5,6,7), (0,1,2,3,4,5,6,7)]

    # calling case functions with respective masked arguments
    for cid,cfkt,cargs in zip(case_id, case_fkt, case_args):
        mask = cases==cid
        if any(mask):
            result[mask] = cfkt(*[allargs[aid][mask] for aid in cargs])

    # sum up contributions from different boundary cases (ax1) and different face types (ax3)
    result = np.reshape(result, (-1,8,3,3))
    result = np.sum(result[:,(1,2,4,7)] - result[:,(0,3,5,6)], axis=(1,3))

    # multiply with magnetization amplitude
    result = result.T*mag[:,0]/(4*np.pi)

    return result.T


def field_BH_cylinder2(
        bh: bool,
        mag: np.ndarray,
        dim: np.ndarray,
        pos_obs: np.ndarray
        ) -> np.ndarray:
    """
    ### Args:
    - bh (boolean): True=B, False=H
    - mag (ndarray Nx3): homogeneous magnetization vector in cartesian CS units of [mT]
    - dim (ndarray Nx2): dimension of Cylinder (r1,r2,phi1,phi2,z1,z2) in units of [mm], [deg]
    - pos_obs (ndarray Nx3): position of observer in units of [mm]

    ### Returns:
    - B/H-field (ndarray Nx3): magnetic field vectors in cartesian CS at pos_obs
        in units of [mT] or [kA/m]
    """
    # edgesize = Config.EDGESIZE

    # transform dim deg->rad
    dim[:,2] = dim[:,2]/180*np.pi
    dim[:,3] = dim[:,3]/180*np.pi

    # transform obs_pos to Cy CS --------------------------------------------
    x, y, z = pos_obs.T
    r, phi = np.sqrt(x**2+y**2), np.arctan2(y, x)
    pos_obs_cy = np.concatenate(((r,),(phi,),(z,)),axis=0).T

    # transform mag to spherical CS -----------------------------------------
    m = np.sqrt(mag[:,0]**2 + mag[:,1]**2 + mag[:,2]**2)
    phi_m = np.arctan2(mag[:,1], mag[:,0])
    th_m = np.arctan2(np.sqrt(mag[:,0]**2+mag[:,1]**2), mag[:,2])
    mag_sph = np.concatenate(((m,),(phi_m,),(th_m,)),axis=0).T

    # compute H and transform to cart CS -------------------------------------
    H_cy = field_H_cylinder_tile(pos_obs_cy, dim, mag_sph)
    Hr, Hphi, Hz = H_cy.T
    Hx = Hr*np.cos(phi) - Hphi*np.sin(phi)
    Hy = Hr*np.sin(phi) + Hphi*np.cos(phi)
    H = np.concatenate(((Hx,),(Hy,),(Hz,)),axis=0).T*10/4/np.pi

    # return B or H --------------------------------------------------------
    if not bh:
        return H

    B = H/(10/4/np.pi) # kA/m -> mT
    r1,r2,phi1,phi2,z1,z2 = dim.T
    phi[phi<0] += 2*np.pi  # map phi on interval [0,2pi]
    mask_inside = (r1<r) * (r<r2) * (phi1<phi) * (phi<phi2) * (z1<z) * (z<z2)
    B[mask_inside] += mag[mask_inside]
    return B
