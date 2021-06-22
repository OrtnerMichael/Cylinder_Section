# pylint: disable=no-member
# pylint: disable=too-many-lines
# pylint: disable=line-too-long
# pylint: disable=missing-function-docstring
import numpy as np
import scipy.special
import magpy_special as icels


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

    result = np.zeros(phi.shape)

    case1 = close(phi_red, -np.pi)
    result[case1] = full_periods[case1] * np.pi - np.pi / 2.0
    case2 = close(phi_red, np.pi)
    result[case2] = full_periods[case2] * np.pi + np.pi / 2.0
    case3 = ~(case1 + case2)
    result[case3] = full_periods[case3] * np.pi + np.arctan(k[case3] * np.tan(phi_red[case3] / 2.0))

    return result


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
    t1 = r_i + r * np.log(np.abs(r_bar_i))
    t1_coef = -np.cos(theta_M) * np.sign(r_bar_i) / r
    return t1_coef * t1

def Hz_ri_case115(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M):
    t = r_bar_i**2
    t1 = np.abs(r_bar_i) / r
    t1_coef = np.sin(theta_M) * np.sin(phi_bar_M)
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r * r_i / t)
    E_coef = np.sin(theta_M) * np.cos(phi_bar_M) * np.abs(r_bar_i) / r
    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r * r_i / t)
    F_coef = -np.sin(theta_M) * np.cos(phi_bar_M) * (r**2 + r_i**2) / (r * np.abs(r_bar_i))
    return t1_coef * t1 + E_coef * E + F_coef * F

def Hz_phij_case115(r_bar_i, phi_bar_M, theta_M):
    t1 = np.log(np.abs(r_bar_i))
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_M) * np.sign(r_bar_i)
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
    return (-np.cos(theta_M)*np.sin(phi_bar_j) +
        np.cos(theta_M)*np.sin(phi_bar_j)*np.log(r * (1.0 - np.cos(phi_bar_j))))

def Hphi_zk_case133(phi_bar_j, theta_M):
    return np.cos(theta_M) - np.cos(theta_M)*np.cos(phi_bar_j)*np.arctanh(np.cos(phi_bar_j))

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

def Hz_zk_case134(r):
    return np.zeros(r.shape)

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
    t1 = -np.sin(theta_M) * z_bar_k / np.sqrt(r_i**2 + z_bar_k**2)
    t2 = 1.0/2.0 * phi_j * np.cos(phi_bar_M)
    t3 = 1.0/4.0 * np.sin(phi_bar_M)
    return -t1 * (t2 - t3)

def Hr_phij_case212(r_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t1 = np.arctanh(z_bar_k / np.sqrt(r_i**2 + z_bar_k**2))
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_M) * np.cos(phi_bar_j)
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

def Hr_zk_case213(r, phi_bar_j, theta_M, z_bar_k):
    t = np.sqrt(r**2 + z_bar_k**2)
    t3_coef = np.cos(theta_M) * np.abs(z_bar_k) / r
    t4 = arctan_k_tan_2(t / np.abs(z_bar_k), 2.0 * phi_bar_j)
    t4_coef = -t3_coef
    def t5(sign):
        return arctan_k_tan_2(np.abs(z_bar_k) / np.abs(r + sign * t), phi_bar_j)
    t5_coef = t3_coef
    return t4_coef * t4 + t5_coef * t5(1) + t5_coef * t5(-1)

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

def Hz_zk_case213(r, phi_bar_j, theta_M, z_bar_k):
    t1 = arctan_k_tan_2(np.sqrt(r**2 + z_bar_k**2) / np.abs(z_bar_k), 2.0 * phi_bar_j)
    t1_coef = np.cos(theta_M) * np.sign(z_bar_k)
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
        return icels.el3_angle_vectorized(phi_bar_j/2, 2*r/(r+sign*t), -4*r**2/z_bar_k**2)
    def Pi1_coef(sign):
        return -np.cos(theta_M) / ( r * np.sqrt((r**2 +  z_bar_k**2) * z_bar_k**2) ) * (t - sign * r) * (r + sign * t)**2
    def Pi2(sign):
        return icels.el3_angle_vectorized(arctan_k_tan_2(np.sqrt((4.0 * r**2 + z_bar_k**2) / z_bar_k**2), phi_bar_j), 1.0 - z_bar_k**4 / ( (4.0 * r**2 + z_bar_k**2) * (2.0 * r**2 + z_bar_k**2 + sign*2.0*r*t) ), 4.0*r**2 / ( 4.0*r**2 + z_bar_k**2 ))
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

def Hphi_phij_case214(r):
    return np.zeros(r.shape)

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
        return icels.el3_angle_vectorized(phi_bar_j/2, 2*r/(r+sign*t), -4*r**2/z_bar_k**2)
    def Pi_coef(sign):  #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<bugggg?
        return np.cos(theta_M) * np.sign(z_bar_k)
    return Pi_coef(1) * Pi(1) + Pi_coef(-1) * Pi(-1)

# 215 ##############

def Hr_ri_case215(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t2 = np.arctanh(np.abs(z_bar_k) / np.sqrt(r_bar_i**2 + z_bar_k**2))
    t2_coef = np.sin(theta_M) * np.sin(phi_bar_M) * np.sign(z_bar_k) / 2.0 * (1-r_i**2/r**2)
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    E_coef = -np.sin(theta_M) * np.cos(phi_bar_M) * z_bar_k * np.sqrt(r_bar_i**2 + z_bar_k**2) / (2*r**2)
    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    F_coef = np.sin(theta_M) * np.cos(phi_bar_M) * z_bar_k * (2.0 * r_i**2 + z_bar_k**2) / (2*r**2 * np.sqrt(r_bar_i**2 + z_bar_k**2) )
    Pi = icels.el3_angle_vectorized(phi_bar_j/2.0, -4.0 * r * r_i / r_bar_i**2, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
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
        return icels.el3_angle_vectorized(phi_bar_j/2.0, 2.0 * r / (r + sign * t), -4*r*r_i / (r_bar_i**2 + z_bar_k**2))
    def Pi1_coef(sign):
        return -np.cos(theta_M) / ( r * np.sqrt((r**2+z_bar_k**2) * (r_bar_i**2+z_bar_k**2)) ) * (t - sign * r) * (r_i + sign * t)**2
    def Pi2(sign):
        return icels.el3_angle_vectorized(arctan_k_tan_2(np.sqrt(((r_i + r)**2 + z_bar_k**2)/(r_bar_i**2 + z_bar_k**2)),phi_bar_j), 1.0 - z_bar_k**2 * (r_bar_i**2 + z_bar_k**2) / (((r + r_i)**2 + z_bar_k**2) * (2.0 * r**2 + z_bar_k**2 + sign*2*r*t) ), 4*r*r_i / ((r + r_i)**2 + z_bar_k**2 ))
    def Pi2_coef(sign):
        return sign * np.cos(theta_M) * z_bar_k**2 * (r_bar_i**2 + z_bar_k**2) / ( r * np.sqrt((r**2 + z_bar_k**2) * ((r + r_i)**2 + z_bar_k**2)) * (r + sign * t) )
    return E_coef*E + F_coef*F + Pi1_coef(1)*Pi1(1) + Pi1_coef(-1)*Pi1(-1) + Pi2_coef(1)*Pi2(1) + Pi2_coef(-1)*Pi2(-1)


def Hphi_ri_case215(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t1 = np.sqrt(r_bar_i**2 + z_bar_k**2) * z_bar_k / (2.0 * r**2)
    t1_coef = -np.sin(theta_M) * np.cos(phi_bar_M)
    t2 = np.arctanh(np.abs(z_bar_k) / np.sqrt(r_bar_i**2 + z_bar_k**2))
    t2_coef = -np.sin(theta_M) * np.cos(phi_bar_M) * (r**2 + r_i**2) * np.sign(z_bar_k) / (2*r**2)
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    E_coef = np.sin(theta_M) * np.sin(phi_bar_M) * z_bar_k * np.sqrt(r_bar_i**2+z_bar_k**2) / (2*r**2)
    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    F_coef = -np.sin(theta_M) * np.sin(phi_bar_M) * z_bar_k  * (2.0 * r**2 + 2.0 * r_i**2 + z_bar_k**2) / ( 2.0 * r**2 * np.sqrt(r_bar_i**2 + z_bar_k**2) )
    Pi = icels.el3_angle_vectorized(phi_bar_j/2.0, -4.0 * r * r_i / r_bar_i**2, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
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
        return icels.el3_angle_vectorized(phi_bar_j/2.0, 2.0 * r / (r + sign * t), -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
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
    t1 = -np.sin(theta_M) * z_bar_k / np.sqrt(r_i**2 + z_bar_k**2)
    t2 = 1.0/2.0 * phi_j * np.cos(phi_bar_M)
    t3 = 1.0/4.0 * np.sin(phi_bar_M)
    return -t1 * (t2 - t3)

def Hr_phij_case222(r_i, phi_bar_M, theta_M, z_bar_k):
    t1 = np.arctanh(z_bar_k / np.sqrt(r_i**2 + z_bar_k**2))
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_M)
    return t1_coef * t1

def Hphi_ri_case222(r_i, phi_j, phi_bar_M, theta_M, z_bar_k):
    t1 = np.sin(theta_M) * z_bar_k / np.sqrt(r_i**2 + z_bar_k**2)
    t2 = 1.0/4.0 * np.cos(phi_bar_M)
    t3 = 1.0/2.0 * phi_j * np.sin(phi_bar_M)
    return t1 * (-t2 + t3)

def Hphi_phij_case222(r_i):
    return np.zeros(r_i.shape)

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
    t1 = phi_j / np.sqrt(r_i**2 + z_bar_k**2)
    t1_coef = -np.cos(theta_M) * z_bar_k
    return t1_coef * t1

# 223 ##############

def Hr_phij_case223(r, phi_bar_M, theta_M, z_bar_k):
    t1 = np.arctanh(z_bar_k / np.sqrt(r**2 + z_bar_k**2))
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_M)
    return t1_coef * t1

def Hr_zk_case223(r, phi_bar_j, theta_M, z_bar_k):
    t = np.sqrt(r**2 + z_bar_k**2)
    t3_coef = np.cos(theta_M) * np.abs(z_bar_k) / r
    t4 = arctan_k_tan_2(t / np.abs(z_bar_k), 2.0 * phi_bar_j)
    t4_coef = -t3_coef
    def t5(sign):
        return arctan_k_tan_2(np.abs(z_bar_k) / np.abs(r + sign * t), phi_bar_j)
    t5_coef = t3_coef
    return t4_coef * t4 + t5_coef * t5(1) + t5_coef * t5(-1)

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
        return icels.el3_angle_vectorized(phi_bar_j/2.0, 2.0 * r / (r + sign * t), -4.0 * r**2 / z_bar_k**2)
    def Pi1_coef(sign):
        return -np.cos(theta_M) / ( r * np.sqrt((r**2 + z_bar_k**2) * z_bar_k**2) ) * (t - sign * r) * (r + sign * t)**2
    def Pi2(sign):
        return icels.el3_angle_vectorized(arctan_k_tan_2(np.sqrt((4.0 * r**2 + z_bar_k**2) / z_bar_k**2), phi_bar_j), 1.0 - z_bar_k**4 / ( (4.0 * r**2 + z_bar_k**2) * (2.0 * r**2 + z_bar_k**2 + sign * 2.0 * r * t) ), 4.0 * r**2 / ( 4.0 * r**2 + z_bar_k**2 ))
    def Pi2_coef(sign):
        return sign * np.cos(theta_M) * z_bar_k**4 / ( r * np.sqrt((r**2 + z_bar_k**2) * (4.0 * r**2 + z_bar_k**2)) * (r + sign * t) )
    return E_coef * E + F_coef * F + Pi1_coef(1) * Pi1(1) + Pi1_coef(-1) * Pi1(-1) + Pi2_coef(1) * Pi2(1) + Pi2_coef(-1) * Pi2(-1)

def Hphi_ri_case224(r, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t1 = np.sqrt(4.0 * r**2 + z_bar_k**2) * z_bar_k / (2.0 * r**2)
    t1_coef = -np.sin(theta_M) * np.cos(phi_bar_M)
    t2 = np.arctanh(np.abs(z_bar_k) / np.sqrt(4.0 * r**2 + z_bar_k**2))
    t2_coef = -np.sin(theta_M) * np.cos(phi_bar_M) * np.sign(z_bar_k)
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r**2 / z_bar_k**2)
    E_coef = np.sin(theta_M) * np.sin(phi_bar_M) * z_bar_k**2 * np.sign(z_bar_k) / ( 2.0 * r**2 )
    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r**2 / z_bar_k**2)
    F_coef = -np.sin(theta_M) * np.sin(phi_bar_M) * np.sign(z_bar_k) * (4.0 * r**2 + z_bar_k**2) / ( 2.0 * r**2 )
    return t1_coef * t1 + t2_coef * t2 + E_coef * E + F_coef * F

def Hphi_zk_case224(r, theta_M, z_bar_k):
    t1 = np.sqrt(4.0 * r**2 + z_bar_k**2)
    t1_coef = np.cos(theta_M) / r
    t2 = np.arctanh(-2.0 * r / t1)
    t2_coef = np.cos(theta_M)
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
        return icels.el3_angle_vectorized(phi_bar_j/2.0, 2.0 * r / (r + sign * t), -4.0 * r**2 / z_bar_k**2)
    def Pi_coef(sign):
        return np.cos(theta_M) * z_bar_k * (r + sign * t) / ( np.abs(z_bar_k) * (r + sign * t) )
    return Pi_coef(1) * Pi(1) + Pi_coef(-1) * Pi(-1)

# 225 ##############

def Hr_ri_case225(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t1 = np.sqrt((r + r_i)**2 + z_bar_k**2) * z_bar_k / (2.0 * r**2)
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_M)
    t2 = np.arctanh(np.abs(z_bar_k) / np.sqrt((r + r_i)**2 + z_bar_k**2))
    t2_coef = np.sin(theta_M) * np.sin(phi_bar_M) * np.sign(z_bar_k) / 2.0 * (1.0 - r_i**2 / r**2)
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    E_coef = -np.sin(theta_M) * np.cos(phi_bar_M) * z_bar_k * np.sqrt(r_bar_i**2 + z_bar_k**2) / ( 2.0 * r**2 )
    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    F_coef = np.sin(theta_M) * np.cos(phi_bar_M) * z_bar_k * (2.0 * r_i**2 + z_bar_k**2) / ( 2.0 * r**2 * np.sqrt(r_bar_i**2 + z_bar_k**2) )
    Pi = icels.el3_angle_vectorized(phi_bar_j/2.0, -4.0 * r * r_i / r_bar_i**2, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
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
        return icels.el3_angle_vectorized(phi_bar_j/2.0, 2.0 * r / (r + sign * t), -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    def Pi1_coef(sign):
        return -np.cos(theta_M) / ( r * np.sqrt((r**2 +  z_bar_k**2) * (r_bar_i**2 + z_bar_k**2)) ) * (t - sign * r) * (r_i + sign * t)**2
    def Pi2(sign):
        return icels.el3_angle_vectorized(arctan_k_tan_2(np.sqrt(((r_i + r)**2 + z_bar_k**2)/(r_bar_i**2 + z_bar_k**2)),phi_bar_j), 1.0 - z_bar_k**2 * (r_bar_i**2 + z_bar_k**2) / ( ((r + r_i)**2 + z_bar_k**2) * (2.0 * r**2 + z_bar_k**2 + sign * 2.0 * r * t) ), 4.0 * r * r_i / ( (r + r_i)**2 + z_bar_k**2 ))
    def Pi2_coef(sign):
        return sign * np.cos(theta_M) * z_bar_k**2 * (r_bar_i**2 + z_bar_k**2) / ( r * np.sqrt((r**2 + z_bar_k**2) * ((r + r_i)**2 + z_bar_k**2)) * (r + sign * t) )
    return E_coef * E + F_coef * F + Pi1_coef(1) * Pi1(1) + Pi1_coef(-1) * Pi1(-1) + Pi2_coef(1) * Pi2(1) + Pi2_coef(-1) * Pi2(-1)

def Hphi_ri_case225(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t1 = np.sqrt((r + r_i)**2 + z_bar_k**2) * z_bar_k / (2.0 * r**2)
    t1_coef = -np.sin(theta_M) * np.cos(phi_bar_M)
    t2 = np.arctanh(np.abs(z_bar_k) / np.sqrt((r + r_i)**2 + z_bar_k**2))
    t2_coef = -np.sin(theta_M) * np.cos(phi_bar_M) * (r**2 + r_i**2) * np.sign(z_bar_k) / ( 2.0 * r**2 )
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    E_coef = np.sin(theta_M) * np.sin(phi_bar_M) * z_bar_k * np.sqrt(r_bar_i**2 + z_bar_k**2) / ( 2.0 * r**2 )
    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    F_coef = -np.sin(theta_M) * np.sin(phi_bar_M) * z_bar_k  * (2.0 * r**2 + 2.0 * r_i**2 + z_bar_k**2) / ( 2.0 * r**2 * np.sqrt(r_bar_i**2 + z_bar_k**2) )
    Pi = icels.el3_angle_vectorized(phi_bar_j/2.0, -4.0 * r * r_i / r_bar_i**2, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
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
        return icels.el3_angle_vectorized(phi_bar_j/2.0, 2.0 * r / (r + sign * t), -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
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
    t1 = -np.sin(theta_M) * z_bar_k / np.sqrt(r_i**2 + z_bar_k**2)
    t2 = 1.0/2.0 * phi_j * np.cos(phi_bar_M)
    t3 = 1.0/4.0 * np.sin(phi_bar_Mj + phi_bar_j)
    return -t1 * (t2 - t3)

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
    t1 = phi_j / np.sqrt(r_i**2 + z_bar_k**2)
    t1_coef = -np.cos(theta_M) * z_bar_k
    return t1_coef * t1

# 233 ##############

def Hr_phij_case233(r, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k):
    t1 = np.arctanh(z_bar_k / np.sqrt(r**2 + z_bar_k**2))
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_Mj) * np.cos(phi_bar_j)
    t2 = np.arctan(z_bar_k * np.abs(np.cos(phi_bar_j) / np.sin(phi_bar_j)) / np.sqrt(r**2 + z_bar_k**2))
    t2_coef = np.sin(theta_M) * np.sin(phi_bar_Mj) * np.abs(np.sin(phi_bar_j)) * np.sign(np.cos(phi_bar_j))
    return t1_coef * t1 + t2_coef * t2

def Hr_zk_case233(r, phi_bar_j, theta_M, z_bar_k):
    t = np.sqrt(r**2 + z_bar_k**2)
    t1 = np.sin(phi_bar_j)
    t1_coef = -np.cos(theta_M)
    t2 = np.log(-r * np.cos(phi_bar_j) + t)
    t2_coef = np.cos(theta_M) * np.sin(phi_bar_j)
    t3 = np.arctan(r * np.sin(phi_bar_j) / np.abs(z_bar_k))
    t3_coef = np.cos(theta_M) * np.abs(z_bar_k) / r
    t4 = arctan_k_tan_2(t / np.abs(z_bar_k), 2.0 * phi_bar_j)
    t4_coef = -t3_coef
    def t5(sign):
        return arctan_k_tan_2(np.abs(z_bar_k) / np.abs(r + sign * t), phi_bar_j)
    t5_coef = t3_coef
    return t1_coef * t1 + t2_coef * t2 + t3_coef * t3 + t4_coef * t4 + t5_coef * t5(1) + t5_coef * t5(-1)

def Hphi_phij_case233(r, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k):
    t = np.sqrt(r**2 + z_bar_k**2)
    t1 = np.arctan(np.abs(z_bar_k) * np.cos(phi_bar_j) / ( np.abs(np.sin(phi_bar_j)) * t ))
    t1_coef = np.sin(theta_M) * np.sin(phi_bar_Mj) * np.cos(phi_bar_j) * np.sign(np.sin(phi_bar_j)) * np.sign(z_bar_k)
    t2 = np.arctanh(np.abs(z_bar_k) / t)
    t2_coef = np.sin(theta_M) * np.sin(phi_bar_Mj) * np.sin(phi_bar_j) * np.sign(z_bar_k)
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
    t1 = np.sqrt(1.0 + z_bar_k**2 / (2.0 * r**2) - np.cos(phi_bar_j))
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_M) * z_bar_k / ( np.sqrt(2) * r )
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r**2 / z_bar_k**2)
    E_coef = -np.sin(theta_M) * np.cos(phi_bar_M) * z_bar_k**2 * np.sign(z_bar_k) / ( 2.0 * r**2 )
    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r**2 / z_bar_k**2)
    F_coef = np.sin(theta_M) * np.cos(phi_bar_M) * np.sign(z_bar_k) * (2.0 * r**2 + z_bar_k**2) / ( 2.0 * r**2 )
    return t1_coef * t1 + E_coef * E + F_coef * F

def Hr_phij_case234(r, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k):
    t1 = np.arctanh(z_bar_k / np.sqrt(2.0 * r**2 * (1.0 - np.cos(phi_bar_j)) + z_bar_k**2))
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_Mj) * np.cos(phi_bar_j)
    t2 = np.arctan(z_bar_k * (1.0 - np.cos(phi_bar_j)) / ( np.abs(np.sin(phi_bar_j)) * np.sqrt(2.0 * r**2 * (1.0 - np.cos(phi_bar_j)) + z_bar_k**2) ))
    t2_coef = -np.sin(theta_M) * np.sin(phi_bar_Mj) * np.abs(np.sin(phi_bar_j))
    return t1_coef * t1 + t2_coef * t2

def Hr_zk_case234(r, phi_bar_j, theta_M, z_bar_k):
    t1 = np.sin(phi_bar_j)
    t1_coef = -np.cos(theta_M)
    t2 = np.log( r * (1.0 - np.cos(phi_bar_j)) + np.sqrt(2.0 * r**2 * (1.0 - np.cos(phi_bar_j)) + z_bar_k**2) )
    t2_coef = np.cos(theta_M) * np.sin(phi_bar_j)
    t3 = np.arctan(r * np.sin(phi_bar_j) / np.abs(z_bar_k))
    t3_coef = np.cos(theta_M) * np.abs(z_bar_k) / r
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r**2 / z_bar_k**2)
    E_coef = np.cos(theta_M) * np.abs(z_bar_k) / r
    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r**2/ z_bar_k**2)
    F_coef = -np.cos(theta_M) * (2.0 * r**2 + z_bar_k**2) / ( r * np.abs(z_bar_k) )
    t = np.sqrt(r**2 + z_bar_k**2)
    def Pi1(sign):
        return icels.el3_angle_vectorized(phi_bar_j/2.0, 2.0 * r / (r + sign * t), -4.0 * r**2 / z_bar_k**2)
    def Pi1_coef(sign):
        return -np.cos(theta_M) / ( r * np.sqrt((r**2 + z_bar_k**2) * z_bar_k**2) ) * (t - sign * r) * (r + sign * t)**2
    def Pi2(sign):
        return icels.el3_angle_vectorized(arctan_k_tan_2(np.sqrt((4.0 * r**2 + z_bar_k**2)/z_bar_k**2),phi_bar_j), 1.0 - z_bar_k**4 / ( (4.0 * r**2 + z_bar_k**2) * (2.0 * r**2 + z_bar_k**2 + sign * 2.0 * r * t) ), 4.0 * r**2 / ( 4.0 * r**2 + z_bar_k**2 ))
    def Pi2_coef(sign):
        return sign * np.cos(theta_M) * z_bar_k**4 / ( r * np.sqrt((r**2 + z_bar_k**2) * (4.0 * r**2 + z_bar_k**2)) * (r + sign * t) )
    return t1_coef * t1 + t2_coef * t2 + t3_coef * t3 + E_coef * E + F_coef * F + Pi1_coef(1) * Pi1(1) + Pi1_coef(-1) * Pi1(-1) + Pi2_coef(1) * Pi2(1) + Pi2_coef(-1) * Pi2(-1)

def Hphi_ri_case234(r, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t1 = np.sqrt(1.0 + z_bar_k**2 / (2.0 * r**2) - np.cos(phi_bar_j))
    t1_coef = -np.sin(theta_M) * np.cos(phi_bar_M) * z_bar_k / ( np.sqrt(2.0) * r )
    t2 = np.arctanh(np.abs(z_bar_k) / np.sqrt(2.0 * r**2 * (1.0 - np.cos(phi_bar_j)) + z_bar_k**2))
    t2_coef = -np.sin(theta_M) * np.cos(phi_bar_M) * np.sign(z_bar_k)
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r**2 / z_bar_k**2)
    E_coef = np.sin(theta_M) * np.sin(phi_bar_M) * z_bar_k**2 * np.sign(z_bar_k) / ( 2.0 * r**2 )
    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r**2 / z_bar_k**2)
    F_coef = -np.sin(theta_M) * np.sin(phi_bar_M) * np.sign(z_bar_k) * (4.0 * r**2 + z_bar_k**2) / ( 2.0 * r**2 )
    return t1_coef * t1 + t2_coef * t2 + E_coef * E + F_coef * F

def Hphi_phij_case234(r, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k):
    t = np.sqrt(2.0 * r**2 * (1.0 - np.cos(phi_bar_j)) + z_bar_k**2)
    t1 = np.arctan(np.abs(z_bar_k) * (1.0 - np.cos(phi_bar_j)) / ( np.abs(np.sin(phi_bar_j)) * t ))
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_Mj) * np.cos(phi_bar_j) * np.sign(np.sin(phi_bar_j)) * np.sign(z_bar_k)
    t2 = np.arctanh(np.abs(z_bar_k) / t)
    t2_coef = np.sin(theta_M) * np.sin(phi_bar_Mj) * np.sin(phi_bar_j) * np.sign(z_bar_k)
    return t1_coef * t1 + t2_coef * t2

def Hphi_zk_case234(r, phi_bar_j, theta_M, z_bar_k):
    t1 = np.sqrt(2.0 * r**2 * (1.0 - np.cos(phi_bar_j)) + z_bar_k**2)
    t1_coef = np.cos(theta_M) / r
    t2 = np.arctanh(r * (1.0 - np.cos(phi_bar_j)) / t1)
    t2_coef = np.cos(theta_M) * np.cos(phi_bar_j)
    return t1_coef * t1 + t2_coef * t2

def Hz_ri_case234(r, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t1 = np.sqrt(2.0 * r**2 * (1.0 - np.cos(phi_bar_j)) + z_bar_k**2) / r
    t1_coef = np.sin(theta_M) * np.sin(phi_bar_M)
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
        return icels.el3_angle_vectorized(phi_bar_j/2.0, 2.0 * r / (r + sign * t), -4.0 * r**2 / z_bar_k**2)
    def Pi_coef(sign): # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< buuuuuuuuggggggssss
        return np.cos(theta_M) * np.sign(z_bar_k)
    return Pi_coef(1) * Pi(1) + Pi_coef(-1) * Pi(-1)

# 235 ##############

def Hr_ri_case235(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t1 = np.sqrt(r**2 + r_i**2 - 2.0 * r * r_i * np.cos(phi_bar_j) + z_bar_k**2) * z_bar_k / (2.0 * r**2)
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_M)
    t2 = np.arctanh(np.abs(z_bar_k) / np.sqrt(r**2 + r_i**2 - 2.0 * r * r_i * np.cos(phi_bar_j) + z_bar_k**2))
    t2_coef = np.sin(theta_M) * np.sin(phi_bar_M) * np.sign(z_bar_k) / 2.0 * (1.0 - r_i**2 / r**2)
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    E_coef = -np.sin(theta_M) * np.cos(phi_bar_M) * z_bar_k * np.sqrt(r_bar_i**2 + z_bar_k**2) / ( 2.0 * r**2 )
    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    F_coef = np.sin(theta_M) * np.cos(phi_bar_M) * z_bar_k * (2.0 * r_i**2 + z_bar_k**2) / ( 2.0 * r**2 * np.sqrt(r_bar_i**2 + z_bar_k**2) )
    Pi = icels.el3_angle_vectorized(phi_bar_j/2.0, -4.0 * r * r_i / r_bar_i**2, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    Pi_coef = np.sin(theta_M) * np.cos(phi_bar_M) * z_bar_k * (r**2 + r_i**2) * (r + r_i) / ( 2.0 * r**2 * r_bar_i * np.sqrt(r_bar_i**2 + z_bar_k**2) )
    return t1_coef * t1 + t2_coef * t2 + E_coef * E + F_coef * F + Pi_coef * Pi

def Hr_phij_case235(r, r_i, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k):
    t1 = np.arctanh(z_bar_k / np.sqrt(r**2 + r_i**2 - 2.0 * r * r_i * np.cos(phi_bar_j) + z_bar_k**2))
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_Mj) * np.cos(phi_bar_j)
    t2 = np.arctan(z_bar_k * np.abs(r * np.cos(phi_bar_j) - r_i) / ( r * np.abs(np.sin(phi_bar_j)) * np.sqrt(r**2 + r_i**2 - 2.0 * r * r_i * np.cos(phi_bar_j) + z_bar_k**2) ))
    t2_coef = -np.sin(theta_M) * np.sin(phi_bar_Mj) * np.abs(np.sin(phi_bar_j)) * np.sign(r_i - r * np.cos(phi_bar_j))
    return t1_coef * t1 + t2_coef * t2

def Hr_zk_case235(r, r_i, r_bar_i, phi_bar_j, theta_M, z_bar_k):
    t1 = np.sin(phi_bar_j)
    t1_coef = -np.cos(theta_M)
    t2 = np.log( r_i - r * np.cos(phi_bar_j) + np.sqrt(r_i**2 + r**2 - 2.0 * r_i * r * np.cos(phi_bar_j) + z_bar_k**2) )
    t2_coef = np.cos(theta_M) * np.sin(phi_bar_j)
    t3 = np.arctan(r * np.sin(phi_bar_j) / np.abs(z_bar_k))
    t3_coef = np.cos(theta_M) * np.abs(z_bar_k) / r
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    E_coef = np.cos(theta_M) * np.sqrt(r_bar_i**2 + z_bar_k**2) / r
    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    F_coef = -np.cos(theta_M) * (r**2 + r_i**2 + z_bar_k**2) / ( r * np.sqrt(r_bar_i**2 + z_bar_k**2) )
    t = np.sqrt(r**2 + z_bar_k**2)
    def Pi1(sign):
        return icels.el3_angle_vectorized(phi_bar_j/2.0, 2.0 * r / (r + sign * t), -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    def Pi1_coef(sign):
        return -np.cos(theta_M) / ( r * np.sqrt((r**2 +  z_bar_k**2) * (r_bar_i**2 + z_bar_k**2)) ) * (t - sign * r) * (r_i + sign * t)**2
    def Pi2(sign):
        return icels.el3_angle_vectorized(arctan_k_tan_2(np.sqrt(((r_i + r)**2 + z_bar_k**2)/(r_bar_i**2 + z_bar_k**2)),phi_bar_j), 1.0 - z_bar_k**2 * (r_bar_i**2 + z_bar_k**2) / ( ((r + r_i)**2 + z_bar_k**2) * (2.0 * r**2 + z_bar_k**2 + sign * 2.0 * r * t) ), 4.0 * r * r_i / ( (r + r_i)**2 + z_bar_k**2 ))
    def Pi2_coef(sign):
        return sign * np.cos(theta_M) * z_bar_k**2 * (r_bar_i**2 + z_bar_k**2) / ( r * np.sqrt((r**2 + z_bar_k**2) * ((r + r_i)**2 + z_bar_k**2)) * (r + sign * t) )
    return t1_coef * t1 + t2_coef * t2 + t3_coef * t3 + E_coef * E + F_coef * F + Pi1_coef(1) * Pi1(1) + Pi1_coef(-1) * Pi1(-1) + Pi2_coef(1) * Pi2(1) + Pi2_coef(-1) * Pi2(-1)

def Hphi_ri_case235(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t1 = np.sqrt(r**2 + r_i**2 - 2.0 * r * r_i * np.cos(phi_bar_j) + z_bar_k**2) * z_bar_k / (2.0 * r**2)
    t1_coef = -np.sin(theta_M) * np.cos(phi_bar_M)
    t2 = np.arctanh(np.abs(z_bar_k) / np.sqrt(r**2 + r_i**2 - 2.0 * r * r_i * np.cos(phi_bar_j) + z_bar_k**2))
    t2_coef = -np.sin(theta_M) * np.cos(phi_bar_M) * (r**2 + r_i**2) * np.sign(z_bar_k) / ( 2.0 * r**2 )
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    E_coef = np.sin(theta_M) * np.sin(phi_bar_M) * z_bar_k * np.sqrt(r_bar_i**2 + z_bar_k**2) / ( 2.0 * r**2 )
    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    F_coef = -np.sin(theta_M) * np.sin(phi_bar_M) * z_bar_k  * (2.0 * r**2 + 2.0 * r_i**2 + z_bar_k**2) / ( 2.0 * r**2 * np.sqrt(r_bar_i**2 + z_bar_k**2) )
    Pi = icels.el3_angle_vectorized(phi_bar_j/2.0, -4.0 * r * r_i / r_bar_i**2, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    Pi_coef = np.sin(theta_M) * np.sin(phi_bar_M) * z_bar_k * (r + r_i)**2 / ( 2.0 * r**2 * np.sqrt(r_bar_i**2 + z_bar_k**2) )
    return t1_coef * t1 + t2_coef * t2 + E_coef * E + F_coef * F + Pi_coef * Pi

def Hphi_phij_case235(r, r_i, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k):
    t = np.sqrt(r**2 + r_i**2 - 2.0 * r * r_i * np.cos(phi_bar_j) + z_bar_k**2)
    t1 = np.arctan(np.abs(z_bar_k) * (r * np.cos(phi_bar_j) - r_i) / ( r * np.abs(np.sin(phi_bar_j)) * t ))
    t1_coef = np.sin(theta_M) * np.sin(phi_bar_Mj) * np.cos(phi_bar_j) * np.sign(np.sin(phi_bar_j)) * np.sign(z_bar_k)
    t2 = np.arctanh(np.abs(z_bar_k) / t)
    t2_coef = np.sin(theta_M) * np.sin(phi_bar_Mj) * np.sin(phi_bar_j) * np.sign(z_bar_k)
    return t1_coef * t1 + t2_coef * t2

def Hphi_zk_case235(r, r_i, phi_bar_j, theta_M, z_bar_k):
    t1 = np.sqrt(r**2 + r_i**2 - 2.0 * r * r_i * np.cos(phi_bar_j) + z_bar_k**2)
    t1_coef = np.cos(theta_M) / r
    t2 = np.arctanh((r * np.cos(phi_bar_j) - r_i) / t1)
    t2_coef = -np.cos(theta_M) * np.cos(phi_bar_j)
    return t1_coef * t1 + t2_coef * t2

def Hz_ri_case235(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t = r_bar_i**2 + z_bar_k**2
    t1 = np.sqrt(r**2 + r_i**2 - 2.0 * r * r_i * np.cos(phi_bar_j) + z_bar_k**2) / r
    t1_coef = np.sin(theta_M) * np.sin(phi_bar_M)
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
        return icels.el3_angle_vectorized(phi_bar_j/2.0, 2.0 * r / (r + sign * t), -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    def Pi_coef(sign):
        return np.cos(theta_M) * z_bar_k * (r_i + sign * t) / ( np.sqrt(r_bar_i**2 + z_bar_k**2) * (r + sign * t) )
    return Pi_coef(1) * Pi(1) + Pi_coef(-1) * Pi(-1)


####################
####################
####################
#calculation of all field components for each case
#especially these function show, which inputs are needed for the calculation
#full vectorization for all cases could be implemented here

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

def case212(r_i, phi_j, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    results = np.zeros((len(r_i), 3, 3))
    results[:,0,0] = Hr_ri_case212(r_i, phi_j, phi_bar_M, theta_M, z_bar_k)
    results[:,0,1] = Hr_phij_case212(r_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k)
    results[:,1,0] = Hphi_ri_case212(r_i, phi_j, phi_bar_M, theta_M, z_bar_k)
    results[:,1,2] = Hphi_zk_case212(r_i, theta_M, z_bar_k)
    results[:,2,0] = Hz_ri_case212(r_i, phi_bar_M, theta_M, z_bar_k)
    results[:,2,1] = Hz_phij_case212(r_i, phi_bar_M, theta_M, z_bar_k)
    results[:,2,2] = Hz_zk_case212(r_i, phi_j, theta_M, z_bar_k)
    return results

def case213(r, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    results = np.zeros((len(r), 3, 3))
    results[:,0,1] = Hr_phij_case213(r, phi_bar_M, theta_M, z_bar_k)
    results[:,0,2] = Hr_zk_case213(r, phi_bar_j, theta_M, z_bar_k)
    results[:,1,2] = Hphi_zk_case213(r, theta_M, z_bar_k)
    results[:,2,1] = Hz_phij_case213(r, phi_bar_M, theta_M, z_bar_k)
    results[:,2,2] = Hz_zk_case213(r, phi_bar_j, theta_M, z_bar_k)
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
    results[:,0,2] = Hr_zk_case223(r, phi_bar_j, theta_M, z_bar_k)
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

############
############
############
# calculates the antiderivate for a certain parameter set
# for the real field, this has to be evaluated 8 times at all bounds r_i, phi_j, z_k of the cylinder tile
# for vectorized computing, all input values could be 1D arrays

def antiderivate_final(r, phi, z, r_i, phi_j, z_k, phi_M, theta_M):

    r_bar_i = r - r_i
    phi_bar_j = phi - phi_j
    phi_bar_M = phi_M - phi
    phi_bar_Mj = phi_M - phi_j
    z_bar_k = z - z_k

    case_id = np.array([111, 112, 113, 114, 115, 121, 122, 123, 124, 125, 131, 132, 133, 134, 135, 211, 212, 213, 214, 215, 221, 222, 223, 224, 225, 231, 232, 233, 234, 235], dtype = np.uint8)

    n = len(r)
    results = np.zeros((n, 3, 3))

    cases = determine_cases(r, phi, z, r_i, phi_j, z_k)

    # all cases:
    # mask for each case is created and corresponding values are passed to functions

    #for id in case_id:
    #    mask = cases==id
    #    if any(mask):
    #        results[mask] = supercase(id,r, phi, z, r_i, phi_j, z_k, phi_M, theta_M)

    m0 = cases == case_id[0]
    if np.any(m0):
        results[m0,:,:] = np.nan

    m0 = cases == case_id[1]
    if np.any(m0):
        results[m0,:,:] = case112(r_i[m0], phi_bar_M[m0], theta_M[m0])

    m0 = cases == case_id[2]
    if np.any(m0):
        results[m0,:,:] = case113(r[m0], phi_bar_M[m0], theta_M[m0])

    m0 = cases == case_id[3]
    if np.any(m0):
        results[m0,:,:] = np.nan

    m0 = cases == case_id[4]
    if np.any(m0):
        results[m0,:,:] = case115(r[m0], r_i[m0], r_bar_i[m0], phi_bar_j[m0], phi_bar_M[m0],
            theta_M[m0])

    m0 = cases == case_id[5]
    if(np.any(m0)):
        results[m0,:,:] = np.nan

    m0 = cases == case_id[6]
    if np.any(m0):
        results[m0,:,:] = case122(r_i[m0], phi_bar_M[m0], theta_M[m0])

    m0 = cases == case_id[7]
    if np.any(m0):
        results[m0,:,:] = case123(r[m0], phi_bar_M[m0], theta_M[m0])

    m0 = cases == case_id[8]
    if np.any(m0):
        results[m0,:,:] = case124(r[m0], phi_bar_M[m0], theta_M[m0])

    m0 = cases == case_id[9]
    if np.any(m0):
        results[m0,:,:] = case125(r[m0], r_i[m0], r_bar_i[m0], phi_bar_j[m0], phi_bar_M[m0],
            theta_M[m0])

    m0 = cases == case_id[10]
    if(np.any(m0)):
        results[m0,:,:] = np.nan

    m0 = cases == case_id[11]
    if np.any(m0):
        results[m0,:,:] = case132(r[m0], r_i[m0], phi_bar_j[m0], phi_bar_Mj[m0], theta_M[m0])

    m0 = cases == case_id[12]
    if np.any(m0):
        results[m0,:,:] = case133(r[m0], phi_bar_j[m0], phi_bar_Mj[m0], theta_M[m0])

    m0 = cases == case_id[13]
    if np.any(m0):
        results[m0,:,:] = case134(r[m0], phi_bar_j[m0], phi_bar_M[m0], phi_bar_Mj[m0], theta_M[m0])

    m0 = cases == case_id[14]
    if np.any(m0):
        results[m0,:,:] = case135(r[m0], r_i[m0], r_bar_i[m0], phi_bar_j[m0],
            phi_bar_M[m0], phi_bar_Mj[m0], theta_M[m0])

    m0 = cases == case_id[15]
    if np.any(m0):
        results[m0,:,:] = case211(phi_j[m0], phi_bar_M[m0], theta_M[m0], z_bar_k[m0])

    m0 = cases == case_id[16]
    if np.any(m0):
        results[m0,:,:] = case212(r_i[m0], phi_j[m0], phi_bar_j[m0], phi_bar_M[m0],
            theta_M[m0], z_bar_k[m0])

    m0 = cases == case_id[17]
    if np.any(m0):
        results[m0,:,:] = case213(r[m0], phi_bar_j[m0], phi_bar_M[m0], theta_M[m0], z_bar_k[m0])

    m0 = cases == case_id[18]
    if np.any(m0):
        results[m0,:,:] = case214(r[m0], phi_j[m0], phi_bar_j[m0], phi_bar_M[m0],
            theta_M[m0], z_bar_k[m0])

    m0 = cases == case_id[19]
    if np.any(m0):
        results[m0,:,:] = case215(r[m0], r_i[m0], r_bar_i[m0], phi_bar_j[m0],
            phi_bar_M[m0], theta_M[m0], z_bar_k[m0])

    m0 = cases == case_id[20]
    if np.any(m0):
        results[m0,:,:] = case221(phi_j[m0], phi_bar_M[m0], theta_M[m0], z_bar_k[m0])

    m0 = cases == case_id[21]
    if np.any(m0):
        results[m0,:,:] = case222(r_i[m0], phi_j[m0], phi_bar_M[m0], theta_M[m0], z_bar_k[m0])

    m0 = cases == case_id[22]
    if np.any(m0):
        results[m0,:,:] = case223(r[m0], phi_bar_j[m0], phi_bar_M[m0], theta_M[m0], z_bar_k[m0])

    m0 = cases == case_id[23]
    if np.any(m0):
        results[m0,:,:] = case224(r[m0], phi_bar_j[m0], phi_bar_M[m0], theta_M[m0], z_bar_k[m0])

    m0 = cases == case_id[24]
    if np.any(m0):
        results[m0,:,:] = case225(r[m0], r_i[m0], r_bar_i[m0], phi_bar_j[m0],
            phi_bar_M[m0], theta_M[m0], z_bar_k[m0])

    m0 = cases == case_id[25]
    if np.any(m0):
        results[m0,:,:] = case231(phi_j[m0], phi_bar_j[m0], phi_bar_Mj[m0], theta_M[m0],
            z_bar_k[m0])

    m0 = cases == case_id[26]
    if np.any(m0):
        results[m0,:,:] = case232(r_i[m0], phi_j[m0], phi_bar_j[m0], phi_bar_M[m0],
            phi_bar_Mj[m0], theta_M[m0], z_bar_k[m0])

    m0 = cases == case_id[27]
    if np.any(m0):
        results[m0,:,:] = case233(r[m0], phi_bar_j[m0], phi_bar_Mj[m0], theta_M[m0], z_bar_k[m0])

    m0 = cases == case_id[28]
    if np.any(m0):
        results[m0,:,:] = case234(r[m0], phi_bar_j[m0], phi_bar_M[m0],
            phi_bar_Mj[m0], theta_M[m0], z_bar_k[m0])

    m0 = cases == case_id[29]
    if np.any(m0):
        results[m0,:,:] = case235(r[m0], r_i[m0], r_bar_i[m0], phi_bar_j[m0],
            phi_bar_M[m0], phi_bar_Mj[m0], theta_M[m0], z_bar_k[m0])

    return results

############
############
############
#final function for the field evaluation
#(r, phi, z): point for field evaluation in cylinder coordinates
#(r_i12, phi_j12, z_k12): limits of the cylinder tiles in cylinder coordinates
#(M, phi_M, theta_M): spherical coordinates of the homogenious magnetization: amplitude of magnetization, azimuthal angle, polar angle
#
#for vectorized computation, the input values have to be the following form
#if m denotes the number of different cylinder tiles and n the number of field evaluation points, the input parameters must be arrays with dimensions:
#r, phi, z: m x n
#r_i12, phi_j12, z_k12: m x 2
#M, phi_M, theta_M: m
#
#output is a m x n x 3 array, which contains the three components of the field in cylindrical coordinates for m cylinder tiles at n positions each

def getH_cy_section(obs_pos, dim, mag):
    """
    Cylinder section field
    obs_pos : ndarray, shape (N,3)
        observer positions (r,phi,z) in cy CS, units: [mm] [rad]
    dim: ndarray, shape (N,6)
        section dimensions (r1,r2,phi1,phi2,z1,z2) in cy CS , units: [mm] [rad]
    mag: ndarray, shape (N,3)
        magnetization vector (amp, phi, th) in spherical CS, units: [mT] [rad]
    """

    # dimension inputs
    r_i12, phi_j12, z_k12 = dim[:,:2], dim[:,2:4], dim[:,4:6]

    # tile up obs_pos
    #n_cy = len(r_i12)
    #obs_pos_tile = np.tile(obs_pos, (n_cy,1,1))
    #r, phi, z = np.moveaxis(obs_pos_tile, 2, 0)
    r, phi, z = obs_pos.T
    r = np.array([r]).T
    phi = np.array([phi]).T
    z = np.array([z]).T

    # magnetization
    M, phi_M, theta_M = mag.T

    ###
    #here, some reshapes have to be done to convert the input data to the proper vectorizable format
    #eventually also special cases (like e.g. scalar input) should be handled here in
    r_i12_concat = np.stack((r_i12[:,0], r_i12[:,0], r_i12[:,0], r_i12[:,0], r_i12[:,1], r_i12[:,1], r_i12[:,1], r_i12[:,1]), axis = 1)
    phi_j12_concat = np.stack((phi_j12[:,0], phi_j12[:,0], phi_j12[:,1], phi_j12[:,1], phi_j12[:,0], phi_j12[:,0], phi_j12[:,1], phi_j12[:,1]), axis = 1)
    z_k12_concat = np.stack((z_k12[:,0], z_k12[:,1], z_k12[:,0], z_k12[:,1], z_k12[:,0], z_k12[:,1], z_k12[:,0], z_k12[:,1]), axis = 1)

    m, n = np.shape(r)
    result_final = np.zeros((m, n, 3, 3))

    r_full = np.tile(r[:,:,np.newaxis],(1,1,8))
    phi_full = np.tile(phi[:,:,np.newaxis],(1,1,8))
    z_full = np.tile(z[:,:,np.newaxis],(1,1,8))

    r_i12_full = np.tile(r_i12_concat[:,np.newaxis,:],(1,n,1))
    phi_j12_full = np.tile(phi_j12_concat[:,np.newaxis,:],(1,n,1))
    z_k12_full = np.tile(z_k12_concat[:,np.newaxis,:],(1,n,1))

    M_full = np.tile(M[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis],(1,n,8,3,3))
    phi_M_full = np.tile(phi_M[:,np.newaxis,np.newaxis],(1,n,8))
    theta_M_full = np.tile(theta_M[:,np.newaxis,np.newaxis],(1,n,8))
    ###

    result = antiderivate_final(np.reshape(r_full, m * n * 8), np.reshape(phi_full, m * n * 8), np.reshape(z_full, m * n * 8), np.reshape(r_i12_full, m * n * 8), np.reshape(phi_j12_full, m * n * 8), np.reshape(z_k12_full, m * n * 8), np.reshape(phi_M_full, m * n * 8), np.reshape(theta_M_full, m * n * 8)) 

    result = np.reshape(result, (m, n, 8, 3, 3)) * M_full / (4.0 * np.pi)

    result_final = result[:,:,7,:,:] - result[:,:,6,:,:] - result[:,:,5,:,:] + result[:,:,4,:,:] - result[:,:,3,:,:] + result[:,:,2,:,:] + result[:,:,1,:,:] - result[:,:,0,:,:]

    return np.squeeze(np.sum(result_final, axis = -1))
