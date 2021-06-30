import numpy as np
# pylint: disable=no-member
import scipy.special
import florian_ell3_paper as ell3
#equality tolerance
tol = 1.0e-10
def tolerance(arg1, arg2):

    return np.abs(arg1 - arg2) < tol



#############
#help function for periodic continuation

def arctan_k_tan_2(k, phi):

    full_periods = np.round(phi / (2.0 * np.pi))
    phi_red = phi - full_periods * 2.0 * np.pi

    result = np.zeros(phi.shape)

    case1 = tolerance(phi_red, -np.pi)
    result[case1] = full_periods[case1] * np.pi - np.pi / 2.0
    case2 = tolerance(phi_red, np.pi)
    result[case2] = full_periods[case2] * np.pi + np.pi / 2.0
    case3 = ~(case1 + case2)
    result[case3] = full_periods[case3] * np.pi + np.arctan(k[case3] * np.tan(phi_red[case3] / 2.0))

    return result


#########################
#help function that determines the special case for each input parameter set
#the case number is a three digits integer, where the digits can be the following values
#first digit: 1 -> z = z_k, 2 -> z != z_k
# second digit: 1 -> phi - phi_j = n*pi with even n, 2 -> phi - phi_j = n*pi with odd n, 3 -> else
# third digit: 1 -> r = r_i = 0, 2 -> r = 0, r_i > 0, 3 -> r > 0, r_i = 0, 4 -> r = r_i > 0, 5 -> r, r_i >0  and r != r_i

def determine_cases(r, phi, z, r_i, phi_j, z_k, phi_M, theta_M):

    phi_bar_j = phi - phi_j

    if(tolerance(z, z_k)):

        if(tolerance(phi_bar_j % (2.0 * np.pi), 0.0)):

            if(tolerance(r, 0.0)):

                if(tolerance(r_i, 0.0)):

                    #print('on surface')
                    return 111

                else:#(r_i > 0.0):

                    return 112

            else:#(r > 0.0):

                if(tolerance(r_i, 0.0)):

                    return 113

                else:#(r_i > 0.0):
                    if(tolerance(r, r_i)):

                        #print('on surface')
                        return 114

                    else:#(r != r_i):

                        return 115

        elif(tolerance(phi_bar_j % (2.0 * np.pi), np.pi)):
            if(tolerance(r, 0.0)):
                if(tolerance(r_i, 0.0)):

                    #print('on surface')
                    return 121

                else:#(r_i > 0.0):

                    return 122

            else:#(r > 0.0):
                if(tolerance(r_i, 0.0)):

                    return 123

                else:#(r_i > 0.0):
                    if(tolerance(r, r_i)):

                        return 124

                    else:#(r != r_i)

                        return 125

        else:#(phi_bar_j % np.pi != np.pi):
            if(tolerance(r, 0.0)):
                if(tolerance(r_i, 0.0)):

                    #print('on surface')
                    return 131

                else:#(r_i > 0.0):

                    return 132

            else:#(r > 0.0):
                if(tolerance(r_i, 0.0)):

                    return 133

                else:#(r_i > 0.0):
                    if(tolerance(r, r_i)):

                        return 134

                    else:#(r != r_i):

                            return 135

    else:#(z != z_k):
        if(tolerance(phi_bar_j % (2.0 * np.pi), 0.0)):
            if(tolerance(r, 0.0)):
                if(tolerance(r_i, 0.0)):

                    return 211

                else:#(r_i > 0.0):

                    return 212

            else:#(r > 0.0):
                if(tolerance(r_i, 0.0)):

                    return 213

                else:#(r_i > 0.0):
                    if(tolerance(r, r_i)):

                        return 214

                    else:#(r != r_i):

                        return 215

        elif(tolerance(phi_bar_j % (2.0 * np.pi), np.pi)):
            if(tolerance(r, 0.0)):
                if(tolerance(r_i, 0.0)):

                    return 221

                else:#(r_i > 0.0):

                    return 222

            else:#(r > 0.0):
                if(tolerance(r_i, 0.0)):

                    return 223

                else:#(r_i > 0.0):
                    if(tolerance(r, r_i)):

                        return 224

                    else:#(r != r_i)

                        return 225

        else:#(phi_bar_j % np.pi != 0.0):
            if(tolerance(r, 0.0)):
                if(tolerance(r_i, 0.0)):

                    return 231

                else:#(r_i > 0.0):

                    return 232

            else:#(r > 0.0):
                if(tolerance(r_i, 0.0)):

                    return 233

                else:#(r_i > 0.0):
                    if(tolerance(r, r_i)):

                        return 234

                    else:#(r != r_i):

                        return 235
            

#########################


#Implementation of all non-zero field components in every special case
#e.g. Hphi_zk stands for field component in phi-direction originating from the cylinder tile face at zk
########################

def Hphi_zk_case112(r_i, theta_M):

    return np.cos(theta_M) * np.log(r_i)

def Hz_ri_case112(phi_bar_M, theta_M):

    return -np.sin(theta_M) * np.sin(phi_bar_M)

def Hz_phij_case112(r_i, phi_bar_M, theta_M):

    return np.sin(theta_M) * np.sin(phi_bar_M) * np.log(r_i)

###################

def Hphi_zk_case113(r, theta_M):

    return -np.cos(theta_M) * np.log(r)

def Hz_phij_case113(r, phi_bar_M, theta_M):

    return -np.sin(theta_M) * np.sin(phi_bar_M) * np.log(r)

###################

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

def Hz_phij_case115(r, r_i, r_bar_i, phi_bar_M, theta_M):

    t1 = np.log(np.abs(r_bar_i)) * np.sign(r_bar_i)
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_M) 

    return t1_coef * t1

###################

def Hphi_zk_case122(r_i, theta_M):

    return -np.cos(theta_M) * np.log(r_i)

def Hz_ri_case122(phi_bar_M, theta_M):

    return np.sin(theta_M) * np.sin(phi_bar_M)

def Hz_phij_case122(r_i, phi_bar_M, theta_M):

    return -np.sin(theta_M) * np.sin(phi_bar_M) * np.log(r_i)

###################

def Hphi_zk_case123(r, theta_M):

    return -np.cos(theta_M) * np.log(r)

def Hz_phij_case123(r, phi_bar_M, theta_M):

    return -np.sin(theta_M) * np.sin(phi_bar_M) * np.log(r)

###################

def Hphi_zk_case124(r, theta_M):

    return np.cos(theta_M) * (1.0 - np.log(2.0 * r))

def Hz_ri_case124(phi_bar_M, theta_M):

    return 2.0 * np.sin(theta_M) * np.sin(phi_bar_M)

def Hz_phij_case124(r, phi_bar_M, theta_M):

    return -np.sin(theta_M) * np.sin(phi_bar_M) * np.log(2.0 * r)

###################

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

###################

def Hr_zk_case132(r_i, phi_bar_j, theta_M):

    return np.cos(theta_M) * np.sin(phi_bar_j) * np.log(r_i)

def Hphi_zk_case132(r_i, phi_bar_j, theta_M):

    return np.cos(theta_M) * np.cos(phi_bar_j) * np.log(r_i)

def Hz_ri_case132(phi_bar_Mj, theta_M):

    return -np.sin(theta_M) * np.sin(phi_bar_Mj)

def Hz_phij_case132(r_i, phi_bar_Mj, theta_M):

    return np.sin(theta_M) * np.sin(phi_bar_Mj) * np.log(r_i)

###################

def Hr_zk_case133(r, phi_bar_j, theta_M):

    return -np.cos(theta_M) * np.sin(phi_bar_j) + np.cos(theta_M) * np.sin(phi_bar_j) * np.log(r * (1.0 - np.cos(phi_bar_j)))

def Hphi_zk_case133(phi_bar_j, theta_M):

    return np.cos(theta_M) - np.cos(theta_M) * np.cos(phi_bar_j) * np.arctanh(np.cos(phi_bar_j))

def Hz_phij_case133(phi_bar_j, phi_bar_Mj, theta_M):

    return -np.sin(theta_M) * np.sin(phi_bar_Mj) * np.arctanh(np.cos(phi_bar_j))

###################

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

    return np.sqrt(2.0) * np.cos(theta_M) * np.sqrt(1.0 - np.cos(phi_bar_j)) + np.cos(theta_M) * np.cos(phi_bar_j) * np.arctanh(np.sqrt((1.0 - np.cos(phi_bar_j)) / 2.0))

def Hz_ri_case134(phi_bar_j, phi_bar_M, theta_M):

    t1 = np.sqrt(1.0 - np.cos(phi_bar_j))
    t1_coef = np.sqrt(2.0) * np.sin(theta_M) * np.sin(phi_bar_M)

    t2 = np.sin(phi_bar_j) / t1
    t2_coef = -np.sqrt(2.0) * np.sin(theta_M) * np.cos(phi_bar_M)

    t3 = np.arctanh(t2 / np.sqrt(2.0))
    t3_coef = np.sin(theta_M) * np.cos(phi_bar_M)

    return t1_coef * t1 + t2_coef * t2 + t3_coef * t3

def Hz_phij_case134(phi_bar_j, phi_bar_Mj, theta_M):

    return np.sin(theta_M) * np.sin(phi_bar_Mj) * np.arctanh(np.sqrt((1.0 - np.cos(phi_bar_j)) / 2.0))

###################

def Hr_zk_case135(r, r_i, r_bar_i, phi_bar_j, theta_M):

    t1 = np.sin(phi_bar_j)
    t1_coef = -np.cos(theta_M)

    t2 = np.log( r_i - r * np.cos(phi_bar_j) + np.sqrt(r_i**2 + r**2 - 2.0 * r_i * r * np.cos(phi_bar_j)) )
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

    t1 = np.arctanh((r * np.cos(phi_bar_j) - r_i) / np.sqrt(r**2 + r_i**2 - 2.0 * r * r_i * np.cos(phi_bar_j)))
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_Mj)

    return t1_coef * t1

###################

def Hr_phij_case211(phi_bar_M, theta_M, z_bar_k):

    return -np.sin(theta_M) * np.sin(phi_bar_M) * np.sign(z_bar_k) * np.log(np.abs(z_bar_k))

def Hz_zk_case211(phi_j, theta_M, z_bar_k):

    return -np.cos(theta_M) * np.sign(z_bar_k) * phi_j

###################

def Hr_ri_case212(r_i, phi_j, phi_bar_M, theta_M, z_bar_k):

    #terms
    t1 = np.sin(theta_M) * z_bar_k / np.sqrt(r_i**2 + z_bar_k**2)
    t2 = 1.0/2.0 * phi_j * np.cos(phi_bar_M)
    t3 = 1.0/4.0 * np.sin(phi_bar_M)

    return t1 * (t2 - t3)

def Hr_phij_case212(r_i, phi_bar_M, theta_M, z_bar_k):

    t1 = np.arctanh(z_bar_k / np.sqrt(r_i**2 + z_bar_k**2))
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_M)

    return t1_coef * t1 

def Hphi_ri_case212(r_i, phi_j, phi_bar_M, theta_M, z_bar_k):

    #terms
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

    #terms
    t1 = z_bar_k / np.sqrt(r_i**2 + z_bar_k**2)
    t1_coef = -np.cos(theta_M) * phi_j

    return t1_coef * t1

###################

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

def Hz_zk_case213(r, phi_bar_j, theta_M, z_bar_k):

    #terms
    t1 = np.sign(z_bar_k)
    t1_coef = np.cos(theta_M) * phi_bar_j

    return t1_coef * t1


###################

def Hr_ri_case214(r, phi_bar_j, phi_bar_M, theta_M, z_bar_k):

    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r**2 / z_bar_k**2)
    E_coef = -np.sin(theta_M) * np.cos(phi_bar_M) * z_bar_k**2 * np.sign(z_bar_k) / ( 2.0 * r**2 )

    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r**2 / z_bar_k**2)
    F_coef = np.sin(theta_M) * np.cos(phi_bar_M) * np.sign(z_bar_k) * (2.0 * r**2 + z_bar_k**2) / ( 2.0 * r**2 )

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
        return ell3.el3_angle_vectorized(phi_bar_j/2.0, 2.0 * r / (r + sign * t), -4.0 * r**2 / z_bar_k**2)
    def Pi1_coef(sign):
        return -np.cos(theta_M) / ( r * np.sqrt((r**2 +  z_bar_k**2) * z_bar_k**2) ) * (t - sign * r) * (r + sign * t)**2

    def Pi2(sign):
        return ell3.el3_angle_vectorized(phi_bar_j / 2.0, 1.0 - z_bar_k**4 / ( (4.0 * r**2 + z_bar_k**2) * (r + sign * t)**2 ), 4.0 * r**2 / ( 4.0 * r**2 + z_bar_k**2 ))
    def Pi2_coef(sign):
        return sign * np.cos(theta_M) * z_bar_k**4 / ( r * np.sqrt((r**2 + z_bar_k**2) * (4.0 * r**2 + z_bar_k**2)) * (r + sign * t) )

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
    F_coef = -np.sin(theta_M) * np.sin(phi_bar_M) * np.sign(z_bar_k) * (4.0 * r**2 + z_bar_k**2) / ( 2.0 * r**2 )

    return t1 + t2_coef * t2 + t3_coef * t3 + t4_coef * t4 + E_coef * E + F_coef * F


def Hphi_zk_case214(r, theta_M, z_bar_k):

    t1 = np.abs(z_bar_k)
    t1_coef = np.cos(theta_M) / r

    return t1_coef * t1

def Hz_ri_case214(r, phi_bar_j, phi_bar_M, theta_M, z_bar_k):

    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r**2 / z_bar_k**2)
    E_coef = np.sin(theta_M) * np.cos(phi_bar_M) * np.abs(z_bar_k) / r

    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r**2 / z_bar_k**2)
    F_coef = -np.sin(theta_M) * np.cos(phi_bar_M) * (2.0 * r**2 + z_bar_k**2) / (r * np.abs(z_bar_k))

    return np.sin(theta_M) * np.sin(phi_bar_M) * np.abs(z_bar_k) / r + E_coef * E + F_coef * F

def Hz_zk_case214(r, phi_bar_j, theta_M, z_bar_k):

    t = np.sqrt(r**2 + z_bar_k**2)

    #terms
    def Pi(sign):
        return ell3.el3_angle_vectorized(phi_bar_j/2.0, 2.0 * r / (r + sign * t), -4.0 * r**2 / z_bar_k**2)
    Pi_coef = np.cos(theta_M) * np.sign(z_bar_k)

    return Pi_coef * Pi(1) + Pi_coef * Pi(-1)

###################

def Hr_ri_case215(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k):

    t2 = np.arctanh(z_bar_k / np.sqrt(r_bar_i**2 + z_bar_k**2))
    t2_coef = np.sin(theta_M) * np.sin(phi_bar_M) / 2.0 * (1.0 - r_i**2 / r**2)
    
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    E_coef = -np.sin(theta_M) * np.cos(phi_bar_M) * z_bar_k * np.sqrt(r_bar_i**2 + z_bar_k**2) / ( 2.0 * r**2 )

    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    F_coef = np.sin(theta_M) * np.cos(phi_bar_M) * z_bar_k * (2.0 * r_i**2 + z_bar_k**2) / ( 2.0 * r**2 * np.sqrt(r_bar_i**2 + z_bar_k**2) )

    Pi = ell3.el3_angle_vectorized(phi_bar_j/2.0, -4.0 * r * r_i / r_bar_i**2, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    Pi_coef = np.sin(theta_M) * np.cos(phi_bar_M) * z_bar_k * (r**2 + r_i**2) * (r + r_i) / ( 2.0 * r**2 * r_bar_i * np.sqrt(r_bar_i**2 + z_bar_k**2) )

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
        return ell3.el3_angle_vectorized(phi_bar_j/2.0, 2.0 * r / (r + sign * t), -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    def Pi1_coef(sign):
        return -np.cos(theta_M) / ( r * np.sqrt((r**2 +  z_bar_k**2) * (r_bar_i**2 + z_bar_k**2)) ) * (t - sign * r) * (r_i + sign * t)**2

    def Pi2(sign):
        return ell3.el3_angle_vectorized(phi_bar_j / 2.0, 1.0 - z_bar_k**2 * (r_bar_i**2 + z_bar_k**2) / ( ((r + r_i)**2 + z_bar_k**2) * (r + sign * t)**2 ), 4.0 * r * r_i / ( (r + r_i)**2 + z_bar_k**2 ))
    def Pi2_coef(sign):
        return sign * np.cos(theta_M) * z_bar_k**2 * (r_bar_i**2 + z_bar_k**2) / ( r * np.sqrt((r**2 + z_bar_k**2) * ((r + r_i)**2 + z_bar_k**2)) * (r + sign * t) )

    return E_coef * E + F_coef * F + Pi1_coef(1) * Pi1(1) + Pi1_coef(-1) * Pi1(-1) + Pi2_coef(1) * Pi2(1) + Pi2_coef(-1) * Pi2(-1)
    

def Hphi_ri_case215(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k):

    t1 = np.sqrt(r_bar_i**2 + z_bar_k**2) * z_bar_k / (2.0 * r**2)
    t1_coef = -np.sin(theta_M) * np.cos(phi_bar_M)

    t2 = np.arctanh(z_bar_k / np.sqrt(r_bar_i**2 + z_bar_k**2))
    t2_coef = -np.sin(theta_M) * np.cos(phi_bar_M) * (r**2 + r_i**2) / ( 2.0 * r**2 )

    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    E_coef = np.sin(theta_M) * np.sin(phi_bar_M) * z_bar_k * np.sqrt(r_bar_i**2 + z_bar_k**2) / ( 2.0 * r**2 )

    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    F_coef = -np.sin(theta_M) * np.sin(phi_bar_M) * z_bar_k  * (2.0 * r**2 + 2.0 * r_i**2 + z_bar_k**2) / ( 2.0 * r**2 * np.sqrt(r_bar_i**2 + z_bar_k**2) )

    Pi = ell3.el3_angle_vectorized(phi_bar_j/2.0, -4.0 * r * r_i / r_bar_i**2, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
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

    #terms
    def Pi(sign):
        return ell3.el3_angle_vectorized(phi_bar_j/2.0, 2.0 * r / (r + sign * t), -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    def Pi_coef(sign):
        return np.cos(theta_M) * z_bar_k * (r_i + sign * t) / ( np.sqrt(r_bar_i**2 + z_bar_k**2) * (r + sign * t) )

    return Pi_coef(1) * Pi(1) + Pi_coef(-1) * Pi(-1)

###################

def Hr_phij_case221(phi_bar_M, theta_M, z_bar_k):

    return -np.sin(theta_M) * np.sin(phi_bar_M) * np.sign(z_bar_k) * np.log(np.abs(z_bar_k))

def Hz_zk_case221(phi_j, theta_M, z_bar_k):

    return -np.cos(theta_M) * np.sign(z_bar_k) * phi_j

###################

def Hr_ri_case222(r_i, phi_j, phi_bar_M, theta_M, z_bar_k):

    #terms
    t1 = np.sin(theta_M) * z_bar_k / np.sqrt(r_i**2 + z_bar_k**2)
    t2 = 1.0/2.0 * phi_j * np.cos(phi_bar_M)
    t3 = 1.0/4.0 * np.sin(phi_bar_M)

    return t1 * (t2 - t3)

def Hr_phij_case222(r_i, phi_bar_M, theta_M, z_bar_k):

    t1 = np.arctanh(z_bar_k / np.sqrt(r_i**2 + z_bar_k**2))
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_M)

    return t1_coef * t1 
    
def Hphi_ri_case222(r_i, phi_j, phi_bar_M, theta_M, z_bar_k):

    #terms
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

    #terms
    t1 = z_bar_k / np.sqrt(r_i**2 + z_bar_k**2)
    t1_coef = -np.cos(theta_M) * phi_j 

    return t1_coef * t1

###################

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

    #terms
    t1 = np.sign(z_bar_k)
    t1_coef = np.cos(theta_M) * phi_bar_j

    return t1_coef * t1

###################

def Hr_ri_case224(r, phi_bar_j, phi_bar_M, theta_M, z_bar_k):

    #terms
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
        return ell3.el3_angle_vectorized(phi_bar_j/2.0, 2.0 * r / (r + sign * t), -4.0 * r**2 / z_bar_k**2)
    def Pi1_coef(sign):
        return -np.cos(theta_M) / ( r * np.sqrt((r**2 + z_bar_k**2) * z_bar_k**2) ) * (t - sign * r) * (r + sign * t)**2

    def Pi2(sign):
        return ell3.el3_angle_vectorized(phi_bar_j / 2.0, 1.0 - z_bar_k**4 / ( (4.0 * r**2 + z_bar_k**2) * (r + sign * t)**2 ), 4.0 * r**2 / ( 4.0 * r**2 + z_bar_k**2 ))
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

    #terms
    def Pi(sign):
        return ell3.el3_angle_vectorized(phi_bar_j/2.0, 2.0 * r / (r + sign * t), -4.0 * r**2 / z_bar_k**2)
    Pi_coef = np.cos(theta_M) * np.sign(z_bar_k)

    return Pi_coef * Pi(1) + Pi_coef * Pi(-1)

###################

def Hr_ri_case225(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k):

    #terms
    t1 = np.sqrt((r + r_i)**2 + z_bar_k**2) * z_bar_k / (2.0 * r**2)
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_M)

    t2 = np.arctanh(z_bar_k / np.sqrt((r + r_i)**2 + z_bar_k**2))
    t2_coef = np.sin(theta_M) * np.sin(phi_bar_M) / 2.0 * (1.0 - r_i**2 / r**2)
    
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    E_coef = -np.sin(theta_M) * np.cos(phi_bar_M) * z_bar_k * np.sqrt(r_bar_i**2 + z_bar_k**2) / ( 2.0 * r**2 )

    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    F_coef = np.sin(theta_M) * np.cos(phi_bar_M) * z_bar_k * (2.0 * r_i**2 + z_bar_k**2) / ( 2.0 * r**2 * np.sqrt(r_bar_i**2 + z_bar_k**2) )

    Pi = ell3.el3_angle_vectorized(phi_bar_j/2.0, -4.0 * r * r_i / r_bar_i**2, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
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
        return ell3.el3_angle_vectorized(phi_bar_j/2.0, 2.0 * r / (r + sign * t), -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    def Pi1_coef(sign):
        return -np.cos(theta_M) / ( r * np.sqrt((r**2 +  z_bar_k**2) * (r_bar_i**2 + z_bar_k**2)) ) * (t - sign * r) * (r_i + sign * t)**2

    def Pi2(sign):
        return ell3.el3_angle_vectorized(phi_bar_j / 2.0, 1.0 - z_bar_k**2 * (r_bar_i**2 + z_bar_k**2) / ( ((r + r_i)**2 + z_bar_k**2) * (r + sign * t)**2 ), 4.0 * r * r_i / ( (r + r_i)**2 + z_bar_k**2 ))
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

    Pi = ell3.el3_angle_vectorized(phi_bar_j/2.0, -4.0 * r * r_i / r_bar_i**2, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
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

    #terms
    def Pi(sign):
        return ell3.el3_angle_vectorized(phi_bar_j/2.0, 2.0 * r / (r + sign * t), -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    def Pi_coef(sign):
        return np.cos(theta_M) * z_bar_k * (r_i + sign * t) / ( np.sqrt(r_bar_i**2 + z_bar_k**2) * (r + sign * t) )

    return Pi_coef(1) * Pi(1) + Pi_coef(-1) * Pi(-1)

###################

def Hr_phij_case231(phi_bar_j, phi_bar_Mj, theta_M, z_bar_k):

    return -np.sin(theta_M) * np.sin(phi_bar_Mj) * np.cos(phi_bar_j) * np.sign(z_bar_k) * np.log(np.abs(z_bar_k)) 

def Hphi_phij_case231(phi_bar_j, phi_bar_Mj, theta_M, z_bar_k):

    t1 = np.log(np.abs(z_bar_k))
    t1_coef = np.sin(theta_M) * np.sin(phi_bar_Mj) * np.sin(phi_bar_j) * np.sign(z_bar_k)

    return t1_coef * t1

def Hz_zk_case231(phi_j, theta_M, z_bar_k):

    #terms
    t1 = phi_j * np.sign(z_bar_k)
    t1_coef = -np.cos(theta_M)

    return t1_coef * t1

###################

def Hr_ri_case232(r_i, phi_j, phi_bar_j, phi_bar_M, phi_bar_Mj, theta_M, z_bar_k):

    #terms
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

    #terms
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

    #terms
    t1 = z_bar_k / np.sqrt(r_i**2 + z_bar_k**2)
    t1_coef = -np.cos(theta_M) * phi_j

    return t1_coef * t1

###################

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
    t4_coef = -np.cos(theta_M) * np.abs(z_bar_k) / r

    def t5(sign):
         return arctan_k_tan_2(np.abs(z_bar_k) / np.abs(r + sign * t), phi_bar_j)

    t5_coef = -t4_coef

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

    #terms
    t1 = arctan_k_tan_2(np.sqrt(r**2 + z_bar_k**2) / np.abs(z_bar_k), 2.0 * phi_bar_j)
    t1_coef = np.cos(theta_M) * np.sign(z_bar_k)

    return t1_coef * t1

###################

def Hr_ri_case234(r, phi_bar_j, phi_bar_M, theta_M, z_bar_k):

    #terms
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
        return ell3.el3_angle_vectorized(phi_bar_j/2.0, 2.0 * r / (r + sign * t), -4.0 * r**2 / z_bar_k**2)
    def Pi1_coef(sign):
        return -np.cos(theta_M) / ( r * np.sqrt((r**2 + z_bar_k**2) * z_bar_k**2) ) * (t - sign * r) * (r + sign * t)**2

    def Pi2(sign):
        return ell3.el3_angle_vectorized(arctan_k_tan_2(np.sqrt((4.0 * r**2 + z_bar_k**2)/z_bar_k**2),phi_bar_j), 1.0 - z_bar_k**4 / ( (4.0 * r**2 + z_bar_k**2) * (r + sign * t)**2 ), 4.0 * r**2 / ( 4.0 * r**2 + z_bar_k**2 ))
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

    #terms
    def Pi(sign):
        return ell3.el3_angle_vectorized(phi_bar_j/2.0, 2.0 * r / (r + sign * t), -4.0 * r**2 / z_bar_k**2)
    Pi_coef = np.cos(theta_M) * np.sign(z_bar_k)

    return Pi_coef * Pi(1) + Pi_coef * Pi(-1)

###################

def Hr_ri_case235(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k):

    #terms
    t1 = np.sqrt(r**2 + r_i**2 - 2.0 * r * r_i * np.cos(phi_bar_j) + z_bar_k**2)
    t1_coef = -np.sin(theta_M) * np.sin(phi_bar_M) * z_bar_k / (2.0 * r**2)

    t2 = np.arctanh(z_bar_k / np.sqrt(r**2 + r_i**2 - 2.0 * r * r_i * np.cos(phi_bar_j) + z_bar_k**2))
    t2_coef = np.sin(theta_M) * np.sin(phi_bar_M) / 2.0 * (1.0 - r_i**2 / r**2)
    
    E = scipy.special.ellipeinc(phi_bar_j/2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    E_coef = -np.sin(theta_M) * np.cos(phi_bar_M) * z_bar_k * np.sqrt(r_bar_i**2 + z_bar_k**2) / ( 2.0 * r**2 )

    F = scipy.special.ellipkinc(phi_bar_j/2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    F_coef = np.sin(theta_M) * np.cos(phi_bar_M) * z_bar_k * (2.0 * r_i**2 + z_bar_k**2) / ( 2.0 * r**2 * np.sqrt(r_bar_i**2 + z_bar_k**2) )

    Pi = ell3.el3_angle_vectorized(phi_bar_j/2.0, -4.0 * r * r_i / r_bar_i**2, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
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
        return ell3.el3_angle_vectorized(phi_bar_j/2.0, 2.0 * r / (r + sign * t), -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    def Pi1_coef(sign):
        return -np.cos(theta_M) / ( r * np.sqrt((r**2 +  z_bar_k**2) * (r_bar_i**2 + z_bar_k**2)) ) * (t - sign * r) * (r_i + sign * t)**2

    def Pi2(sign):
        return ell3.el3_angle_vectorized(arctan_k_tan_2(np.sqrt(((r_i + r)**2 + z_bar_k**2)/(r_bar_i**2 + z_bar_k**2)),phi_bar_j), 1.0 - z_bar_k**2 * (r_bar_i**2 + z_bar_k**2) / ( ((r + r_i)**2 + z_bar_k**2) * (r + sign * t)**2 ), 4.0 * r * r_i / ( (r + r_i)**2 + z_bar_k**2 ))
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

    Pi = ell3.el3_angle_vectorized(phi_bar_j/2.0, -4.0 * r * r_i / r_bar_i**2, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
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

    #terms
    def Pi(sign):
        return ell3.el3_angle_vectorized(phi_bar_j/2.0, 2.0 * r / (r + sign * t), -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
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

    n = len(r_i)

    results = np.zeros((n, 3, 3))
    
    results[:,1,2] = Hphi_zk_case112(r_i, theta_M)
    results[:,2,0] = Hz_ri_case112(phi_bar_M, theta_M)
    results[:,2,1] = Hz_phij_case112(r_i, phi_bar_M, theta_M)

    return results

def case113(r, phi_bar_M, theta_M):

    n = len(r)

    results = np.zeros((n, 3, 3))
    
    results[:,1,2] = Hphi_zk_case113(r, theta_M)
    results[:,2,1] = Hz_phij_case113(r, phi_bar_M, theta_M)

    return results

def case115(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M):

    n = len(r)

    results = np.zeros((n, 3, 3))
    
    results[:,0,2] = Hr_zk_case115(r, r_i, r_bar_i, phi_bar_j, theta_M)
    results[:,1,2] = Hphi_zk_case115(r, r_i, r_bar_i, theta_M)
    results[:,2,0] = Hz_ri_case115(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M)
    results[:,2,1] = Hz_phij_case115(r, r_i, r_bar_i, phi_bar_M, theta_M)

    return results

def case122(r_i, phi_bar_M, theta_M):

    n = len(r_i)

    results = np.zeros((n, 3, 3))
    
    results[:,1,2] = Hphi_zk_case122(r_i, theta_M)
    results[:,2,0] = Hz_ri_case122(phi_bar_M, theta_M)
    results[:,2,1] = Hz_phij_case122(r_i, phi_bar_M, theta_M)

    return results

def case123(r, phi_bar_M, theta_M):

    n = len(r)

    results = np.zeros((n, 3, 3))
    
    results[:,1,2] = Hphi_zk_case123(r, theta_M)
    results[:,2,1] = Hz_phij_case123(r, phi_bar_M, theta_M)

    return results

def case124(r, phi_bar_M, theta_M):

    n = len(r)

    results = np.zeros((n, 3, 3))
    
    results[:,1,2] = Hphi_zk_case124(r, theta_M)
    results[:,2,0] = Hz_ri_case124(phi_bar_M, theta_M)
    results[:,2,1] = Hz_phij_case124(r, phi_bar_M, theta_M)

    return results

def case125(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M):

    n = len(r)

    results = np.zeros((n, 3, 3))
    
    results[:,0,2] = Hr_zk_case125(r, r_i, r_bar_i, phi_bar_j, theta_M)
    results[:,1,2] = Hphi_zk_case125(r, r_i, theta_M)
    results[:,2,0] = Hz_ri_case125(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M)
    results[:,2,1] = Hz_phij_case125(r, r_i, phi_bar_M, theta_M)

    return results

def case132(r, r_i, r_bar_i, phi_bar_j, phi_bar_Mj, phi_bar_M, theta_M):

    n = len(r)

    results = np.zeros((n, 3, 3))
    
    results[:,0,2] = Hr_zk_case132(r_i, phi_bar_j, theta_M)
    results[:,1,2] = Hphi_zk_case132(r_i, phi_bar_j, theta_M)
    results[:,2,0] = Hz_ri_case132(phi_bar_Mj, theta_M)
    results[:,2,1] = Hz_phij_case132(r_i, phi_bar_Mj, theta_M)

    return results

def case133(r, phi_bar_j, phi_bar_Mj, theta_M):

    n = len(r)

    results = np.zeros((n, 3, 3))
    
    results[:,0,2] = Hr_zk_case133(r, phi_bar_j, theta_M)
    results[:,1,2] = Hphi_zk_case133(phi_bar_j, theta_M)
    results[:,2,1] = Hz_phij_case133(phi_bar_j, phi_bar_Mj, theta_M)

    return results

def case134(r, phi_bar_j, phi_bar_M, phi_bar_Mj, theta_M):

    n = len(r)

    results = np.zeros((n, 3, 3))
    
    results[:,0,2] = Hr_zk_case134(r, phi_bar_j, theta_M)
    results[:,1,2] = Hphi_zk_case134(phi_bar_j, theta_M)
    results[:,2,0] = Hz_ri_case134(phi_bar_j, phi_bar_M, theta_M)
    results[:,2,1] = Hz_phij_case134(phi_bar_j, phi_bar_Mj, theta_M)

    return results

def case135(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, phi_bar_Mj, theta_M):

    n = len(r)

    results = np.zeros((n, 3, 3))
    
    results[:,0,2] = Hr_zk_case135(r, r_i, r_bar_i, phi_bar_j, theta_M)
    results[:,1,2] = Hphi_zk_case135(r, r_i, phi_bar_j, theta_M)
    results[:,2,0] = Hz_ri_case135(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M)
    results[:,2,1] = Hz_phij_case135(r, r_i, phi_bar_j, phi_bar_Mj, theta_M)

    return results

def case211(phi_j, phi_bar_M, theta_M, z_bar_k):

    n = len(phi_j)

    results = np.zeros((n, 3, 3))
    
    results[:,0,1] = Hr_phij_case211(phi_bar_M, theta_M, z_bar_k)
    results[:,2,2] = Hz_zk_case211(phi_j, theta_M, z_bar_k)

    return results

def case212(r_i, phi_j, phi_bar_j, phi_bar_M, theta_M, z_bar_k):

    n = len(r_i)

    results = np.zeros((n, 3, 3))
    
    results[:,0,0] = Hr_ri_case212(r_i, phi_j, phi_bar_M, theta_M, z_bar_k)
    results[:,0,1] = Hr_phij_case212(r_i, phi_bar_M, theta_M, z_bar_k)
    results[:,1,0] = Hphi_ri_case212(r_i, phi_j, phi_bar_M, theta_M, z_bar_k)
    results[:,1,2] = Hphi_zk_case212(r_i, theta_M, z_bar_k)
    results[:,2,0] = Hz_ri_case212(r_i, phi_bar_M, theta_M, z_bar_k)
    results[:,2,1] = Hz_phij_case212(r_i, phi_bar_M, theta_M, z_bar_k)
    results[:,2,2] = Hz_zk_case212(r_i, phi_j, theta_M, z_bar_k)

    return results

def case213(r, phi_bar_j, phi_bar_M, theta_M, z_bar_k):

    n = len(r)

    results = np.zeros((n, 3, 3))
    
    results[:,0,1] = Hr_phij_case213(r, phi_bar_M, theta_M, z_bar_k)
    results[:,1,2] = Hphi_zk_case213(r, theta_M, z_bar_k)
    results[:,2,1] = Hz_phij_case213(r, phi_bar_M, theta_M, z_bar_k)
    results[:,2,2] = Hz_zk_case213(r, phi_bar_j, theta_M, z_bar_k)

    return results

def case214(r, phi_j, phi_bar_j, phi_bar_M, theta_M, z_bar_k):

    n = len(r)

    results = np.zeros((n, 3, 3))
    
    results[:,0,0] = Hr_ri_case214(r, phi_bar_j, phi_bar_M, theta_M, z_bar_k)
    results[:,0,1] = Hr_phij_case214(phi_bar_M, theta_M, z_bar_k)
    results[:,0,2] = Hr_zk_case214(r, phi_bar_j, theta_M, z_bar_k)
    results[:,1,0] = Hphi_ri_case214(r, phi_j, phi_bar_j, phi_bar_M, theta_M, z_bar_k)
    results[:,1,2] = Hphi_zk_case214(r, theta_M, z_bar_k)
    results[:,2,0] = Hz_ri_case214(r, phi_bar_j, phi_bar_M, theta_M, z_bar_k)
    results[:,2,2] = Hz_zk_case214(r, phi_bar_j, theta_M, z_bar_k)

    return results

def case215(r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k):

    n = len(r)

    results = np.zeros((n, 3, 3))
    
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

    n = len(phi_j)

    results = np.zeros((n, 3, 3))
    
    results[:,0,1] = Hr_phij_case221(phi_bar_M, theta_M, z_bar_k)
    results[:,2,2] = Hz_zk_case221(phi_j, theta_M, z_bar_k)

    return results

def case222(r_i, phi_j, phi_bar_M, theta_M, z_bar_k):

    n = len(r_i)

    results = np.zeros((n, 3, 3))
    
    results[:,0,0] = Hr_ri_case222(r_i, phi_j, phi_bar_M, theta_M, z_bar_k)
    results[:,0,1] = Hr_phij_case222(r_i, phi_bar_M, theta_M, z_bar_k)
    results[:,1,0] = Hphi_ri_case222(r_i, phi_j, phi_bar_M, theta_M, z_bar_k)
    results[:,1,2] = Hphi_zk_case222(r_i, theta_M, z_bar_k)
    results[:,2,0] = Hz_ri_case222(r_i, phi_bar_M, theta_M, z_bar_k)
    results[:,2,1] = Hz_phij_case222(r_i, phi_bar_M, theta_M, z_bar_k)
    results[:,2,2] = Hz_zk_case222(r_i, phi_j, theta_M, z_bar_k)

    return results

def case223(r, phi_bar_j, phi_bar_M, theta_M, z_bar_k):

    n = len(r)

    results = np.zeros((n, 3, 3))
    
    results[:,0,1] = Hr_phij_case223(r, phi_bar_M, theta_M, z_bar_k)
    results[:,1,2] = Hphi_zk_case223(r, theta_M, z_bar_k)
    results[:,2,1] = Hz_phij_case223(r, phi_bar_M, theta_M, z_bar_k)
    results[:,2,2] = Hz_zk_case223(r, phi_bar_j, theta_M, z_bar_k)

    return results

def case224(r, phi_bar_j, phi_bar_M, theta_M, z_bar_k):

    n = len(r)

    results = np.zeros((n, 3, 3))

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

    n = len(r)

    results = np.zeros((n, 3, 3))

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

    n = len(phi_j)

    results = np.zeros((n, 3, 3))

    results[:,0,1] = Hr_phij_case231(phi_bar_j, phi_bar_Mj, theta_M, z_bar_k)
    results[:,1,1] = Hphi_phij_case231(phi_bar_j, phi_bar_Mj, theta_M, z_bar_k)
    results[:,2,2] = Hz_zk_case231(phi_j, theta_M, z_bar_k)

    return results

def case232(r_i, phi_j, phi_bar_j, phi_bar_M, phi_bar_Mj, theta_M, z_bar_k):

    n = len(r_i)

    results = np.zeros((n, 3, 3))

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

    n = len(r)

    results = np.zeros((n, 3, 3))

    results[:,0,1] = Hr_phij_case233(r, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k)
    results[:,0,2] = Hr_zk_case233(r, phi_bar_j, theta_M, z_bar_k)
    results[:,1,1] = Hphi_phij_case233(r, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k)
    results[:,1,2] = Hphi_zk_case233(r, phi_bar_j, theta_M, z_bar_k)
    results[:,2,1] = Hz_phij_case233(r, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k)
    results[:,2,2] = Hz_zk_case233(r, phi_bar_j, theta_M, z_bar_k)

    return results

def case234(r, phi_bar_j, phi_bar_M, phi_bar_Mj, theta_M, z_bar_k):

    n = len(r)

    results = np.zeros((n, 3, 3))

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

    n = len(r)

    results = np.zeros((n, 3, 3))

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
#calculates the antiderivate for a certain parameter set
#for the real field, this has to be evaluated 8 times at all bounds r_i, phi_j, z_k of the cylinder tile
#for vectorized computing, all input values could be 1D arrays

def antiderivate_final(r, phi, z, r_i, phi_j, z_k, phi_M, theta_M):

    r_bar_i = r - r_i
    phi_bar_j = phi - phi_j
    phi_bar_M = phi_M - phi
    phi_bar_Mj = phi_M - phi_j
    z_bar_k = z - z_k
    
    case_id = np.array([111, 112, 113, 114, 115, 121, 122, 123, 124, 125, 131, 132, 133, 134, 135, 211, 212, 213, 214, 215, 221, 222, 223, 224, 225, 231, 232, 233, 234, 235], dtype = np.uint8)

    n = len(r)
    cases = np.zeros(n, dtype = np.uint8)
    results = np.zeros((n, 3, 3))

    for i in range(n):
        cases[i] = determine_cases(r[i], phi[i], z[i], r_i[i], phi_j[i], z_k[i], phi_M[i], theta_M[i])

    # all cases:
    # mask for each case is created and corresponding values are passed to functions

    cases_mask = cases == case_id[0]

    if(np.any(cases_mask)):
        #print('Warning...some Points on Surface')
        results[cases_mask,:,:] = np.nan

############

    cases_mask = cases == case_id[1]

    results[cases_mask,:,:] = case112(r_i[cases_mask], phi_bar_M[cases_mask], theta_M[cases_mask])
 
############

    cases_mask = cases == case_id[2]

    results[cases_mask,:,:] = case113(r[cases_mask], phi_bar_M[cases_mask], theta_M[cases_mask])

############

    cases_mask = cases == case_id[3]

    if(np.any(cases_mask)):
        #print('Warning...some Points on Surface')
        results[cases_mask,:,:] = np.nan

############

    cases_mask = cases == case_id[4]

    results[cases_mask,:,:] = case115(r[cases_mask], r_i[cases_mask], r_bar_i[cases_mask], phi_bar_j[cases_mask], phi_bar_M[cases_mask], theta_M[cases_mask])

############

    cases_mask = cases == case_id[5]

    if(np.any(cases_mask)):
        #print('Warning...some Points on Surface')
        results[cases_mask,:,:] = np.nan

############

    cases_mask = cases == case_id[6]

    results[cases_mask,:,:] = case122(r_i[cases_mask], phi_bar_M[cases_mask], theta_M[cases_mask])

############

    cases_mask = cases == case_id[7]

    results[cases_mask,:,:] = case123(r[cases_mask], phi_bar_M[cases_mask], theta_M[cases_mask])

############

    cases_mask = cases == case_id[8]

    results[cases_mask,:,:] = case124(r[cases_mask], phi_bar_M[cases_mask], theta_M[cases_mask])

############

    cases_mask = cases == case_id[9]

    results[cases_mask,:,:] = case125(r[cases_mask], r_i[cases_mask], r_bar_i[cases_mask], phi_bar_j[cases_mask], phi_bar_M[cases_mask], theta_M[cases_mask])

############

    cases_mask = cases == case_id[10]

    if(np.any(cases_mask)):
        #print('Warning...some Points on Surface')
        results[cases_mask,:,:] = np.nan

############

    cases_mask = cases == case_id[11]

    results[cases_mask,:,:] = case132(r[cases_mask], r_i[cases_mask], r_bar_i[cases_mask], phi_bar_j[cases_mask], phi_bar_Mj[cases_mask], phi_bar_M[cases_mask], theta_M[cases_mask])

############

    cases_mask = cases == case_id[12]

    results[cases_mask,:,:] = case133(r[cases_mask], phi_bar_j[cases_mask], phi_bar_Mj[cases_mask], theta_M[cases_mask])

############

    cases_mask = cases == case_id[13]

    results[cases_mask,:,:] = case134(r[cases_mask], phi_bar_j[cases_mask], phi_bar_M[cases_mask], phi_bar_Mj[cases_mask], theta_M[cases_mask])

############

    cases_mask = cases == case_id[14]

    results[cases_mask,:,:] = case135(r[cases_mask], r_i[cases_mask], r_bar_i[cases_mask], phi_bar_j[cases_mask], phi_bar_M[cases_mask], phi_bar_Mj[cases_mask], theta_M[cases_mask])

############

    cases_mask = cases == case_id[15]

    results[cases_mask,:,:] = case211(phi_j[cases_mask], phi_bar_M[cases_mask], theta_M[cases_mask], z_bar_k[cases_mask])

############

    cases_mask = cases == case_id[16]

    results[cases_mask,:,:] = case212(r_i[cases_mask], phi_j[cases_mask], phi_bar_j[cases_mask], phi_bar_M[cases_mask], theta_M[cases_mask], z_bar_k[cases_mask])

############

    cases_mask = cases == case_id[17]

    results[cases_mask,:,:] = case213(r[cases_mask], phi_bar_j[cases_mask], phi_bar_M[cases_mask], theta_M[cases_mask], z_bar_k[cases_mask])

############

    cases_mask = cases == case_id[18]

    results[cases_mask,:,:] = case214(r[cases_mask], phi_j[cases_mask], phi_bar_j[cases_mask], phi_bar_M[cases_mask], theta_M[cases_mask], z_bar_k[cases_mask])

############

    cases_mask = cases == case_id[19]

    results[cases_mask,:,:] = case215(r[cases_mask], r_i[cases_mask], r_bar_i[cases_mask], phi_bar_j[cases_mask], phi_bar_M[cases_mask], theta_M[cases_mask], z_bar_k[cases_mask])

############

    cases_mask = cases == case_id[20]

    results[cases_mask,:,:] = case221(phi_j[cases_mask], phi_bar_M[cases_mask], theta_M[cases_mask], z_bar_k[cases_mask])

############

    cases_mask = cases == case_id[21]

    results[cases_mask,:,:] = case222(r_i[cases_mask], phi_j[cases_mask], phi_bar_M[cases_mask], theta_M[cases_mask], z_bar_k[cases_mask])

############

    cases_mask = cases == case_id[22]

    results[cases_mask,:,:] = case223(r[cases_mask], phi_bar_j[cases_mask], phi_bar_M[cases_mask], theta_M[cases_mask], z_bar_k[cases_mask])

############

    cases_mask = cases == case_id[23]

    results[cases_mask,:,:] = case224(r[cases_mask], phi_bar_j[cases_mask], phi_bar_M[cases_mask], theta_M[cases_mask], z_bar_k[cases_mask])

############

    cases_mask = cases == case_id[24]

    results[cases_mask,:,:] = case225(r[cases_mask], r_i[cases_mask], r_bar_i[cases_mask], phi_bar_j[cases_mask], phi_bar_M[cases_mask], theta_M[cases_mask], z_bar_k[cases_mask])

############

    cases_mask = cases == case_id[25]

    results[cases_mask,:,:] = case231(phi_j[cases_mask], phi_bar_j[cases_mask], phi_bar_Mj[cases_mask], theta_M[cases_mask], z_bar_k[cases_mask])

############

    cases_mask = cases == case_id[26]

    results[cases_mask,:,:] = case232(r_i[cases_mask], phi_j[cases_mask], phi_bar_j[cases_mask], phi_bar_M[cases_mask], phi_bar_Mj[cases_mask], theta_M[cases_mask], z_bar_k[cases_mask])

############

    cases_mask = cases == case_id[27]

    results[cases_mask,:,:] = case233(r[cases_mask], phi_bar_j[cases_mask], phi_bar_Mj[cases_mask], theta_M[cases_mask], z_bar_k[cases_mask])

############

    cases_mask = cases == case_id[28]

    results[cases_mask,:,:] = case234(r[cases_mask], phi_bar_j[cases_mask], phi_bar_M[cases_mask], phi_bar_Mj[cases_mask], theta_M[cases_mask], z_bar_k[cases_mask])

############

    cases_mask = cases == case_id[29]

    results[cases_mask,:,:] = case235(r[cases_mask], r_i[cases_mask], r_bar_i[cases_mask], phi_bar_j[cases_mask], phi_bar_M[cases_mask], phi_bar_Mj[cases_mask], theta_M[cases_mask], z_bar_k[cases_mask])

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

def H_total_final(obs_pos, dim, mag):

    # dimension inputs
    r_i12, phi_j12, z_k12 = dim[:,:2], dim[:,2:4], dim[:,4:6]
    
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



