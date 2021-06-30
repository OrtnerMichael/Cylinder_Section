"""
what is going on ?
"""

import numpy as np
import scipy.integrate as integrate

def xi(r, phi, z, r_dash, phi_dash, z_dash):
    """
    Hilfsterm der oft vorkommt, see xi in paper
    """
    return np.sqrt(r**2 + r_dash**2 - 2.0 * r * r_dash * np.cos(phi - phi_dash) + (z - z_dash)**2)


def H_r_ri(r, phi, z, r_i12, phi_j12, z_k12, phi_M, theta_M):

    def integrand(z_dash, phi_dash, r_i):

        t1 = np.sin(theta_M) * np.cos(phi_M - phi_dash)
        t2 = r - r_i * np.cos(phi - phi_dash)

        return t1 * t2 / xi(r, phi, z, r_i, phi_dash, z_dash)**3 * r_i

    result = 0.0
    
    for i in range(len(r_i12)):

        result -= (-1)**i * integrate.dblquad(integrand, phi_j12[0], phi_j12[1], lambda x: z_k12[0], lambda x: z_k12[1], args=(r_i12[i],), epsabs=1.49e-08, epsrel=1.49e-08)[0]

    return result



def H_r_phij(r, phi, z, r_i12, phi_j12, z_k12, phi_M, theta_M):

    def integrand(z_dash, r_dash, phi_j):

        t1 = np.sin(theta_M) * np.sin(phi_M - phi_j)
        t2 = r - r_dash * np.cos(phi - phi_j)

        return t1 * t2 / xi(r, phi, z, r_dash, phi_j, z_dash)**3 

    result = 0.0
    
    for j in range(len(phi_j12)):

        result -= (-1)**j * integrate.dblquad(integrand, r_i12[0], r_i12[1], lambda x: z_k12[0], lambda x: z_k12[1], args=(phi_j12[j],), epsabs=1.49e-08, epsrel=1.49e-08)[0]

    return result


def H_r_zk(r, phi, z, r_i12, phi_j12, z_k12, phi_M, theta_M):

    def integrand(r_dash, phi_dash, z_k):

        t1 = np.cos(theta_M)
        t2 = r - r_dash * np.cos(phi - phi_dash)

        return t1 * t2 / xi(r, phi, z, r_dash, phi_dash, z_k)**3 * r_dash

    result = 0.0
    
    for k in range(len(z_k12)):

        result -= (-1)**k * integrate.dblquad(integrand, phi_j12[0], phi_j12[1], lambda x: r_i12[0], lambda x: r_i12[1], args=(z_k12[k],), epsabs=1.49e-08, epsrel=1.49e-08)[0]

    return result



def H_phi_ri(r, phi, z, r_i12, phi_j12, z_k12, phi_M, theta_M):

    def integrand(z_dash, phi_dash, r_i):

        t1 = np.sin(theta_M) * np.cos(phi_M - phi_dash)
        t2 = r_i**2 * np.sin(phi - phi_dash)

        return t1 * t2 / xi(r, phi, z, r_i, phi_dash, z_dash)**3

    result = 0.0
    
    for i in range(len(r_i12)):

        result -= (-1)**i * integrate.dblquad(integrand, phi_j12[0], phi_j12[1], lambda x: z_k12[0], lambda x: z_k12[1], args=(r_i12[i],), epsabs=1.49e-08, epsrel=1.49e-08)[0]

    return result


def H_phi_phij(r, phi, z, r_i12, phi_j12, z_k12, phi_M, theta_M):

    def integrand(z_dash, r_dash, phi_j):

        t1 = np.sin(theta_M) * np.sin(phi_M - phi_j)
        t2 = r_dash * np.sin(phi - phi_j)

        return t1 * t2 / xi(r, phi, z, r_dash, phi_j, z_dash)**3 

    result = 0.0
    
    for j in range(len(phi_j12)):

        result -= (-1)**j * integrate.dblquad(integrand, r_i12[0], r_i12[1], lambda x: z_k12[0], lambda x: z_k12[1], args=(phi_j12[j],), epsabs=1.49e-08, epsrel=1.49e-08)[0]

    return result


def H_phi_zk(r, phi, z, r_i12, phi_j12, z_k12, phi_M, theta_M):

    def integrand(r_dash, phi_dash, z_k):

        t1 = np.cos(theta_M)
        t2 = r_dash**2 * np.sin(phi - phi_dash)

        return t1 * t2 / xi(r, phi, z, r_dash, phi_dash, z_k)**3

    result = 0.0
    
    for k in range(len(z_k12)):

        result -= (-1)**k * integrate.dblquad(integrand, phi_j12[0], phi_j12[1], lambda x: r_i12[0], lambda x: r_i12[1], args=(z_k12[k],), epsabs=1.49e-08, epsrel=1.49e-08)[0]

    return result




def H_z_ri(r, phi, z, r_i12, phi_j12, z_k12, phi_M, theta_M):

    def integrand(z_dash, phi_dash, r_i):

        t1 = np.sin(theta_M) * np.cos(phi_M - phi_dash)
        t2 = (z - z_dash)

        return t1 * t2 / xi(r, phi, z, r_i, phi_dash, z_dash)**3 * r_i

    result = 0.0
    
    for i in range(len(r_i12)):

        result -= (-1)**i * integrate.dblquad(integrand, phi_j12[0], phi_j12[1], lambda x: z_k12[0], lambda x: z_k12[1], args=(r_i12[i],), epsabs=1.49e-08, epsrel=1.49e-08)[0]

    return result


def H_z_phij(r, phi, z, r_i12, phi_j12, z_k12, phi_M, theta_M):

    def integrand(z_dash, r_dash, phi_j):

        t1 = np.sin(theta_M) * np.sin(phi_M - phi_j)
        t2 = (z - z_dash)

        return t1 * t2 / xi(r, phi, z, r_dash, phi_j, z_dash)**3 

    result = 0.0
    
    for j in range(len(phi_j12)):

        result -= (-1)**j * integrate.dblquad(integrand, r_i12[0], r_i12[1], lambda x: z_k12[0], lambda x: z_k12[1], args=(phi_j12[j],), epsabs=1.49e-08, epsrel=1.49e-08)[0]

    return result


def H_z_zk(r, phi, z, r_i12, phi_j12, z_k12, phi_M, theta_M):

    def integrand(r_dash, phi_dash, z_k):

        t1 = np.cos(theta_M)
        t2 = (z - z_k)

        return t1 * t2 / xi(r, phi, z, r_dash, phi_dash, z_k)**3 * r_dash

    result = 0.0
    
    for k in range(len(z_k12)):

        result -= (-1)**k * integrate.dblquad(integrand, phi_j12[0], phi_j12[1], lambda x: r_i12[0], lambda x: r_i12[1], args=(z_k12[k],), epsabs=1.49e-08, epsrel=1.49e-08)[0]

    return result


