import numpy as np
#from numba import jit, njit, float64, types
import os

os.environ['NUMBA_DISABLE_JIT'] = '1'


#@njit(float64(float64,float64,float64,float64))
def elliptic(kc, p, c, s):
    if kc == 0:
        raise RuntimeError('FAIL')
    errtol = .000001
    k = abs(kc)
    pp = p
    cc = c
    ss = s
    em = 1.
    if p > 0:
        pp = np.sqrt(p)
        ss = s/pp
    else:
        f = kc*kc
        q = 1.-f
        g = 1. - pp
        f = f - pp
        q = q*(ss - c*pp)
        pp = np.sqrt(f/g)
        cc = (c-ss)/g
        ss = -q/(g*g*pp) + cc*pp
    f = cc
    cc = cc + ss/pp
    g = k/pp
    ss = 2*(ss + f*g)
    pp = g + pp
    g = em
    em = k + em
    kk = k
    while abs(g-k) > g*errtol:
        k = 2*np.sqrt(kk)
        kk = k*em
        f = cc
        cc = cc + ss/pp
        g = kk/pp
        ss = 2*(ss + f*g)
        pp = g + pp
        g = em
        em = k+em
    return(np.pi/2)*(ss+cc*em)/(em*(em+pp))









#@njit(float64(float64,float64,float64,float64,float64))
def ell3(x, m, n, kc, cc):
    """
    Numerical Calculation of Elliptic Integrals and Elliptic Functions
    ROLAND BULIRSCH
    Numerische Mathematik 7, 78--90 (t965)
    """
    
    nm = n + m

    if (kc != 0 and m <= 1 and n >= 0 and nm >= 0):
        
        d = np.zeros(12)            #fix magic number 12
        b = x*x
        dn = np.sqrt((1 + kc*kc * b)/(1 + b))
        i = 0
        e = 0.0
        y = 0.0
        y1 = 0.0
        q = 1.0
        m0 = 1.0
        kc = np.abs(kc)

        while True:
            e = y1 + e
            y1 = y
            k = m/4
            m1 = (m0 + kc) / 2
            b = np.sqrt(n + b)
            y = np.arctan(b * x) / b
            dn = np.sqrt((kc + m0 * dn)/((1 + dn) * m1))
            x = dn * x
            b = (np.sqrt(n * nm) + n)/2
            n = b + k
            q = q * 2 / (k+n)
            d[i] = k * q
            q = b * q
            y = q * y
            b = m1*m1
            m = k*k / b
            nm = n + m

            if(m > cc * b):
                kc = np.sqrt(m0 * kc)
                m0 = m1
                i += 1
            else:
                break

        y /= 2
        y = (y - y1) + y                        #seltsam...
        b = q*m / (np.sqrt(n*nm) + nm)

        for j in range(i,-1,-1):
            b = d[j] + b
            result = np.arctan(x*m1) * b/m1 + y/2 - e/2

    else:
        raise RuntimeError('FAIL')

    return result


#@njit(types.Tuple((float64, types.int64))(float64))
#def crack(x):
#    b=np.log(2)
#    a=np.log(x)
#    k=int(a/b)+1
#    a=a-k*b
#    m=np.exp(a)
#    return m,k
    
#def arsinh(x):
#    a=x*x
#    if(a<1):
#        arsinh=x*np.polynomial.chebyshev.Chebyshev(a+a-1,'cAs','nAs')
#    else:
#        y=abs(x)
#        if(a>10e16):
#            a=np.log(y)+0.6931471805599453
#        else:
#            a=np.log(y+np.sqrt(1+a))
#        if(x>0):
#            arsinh=a
#        else:
#            arsinh=-a
#    return arsinh







#@njit(float64(float64,float64,float64))       
def el3(x, kc, p):
    if(x==0.0):
        return 0.0
    else:

#        t1=time.time()
        ############
        ##init

        #double
        am=0.0
        ap=0.0
        c=0.0
        d=0.0
        de=0.0
        fa=0.0
        g=0.0
        p1=0.0
        pz=0.0
        q=0.0
        u=0.0
        v=0.0
        w=0.0
        y=0.0
        ye=0.0
        zd=0.0

        #int
        k=0
        km2=0
        l=0
        m=0
        n=0

        #boolean
        bo=False
        bk=False
        ################
        D=8
        CA=10.0**(-D/2)
        CB=10.0**(-D-2)
        ND=D-2
        ln2=np.log(2)
        ra=np.zeros(ND-1,dtype=np.float64)
        rb=np.zeros(ND-1,dtype=np.float64)
        rr=np.zeros(ND-1,dtype=np.float64)
        pi=np.pi
        hh=x*x
        f=p*hh
        s=CA/(1.0+np.abs(x)) if (kc==0.0) else kc
        t=s*s
        pm=0.5*t
        e=hh*t
        z=np.abs(f)
        r=np.abs(p)
        h=1.0+hh

#        t2=time.time()
#        print('t2-t1',t2-t1)
        
        if(e < 0.1 and z < 0.1 and t < 1.0 and r < 1.0):
            for k in range(2,ND+1):
                km2=int(k-2)
                rb[km2]=0.5/k
                ra[km2]=1.0-rb[km2]
            zd=0.5/(ND+1)
            s=p+pm
            for k in range(2,ND+1):
                km2=int(k-2)
                rr[km2]=s
                pm=pm*t*ra[km2]
                s=s*p+pm
            u=s*zd
            s=u
            bo=False
            for k in range(ND,1,-1):
                km2=int(k-2)
                u=u+(rr[km2]-u)*rb[km2]
                bo=not bo
                v=-u if (bo) else u
                s=s*hh+v
            if(bo):
                s=-s
            u=(u+1.0)*0.5
            return (u-s*h)*np.sqrt(h)*x+u*np.arcsinh(x)
        w=1.0+f
        if(w==0.0):
            raise RuntimeError('FAIL')
        p1=CB/hh if (p==0.0) else p
        s=np.abs(s)
        y=np.abs(x)
        g=p1-1.0
        if(g==0.0):
            g=CB
        f=p1-t
        if(f==0.0):
            f=CB*t
        am=1.0-t
        ap=1.0+e
        r=p1*h
        fa=g/(f*p1)
        bo=fa>0.0
        fa=np.abs(fa)
        pz=np.abs(g*f)
        de=np.sqrt(pz)
        q=np.sqrt(np.abs(p1))
        if(pm>0.5):
            pm=0.5
        pm=p1-pm

#        t3=time.time()
#        print('t3-t2',t3-t2)
        if(pm>=0.0):
            u=np.sqrt(r*ap)
            v=y*de
            if(g<0.0):
                v=-v
            d=1.0/q
            c=1.0
        else:
            u=np.sqrt(h*ap*pz)
            ye=y*q
            v=am*ye
            q=-de/g
            d=-am/de
            c=0.0
            pz=ap-r
        if(bo):
            r=v/u
            z=1.0
            k=1
            if(pm<0.0):
                h=y*np.sqrt(h/(ap*fa))
                h=1.0/h-h
                z=h-r-r
                r=2.0+r*h
                if(r==0.0):
                    r=CB
                if(z==0.0):
                    z=h*CB
                z=r=r/z
                w=pz
            u=u/w
            v=v/w
        else:
            t=u+np.abs(v)
            bk=True
            if p1<0.0:
                de=v/pz
                ye=u*ye
                ye=ye+ye
                u=t/pz
                v=(-f-g*e)/t
                t=pz*np.abs(w)
                z=(hh*r*f-g*ap+ye)/t
                ye=ye/t
            else:
                de=v/w
                ye=0.0
                u=(e+p1)/t
                v=t/w
                z=1.0
            if(s>1.0):
                h=u
                u=v
                v=h
        y=1.0/y
        e=s
        n=1
        t=1.0
        l=0
        m=0

#        t4=time.time()


#MAR
        while True:
#            t5=time.time()
#            print('t5-t4',t5-t4)
            y=y-e/y
            if(y==0.0):
                y=np.sqrt(e)*CB
            f=c
            c=d/q+c
            g=e/q
            d=f*g+d
            d=d+d
            q=g+q
            g=t
            t=s+t
            n=n+n
            m=m+m
            if(bo):
                if(z<0.0):
                    m=k+m
                k=np.sign(r)
                h=e/(u*u+v*v)
                u=u*(1.0+h)
                v=v*(1.0-h)
            else:
                r=u/v
                h=z*r
                z=h*z
                hh=e/v
                if(bk):
                    de=de/u
                    ye=ye*(h+1.0/h)+de*(1.0+r)
                    de=de*(u-hh)
                    bk=np.abs(ye)<1.0
                else:
                    ######################                    
                    #crack
                    b_crack=ln2
                    a_crack=np.log(x)
                    k=int(a_crack/b_crack)+1
                    a_crack=a_crack-k*b_crack
                    m=np.exp(a_crack)
                    ######################

                    m=m+k
            if(np.abs(g-s)>CA*g):
                if(bo):
                    g=(1.0/r-r)*0.5
                    hh=u+v*g
                    h=g*u-v
                    if(hh==0.0):
                        hh=u*CB
                    if(h==0.0):
                        h=v*CB
                    z=r*h
                    r=hh/h
                else:
                    u=u+e/u
                    v=v+hh
                s=np.sqrt(e)
                s=s+s
                e=s*t
                l=l+l
                if(y<0.0):
                    l+=1
            else:
                break

#        t5=time.time()
#        print('t5-t4',t5-t4)
        
        if(y<0.0):
            l+=1
        e=np.arctan(t/y)+pi*l
        e=e*(c*t+d)/(t*(t+q))
        if(bo):
            h=v/(t+u)
            z=1.0-r*h
            h=r+h
            if(z==0.0):
                z=CB
            if(z<0.0):
                m=m+np.sign(h)
            s=np.arctan(h/z)+m*pi
        else:
            s=np.arcsinh(ye) if (bk) else np.log(z)+m*ln2
            s=s*0.5
        e=(e+np.sqrt(fa)*s)/n        
        return e if (x>0.0) else -e


#@njit(float64(float64,float64,float64))       
def el3_angle(phi, n, m):
    kc = np.sqrt(1.-m)
    p = 1.-n

    D=8
    n = int(phi / np.pi)
    phi_red = phi - n*np.pi

    if(n <= 0 and phi_red < -np.pi/2.0):
        n -= 1
        phi_red += np.pi
    elif(n >= 0 and phi_red > np.pi/2.0):
        n += 1
        phi_red -= np.pi
    
    if(n!=0):
        cel3_res = elliptic(kc, p, 1.0, 1.0)
        if(phi_red>np.pi/2-10**(-D)):
            return (2*n+1)*cel3_res
        elif(phi_red<-np.pi/2+10**(-D)):
            return (2*n-1)*cel3_res
        else:
            return 2*n*cel3_res + el3(np.tan(phi), kc, p)
    else:
        if(phi_red>np.pi/2-10.0**(-D)):
            return elliptic(kc, p, 1.0, 1.0)
        elif(phi_red<-np.pi/2+10.0**(-D)):
            return -elliptic(kc, p, 1.0, 1.0)
        else:
            return el3(np.tan(phi), kc, p)
        

def el3_angle_vectorized(phi, n, m):

    results = np.zeros(len(phi))

    for i in range(len(phi)):
        results[i] = el3_angle(phi[i], n[i], m[i])

    return results

            
#n=3
#phi=1
#m=0.1
#
#x=np.tan(phi)
#kc=np.sqrt(1-m)
#p=-n+1  
#print(el3(x,kc,p))



#@jit(float64(float64,float64,float64))
#def ell3_2(x, kc, p):
#    """
#    Numerical Calculation of Elliptic Integrals and Elliptic Functions
#    ROLAND BULIRSCH
#    Numerische Mathematik 7, 78--90 (t965)
#    """
#    
#    if(x == 0):
#
#        el3 = 0
#    
#    else:
#
#        CA = 1.0e-6
#        CB = 1.0e-14
#        ND = 10
#
#        ra = np.zeros(ND-1)
#        rb = np.zeros(ND-1)
#        rr = np.zeros
#
#        ln2 = np.log(2)
#        hh = x*x
#        f = p*hh
#        s = CA/(1+np.abs(x)) if kc == 0 else kc
#        t = s*s
#        pm = t*0.5
#        e = hh*t
#        z = np.abs(f)
#        r = np.abs(p)
#        h = 1+hh
#
#        if(e<0.1 and z<0.1 and t<1 and r<1):
#            for k in range(2,ND+1):
#                
#
#
#    return el3




######################

#import matplotlib.pyplot as plt
#
#values=20
#n = np.array([0,1,3,10])
#k = np.linspace(0.01,1,values)
#res = np.zeros(values)
#
#for i in range(len(n)):
#    for j in range(values):
#        res[j] = ell3(1, k[j]**2, n[i], np.sqrt(1-k[j]**2), 1.0e-8)
#    
#    plt.plot(k,res)
#    plt.show()
#

#import time
#
#m=-5.0
#n=3.0
#x=-10.0

#
#res = ell3(x, m, n, np.sqrt(1-m), 1.0e-8)
#
#start = time.time()
#for i in range(10):
#    res = ell3(x, m, n, np.sqrt(1-m), 1.0e-8)
#end = time.time()
#print('time 3rd',end - start)
#
#print(res)
#
#res = el3(x, np.sqrt(1-m), n+1)


#
#print(res)
#
#
#phi = -30.0
#
#import scipy.integrate as integrate
#print(integrate.quad(lambda theta: 1/(1+n*np.sin(theta)**2)/np.sqrt(1-m*np.sin(theta)**2), 0, phi, epsabs=1.49e-12, epsrel=1.49e-12, limit=100, points=None, weight=None, wvar=None, wopts=None, maxp1=100, limlst=100)[0])
#
#print(el3_angle(phi, np.sqrt(1-m), n+1))
#
#
#
#for i in range(10):
#    start = time.time()
#    res = el3_2(x, np.sqrt(1-m), n+1)
#    end = time.time()
#    print('time 3rd',end - start)
#
#print(res)


###################


#import scipy.special
#
#start = time.time()
#for i in range(1):
#    res = scipy.special.ellipkinc(np.arctan(x), m)
#end = time.time()
#print('time 1st',end - start)
#
#start = time.time()
#for i in range(1):
#    res = scipy.special.ellipeinc(np.arctan(x), m)
#end = time.time()
#print('time 2nd',end - start)
#
#start = time.time()
#for i in range(1):
#    res = el3_angle(np.arctan(x), np.sqrt(1-m), n+1)
#end = time.time()
#print('time 3rd',end - start)
#
#start = time.time()
#for i in range(1):
#    res = scipy.special.ellipk( m)
#end = time.time()
#print('time complete 1st',end - start)
#
#start = time.time()
#for i in range(1):
#    res = scipy.special.ellipe( m)
#end = time.time()
#print('time complete 2nd',end - start)
#
#start = time.time()
#for i in range(1):
#    res = elliptic(np.sqrt(1-m), n+1, 1, 1)
#end = time.time()
#print('time complete 3rd',end - start)

###################

#ellipk_vec = np.vectorize(scipy.special.ellipk)
#
#m_vec = m*np.ones(5)
#
#start = time.time()
#for i in range(1):
#    res = ellipk_vec( m_vec)
#end = time.time()
#print('time complete 1st vec',end - start)


#print('mpmath')
#import mpmath
#
#
#start = time.time()
#for i in range(1):
#    res = mpmath.ellipf(np.arctan(x), m)
#end = time.time()
#print('time 1st',end - start)
#
#start = time.time()
#for i in range(1):
#    res = mpmath.ellipe(np.arctan(x), m)
#end = time.time()
#print('time 2nd',end - start)
#
#start = time.time()
#for i in range(1):
#    res = mpmath.ellippi(n,np.arctan(x), m)
#end = time.time()
#print('time 3rd',end - start)
#
#start = time.time()
#for i in range(1):
#    res = mpmath.ellipk( m)
#end = time.time()
#print('time complete 1st',end - start)
#
#start = time.time()
#for i in range(1):
#    res = mpmath.ellipe( m)
#end = time.time()
#print('time complete 2nd',end - start)
#
#start = time.time()
#for i in range(1):
#    res = mpmath.ellippi(n, m)
#end = time.time()
#print('time complete 3rd',end - start)
#

