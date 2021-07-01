import numpy as np
from magpylib._lib.fields.field_BH_cylinder import field_BH_cylinder
from cylinder_tile.magpy_getH import field_BH_cylinder2
from time import perf_counter as pf
import magpylib as mag3

mag3.Config.ITER_CYLINDER = 100
mag3.Config.EDGESIZE = 1e-10
N = 100000

mag = (np.random.rand(N,3)-.5)*1000
poso = (np.random.rand(N,3)-.5)
d, h = np.random.rand(2,N)

mag[:,0] = 0
mag[:,1] = 0

N = 1
mag = np.array([[ 0.,  0., 100]])
poso = np.array([[-0.49872588, -0.16649879,  0.06560895]])
d = np.array([0.025470189348666072])
h = np.array([0.1311427223360241])

print(h/2)

dim = np.array([d,h]).T

T0 = pf()
H0 = field_BH_cylinder(True, mag, dim, poso)
T1 = pf()
print(f'time - iter: {T1-T0}')

null = np.zeros(N)
eins = np.ones(N)
dim = np.array([null, d/2, null, eins*360, -h/2, h/2]).T
T0 = pf()
H1 = field_BH_cylinder2(True, mag, dim, poso)
T1 = pf()
print(f'time - ana: {T1-T0}')


for i,(h0,h1) in enumerate(zip(H0,H1)):

    if not np.allclose(h0,h1):
        print((h0-h1)/np.linalg.norm(h0))
        #print(mag[i])
        #print(poso[i])
        #print(d[i])
        #print(h[i])

#assert np.allclose(H0, H1)
