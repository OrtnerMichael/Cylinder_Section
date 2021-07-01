import numpy as np
from magpylib._lib.fields.field_BH_cylinder import field_BH_cylinder
from cylinder_tile.magpy_getH import field_BH_cylinder2
import magpylib as mag3
import matplotlib.pyplot as plt

mag3.Config.ITER_CYLINDER = 100
mag3.Config.EDGESIZE = 1e-10

N = 100
ezs = np.linspace(4,12,N)
pz = np.array([1-10**-ez for ez in ezs])

mag = np.array([[ 0.,  0., 100]]*N)
poso = np.array([[1, 1, z] for z in pz])
d = np.array([1]*N)
h = np.array([2]*N)

dim = np.array([d,h]).T

H0 = field_BH_cylinder(True, mag, dim, poso)

null = np.zeros(N)
eins = np.ones(N)
dim = np.array([null, d/2, null, eins*360, -h/2, h/2]).T
H1 = field_BH_cylinder2(True, mag, dim, poso)

#assert np.allclose(H0, H1)

plt.plot(ezs, H1[:,0])
plt.plot(ezs, H1[:,1])
plt.plot(ezs, H1[:,2])
plt.show()