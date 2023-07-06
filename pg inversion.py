import pygimli.meshtools as mt
from pygimli.physics import ert
import pygimli as pg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygimli.frameworks.timelapse
cube=mt.createWorld(start=[0,-20], end=[63,0],worldMarker=True,marker=1)
pg.show(cube)

filename = "9.1.dat"
shm = pg.DataContainerERT(filename)
for s in shm.sensors():
    cube.createNode(s)
    cube.createNode(s - [0, 0.1])
mesh=mt.createMesh(cube,quality=34,smooth=True,area=1)
pg.show(mesh,showMesh=True)

# pg.wait()
data=ert.load('9.1.dat')
data['k']=ert.geometricFactors(data)
# print(data['k'])
#
data['rhoa']=data['k']*data['u']/data['i']*1000
# print(data['rhoa'])
# print(np.mean(data['rhoa']))
#
data['err']=ert.estimateError(data)
data.remove(data['rhoa']<0)
# data.remove(data['rhoa']>30)
ert.show(data)
# plt.show()
# data.save('simple.dat')
#
#
mgr=ert.ERTManager()
mgr.setData(data)
mgr.setMesh(mesh)
mgr.invert(lam=20,verbose=True)
# np.testing.assert_approx_equal(mgr.inv.chi2(), 0.7, significant=1)
mgr.saveResult(cMax=20)
# meshPD = pg.Mesh(mgr.paraDomain)
# mgr.saveResult()
