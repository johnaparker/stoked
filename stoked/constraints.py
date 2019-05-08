import numpy as np
import quaternion

def constrain_position(x=None, y=None, z=None, idx=None):
    if idx is None:
        idx = np.s_[:]

    idy = []
    vals = []
    if x is not None:
        idy.append(0)
        vals.append(x)

    if y is not None:
        idy.append(1)
        vals.append(y)

    if z is not None:
        idy.append(2)
        vals.append(z)

    index = np.s_[idx,idy]
    vals = np.asarray(vals)

    def contraint(position, orientation):
        position[index] = vals

    return contraint

def constrain_rotation(x=False, y=False, z=False):
    idy = []
    if x:
        idy.append(0)

    if y:
        idy.append(1)

    if z:
        idy.append(2)


    def contraint(position, orientation):
        for i in range(len(orientation)):
            vec = orientation[i].vec
            vec[idy] = 0
            q = quaternion.quaternion(orientation[i].real, *vec)
            orientation[i] = q.normalized()

    return contraint
