import numpy as np


def glog(x, lamdba):
    xarr = np.array(x)
    y = np.log10(xarr + np.sqrt(xarr ** 2 + lamdba))
    return y
