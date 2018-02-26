import numpy as np
import scipy.io as sp

mat = sp.loadmat('/Users/User/project/data/chair.mat', squeeze_me=True)
coord3d = []

for c in range(1,193):
	val = mat['X'][c-1]
	if val.shape==(3,10) and not np.isnan(val).any():
		coord3d.append(val)

















