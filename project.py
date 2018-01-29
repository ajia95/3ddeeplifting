import numpy as np
import random

def projCoords(chair,f):
	"get 2D projected coords from 3D coords"
	projC = np.empty([2,10], dtype = float)
	#centre of image pixel coords
	u = 0
	v = 0
	pMat = np.array([[f,0,u,0],
					 [0,f,v,0],
					 [0,0,1,0]])

	for c in range(0,10):
		#homogenous coords
		hCoord = np.array([chair[0,c],chair[1,c],chair[2,c],1]).reshape(4,1)
		#x = np.matmul(pMat, hCoord)
		#print (projC[:,c].shape)
		#print (np.matmul(pMat, hCoord)[0:2, :].shape)
		projC[:,c] = (np.matmul(pMat, hCoord)[0:2, :]).reshape(2,)

	print (projC,"\n")
	return projC;
