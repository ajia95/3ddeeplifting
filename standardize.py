import numpy as np 

def standardize3DCoords(chair):
	xMean = np.mean(chair[0,:])
	yMean = np.mean(chair[1,:])
	zMean = np.mean(chair[2,:])
	xSD = np.std(chair[0,:])
	ySD = np.std(chair[1,:])
	dom = (xSD + ySD)/2

	sChair = np.empty([3,10], dtype = float)

	for c in range(0,10):

		sChair[0,c] = (chair[0,c] - xMean)/dom
		sChair[1,c] = (chair[1,c] - yMean)/dom
		sChair[2,c] = (chair[2,c] - zMean)/dom

	return sChair;

def standardize2DCoords(chair):
	uMean = np.mean(chair[0,:])
	vMean = np.mean(chair[1,:])
	uSD = np.std(chair[0,:])
	vSD = np.std(chair[1,:])
	dom = (uSD + vSD)/2

	sChair = np.empty([2,10], dtype = float)

	for c in range(0,10):

		sChair[0,c] = (chair[0,c] - uMean)/dom
		sChair[1,c] = (chair[1,c] - vMean)/dom

	return sChair;

