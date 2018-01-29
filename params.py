import numpy as np
import random
#unit - cm


bases = np.array([[-100,-200,100,100,-200,100,100,-200,-100,-100,-200,-100,-100,0,100,100,0,100,100,0,-100,-100,0,-100,-100,200,-100,100,200,-100],
				  [0,-100,0,0,-100,0,0,-100,0,0,-100,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,100,0,0,100,0],
				  [-100,0,0,100,0,0,100,0,0,-100,0,0,-100,0,0,100,0,0,100,0,0,-100,0,0,-100,0,0,100,0,0],
				  [-100,0,100,100,0,100,100,0,-100,-100,0,-100,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]);


def genWeights():
	"generate random weights"
	weights = abs(np.random.normal(1,1,size=4))
	total = np.sum(weights)
	weights = np.divide(weights, total)
	print (weights,"\n")
	#weights = np.array([1,0,0,0,0])
	return weights;

def genFocalLen():
	"generate random focal length"
	#focal len in pixel units?
	F = random.uniform(2.0,6.0)
	print (F,"\n")
	return F;




