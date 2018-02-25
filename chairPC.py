import numpy as np

def randChair(bases, w): 
	"create new chair using weights"
	chair = bases[0,:].reshape(1,30)
	for x in range(0,4):
		chair = np.add(np.multiply(bases[x+1,:],w[x]).reshape(1,30),chair)
	return chair.reshape(30,);



