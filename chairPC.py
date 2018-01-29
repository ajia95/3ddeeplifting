import numpy as np

def getData(N):
	"create list of matrices from ikea dataset"
	data = np.empty([N,30], dtype = int)
	for x in range(0, N):
		#change to read in data from ikea dataset
		a = np.array([1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3])
		data[x,:] = a;
	return data;


def svd(covM):
	"singular value decomposition"
	U, s, VT = np.linalg.svd(covM, full_matrices=False)
	#print ("U:\n {}".format(U))
	#print ("s:\n {}".format(s))
	#print ("VT:\n {}".format(VT))
	return U;

def randChair(bases, w): 
	"create new chair using weights"
	chair = bases[0,:].reshape(1,30)
	for x in range(0,4):
		chair = np.add(np.multiply(bases[x+1,:],w[x]).reshape(1,30),chair)
	return chair.reshape(30,);

#def randChair(U):
#	"create new chair using weights"
#	w = np.array([[1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30]])
#	chair = np.zeros([30,1], dtype = int)
#	for x in range(0,30):
#		chair = np.add(chair,np.multiply(U[:,x], w[0,x]).reshape(30,1))
#	print(chair)
#	return chair;


#N=5
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)
#data = genData(N)
#covM = np.cov(data, rowvar=False)
#U = svd(covM)

