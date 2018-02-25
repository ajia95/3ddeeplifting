import test 
import testImg as ti
import math
import matplotlib.pyplot as plt
import numpy as np

def eucDist(ch1, ch2):
	dist = 0
	for c in range(0,np.size(ch1,1)):
		dist = dist + math.sqrt((ch1[0,c]-ch2[0,c])**2 + (ch1[1,c]-ch2[1,c])**2 + (ch1[2,c]-ch2[2,c])**2)
	dist = dist/np.size(ch1,1)
	return dist

def errorGraph(netOut,x,y,sess, testSize):

	errors = np.zeros([5], dtype = float)
	noise = np.arange(0,5)
	for c in range(0,5):
		errors[c] = (test.test(netOut,x,y,sess, testSize,c))
	print (errors)
	plt.plot(noise, errors)
	plt.axis([0, 4, 0, 25])
	plt.xlabel("noise value")
	plt.ylabel("average error in cm")
	plt.show()


def rmseGraph(netOut,x,y,sess, testSize):

	errors = np.zeros([4], dtype = float)
	noise = np.arange(0,4)

	for c in range(0,4):
		#errors[c] = (ti.test(netOut,x,sess,c))
		errors[c] = test.test(netOut,x,y,sess, testSize,c)

	print (errors)
	plt.plot(noise, errors)
	plt.axis([0, 3, 0, 30])
	plt.xlabel("noise value")
	plt.ylabel("RMSE (cm)")
	plt.show()
