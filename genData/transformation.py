import numpy as np
import random
import math

def randRotation():
	"generate random rotation matrix"
	thetax = random.uniform(0,math.pi*2)
	thetay = random.uniform(0,math.pi*2)
	thetaz = random.uniform(0,math.pi*2)

	Rx = np.array([[1,0,0], 
	[0,math.cos(thetax),-math.sin(thetax)], 
	[0,math.sin(thetax),math.cos(thetax)]]);

	Ry = np.array([[math.cos(thetay),0,math.sin(thetay)], 
	[0,1,0],
	[-math.sin(thetay),0,math.cos(thetay)]]);

	Rz = np.array([[math.cos(thetaz),-math.sin(thetaz),0], 
	[math.sin(thetaz),math.cos(thetaz),0],
	[0,0,1]]);
	#print ("angles = ", Rx, Ry, Rz)
	R = np.matmul(np.matmul(Rz,Ry),Rx)
	print (R,"\n")
	return R;


def rotateChair(R, chair):
	"apply rotation matrix"
	return np.transpose(np.array([[np.matmul(R, chair[:,0])],
		[np.matmul(R, chair[:,1])],
		[np.matmul(R, chair[:,2])],
		[np.matmul(R, chair[:,3])],
		[np.matmul(R, chair[:,4])],
		[np.matmul(R, chair[:,5])],
		[np.matmul(R, chair[:,6])],
		[np.matmul(R, chair[:,7])],
		[np.matmul(R, chair[:,8])],
		[np.matmul(R, chair[:,9])]])).reshape(3,10);

def randTranslation():
	"generate random translation vector"
	T = np.zeros([3,1], dtype = float)
	for c in range(0,3):
		T[c,:] = random.uniform(-100,100)
	print (T,"\n")
	return T;

def translateChair(T,chair):
	"apply translation matrix"
	transC = np.zeros([3,10], dtype = float)
	for c in range(0,10):
		#transC[:,[c]] = np.add(T, chair[:,c])
		transC[:,c] = chair[:,c].reshape(3) + T.reshape(3)
	#print (transC.shape)
	return transC;

	#return np.transpose(
	#	np.array([[np.add(T, chair[:,0])],
	#	[np.add(T, chair[:,1])],
	#	[np.add(T, chair[:,2])],
	#	[np.add(T, chair[:,3])],
	#	[np.add(T, chair[:,4])],
	#	[np.add(T, chair[:,5])],
	#	[np.add(T, chair[:,6])],
	#	[np.add(T, chair[:,7])],
	#	[np.add(T, chair[:,8])],
	#	[np.add(T, chair[:,9])]])
	#	).reshape(3,10);