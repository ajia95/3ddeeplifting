import numpy as np
import skeleton as sk
import standardize as sd
import procrustes as p 
import tensorflow as tf
import network as net
import imgCoord as im
import standardize as sd
import project as pr
import params as pa
import evalGraph as ev
import math


def test(netOut,x,sess,noise):

	coord2d = []
	#ave difference in matrcies
	#ave = np.zeros((3,10), dtype=float)
	#average error in euclidean distance - one value
	#av = 0
	#ave euclidean dist matrix - one for each keypoint
	#euMat = np.zeros([10], dtype = float)
	#ave euclidean dist over entire dataset
	euAv = 0

	base = np.transpose(pa.bases[0,:].reshape(10,3))/4

	for _3d in im.coord3d:
		coord2d.append(pr.projCoords(_3d,1))


	for _2d, _3d in zip(coord2d, im.coord3d):

		uv = sd.standardize2DCoords(_2d).flatten('F').reshape(1,20)

		#add gausssian noise
		#noise = np.random.normal(0, 0.1, [1,20])
		noiseMat = np.zeros([1,20], dtype = float)
		noiseMat.fill(noise)

		uv = np.add(uv,noiseMat)

		outZ = netOut.eval({x: uv})

		ch = np.concatenate((sd.standardize2DCoords(_2d), outZ), axis=0)
		ch = p.align_data(_3d,ch)

		ch1 = p.align_data(base,ch)
		ch2 = p.align_data(base,_3d)

		#ave = ave + abs(ch1-ch2)
		#av = av + (ev.eucDist(ch1,ch2)**2)
		euAv = euAv + ev.eucDist(ch1,ch2)

		#for k in range(0,10):
			#euMat[k] = euMat[k] + ev.eucDist(ch1[:,k].reshape(3,1),ch2[:,k].reshape(3,1))
		

		#sk.drawChairs(ch2,ch1)
		
	#ave = (ave)/len(coord2d)
	#print (ave)
	#return math.sqrt(av/len(coord2d))
	print (euAv/len(coord2d))
	#print (euMat/len(coord2d))

	