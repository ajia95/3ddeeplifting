import numpy as np
from draw import skeleton as sk
from genData import standardize as sd
from testing import procrustes as p 
import tensorflow as tf
from architecture import network as net
from data import imgCoord as im
from genData import standardize as sd
from genData import project as pr
from genData import params as pa
from evaluation import evalGraph as ev
import math
import sys

sys.path.append("..")

def test(netOut,x,sess,noise):

	coord2d = []
	
	#ave euclidean dist over entire dataset
	av = 0

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

		av = av + ev.eucDist(ch1,ch2)
		
		#uncomment to visualise
		#sk.drawChairs(ch2,ch1)
		

	print ("average euclidean distance")
	print (av/len(coord2d))
	

	