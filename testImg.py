import numpy as np
import skeleton as sk
import standardize as sd
import procrustes as p 
import tensorflow as tf
import network as net
import imgCoord as im
import standardize as sd



def test(netOut,x,y,sess):

	for _2d, _3d in zip(im.coord2d, im.coord3d):
		uv = sd.standardize2DCoords(_2d).flatten('F').reshape(1,20)
		outZ = netOut.eval({x: uv})

		ch = p.align_data(_3d,np.concatenate((sd.standardize2DCoords(_2d), outZ), axis=0))
		sk.drawProj(_2d)
		sk.drawChairs(ch*10, _3d*10)
	