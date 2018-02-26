import numpy as np
from genData import params as p
from genData import chairPC as cpc
from genData import transformation as t
from genData import project as pr
import tensorflow as tf
from genData import standardize as sd
import sys
sys.path.append("..")


def check(coords):
	"check if every keypoint is in the frame"
	for c in range(0,10):
		if not (coords[0,c]>-1000 and coords[0,c]<1000 and coords[1,c]>-1000 and coords[1,c]<1000):
			return False;
	return True;


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten('F')))

#set 0 = train
#set 1 = test
def genData(N,cN, filename,set):
	count = 0
	#train_filename = 'train.tfrecords'  # address to save the TFRecords file
	writer = tf.python_io.TFRecordWriter(filename)
	while count<N:
		cCount = 0
		print ("-----------------------")
		w = p.genWeights()
		chair = cpc.randChair(p.bases,w)
		chair = chair.reshape(-1,3).T
		print (chair,"\n")
	

		while cCount<cN:
			rC = t.randRotation()
			tC = t.randTranslation()
			tChair = t.translateChair(tC,t.rotateChair(rC,chair))
			print (tChair,"\n")
			#sk.drawChair(tChair)
			f = p.genFocalLen()
			projC = pr.projCoords(tChair,f)
	
			if check(projC)==True:
				#sk.drawProj(projC)
				count = count + 1
				cCount = cCount + 1

				
				
				sChair = sd.standardize3DCoords(tChair)

				if set == 0:

					#train data	
					feature = {'xy': _floats_feature(sChair[0:2,:]),
					   			'z': _floats_feature(sChair[2,:])} 
					sample = tf.train.Example(features=tf.train.Features(feature=feature))
					writer.write(sample.SerializeToString())	

				else:

					#test data
					sProjC = sd.standardize2DCoords(projC)

					feature = {'xyz': _floats_feature(tChair),
								'uv': _floats_feature(sProjC),
					  		    'z': _floats_feature(sChair[2,:])}
					sample = tf.train.Example(features=tf.train.Features(feature=feature))
					writer.write(sample.SerializeToString())		
  
	writer.close()


train_filename = '/Users/User/project/data/train.tfrecords'
trainSize = 4000
trainC = 10

test_filename = '/Users/User/project/data/test.tfrecords'
testSize = 1000
testC = 5

genData(trainSize, trainC, train_filename,0)
genData(testSize, testC, test_filename,1)



