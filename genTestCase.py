import numpy as np
import params as pa
import chairPC as cpc
import transformation as t
import tensorflow as tf
import procrustes as p 



def check(coords):
	"check if every keypoint is in the frame"
	for c in range(0,10):
		if not (coords[0,c]>-1000 and coords[0,c]<1000 and coords[1,c]>-1000 and coords[1,c]<1000):
			return False;
	return True;


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten('F')))


def genData(N, cN, filename, correct):
	count = 0
	base = np.transpose(pa.bases[0,:].reshape(10,3))/4

	#train_filename = 'train.tfrecords'  # address to save the TFRecords file
	writer = tf.python_io.TFRecordWriter(filename)
	while count<N:
		cCount = 0
		print ("-----------------------")
		w = pa.genWeights()
		chair = cpc.randChair(pa.bases,w)
		chair = chair.reshape(-1,3).T
	

		while cCount<cN:
			rC = t.randRotation()
			tC = t.randTranslation()
			tChair = t.translateChair(tC,t.rotateChair(rC,chair))
			tChair = p.align_data(base, tChair)
			#print (tChair,"\n")
	
			count = count + 1
			cCount = cCount + 1

			clas = np.zeros((1,1), dtype=float)
			tChair2 = np.zeros((1,10), dtype=float)

			if count<correct:
				clas[0,0] = 1.0
				noise = np.random.uniform(-2,2,[1,10])
				print (noise)
				tChair2= np.add(tChair[2,:], noise)
				print (tChair[2,:])
				print (tChair2,"\n")
			else:
				clas[0,0] = 0.0
				noise = np.random.uniform(2.1,100,[1,10])
				print (noise)
				tChair2 = np.add(tChair[2,:], noise)
				print (tChair[2,:])
				print (tChair2,"\n")


			feature = {'xyzGT': _floats_feature(tChair[2,:]),
				  		'xyzP': _floats_feature(tChair2),
				  		'class': _floats_feature(clas)}
			
			
			sample = tf.train.Example(features=tf.train.Features(feature=feature))
			writer.write(sample.SerializeToString())		
  
	writer.close()


testCase_filename = 'testCases.tfrecords'
tcSize = 1000
correct = 750



genData(tcSize, 1, testCase_filename, correct)




