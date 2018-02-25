import tensorflow as tf
import readDataTF as d
import skeleton as sk
import numpy as np
import standardize as st
import procrustes as p 
import params as pa
import evalGraph as ev
import math

def test(netOut,x,y,sess, testSize,noise):
  test = ["test.tfrecords"]
  dataset = d.getTestData(test, testSize, 1)
  iterator = dataset.make_initializable_iterator()
  chairBatch, x_batch, y_batch = iterator.get_next()
  sess.run(iterator.initializer)
 
  #ave difference in matrcies
  #ave = np.zeros([3,10], dtype = float)

  noiseMat = np.zeros([1,20], dtype = float)
  noiseMat.fill(noise)
  ac = 0

  #average error in euclidean distance
  #av = 0

  #ave euclidean dist over entire dataset
  #euAv = 0

  #ave euclidean dist matrix - one for each keypoint
  #euMat = np.zeros([10], dtype = float)
 
  base = np.transpose(pa.bases[0,:].reshape(10,3))/4

  try:
    while True:
      ch, uv, z = sess.run((chairBatch, x_batch, y_batch))
      uv =  uv + noiseMat
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(netOut, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      accur = accuracy.eval({x: uv, y: z})
      ac = ac + accur
      print ("Accuracy:", accur)

      
      outZ = netOut.eval({x: uv})

      #print ("GroundTruth z: ", z)
      #print ("Predicted z: ", outZ)
      #print ("Difference: ", abs(z-outZ))

      
      #standardized x,y,z
      ch1 = np.transpose(ch.reshape(10,3))
      #chair u,v,predictedZ
      ch2 = np.concatenate((np.transpose(uv.reshape(10,2)), outZ), axis=0)

      ch2 = p.align_data(ch1, ch2) 
      ch1 = p.align_data(base, ch1)
      ch2 = p.align_data(base, ch2)
      
      #ave = ave + abs(ch1-ch2)
      #av = av + (ev.eucDist(ch1,ch2)**2)
      #for k in range(0,10):
      #  euMat[k] = euMat[k] + ev.eucDist(ch1[:,k].reshape(3,1),ch2[:,k].reshape(3,1))
      #euAv = euAv + ev.eucDist(ch1,ch2)
 

      
  except tf.errors.OutOfRangeError:
    pass
  
  #print ("average difference")
  #print ((ave)/testSize, "\n\n")

  print ("accuracy")
  print (ac/testSize)
  #return math.sqrt(av/testSize)
  #print (euMat/testSize)
  #print (euAv/testSize)
  

  


