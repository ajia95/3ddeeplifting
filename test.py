import tensorflow as tf
import readDataTF as d
import skeleton as sk
import numpy as np
import standardize as st
import procrustes as p 

def test(netOut,x,y,sess, testSize):
  test = ["test.tfrecords"]
  dataset = d.getTestData(test, testSize, 1)
  iterator = dataset.make_initializable_iterator()
  chairBatch, x_batch, y_batch = iterator.get_next()
  sess.run(iterator.initializer)
 
  '''ave1 = np.zeros([3,10], dtype = float)
  ave2 = np.zeros([3,10], dtype = float)
  ac = 0'''

  try:
    while True:
      ch, uv, z = sess.run((chairBatch, x_batch, y_batch))
      #compare ch to xt:netOut instead of yt to netOut
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(netOut, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      accur = accuracy.eval({x: uv, y: z})
      #ac = ac + accur
      print ("Accuracy:", accur)
      #sk.drawProj(np.transpose(uv.reshape(10,2)))
      #print (np.transpose(uv.reshape(10,2)))

      '''
      outZ = netOut.eval({x: uv})

      #print ("GroundTruth z: ", z)
      #print ("Predicted z: ", outZ)
      #print ("Difference: ", abs(z-outZ))

          
      #standardized x,y,z
      ch1 = np.transpose(ch.reshape(10,3))
      #chair u,v,predictedZ
      ch2 = np.concatenate((np.transpose(uv.reshape(10,2)), outZ), axis=0)

      ch3 = st.standardize3DCoords(ch1)

      ch4 = p.align_data(ch1, ch2) 

      #if accur == 1:
      #sk.drawChairs(ch1,ch4)
      #sk.drawChair(ch4)
      
      #compare ch1 and ch4 

      #compare ch2 and ch3

      for r in range(0,3):
        for c in range(0,10):
          ave1[r,c] = ave1[r,c] + abs(ch1[r,c]-ch4[r,c])

      for r in range(0,3):
        for c in range(0,10):
          ave2[r,c] = ave2[r,c] + abs(ch2[r,c]-ch3[r,c])

      #print ("ground truth")
      #print (ch1,"\n\n")

      #print ("predicted")
      #print (ch4, "\n\n\n")

      #print ("difference")
      #print (abs(ch1-ch4),"\n\n")

      '''
  except tf.errors.OutOfRangeError:
    pass

  #print ("average difference")
  #print (ave1/testSize)
 # print (p.align_data(ave1/testSize, ave2/testSize))
  #print (ac/testSize)


