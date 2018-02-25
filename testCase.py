import tensorflow as tf
import readDataTF as d
import skeleton as sk
import numpy as np
import standardize as st
import procrustes as p 
import params as pa

def test():
  test = ["testCases.tfrecords"]
  dataset = d.getTestCaseData(test, 1)
  iterator = dataset.make_initializable_iterator()
  zGT, zP, clas = iterator.get_next()
  with tf.Session() as sess:
    sess.run(iterator.initializer)
 
  
    ac = 0

    t1 = 0
    t2 = 0
    t3 = 0
    t4 = 0
    t5 = 0
    t6 = 0

    tt1 = 0
    tt2 = 0

    try:
      while True:
        z_GT, z_P, classif = sess.run((zGT, zP, clas))

        correct_prediction = tf.equal(tf.argmax(z_GT, 1), tf.argmax(z_P, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accur = accuracy.eval()
        ac = ac + accur
        print ("Accuracy:", accur)
        print (z_GT)
        print (z_P)


      
        
        if classif == 1:
          tt1 = tt1 + 1

        else:
          tt2 = tt2 + 1

        #correct classified as correct
        if classif == 1 and accur == 1:
          t1 = t1 + 1
        #correct classified as incorrect
        if classif == 1 and accur == 0:
          t2 = t2 + 1

        #incorrect classified as correct
        if classif == 0 and accur == 1:
          t3 = t3 + 1

        #incorrect classified as incorrect
        if classif == 0 and accur == 0:
          t4 = t4 + 1
      
        #total cooreclty classified
        if (classif == 1 and accur == 1) or (classif == 0 and accur == 0):
          t5 = t5 + 1

        #total incooreclty classified
        if (classif == 1 and accur == 0) or (classif == 0 and accur == 1):
          t6 = t6 + 1

 
 

      
    except tf.errors.OutOfRangeError:
      pass
  
  #print ("average difference")
  #print ((ave)/testSize, "\n\n")

  #print ((ave2)/testSize)

  #print ("accuracy")
  print (ac/1000)
  
  print (tt1/1000)


  print (t1/tt1)

  print (t2/tt1)

  print (t3/tt2)

  print (t4/tt2)

  print (t5/1000)

  print (t6/1000)

test()


