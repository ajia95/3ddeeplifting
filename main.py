import tensorflow as tf
import train as tr
import test as ts
import network as net
import testImg as ti
import evalGraph as ev

#no. iterations = no. batches * no. epochs
#Iterations is the number of batches needed to complete one epoch.

#dataset = batchsize * itertions

learning_rate = 0.01
epochs = 30
batch_size = 50
trainSize = 4000
testSize = 1000
#batch_size = int(dataSize/iterations) 
#total_batch = int(dataSize / batch_size) 
with tf.name_scope("inputs") as scope:
	x = tf.placeholder(shape=(None,20), dtype=tf.float32)
	y = tf.placeholder(shape=(None,10), dtype=tf.float32)

with tf.name_scope("Layers") as scope:
	netOut = net.mlp(x)

#run trainiing - comment out if just want to test
#tr.train(learning_rate, epochs, batch_size, trainSize,x,y, netOut)

saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "/Users/User/project/model/model.ckpt")
  print("Model restored.")

  #run test on synthetic data
  ts.test(netOut,x,y,sess, testSize,0)

  #run test on real images
  #ti.test(netOut,x,sess,0)

  #ev.errorGraph(netOut,x,y,sess, testSize)
  #ev.rmseGraph(netOut,x,y,sess, testSize)







