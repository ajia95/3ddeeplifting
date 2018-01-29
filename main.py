import tensorflow as tf
import train as tr
import test as ts
import network as net
import testImg as ti

#no. iterations = no. batches * no. epochs
#Iterations is the number of batches needed to complete one epoch.

#dataset = batchsize * itertions
#no. of iterations = 10
learning_rate = 0.01
epochs = 50 #can pick anything?
#iterations = 10
batch_size = 50
trainSize = 4000
testSize = 1000
#batch_size = int(dataSize/iterations) #100
#total_batch = int(dataSize / batch_size) #total 10 batches
with tf.name_scope("inputs") as scope:
	x = tf.placeholder(shape=(None,20), dtype=tf.float32)
	y = tf.placeholder(shape=(None,10), dtype=tf.float32)

with tf.name_scope("Layers") as scope:
	netOut = net.mlp(x)

#tr.train(learning_rate, epochs, batch_size, trainSize,x,y, netOut)

saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "/Users/User/project/model/model.ckpt")
  print("Model restored.")

  #ts.test(netOut,x,y,sess, testSize)
  ti.test(netOut,x,y,sess)






