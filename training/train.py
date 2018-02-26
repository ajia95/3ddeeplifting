import tensorflow as tf
import sys
from readTF import readDataTF as d
#import readDataTF as d
sys.path.append("..")

def train(learning_rate, epochs, batch_size, dataSize,x,y,netOut):
  #batch_size = int(dataSize/iterations) 
  iterations = int(dataSize/batch_size) 
  total_batch = int(dataSize / batch_size) 
 
  #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=netOut))
  
  #cost = -tf.reduce_mean(tf.reduce_sum(y * tf.log(netOut)
                        # + (1 - y) * tf.log(1 - netOut), axis=1))
  with tf.name_scope("cost") as scope:
    cost = tf.reduce_sum(tf.square(netOut - y))
    tf.summary.scalar('cost', cost)

  # add an optimiser
  optimiser = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
  
  # finally setup the initialisation operator
  init_op = tf.global_variables_initializer()

  train = ["/Users/User/project/data/train.tfrecords"]
  dataset = d.getTrainData(train, dataSize, batch_size)
  iterator = dataset.make_initializable_iterator()
  x_batch, y_batch = iterator.get_next()



  with tf.Session() as sess:
    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('/Users/User/project/visualise', sess.graph)

    sess.run(init_op)
    e = 0
    for _ in range(epochs):
      dataset.shuffle(dataSize)
      e = e + 1
      sess.run(iterator.initializer)
      avg_cost = 0
      i = 1
      try:

        while True:
          xy, z = sess.run((x_batch, y_batch))

          summary, _, c = sess.run([merged, optimiser, cost], 
                         feed_dict={x: xy, y: z})

          print ("cost of batch ", i, ": ", c)      
          train_writer.add_summary(summary, e)
          i = i + 1
      except tf.errors.OutOfRangeError:
        
        # Raised when we reach the end of the file.
        pass

      print ("end of epoch ", e)

    train_writer.close()
    save_path = saver.save(sess, "/Users/User/project/model/model.ckpt")
    print("Model saved in file: %s" % save_path)

  return









    



