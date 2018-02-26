import tensorflow as tf

def mlp(x):

  with tf.name_scope("layer1") as scope:
    w1 = tf.Variable(tf.random_normal([20, 20], stddev=0.03), name="w1")
    b1 = tf.Variable(tf.random_normal([20]), name="b1")
    out1 = tf.tanh(tf.add(tf.matmul(x, w1), b1))
    tf.summary.histogram("w1",w1)


  with tf.name_scope("layer2") as scope:
    w2 = tf.Variable(tf.random_normal([20, 20], stddev=0.03), name="w2")
    b2 = tf.Variable(tf.random_normal([20]), name="b2")
    out2 = tf.tanh(tf.add(tf.matmul(out1, w2), b2))
    tf.summary.histogram("w2",w2)

  with tf.name_scope("layer3") as scope:
    w3 = tf.Variable(tf.random_normal([20, 20], stddev=0.03), name="w3")
    b3 = tf.Variable(tf.random_normal([20]), name="b3")
    out3 = tf.tanh(tf.add(tf.matmul(out2, w3), b3))
    tf.summary.histogram("w3",w3)

  with tf.name_scope("layer4") as scope:
    w4 = tf.Variable(tf.random_normal([20, 20], stddev=0.03), name="w4")
    b4 = tf.Variable(tf.random_normal([20]), name="b4")
    out4 = tf.tanh(tf.add(tf.matmul(out3, w4), b4))
    tf.summary.histogram("w4",w4)

  with tf.name_scope("layer5") as scope:
    w5 = tf.Variable(tf.random_normal([20, 10], stddev=0.03), name="w5")
    b5 = tf.Variable(tf.random_normal([10]), name="b5")
    out5 = tf.tanh(tf.add(tf.matmul(out4, w5), b5))
    tf.summary.histogram("w5",w5)

  return out5