import tensorflow as tf


def decodeTrain(serialized_example):
  features = tf.parse_single_example(
      serialized_example,
      features={'xy': tf.FixedLenFeature([20], tf.float32),
                'z': tf.FixedLenFeature([10], tf.float32)}) #change to [1,30] for x,y,z


  # NOTE: No need to cast these features, as they are already `tf.float32` values.
  return features['xy'], features['z']



def decodeTest(serialized_example):
  features = tf.parse_single_example(
      serialized_example,
      features={'xyz': tf.FixedLenFeature([30], tf.float32),
                'uv': tf.FixedLenFeature([20], tf.float32),
                'z': tf.FixedLenFeature([10], tf.float32)}) #change to [1,30] for x,y,z


  # NOTE: No need to cast these features, as they are already `tf.float32` values.
  return features['xyz'], features['uv'], features['z']


def getTestData(filename, dataSize, batch_size):

  dataset = tf.data.TFRecordDataset(filename).map(decodeTest)
  dataset = dataset.shuffle(dataSize)
  # form batch and epoch
  #dataset = dataset.repeat(epochs)
  dataset = dataset.batch(batch_size)
  return dataset

def getTrainData(filename, dataSize, batch_size):

  dataset = tf.data.TFRecordDataset(filename).map(decodeTrain)
  dataset = dataset.shuffle(dataSize)
  # form batch and epoch
  #dataset = dataset.repeat(epochs)
  dataset = dataset.batch(batch_size)
  return dataset