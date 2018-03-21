"""Machine learning model definitions."""

import tensorflow as tf
import keras
from keras.datasets import mnist


IMG_ROWS, IMG_COLS = 28, 28
IMG_MAX_VAL = 255
NUM_CLASSES = 10
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1)


def _preprocess_features(x):
  """Preprocess input features."""
  x = x.reshape(x.shape[0], IMG_ROWS, IMG_COLS, 1)
  x = x.astype('float32')
  x /= IMG_MAX_VAL
  return x


def _preprocess_labels(y, num_classes):
  """Preprocess label vector"""

  # One-hot encode labels based on num_classes
  return keras.utils.to_categorical(y, num_classes)


class MNISTData(object):

  def __init__(self):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    self.x_train = _preprocess_features(x_train)
    self.y_train = _preprocess_labels(y_train, NUM_CLASSES)
    self.x_test = _preprocess_features(x_test)
    self.y_test = _preprocess_labels(y_test, NUM_CLASSES)


data = MNISTData()


def make_input_fn(features, labels=None, shuffle=False):
  """Input function for training/eval."""

  # TODO(cezequiel): implement batching option
  input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'images_input': features},
      y=labels,
      shuffle=shuffle)

  return input_fn


def make_train_input_fn(shuffle=False):
  return make_input_fn(data.x_train, data.y_train, shuffle)


def make_eval_input_fn():
  return make_input_fn(data.x_test, data.y_test)


def make_json_serving_input_fn(input_shape):
  def input_fn():
    images_input = tf.placeholder(shape=input_shape,
                                  dtype=tf.float32)
    inputs = {'images_input': images_input}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

  return input_fn


def build_estimator(config, input_shape, num_classes):
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                                   input_shape=input_shape, name='images'))
  model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
  model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(tf.keras.layers.Dropout(0.25))
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(128, activation='relu'))
  model.add(tf.keras.layers.Dropout(0.5))
  model.add(tf.keras.layers.Dense(num_classes, activation='softmax',
                                  name='labels'))
  model.compile(loss='categorical_crossentropy', optimizer='adadelta',
                metrics=['accuracy'])

  return tf.keras.estimator.model_to_estimator(keras_model=model, config=config)


