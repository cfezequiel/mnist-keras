"""Machine learning model definitions."""

import tensorflow as tf


def make_input_fn(features, labels=None, shuffle=False):
  """Input function for training/eval."""

  input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'images_input': features},
      y=labels,
      shuffle=shuffle)

  return input_fn


def make_json_serving_input_fn(input_shape):
  def input_fn():
    images_input = tf.placeholder(shape=input_shape,
                                  dtype=tf.float32)
    inputs = {'images_input': images_input}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

  return input_fn


def build_estimator(input_shape, num_classes):
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

  return tf.keras.estimator.model_to_estimator(keras_model=model)


