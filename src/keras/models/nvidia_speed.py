import tensorflow as tf

class NvidiaSpeedModel(tf.keras.Model):

  def __init__(self, input_shape, num_outputs):
    super(NvidiaSpeedModel, self).__init__()
    self.conv1 = tf.keras.layers.Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2),
            activation='relu', padding='same',
            kernel_initializer=tf.keras.initializers.he_uniform())

    self.conv2 = tf.keras.layers.Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2),
            activation='relu', padding='same',
            kernel_initializer=tf.keras.initializers.he_uniform())

    self.conv3 = tf.keras.layers.Conv2D(filters=48, kernel_size=(3, 3), strides=(2, 2),
            activation='relu', padding='same',
            kernel_initializer=tf.keras.initializers.he_uniform())

    self.conv4 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2),
            activation='relu', padding='same',
            kernel_initializer=tf.keras.initializers.he_uniform())

    self.flatten1 = tf.keras.layers.Flatten()
    self.dropout1 = tf.keras.layers.Dropout(.2)

    self.dense1 = tf.keras.layers.Dense(512, activation='relu')
    self.dropout2 =tf.keras.layers.Dropout(.5)

    self.dense2 = tf.keras.layers.Dense(256, activation='relu')

    self.dense3 = tf.keras.layers.Dense(128, activation='tanh')

    self.outputs = tf.keras.layers.Dense(num_outputs)

  def call(self, inputs, training=False):
    x = self.conv1(inputs[0])
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)

    x = self.flatten1(x)
    if training:
        x = self.dropout1(x, training)

    x = self.dense1(tf.keras.layers.concatenate([x, inputs[1]], axis=-1))
    if training:
        x = self.dropout2(x, training)

    x = self.dense2(x)
    x = self.dense3(x)

    return self.outputs(x)
