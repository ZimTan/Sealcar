import tensorflow as tf

class AutoencoderModel(tf.keras.Model):

    def __init__(self, input_shape, num_outputs):
        super(AutoencoderModel, self).__init__()

        self.flatten1 = tf.keras.layers.Flatten()

        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.he_uniform())

        self.pooling1 = tf.keras.layers.MaxPooling2D(pool_size=2)

        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.he_uniform())

        self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.he_uniform())

        self.pooling3 = tf.keras.layers.MaxPooling2D(pool_size=2)

        self.conv4 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.he_uniform())

        self.conv5 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(3,3), activation='sigmoid', padding='same', kernel_initializer=tf.keras.initializers.he_uniform())

        self.flatten = tf.keras.layers.Flatten()

        self.outputs = tf.keras.layers.Dense(120 * 160)

        self.reshape = tf.keras.layers.Reshape((120, 160))

    def call(self, inputs, training=False):

        x = self.conv1(inputs)
        x = self.pooling1(x)
        x = self.conv2(x)
        x = self.pooling2(x)
        x = self.conv3(x)
        x = self.pooling3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.flatten(x)

        outputs = self.outputs(x)
        return self.reshape(outputs)
