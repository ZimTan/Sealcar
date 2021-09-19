import tensorflow as tf
from tensorflow.keras.layers import MaxPooling2D, Dropout, Conv2D, GlobalAveragePooling2D, Dropout, Dense, Concatenate



class SqueezenetModel(tf.keras.Model):

    def __init__(self, unput_shape, num_outputs):
        super(SqueezenetModel, self).__init__()

        self.conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu',
                kernel_initializer=tf.keras.initializers.he_uniform())

        self.conv2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu',
          strides=(1, 1), kernel_initializer=tf.keras.initializers.he_uniform())

        self.conv3 = Conv2D(filters=16, kernel_size=(1, 1), activation='relu', kernel_initializer=tf.keras.initializers.he_uniform())
        self.conv4 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.he_uniform())
        self.conv5 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.he_uniform())

        self.conv6 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', kernel_initializer=tf.keras.initializers.he_uniform())
        self.conv7 = Conv2D(filters=128, kernel_size=(1, 1), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.he_uniform())
        self.conv8 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.he_uniform())

        self.conv9 = Conv2D(filters=48, kernel_size=(1, 1), activation='relu', kernel_initializer=tf.keras.initializers.he_uniform())
        self.conv10 = Conv2D(filters=192, kernel_size=(1, 1), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.he_uniform())
        self.conv11 = Conv2D(filters=192, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.he_uniform())

        self.conv12 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', kernel_initializer=tf.keras.initializers.he_uniform())
        self.conv13 = Conv2D(filters=256, kernel_size=(1, 1), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.he_uniform())
        self.conv14 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.he_uniform())

        self.pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')

        self.gavgpool1 = GlobalAveragePooling2D()

        self.dropout1 = Dropout(0.5)

        self.outputs = Dense(num_outputs, activation='tanh')



    def call(self, inputs, training=False):

        x = self.conv1(inputs[0])
        fire = self.pool1(x)

        fire = self.conv3(fire)
        left = self.conv4(fire)
        right = self.conv5(fire)
        fire = Concatenate(axis=-1)([left, right])

        fire = self.conv3(fire)
        left = self.conv4(fire)
        right = self.conv5(fire)
        fire = Concatenate(axis=-1)([left, right])

        fire = self.pool1(fire)

        fire = self.conv6(fire)
        left = self.conv7(fire)
        right = self.conv8(fire)
        fire = Concatenate(axis=-1)([left, right])

        fire = self.conv6(fire)
        left = self.conv7(fire)
        right = self.conv8(fire)
        fire = Concatenate(axis=-1)([left, right])

        fire = self.pool1(fire)

        fire = self.conv9(fire)
        left = self.conv10(fire)
        right = self.conv11(fire)
        fire = Concatenate(axis=-1)([left, right])

        fire = self.conv9(fire)
        left = self.conv10(fire)
        right = self.conv11(fire)
        fire = Concatenate(axis=-1)([left, right])

        fire = self.conv12(fire)
        left = self.conv13(fire)
        right = self.conv14(fire)
        fire = Concatenate(axis=-1)([left, right])

        fire = self.conv12(fire)
        left = self.conv13(fire)
        right = self.conv14(fire)
        fire = Concatenate(axis=-1)([left, right])

        dropout = self.dropout1(fire, training)

        conv2 = self.conv2(dropout)

        avg_pool = self.gavgpool1(conv2)

        outputs = self.outputs(avg_pool)

        return outputs
