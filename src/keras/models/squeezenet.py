import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def squeezenet_model(input_shape, num_outputs):
  def fire_module(x, filters_squeeze, filters_expand):
    x = Conv2D(filters=filters_squeeze, kernel_size=(1, 1), activation='relu', kernel_initializer=tf.keras.initializers.he_uniform())(x)
    left = Conv2D(filters=filters_expand, kernel_size=(1, 1), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.he_uniform())(x)
    right = Conv2D(filters=filters_expand, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.he_uniform())(x)
    x = Concatenate(axis=-1)([left, right])
    return x

  inputs = Input(shape=input_shape)

  # Standalone convolution layer (conv1)
  conv1 = Conv2D(filters=48, kernel_size=3, strides=(2, 2), activation='relu',
          kernel_initializer=tf.keras.initializers.he_uniform())(inputs)

  # Max-pooling
  fire = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv1)

  fire = fire_module(fire, 16, 64)
  fire = fire_module(fire, 16, 64)

  fire = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(fire)

  fire = fire_module(fire, 32, 128)
  fire = fire_module(fire, 32, 128)

  fire = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(fire)

  fire = fire_module(fire, 48, 192)
  fire = fire_module(fire, 48, 192)
  fire = fire_module(fire, 64, 256)
  fire = fire_module(fire, 64, 256)

  dropout = Dropout(0.5)(fire)

  conv10 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu',
          strides=(1, 1), kernel_initializer=tf.keras.initializers.he_uniform())(dropout)

  avg_pool = GlobalAveragePooling2D()(conv10)

  #lin1 = Dense(512, activation='relu')(avg_pool)
  #lin2 = Dense(256, activation='relu')(lin1)
  #lin3 = Dense(128, activation='relu')(lin2)

  outputs = Dense(num_outputs, activation='tanh')(avg_pool)

  model = Model(inputs=inputs, outputs=outputs)

  model.compile(loss='mse',
              optimizer=Adam(lr=1e-3),
              metrics=[tf.keras.metrics.MeanAbsoluteError()])

  return model