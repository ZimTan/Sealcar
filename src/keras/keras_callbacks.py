from tensorflow import keras

class ModelCheckpointTFFormat(keras.callbacks.Callback):
    """Save at each epoch with the tensorflow format a model.

  Arguments:
      path: path ot the folder where it will be saved
  """

    def __init__(self, path):
        super(ModelCheckpointTFFormat, self).__init__()
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(filepath=self.path, save_format='tf')

    def on_train_end(self, logs=None):
        self.model.save(filepath=self.path, save_format='tf')
