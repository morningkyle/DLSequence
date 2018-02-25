import keras


class MyCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        print("Weights before training:", self.model.get_weights())

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        print("Weights after batch:", self.model.get_weights())



