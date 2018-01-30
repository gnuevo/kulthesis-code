import keras.callbacks as kcallbacks


class customModelCheckpoint(kcallbacks.Callback):


    def __init__(self, model, dirpath, period=5):
        self._model = model
        self._dirpath = dirpath
        self._period = period
        self._epochs_since_last_save = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self._epochs_since_last_save += 1
        if self._epochs_since_last_save >= self._period:
            self._epochs_since_last_save = 0
            print("saving model")
            self._model.save(self._dirpath, {"last_epoch": epoch})