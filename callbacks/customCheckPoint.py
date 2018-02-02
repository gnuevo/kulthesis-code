import keras.callbacks as kcallbacks


class customModelCheckpoint(kcallbacks.Callback):


    def __init__(self, model, dirpath, period=5):
        self._model = model
        self._dirpath = dirpath
        self._period = period
        self._epochs_since_last_save = 0

    def _save_model(self, epoch, logs=None):
        self._model.save(self._dirpath, {"last_epoch": epoch})

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self._last_epoch = epoch
        self._epochs_since_last_save += 1
        if self._epochs_since_last_save >= self._period:
            self._epochs_since_last_save = 0
            print("saving model")
            self._save_model(epoch, logs)

    def on_train_end(self, logs=None):
        self._save_model(self._last_epoch, logs)