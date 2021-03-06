"""This file was copied from keras.callbacks.TensorBoard

The idea is to custom it but reusing most part of the code
"""

import keras.callbacks as kcallbacks
import keras.backend as K
import os

if K.backend() == 'tensorflow':
    import tensorflow as tf
    from tensorflow.contrib.tensorboard.plugins import projector

class customTensorBoard(kcallbacks.Callback):
    """Tensorboard basic visualizations.

    This callback writes a log for TensorBoard, which allows
    you to visualize dynamic graphs of your training and test
    metrics, as well as activation histograms for the different
    layers in your model.

    TensorBoard is a visualization tool provided with TensorFlow.

    If you have installed TensorFlow with pip, you should be able
    to launch TensorBoard from the command line:
    ```
    tensorboard --logdir=/full_path_to_your_logs
    ```
    You can find more information about TensorBoard
    [here](https://www.tensorflow.org/versions/master/how_tos/summaries_and_tensorboard/index.html).

    # Arguments
        log_dir: the path of the directory where to save the log
            files to be parsed by Tensorboard.
        histogram_freq: frequency (in epochs) at which to compute activation
            histograms for the layers of the model. If set to 0,
            histograms won't be computed.
        write_graph: whether to visualize the graph in Tensorboard.
            The log file can become quite large when
            write_graph is set to True.
        write_images: whether to write model weights to visualize as
            image in Tensorboard.
        embeddings_freq: frequency (in epochs) at which selected embedding
            layers will be saved.
        embeddings_layer_names: a list of names of layers to keep eye on. If
            None or empty list all the embedding layer will be watched.
        embeddings_metadata: a dictionary which maps layer name to a file name
            in which metadata for this embedding layer is saved. See the
            [details](https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
            about metadata files format. In case if the same metadata file is
            used for all embedding layers, string can be passed.
            
        batch_freq: frequency in epochs in which you want to write the
            summaries to disk. If none then the summaries are written for
            every epoch.
    """

    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 write_graph=True,
                 write_images=False,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None,
                 batch_freq=None,
                 batches_per_epoch=None,
                 variables=['loss', 'val_loss']):
        super(customTensorBoard, self).__init__()
        if K.backend() != 'tensorflow':
            raise RuntimeError('TensorBoard callback only works '
                               'with the TensorFlow backend.')
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.merged = None
        self.write_graph = write_graph
        self.write_images = write_images
        self.embeddings_freq = embeddings_freq
        self.embeddings_layer_names = embeddings_layer_names
        self.embeddings_metadata = embeddings_metadata or {}
        self.variables = variables
        self.batch_freq = batch_freq
        self.batches_per_epoch = batches_per_epoch
        self.batch_count = -1

    def set_model(self, model):
        self.model = model
        self.sess = K.get_session()
        if self.histogram_freq and self.merged is None:
            for layer in self.model.layers:

                for weight in layer.weights:
                    tf.summary.histogram(weight.name, weight)
                    if self.write_images:
                        w_img = tf.squeeze(weight)
                        shape = w_img.get_shape()
                        if len(shape) > 1 and shape[0] > shape[1]:
                            w_img = tf.transpose(w_img)
                        if len(shape) == 1:
                            w_img = tf.expand_dims(w_img, 0)
                        w_img = tf.expand_dims(tf.expand_dims(w_img, 0), -1)
                        tf.summary.image(weight.name, w_img)

                if hasattr(layer, 'output'):
                    tf.summary.histogram('{}_out'.format(layer.name),
                                         layer.output)
        self.merged = tf.summary.merge_all()

        self.writer = {}
        for var in self.variables:
            if self.write_graph:
                self.writer[var] = tf.summary.FileWriter(self.log_dir+'/'+var,
                                                    self.sess.graph)
            else:
                self.writer[var] = tf.summary.FileWriter(self.log_dir+'/'+var)

    def write_summaries(self, index, logs=None, variables=None):
        logs = logs or {}

        if self.validation_data and self.histogram_freq:
            if index % self.histogram_freq == 0:
                # TODO: implement batched calls to sess.run
                # (current call will likely go OOM on GPU)
                if self.model.uses_learning_phase:
                    cut_v_data = len(self.model.inputs)
                    val_data = self.validation_data[:cut_v_data] + [0]
                    tensors = self.model.inputs + [K.learning_phase()]
                else:
                    val_data = self.validation_data
                    tensors = self.model.inputs
                feed_dict = dict(zip(tensors, val_data))
                result = self.sess.run([self.merged], feed_dict=feed_dict)
                summary_str = result[0]
                self.writer.add_summary(summary_str, index)

        for name, value in logs.items():
            if name not in variables:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = "value"
            self.writer[name].add_summary(summary, index)
        for key, writer in self.writer.items():
            writer.flush()

    def on_batch_end(self, batch, logs=None):
        self.batch_count += 1
        index = self.batch_count
        if not self.batch_freq == None:
            if batch % self.batch_freq == 0:
                self.write_summaries(index, logs, variables=["loss"])

    def on_epoch_end(self, epoch, logs=None):
        if self.batch_freq == None:
            self.write_summaries(epoch, logs, variables=self.variables)
        else:
            self.write_summaries(self.batch_count, logs, variables=self.variables)

    def on_train_end(self, _):
        for key, writer in self.writer.items():
            writer.close()