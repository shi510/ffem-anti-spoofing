import os

import tensorflow as tf
from tensorflow.python.keras import backend as K


class LossTensorBoard(tf.keras.callbacks.Callback):
    """
    This callback writes loss and learning rate.
    """
    def __init__(self, update_freq, log_dir='./logs'):
        self.train_dir = os.path.join(log_dir, 'train')
        self.test_dir = os.path.join(log_dir, 'test')
        self.update_freq = update_freq
        self.step = 0
        self.prev_lr = 0.
        self.writer_train = tf.summary.create_file_writer(self.train_dir)
        self.writer_test = tf.summary.create_file_writer(self.test_dir)


    def on_train_end(self, logs=None):
        self.writer_train.close()
        self.writer_test.close()


    def on_train_batch_end(self, batch, logs=None):
        if batch % self.update_freq == 0:
            logs = logs or {}
            with self.writer_train.as_default():
                for name, value in logs.items():
                    tf.summary.scalar(name, value, step=self.step)
            self.writer_train.flush()
        self.step += 1


    def on_test_end(self, logs=None):
        with self.writer_test.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=self.step)
        self.writer_test.flush()


    def on_train_batch_begin(self, batch, logs=None):
        if isinstance(self.model.optimizer.lr, tf.Variable):
            cur_lr = float(K.get_value(self.model.optimizer.lr))
        elif callable(self.model.optimizer.lr):
            cur_lr = self.model.optimizer.lr(self.step).numpy()
        else:
            raise '{} is not supported type for learning rate.'.format(self.model.optimizer.lr)
        if batch % self.update_freq == 0 or self.prev_lr != cur_lr:
            self.prev_lr = cur_lr
            with self.writer_train.as_default():
                tf.summary.scalar('lr', cur_lr, step=self.step)
            self.writer_train.flush()
