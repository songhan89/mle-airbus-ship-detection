import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

class Augment(tf.keras.layers.Layer):
    """"
    Preprocessing object class wrapper
    """
    def __init__(self,  resize_shape=(768, 768), train=True, seed=42):
        super().__init__()

    # both use the same seed, so they'll make the same random changes.
        seed = np.random.randint(1000)
        #apply image augmentation during training stage only
        if train:
            self.augment_inputs = tf.keras.Sequential(
                                    [
                                        layers.experimental.preprocessing.RandomFlip(seed=seed),
                                        layers.experimental.preprocessing.RandomRotation(0.1, seed=seed),
                                        layers.experimental.preprocessing.RandomHeight(0.1, seed=seed),
                                        layers.experimental.preprocessing.RandomWidth(0.1, seed=seed),
                                        layers.experimental.preprocessing.RandomZoom(0.9, seed=seed),
                                        layers.experimental.preprocessing.Rescaling(1.0 / 255),
                                        layers.experimental.preprocessing.Resizing(resize_shape[0], resize_shape[0])
                                    ]
                                )

            self.augment_labels = tf.keras.Sequential(
                                    [
                                        layers.experimental.preprocessing.RandomFlip(seed=seed),
                                        layers.experimental.preprocessing.RandomRotation(0.1, seed=seed),
                                        layers.experimental.preprocessing.RandomHeight(0.1, seed=seed),
                                        layers.experimental.preprocessing.RandomWidth(0.1, seed=seed),
                                        layers.experimental.preprocessing.RandomZoom(0.9, seed=seed),
                                        layers.experimental.preprocessing.Resizing(resize_shape[0], resize_shape[0])
                                    ]
                                )
        #only normalization and resizing required for inference
        else:
            self.augment_inputs = tf.keras.Sequential(
                                    [
                                        layers.experimental.preprocessing.Rescaling(1.0 / 255),
                                        layers.experimental.preprocessing.Resizing(resize_shape[0], resize_shape[0])
                                    ]
                                )

            self.augment_labels = tf.keras.Sequential(
                                    [
                                        layers.experimental.preprocessing.Resizing(resize_shape[0], resize_shape[0])
                                    ]
                                )

    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels