"""Functions for reading data as tf.data.Dataset."""

import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.keras import layers

from google.cloud import storage

class Augment(tf.keras.layers.Layer):
    def __init__(self,  resize_shape=(768, 768), train=True, seed=42):
        super().__init__()
    # both use the same seed, so they'll make the same random changes.
        seed = np.random.randint(1000)
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


def rle_decode_tf(mask_rle, shape=(768, 768)):

    shape = tf.convert_to_tensor(shape, tf.int64)
    size = tf.math.reduce_prod(shape)
    # Split string
    s = tf.strings.split(mask_rle)
    s = tf.strings.to_number(s, tf.int64)
    # Get starts and lengths
    starts = s[::2] - 1
    lens = s[1::2]
    # Make ones to be scattered
    total_ones = tf.reduce_sum(lens)
    ones = tf.ones([total_ones], tf.uint8)
    # Make scattering indices
    r = tf.range(total_ones)
    lens_cum = tf.math.cumsum(lens)
    s = tf.searchsorted(lens_cum, r, 'right')
    idx = r + tf.gather(starts - tf.pad(lens_cum[:-1], [(1, 0)]), s)
    # Scatter ones into flattened mask
    mask_flat = tf.scatter_nd(tf.expand_dims(idx, 1), ones, [size])
    return tf.expand_dims(tf.transpose(tf.reshape(mask_flat, shape)), axis=2)

def multi_rle_encode(img):
    labels = label(img[:, :, 0])
    return [rle_encode(labels==k) for k in np.unique(labels[labels>0])]

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T   # Needed to align to RLE direction

def masks_as_image(in_mask_list):
    #in_mask_list = tf.compat.as_str_any(in_mask_list)
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype = np.int16)
    #if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)

def merge_rle_encode(mask_rle, shape=(768, 768)):
    img = np.zeros(shape=shape, dtype=np.uint8)

    for rle in mask_rle.split(";"):
        img += rle_decode(rle)

    return rle_encode(img)

def parse_db_to_img(filename, label):
    file_path = filename
    img = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(img, channels=3)
    label_img = rle_decode_tf(label)
    return image, label_img

def get_dataset(file_name, feature_spec):
    """Generates features and label for tuning/training.
    Args:
      file_pattern: input file path
      feature_spec: a dictionary of feature specifications.
      batch_size: representing the number of consecutive elements of returned
        dataset to combine in a single batch
    Returns:
      A dataset that contains (features, indices) tuple where features is a
        dictionary of Tensors, and indices is a single Tensor of label indices.
    """
    
    GCS_IMAGES = feature_spec['gcs_image']
    IMG_SHAPE = feature_spec['img_shape']
    GCS_BUCKET = feature_spec['gcs_bucket']
    BATCH_SIZE = feature_spec['batch_size']
    
    bucket = storage.Client().bucket(GCS_BUCKET)
    blob = bucket.blob(file_name)
    blob.download_to_filename(file_name)
    
    train_df = pd.read_parquet(file_name)
    dataset = tf.data.Dataset.from_tensor_slices((train_df['ImageId'].values, train_df['EncodedPixels'].values))
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.map(lambda x, y: parse_db_to_img(GCS_IMAGES + x, y))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(Augment(resize_shape=IMG_SHAPE, train=True))
    dataset = dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset
