import kfp
import kfp.dsl as dsl
from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Metrics,
)
from typing import NamedTuple

@component(packages_to_install=["google-cloud-storage", "google-cloud-bigquery", "tensorflow", 
                                "sklearn", "pandas", "scikit-image", "db-dtypes", "google-auth",
                               "fsspec", "pyarrow"],
           output_component_file="import_file_component.yaml",)
def import_file_component(
    project_dict: dict
    ) -> NamedTuple(
    "Outputs",
    [
        ("train_parquet", str),  # Return parameter.
        ("test_parquet", str),  # Return generic Artifact.
    ],
    ):

    import requests
    import os
    import logging
    from sklearn.utils import resample
    from google.cloud import bigquery, storage
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    from google.oauth2 import service_account
    from skimage.segmentation import mark_boundaries
    from skimage.util import montage as montage2d
    from skimage.io import imread
    from skimage.segmentation import mark_boundaries
    from skimage.util import montage
    from skimage.morphology import label

    #TODO: How to improve these functions ?
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

    def parse_db_to_img(filename, label):
        file_path = filename
        img = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(img, channels=3)
        label_img = rle_decode_tf(label)

        return image, label_img
    
    BATCH_SIZE = 16
    EDGE_CROP = 16
    NB_EPOCHS = 10
    GAUSSIAN_NOISE = 0.1
    UPSAMPLE_MODE = 'SIMPLE'
    # downsampling inside the network
    NET_SCALING = None
    # downsampling in preprocessing
    IMG_SCALING = (1, 1)
    # number of validation images to use
    VALID_IMG_COUNT = 10
    # maximum number of steps_per_epoch in training
    MAX_TRAIN_STEPS = 200
    AUGMENT_BRIGHTNESS = False
    N_SAMPLE = 100
    IMG_SHAPE = (128, 128)

    PROJECT_ID = project_dict['PROJECT_ID']
    GCS_BUCKET = project_dict['GCS_BUCKET']
    REGION = project_dict['REGION']
    TABLE_BQ = project_dict['TABLE_BQ']
    
    bucket = storage.Client().bucket(GCS_BUCKET)


    try: 
        bqclient = bigquery.Client(project=PROJECT_ID, location=REGION)
        logging.info("No authentication required!")
    except:
        logging.info("Try a hacky way")
        blob = bucket.blob("mle-airbus-detection-smu-b1f8ee58e814.json")
        blob.download_to_filename("mle-airbus-detection-smu-b1f8ee58e814.json")
        credentials = service_account.Credentials.from_service_account_file(
            "mle-airbus-detection-smu-b1f8ee58e814.json", scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )

        bqclient = bigquery.Client(credentials=credentials, project=PROJECT_ID, location=REGION)
        logging.info("Authenticated!")

    # Download a table.
    table = bigquery.TableReference.from_string(
        #TODO: replace with param
        "mle-airbus-detection-smu.airbus_data.label_data"
    )
    rows = bqclient.list_rows(
        table
    )
    masks = rows.to_dataframe(
        # Optionally, explicitly request to use the BigQuery Storage API. As of
        # google-cloud-bigquery version 1.26.0 and above, the BigQuery Storage
        # API is used by default.
        create_bqstorage_client=True,
    )
    
    #TODO: parame
    masks = masks[:20000]
    masks.replace(to_replace=[None], value='', inplace=True)
    masks = masks.groupby(['ImageId'])['EncodedPixels'].apply(lambda x: ';'.join(x) if x is not None else ';'.join('')).reset_index()
    
    masks['ships'] = masks['EncodedPixels'].map(lambda c_row: c_row.count(";"))
    unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
    unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x>0 else 0.0)
    unique_img_ids['has_ship_vec'] = unique_img_ids['has_ship'].map(lambda x: [x])
    masks.drop(['ships'], axis=1, inplace=True)
    unique_img_ids.sample(5)
    masks.EncodedPixels = masks.EncodedPixels.apply(lambda x: merge_rle_encode(x))
    
    from sklearn.model_selection import train_test_split
    train_ids, valid_ids = train_test_split(unique_img_ids, 
                     test_size = 0.3, 
                     stratify = unique_img_ids['ships'])
    train_df = pd.merge(masks, train_ids)
    valid_df = pd.merge(masks, valid_ids)
    print(train_df.shape[0], 'training masks')
    print(valid_df.shape[0], 'validation masks')
    
    train_df_balanced = pd.DataFrame()
    for ship_num in train_df['ships'].unique():
        train_df_balanced = train_df_balanced.append(resample(train_df.query("ships == {}".format(ship_num)), n_samples=N_SAMPLE))
    train_df_balanced.reset_index(drop=True, inplace=True)

    valid_df_balanced = pd.DataFrame()
    for ship_num in valid_df['ships'].unique():
        valid_df_balanced = valid_df_balanced.append(resample(valid_df.query("ships == {}".format(ship_num)), n_samples=N_SAMPLE//10))

    #TODO: make this nicer , don't hard code
    train_df_balanced.to_parquet(f"train.parquet")
    valid_df_balanced.to_parquet(f"test.parquet")
    
    blob = bucket.blob('train.parquet')
    blob.upload_from_filename('train.parquet')
    blob = bucket.blob('test.parquet')
    blob.upload_from_filename('test.parquet')
    
    return f"gs://{GCS_BUCKET}/train.parquet", f"gs://{GCS_BUCKET}/test.parquet"