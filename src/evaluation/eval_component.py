
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy,BinaryCrossentropy 
from google.cloud import storage
from src.models.preprocessing import Augment
from src.utils.common import *
from src.utils.dataset import *

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_p_bce(in_gt, in_pred):
    return 1e-3*binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)

def true_positive_rate(y_true, y_pred):
    return K.sum(K.flatten(y_true)*K.flatten(K.round(y_pred)))/K.sum(y_true)

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--test_filepath', dest='test_filepath',
                    default='', type=str,
                    help='Validation data file path')
parser.add_argument('--model_filepath', dest='model_filepath',
                    default='', type=str,
                    help='Model file path')
parser.add_argument('--output', dest='metrics',
                    default='', type=str,
                    help='Metrics output')
args = parser.parse_args()

test_filepath: str,
model_filepath: str,
metrics: Output[Metrics]

IMG_SHAPE=(128,128)
GCS_BUCKET="mle_airbus_dataset"
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
bucket = storage.Client().bucket(GCS_BUCKET)

blob = bucket.blob("test.parquet")
blob.download_to_filename("test.parquet")

valid_df = pd.read_parquet(f"test.parquet")
validation = tf.data.Dataset.from_tensor_slices((valid_df['ImageId'].values, valid_df['EncodedPixels'].values))
validation = validation.shuffle(buffer_size=10)
validation = validation.map(lambda x, y: parse_db_to_img("gs://mle_airbus_dataset/train_v2/" + x, y))
validation = validation.batch(BATCH_SIZE)
validation = validation.map(Augment(resize_shape=IMG_SHAPE, train=False))
validation = validation.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
model_eval = tf.keras.models.load_model(args.model_filepath, compile=False)
model_eval.compile(optimizer=Adam(1e-4, decay=1e-6), loss=dice_p_bce, metrics=[dice_coef, 'binary_accuracy', true_positive_rate])
result = model_eval.evaluate(validation)
metrics.log_metric("dice_coef", (result[1]))
metrics.log_metric("binary_accuracy", (result[2]))
metrics.log_metric("true_positive_rate", (result[3]))
    import numpy as np
    import google.cloud.aiplatform as aip
