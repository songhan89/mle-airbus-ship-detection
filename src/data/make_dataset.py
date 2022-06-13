import tensorflow as tf
from src.utils.common import rle_decode_tf

def parse_db_to_img(filename, label):
    file_path = filename
    img = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(img, channels=3)
    label_img = rle_decode_tf(label)
    
    return image, label_img