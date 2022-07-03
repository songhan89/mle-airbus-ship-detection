"""Defaults for the model.
These values can be tweaked to affect model training performance.
"""


NET_SCALING = None
IMG_SHAPE = (128, 128)
GUASSIAN_NOISE = 0.1
BATCH_SIZE = 16
NUM_EPOCHS = 1
NUM_EVAL_STEPS = 10
EDGE_CROP = 16
UPSAMPLE_MODE = "SIMPLE"
LEARNING_RATE = 1e-4 
DECAY_RATE = 1e-6

GCS_IMAGES = "gs://mle_airbus_dataset/train_v2/"
GCS_BUCKET = "mle_airbus_dataset/"

def update_hyperparams(hyperparams: dict) -> dict:
    if "net_scaling" not in hyperparams:
        hyperparams["net_scaling"] = NET_SCALING
    if "img_shape" not in hyperparams:
        hyperparams["img_shape"] = IMG_SHAPE
    if "guassian_noise" not in hyperparams:
        hyperparams["num_epochs"] = GUASSIAN_NOISE
    if "batch_size" not in hyperparams:
        hyperparams["batch_size"] = BATCH_SIZE
    if "num_epochs" not in hyperparams:
        hyperparams["num_epochs"] = NUM_EPOCHS
    if "num_eval_steps" not in hyperparams:
        hyperparams["num_eval_steps"] = NUM_EVAL_STEPS
    if "edge_crop" not in hyperparams:
        hyperparams["edge_crop"] = EDGE_CROP
    if "upsample_mode" not in hyperparams:
        hyperparams["upsample_mode"] = UPSAMPLE_MODE
    if "learning_rate" not in hyperparams:
        hyperparams["learning_rate"] = LEARNING_RATE
    if "decay_rate" not in hyperparams:
        hyperparams["decay_rate"] = DECAY
    
    if "gcs_image" not in hyperparams:
        hyperparams["gcs_image"] = GCS_IMAGES
    if "gcs_bucket" not in hyperparams:
        hyperparams["gcs_bucket"] = GCS_BUCKET
    return hyperparams
