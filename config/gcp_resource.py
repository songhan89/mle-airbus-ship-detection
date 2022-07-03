import os

# Required Parameters, change according to your setups
PROJECT_ID='mle-airbus-detection-smu'
GCS_BUCKET='mle_airbus_dataset'
REGION = 'asia-east1'
ARTIFACT_REGISTRY_REPO="airbus-mle"
CONTAINER_REGISTRY=f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{ARTIFACT_REGISTRY_REPO}/trainer-airbus-model:latest"
SERVICE_ACCOUNT = 'service-account-for-mle@mle-airbus-detection-smu.iam.gserviceaccount.com'

GCS_PATH=f"gs://{GCS_BUCKET}"
GCS_TRAIN_IMAGES=f"gs://{GCS_BUCKET}/train_v2/"
MODEL_DISPLAY_NAME="airbus-mle-model"
ENDPOINT_DISPLAY_NAME="airbus-mle-endpoint"
MODEL_DEPLOY_DISPLAY_NAME="airbus-mle-deploy"
TABLE_BQ="mle-airbus-detection-smu.airbus_data.label_data"

TRAIN_IMAGE = "asia-docker.pkg.dev/vertex-ai/training/tf-gpu.2-8:latest"
#not sure why deploy image does not work if use asia-docker.pkg.dev
DEPLOY_IMAGE = "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-8:latest"

project_dict = {}
project_dict['PROJECT_ID'] = PROJECT_ID
project_dict['GCS_BUCKET'] = GCS_BUCKET
project_dict['REGION'] = REGION
project_dict['TABLE_BQ'] = TABLE_BQ

ACCELERATOR_TYPE = 'NVIDIA_TESLA_K80'
MACHINE_TYPE = "n1-standard"

VCPU = "4"
TRAIN_COMPUTE = MACHINE_TYPE + "-" + VCPU

MACHINE_TYPE = "n1-standard"

VCPU = "4"
DEPLOY_COMPUTE = MACHINE_TYPE + "-" + VCPU

PIPELINE_NAME = 'airbusmlepipeline'

# Path to various pipeline artifact.
PIPELINE_ROOT = 'gs://{}/{}/pipeline_root'.format(
    GCS_BUCKET, PIPELINE_NAME)

# Paths for users' Python module.
MODULE_ROOT = 'gs://{}/{}/pipeline_module'.format(
    GCS_BUCKET, PIPELINE_NAME)

# Paths for users' data.
DATA_ROOT = 'gs://{}/{}/data'.format(GCS_BUCKET, PIPELINE_NAME)

# This is the path where your model will be pushed for serving.
SERVING_MODEL_DIR = 'gs://{}/{}/serving_model'.format(
    GCS_BUCKET, PIPELINE_NAME)

VERSION = 'v01'
DATASET_DISPLAY_NAME = 'airbus-ship-dataset-display'
MODEL_DISPLAY_NAME = f'{DATASET_DISPLAY_NAME}-classifier-{VERSION}'

WORKSPACE = f'gs://{GCS_BUCKET}/{DATASET_DISPLAY_NAME}'
EXPERIMENT_ARTIFACTS_DIR = os.path.join(WORKSPACE, 'experiments')

TENSORBOARD_DISPLAY_NAME = f'tb-{DATASET_DISPLAY_NAME}'
EXPERIMENT_NAME = f'{MODEL_DISPLAY_NAME}'

metrics_thresholds = {
    'dice_coef': 0.1,
    'binary_accuracy': 0.8,
    'true_positive_rate': 0.3
}