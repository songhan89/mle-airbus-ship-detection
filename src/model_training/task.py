"""The entrypoint for the Vertex training job."""

import os
import sys
from datetime import datetime
import logging
import tensorflow as tf
from tensorflow.python.client import device_lib
import argparse
from typing import Optional

from google.cloud import aiplatform as vertex_ai

# from src.model_training 
import defaults, trainer


dirname = os.path.dirname(__file__)
dirname = dirname.replace("/model_training", "")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-dir",
        default="",
        type=str,
    )

    parser.add_argument(
        "--train-data-dir",
        type=str,
    )

    parser.add_argument(
        "--eval-data-dir",
        type=str,
    )
    
    parser.add_argument(
        "--out-model",
        type=str,
    )
    
    parser.add_argument("--net-scaling", default=None, type=Optional[bool])
    parser.add_argument("--image-shape", default=(128,128), type=tuple)
    parser.add_argument("--guassian-noise", default=0.1, type=float)
    parser.add_argument("--batch-size", default=16, type=float)
    parser.add_argument("--num-epochs", default=1, type=int)
    parser.add_argument("--num-eval-steps", default=10, type=int)
    parser.add_argument("--edge-crop", default=16, type=int)
    parser.add_argument("--upsample-mode", default="SIMPLE", type=str)
    parser.add_argument("--learning-rate", default=1e-4, type=float)
    parser.add_argument("--decay-rate", default=1e-6, type=float)

    parser.add_argument("--gcs-bucket", default="mle_airbus_dataset", type=str)
    parser.add_argument("--gcs-image", default="gs://mle_airbus_dataset/train_v2/", type=str)
    
    return parser.parse_args()


def main() -> str:
    args = get_args()
    time = datetime.now()
    hyperparams = vars(args)
    hyperparams = defaults.update_hyperparams(hyperparams)
    logging.info(f"Hyperparameter: {hyperparams}")

#     if args.experiment_name:
#         vertex_ai.init(
#             project=args.project,
#             staging_bucket=args.staging_bucket,
#             experiment=args.experiment_name,
#         )

#         logging.info(f"Using Vertex AI experiment: {args.experiment_name}")

#         run_id = args.run_name
#         if not run_id:
#             run_id = f"run-gcp-{datetime.now().strftime('%Y%m%d%H%M%S')}"

#         vertex_ai.start_run(run_id)
#         logging.info(f"Run {run_id} started.")

#         vertex_ai.log_params(hyperparams)

    seg_model = trainer.train(
        train_data_dir=args.train_data_dir,
        eval_data_dir=args.eval_data_dir,
        hyperparams=hyperparams,
    )

    # val_loss, val_accuracy = trainer.evaluate(
    #     model=classifier,
    #     data_dir=args.eval_data_dir,
    #     raw_schema_location=RAW_SCHEMA_LOCATION,
    #     tft_output_dir=args.tft_output_dir,
    #     hyperparams=hyperparams,
    # )
    
    
    # Report val_accuracy to Vertex hypertuner.
    # logging.info(f'Reporting metric {HYPERTUNE_METRIC_NAME}={val_accuracy} to Vertex hypertuner...')
    # hpt = hypertune.HyperTune()
    # hpt.report_hyperparameter_tuning_metric(
    #     hyperparameter_metric_tag=HYPERTUNE_METRIC_NAME,
    #     metric_value=val_accuracy,
    #     global_step=args.num_epochs * args.batch_size
    # )

    # Log metrics in Vertex Experiments.
    # logging.info(f'Logging metrics to Vertex Experiments...')
    # if args.experiment_name:
    #     vertex_ai.log_metrics({"val_loss": val_loss, "val_accuracy": val_accuracy})
    
    
    import pickle

    logging.info(f"exporting model")
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    export_dir = "gs://mle_airbus_dataset/trained_model/segm_full_{}".format(timestamp)
    print('Exporting to {}'.format(export_dir))
    tf.saved_model.save(seg_model, export_dir)

    # in two lines of code
    # with open(hyperparams['model_dir'] + f"segm_full_{timestamp}/loss.pickle", "wb") as f:
    #     print('Exporting to {}/loss.pickle'.format(export_dir))
    #     pickle.dump(loss_history[0].history, f)

    logging.info(f"exported model: {export_dir}")
    # model_filename = args.model_dir + "model_" + datetime.datetime.now().strftime('%Y%m%d_%H%M')
    # model.save_model(model_filename)

    with open(hyperparams['out_model'], 'w') as f:
        f.write(export_dir)
    # os.stdout(export_dir)
    return (print(export_dir))

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    logging.info(f"Python Version = {sys.version}")
    logging.info(f"TensorFlow Version = {tf.__version__}")
    logging.info(f'TF_CONFIG = {os.environ.get("TF_CONFIG", "Not found")}')
    logging.info(f"DEVICES = {device_lib.list_local_devices()}")
    logging.info(f"Task started...")
    main()
    logging.info(f"Task completed.")
