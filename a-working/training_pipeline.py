import kfp
import kfp.dsl as dsl
from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Metrics,
)
from kfp.v2 import compiler
from data_processing import *

@dsl.pipeline(
    name="training-pipeline",
    description="Training pipeline attempt",
    pipeline_root=PIPELINE_ROOT,
)

def training_pipeline(project_dict: dict = {
    'PROJECT_ID' = 'mle-airbus-detection-smu',
    'GCS_BUCKET' = 'mle_airbus_dataset',
    'REGION' = 'asia-east1',
    'TABLE_BQ' = 'placeholder'}):
    
    import_file = import_file_component(project_dict)

    consumer_task = consumer( # noqa: F841
        import_file.output,
    )


compiler.Compiler().compile(pipeline_func=train_pipeline, package_path="training_pipeline.json")
DISPLAY_NAME = "intro_" + TIMESTAMP

job = aip.PipelineJob(
    display_name=DISPLAY_NAME,
    template_path="training_pipeline.json",
    pipeline_root=PIPELINE_ROOT,
)
job.submit()