@dsl.pipeline(
    name="training-pipeline",
    description="Training pipeline attempt",
    pipeline_root=PIPELINE_ROOT,
    )
    def pipeline(text: str = "hi there"):
        hw_task = hello_world(text)
        two_outputs_task = two_outputs(text)
    consumer_task = consumer( # noqa: F841
        hw_task.output,
        two_outputs_task.outputs["output_one"],
        two_outputs_task.outputs["output_two"],
    )
    from kfp.v2 import compiler # noqa: F811
    compiler.Compiler().compile(pipeline_func=pipeline,
    package_path="intro_pipeline.json")
    DISPLAY_NAME = "intro_" + TIMESTAMP
    job = aip.PipelineJob(
    display_name=DISPLAY_NAME,
    template_path="intro_pipeline.json",
    pipeline_root=PIPELINE_ROOT,
)
job.submit()