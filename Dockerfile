FROM asia-docker.pkg.dev/vertex-ai/training/tf-cpu.2-8:latest
WORKDIR /app
COPY src/model_training src/model_training
ENTRYPOINT ["python","./src/model_training/task.py"]
