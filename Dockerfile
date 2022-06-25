FROM python:3.7-slim
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
WORKDIR /app
COPY app.py /app/app.py
ENTRYPOINT ["python"]
CMD ["/app/app.py"]