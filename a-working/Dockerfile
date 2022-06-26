FROM python:3.7-slim
WORKDIR /app
COPY requirements.txt requirements.txt
COPY app.py /app/app.py
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["/app/app.py"]

# WORKDIR /app
# COPY app.py /app/app.py
# ENTRYPOINT ["python"]
# CMD ["/app/app.py"]