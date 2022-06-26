FROM python:3.7-slim
RUN ls
WORKDIR /a-working
COPY requirements.txt requirements.txt
COPY app.py app.py
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["/a-working/app.py"]