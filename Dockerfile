
FROM python:3.9-slim-buster

COPY requirements.txt /opt/ml/requirements.txt
COPY code/serve.py /opt/ml/code/serve.py

RUN pip3 install -r /opt/ml/requirements.txt
RUN pip3 install flask gunicorn

ENV SAGEMAKER_PROGRAM serve.py
ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["python3", "/opt/ml/code/serve.py"]
