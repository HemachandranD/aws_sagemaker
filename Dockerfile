
FROM python:3.9-slim-buster

COPY requirements.txt /opt/ml/requirements.txt
COPY code/preprocess.py /opt/ml/code/preprocess.py
COPY code/train.py /opt/ml/code/train.py

RUN pip3 install -r /opt/ml/requirements.txt

ENV SAGEMAKER_PROGRAM preprocess.py
ENV SAGEMAKER_PROGRAM train.py
ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["python3"]
