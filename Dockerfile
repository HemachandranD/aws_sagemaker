
FROM python:3.9-slim-buster

RUN pip install pandas==2.2.2 scikit-learn==1.4.2 prophet==1.1.5

COPY code/preprocess.py /opt/ml/code/preprocess.py
COPY code/train.py /opt/ml/code/train.py

ENV SAGEMAKER_PROGRAM preprocess.py
ENV SAGEMAKER_PROGRAM train.py
ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["python3"]
