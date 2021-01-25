FROM python:3 as builder

RUN mkdir /install
WORKDIR /install

COPY requirements.txt /requirements.txt

RUN apt-get update -y && \
    apt-get install -y git && \
    pip install --prefix=/install -r /requirements.txt

FROM python:3-slim

COPY --from=builder /install /usr/local
COPY CRIMAC_preprocess.py /app/CRIMAC_preprocess.py

WORKDIR /app

CMD ["python3", "/app/CRIMAC_preprocess.py"]
