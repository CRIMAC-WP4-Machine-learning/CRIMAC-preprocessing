FROM python:3 as builder

RUN mkdir /install
WORKDIR /install

COPY requirements.txt /requirements.txt

RUN apt-get update -y && \
    apt-get install -y git && \
    pip install --prefix=/install -r /requirements.txt

FROM python:3-slim

COPY --from=builder /install /usr/local
COPY run.sh CRIMAC_preprocess.py /app/

WORKDIR /app

CMD ["./run.sh"]
