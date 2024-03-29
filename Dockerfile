FROM python:3.8 as builder

RUN mkdir /install
WORKDIR /install

COPY requirements.txt /requirements.txt

RUN pip install --prefix=/install -r /requirements.txt

FROM python:3.8-slim

ARG version_number
ARG commit_sha

ENV VERSION_NUMBER=$version_number
ENV COMMIT_SHA=$commit_sha

COPY --from=builder /install /usr/local
COPY CRIMAC_preprocess.py /app/CRIMAC_preprocess.py

WORKDIR /app

CMD ["python3", "/app/CRIMAC_preprocess.py"]
