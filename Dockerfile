FROM clearlinux/numpy-mp as builder

RUN mkdir /install

COPY requirements.txt /tmp/requirements.txt

RUN swupd bundle-add c-basic git && \
    pip install --prefix=/install -r /tmp/requirements.txt

FROM clearlinux/numpy-mp

COPY --from=builder /install /usr
COPY CRIMAC_preprocess.py /app/CRIMAC_preprocess.py

WORKDIR /app

ENTRYPOINT ["docker-entrypoint.sh"]

CMD ["python3", "/app/CRIMAC_preprocess.py"]
