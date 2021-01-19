FROM python:3.8 as builder

COPY requirements.txt /tmp/requirements.txt

ENV PATH="/opt/venv/bin:$PATH"

RUN apt-get update -y && \
    apt-get install -y git lsb-release wget software-properties-common && \
    cd /tmp && wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 10 && \
    export LLVM_CONFIG=$(which llvm-config-10) && \
    python -m venv /opt/venv && \
    python -m pip install --upgrade pip && \
    pip install numpy && \
    git clone git://github.com/numba/numba.git && cd numba && git checkout 0.52.0 && \
    python setup.py install && \
    pip install -r /tmp/requirements.txt

FROM python:3.8-slim

COPY --from=builder /opt/venv /opt/venv
COPY CRIMAC_preprocess.py /app/CRIMAC_preprocess.py

RUN apt-get update -y && \
    apt-get install -y wget gnupg1 && \
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
    echo deb http://apt.llvm.org/buster/ llvm-toolchain-buster-10 main >> /etc/apt/sources.list && \
    apt-get update -y && \
    apt-get install -y libllvm10 libgomp1 && \
    apt-get remove --purge -y wget gnupg1 && \
    apt-get autoremove --purge -y && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PATH="/opt/venv/bin:$PATH"

CMD ["python3", "/app/CRIMAC_preprocess.py"]
