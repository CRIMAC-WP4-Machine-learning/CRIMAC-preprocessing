FROM python:3 as builder

RUN mkdir /install
WORKDIR /install

COPY requirements.txt /requirements.txt

RUN apt-get update -y && \
    # Requirements for building the packages
    apt-get install -y libjemalloc-dev libboost-dev \
                       libboost-filesystem-dev \
                       libboost-system-dev \
                       libboost-regex-dev \
                       python3-dev \
                       autoconf \
                       flex \
                       bison \
                       libssl-dev \
                       curl \
                       cmake \
                       llvm-7 \
                       clang \
                       libnetcdf-dev && \
    pip install six numpy pandas cython pytest psutil && \

    mkdir build && \
    cd build && \
    wget https://github.com/apache/arrow/archive/apache-arrow-3.0.0.tar.gz && \
    tar -zxvf apache-arrow-3.0.0.tar.gz && \
    cd arrow-apache-arrow-3.0.0/cpp && \
    mkdir release && \
    cd release && \
    export ARROW_HOME=/install && \
    export LD_LIBRARY_PATH=/install/lib:$LD_LIBRARY_PATH && \

    cmake -DCMAKE_INSTALL_PREFIX=$ARROW_HOME \
      #-DCMAKE_INSTALL_LIBDIR=lib  \
      -DARROW_FLIGHT=ON \
      -DARROW_GANDIVA=ON  \
      -DARROW_ORC=ON  \
      -DARROW_WITH_BZ2=ON \
      -DARROW_WITH_ZLIB=ON  \
      -DARROW_WITH_ZSTD=ON  \
      -DARROW_WITH_LZ4=ON \
      -DARROW_WITH_SNAPPY=ON \
      -DARROW_WITH_BROTLI=ON \
      -DARROW_PARQUET=ON  \
      -DARROW_PYTHON=ON \
      -DARROW_PLASMA=ON \
      #-DARROW_CUDA=ON \
      #-DARROW_BUILD_TESTS=ON  \
      #-DPYTHON_EXECUTABLE=/usr/bin/python \ 
      .. && \
    make && \
    make install && \

    cd ../.. && \
    cd python/ && \
    pip install -r requirements-build.txt && \
    export PYARROW_WITH_FLIGHT=1 && \
    export PYARROW_WITH_GANDIVA=1 && \
    export PYARROW_WITH_ORC=1 && \
    export PYARROW_WITH_PARQUET=1 && \
    #export PYARROW_WITH_CUDA=1 && \
    export PYARROW_WITH_PLASMA=1 && \
    python setup.py build_ext --inplace && \
    python setup.py install && \
    export CFLAGS="" && \

    cd ../../../ && \
    pip install --prefix=/install -r /requirements.txt

FROM python:3-slim

COPY --from=builder /install /usr/local
COPY CRIMAC_preprocess.py /app/CRIMAC_preprocess.py

WORKDIR /app

CMD ["python", "/app/CRIMAC_preprocess.py"]
