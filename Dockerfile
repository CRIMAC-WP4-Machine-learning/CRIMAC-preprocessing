FROM python:3

# Install linux applications
RUN apt-get update && apt-get install --yes \
    curl \
    zsh \
    git

# Install pip and python packages
RUN pip install --upgrade pip
RUN pip install numpy scipy

# Clone the preprocessing library
RUN git clone https://github.com/CRIMAC-WP4-Machine-learning/CRIMAC-preprocessing
RUN git checkout NOH_pyech

# Clone pyecholab and switch to the RHT-EK80 branch
RUN git clone https://github.com/CI-CMG/pyEcholab
RUN git checkout RHT-EK80

RUN chmod 755 /CRIMAC-preprocessing/masterscript.sh
#CMD zsh
CMD /CRIMAC-preprocessing/masterscript.sh
