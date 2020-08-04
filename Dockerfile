FROM python:3

# Install linux applications
RUN apt-get update && apt-get install --yes \
    fonts-powerline \
    curl \
    zsh \
    git \
    emacs

# Install matlab runtime
RUN mkdir /mcr-install && \
    mkdir /opt/mcr && \
    cd /mcr-install && \
    wget -q https://ssd.mathworks.com/supportfiles/downloads/R2020a/Release/4/deployment_files/installer/complete/glnxa64/MATLAB_Runtime_R2020a_Update_4_glnxa64.zip && \
    unzip MATLAB_Runtime_R2020a_Update_4_glnxa64.zip && \
    rm MATLAB_Runtime_R2020a_Update_4_glnxa64.zip && \
    ./install -destinationFolder /opt/mcr -agreeToLicense yes -mode silent && \
    cd / && \
    rm -rf mcr-install

# Install pip and python packages
RUN pip install --upgrade pip
RUN pip install numpy scipy

# Clone the preprocessing library
RUN git clone https://github.com/CRIMAC-WP4-Machine-learning/CRIMAC-preprocessing

#    wget \
#    xorg \
#    apt-get clean && \
#    rm -rf /var/lib/apt/lists/*

# Matlab runtime environmental variables
# ENV LD_LIBRARY_PATH /opt/mcr/v98/runtime/glnxa64:/opt/mcr/v98/bin/glnxa64:/opt/mcr/v98/sys/os/glnxa64:/opt/mcr/v98/extern/bin/glnxa64
# ENV XAPPLRESDIR /opt/mcr/v93/X11/app-defaults

#CMD python3 /CRIMAC-preprocessing/CRIMAC_preprocess_generate_memmap_files.py

CMD zsh

