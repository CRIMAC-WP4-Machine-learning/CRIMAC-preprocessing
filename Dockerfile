FROM python:3
# Install applications
RUN apt-get update && apt-get install --yes \
    fonts-powerline \
    curl \
    zsh \
    git \
    emacs
#    wget \
#    xorg \
#    apt-get clean && \
#    rm -rf /var/lib/apt/lists/*

# Install the MCR and dependencies
#RUN mkdir /mcr-install && \
#    mkdir /opt/mcr && \
#    cd /mcr-install && \
#    wget -q http://ssd.mathworks.com/supportfiles/downloads/R2020b/deployment_files/R2020b/installers/glnxa64/MCR_Runzip -q MCR_R2020b_glnxa64_installer.zip && \
#    unzip MATLAB_Runtime_R2020a_glnxa64.zip
#        rm -f MCR_R2020b_glnxa64_installer.zip && \
#    ./install -destinationFolder /opt/mcr -agreeToLicense yes -mode silent && \
#    cd / && \
#    rm -rf mcr-install
RUN pip install --upgrade pip
RUN pip install numpy scipy
# Clone the preprocessing library
RUN git clone https://github.com/CRIMAC-WP4-Machine-learning/CRIMAC-preprocessing
CMD python3 /CRIMAC-preprocessing/CRIMAC_preprocess_generate_memmap_files.py