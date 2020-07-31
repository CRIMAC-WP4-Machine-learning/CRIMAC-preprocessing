FROM python:3
RUN apt-get update && apt-get install --yes \
    fonts-powerline \
    curl \
    zsh \
    git \
    emacs
RUN pip install --upgrade pip
RUN pip install jupyterlab
RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" --yes
RUN mkdir repos
RUN cd repos
# Clone the preprocessing library
RUN git clone https://github.com/CRIMAC-WP4-Machine-learning/CRIMAC-preprocessing
#CMD jupyter-notebook --ip=0.0.0.0 --port=8989 --no-browser --allow-root &
CMD zsh