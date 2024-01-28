FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

# Build this Dockerfile with the following command: 
# docker build --no-cache --build-arg USRNM=$(whoami) --build-arg USRUID=$(id -u) --build-arg USRGID=$(id -g) -t sionna-autoencoder:latest .
# This ensures that you will have the right permissions inside docker

# User and group variables
ARG USRNM
ARG USRUID
ARG USRGID

# Create user and add to the specified group
RUN groupadd -g $USRGID $USRNM || echo "Could not create group with GID $USRGID".
RUN useradd -g $USRGID -u $USRUID -m $USRNM

# Set debconf to non interactive
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# Install basic tools
RUN apt-get update && apt-get install -yq apt-utils && apt-get install -y curl nano htop rsync lsof zip unzip wget vim
RUN apt-get update && apt-get install -yq gcc g++ swig build-essential
RUN apt-get update && apt-get install -yq git tini

# install sudo
RUN apt-get update && apt-get install -yq sudo
     
# Add user to sudoers
RUN adduser $USRNM sudo

# Ensure sudo group users are not 
# asked for a password when using 
# sudo command by ammending sudoers file
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Add aliases
RUN echo 'alias ll="ls -lah"' >> /root/.bashrc

# Colored outputs in bash
ENV RED='\033[0;31m'
ENV NC='\033[0m'

RUN rm -rf /var/lib/apt/lists/*

# Install python
RUN apt-get update && apt-get install -yq python3.9 
RUN apt-get update && apt-get install -yq python3-pip

# Install jupyter notebook
RUN pip3 install jupyter==1.0.0 

# notebook password is set to `notebook`
ARG NBPASSCONF='{\
  "NotebookApp": {\
    "password": "sha1:cd11a6ce8f64:7ec0f3eda125e0553b10b77e903cc48c7027034c",\
    "disable_check_xsrf": true\
  }\
}'

# show line numbers
ARG NBCELLCONF='{"CodeCell":{"cm_config":{"lineNumbers": true}}}'

RUN echo 'alias ll="ls -lah"' >> /home/$USRNM/.bashrc

# Configure notebook for user
RUN mkdir -p /home/$USRNM/.jupyter
RUN echo $NBPASSCONF > /home/$USRNM/.jupyter/jupyter_notebook_config.json
RUN mkdir -p /home/$USRNM/.jupyter/nbconfig
RUN echo $NBCELLCONF > /home/$USRNM/.jupyter/nbconfig/notebook.json

# Install required python packages
RUN pip3 install numpy
RUN pip3 install matplotlib
RUN pip3 install importlib_resources
RUN pip3 install pythreejs
RUN pip3 install ipywidgets
RUN pip3 install ipydatawidgets
RUN pip3 install jupyterlab-widgets
RUN pip3 install torch==1.10.1
RUN pip3 install pytorch-lightning
RUN pip3 install PyYAML
RUN pip3 install tensorboard
RUN pip3 install torchsummary
RUN pip3 install torchvision
RUN pip3 install gdown

USER $USRNM

RUN sudo chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
