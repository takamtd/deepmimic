FROM nvidia/cudagl:10.0-devel-ubuntu18.04
LABEL maintainer="Takato Horii"

VOLUME /KHI

#setting
ENV DEBIAN_FRONTEND=noninteractive

# update and install
RUN apt update -y && apt upgrade -y && apt install -y \
    wget \
    curl \
    git \
    make \
    build-essential \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    libssl1.0-dev \
    python-openssl \
    xvfb \
    x11vnc \
    python-opengl \
    ffmpeg \
    nodejs \
    nodejs-dev \
    node-gyp \
    npm \
    icewm \
    python3 \
    python3-pip

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh -O ~/anaconda.sh && \
    #wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

ENV PATH=/opt/conda/bin:$PATH

RUN conda update -n base -c defaults conda

# install jupyter lab
RUN apt-get update -y && apt-get install -y \
    libgl1-mesa-glx wget curl git tmux imagemagick htop libsndfile1 nodejs npm \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# install jupyter lab tensorboard
RUN npm install -g n \
    && n stable
RUN apt purge -y nodejs npm
RUN pip install jupyterlab 
RUN pip install tensorflow-gpu==1.13.1 jupyter-tensorboard==0.2.0 tensorflow-estimator==1.13.0 tensorboard==1.13.1
#RUN pip install tensorflow jupyter-tensorboard tensorflow-estimator tensorboard
RUN jupyter labextension install jupyterlab_tensorboard
RUN jupyter tensorboard enable --system
RUN jupyter lab clean

# install code server
# https://github.com/jupyterhub/jupyter-server-proxy
RUN conda install jupyter-server-proxy -c conda-forge
RUN jupyter labextension install @jupyterlab/server-proxy
RUN pip install jupyter-vscode-proxy
RUN curl -fOL https://github.com/cdr/code-server/releases/download/v3.4.1/code-server_3.4.1_amd64.deb
RUN dpkg -i code-server_3.4.1_amd64.deb
RUN rm -r code-server_3.4.1_amd64.deb

# install jupyter-desktop-server ( noVNC )
# https://github.com/yuvipanda/jupyter-desktop-server
RUN apt-get update -y && \
    env DEBIAN_FRONTEND=noninteractive apt-get install -y dbus-x11 \
    xfce4 \
    xfce4-panel \
    xfce4-session \
    xfce4-settings \
    xorg \
    xubuntu-icon-theme \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
# RUN git clone https://github.com/yuvipanda/jupyter-desktop-server.git /opt/install
# RUN cd /opt/install && \
#    conda env update -n base --file environment.yml

# https://github.com/manics/jupyter-omeroanalysis-desktop/blob/turbovnc/Dockerfile
# RUN apt-get -y update \
#  && apt-get install -y dbus-x11 \
#    firefox \
#    xfce4 \
#    xfce4-panel \
#    xfce4-session \
#    xfce4-settings \
#    xorg \
#    xubuntu-icon-theme
RUN wget 'https://sourceforge.net/projects/turbovnc/files/2.2.5/turbovnc_2.2.5_amd64.deb/download' -O turbovnc_2.2.5_amd64.deb && \
   apt-get install ./turbovnc_2.2.5_amd64.deb && \
   rm ./turbovnc_2.2.5_amd64.deb
# apt-get may result in root-owned directories/files under $HOME
# RUN chown -R $NB_UID:$NB_GID $HOME
RUN git clone https://github.com/manics/jupyter-omeroanalysis-desktop.git /opt/install/jupyter-omeroanalysis-desktop
RUN cd /opt/install/jupyter-omeroanalysis-desktop; git checkout turbovnc
# RUN fix-permissions /opt/install
# USER $NB_USER
RUN cd /opt/install/jupyter-omeroanalysis-desktop && \
   conda env update -n base --file environment.yml

# install pakage for DeepMimic
RUN apt-get update -y && apt-get install -y \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libx11-dev \
    libxrandr-dev \
    libxi-dev \
    freeglut3-dev \
    mesa-utils \
    cmake \
    clang \
    libopenmpi-dev

#python lib
RUN python3 -m pip install --no-cache-dir --upgrade pip && python3 -m pip install --no-cache-dir \
    setuptools \
    moviepy \
    jupyterlab \
    torch torchvision \
    #tensorflow\
    gym \
    pybullet \
    PyOpenGL \
    PyOpenGL_accelerate \
    mpi4py

#install cuDNN7
RUN apt-get update -y && \
    wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libcudnn7_7.4.2.24-1+cuda10.0_amd64.deb && \
    wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libcudnn7-dev_7.4.2.24-1+cuda10.0_amd64.deb && \
    dpkg -i libcudnn7_7.4.2.24-1+cuda10.0_amd64.deb && \
    dpkg -i libcudnn7-dev_7.4.2.24-1+cuda10.0_amd64.deb && \
    rm -r libcudnn7_7.4.2.24-1+cuda10.0_amd64.deb && rm -r libcudnn7-dev_7.4.2.24-1+cuda10.0_amd64.deb 

#install bullet
RUN git clone https://github.com/bulletphysics/bullet3.git && \
    cd bullet3 && \
    sed -i -e 's/DUSE_DOUBLE_PRECISION=ON/DUSE_DOUBLE_PRECISION=OFF/' build_cmake_pybullet_double.sh && \
    ./build_cmake_pybullet_double.sh && \
    cd build_cmake && make install

#install eigen
RUN wget https://gitlab.com/libeigen/eigen/-/archive/3.3.8/eigen-3.3.8.tar.bz2 && \
    tar -jxvf eigen-3.3.8.tar.bz2 && \
    rm -r eigen-3.3.8.tar.bz2 && \
    cd eigen-3.3.8 && \
    mkdir build && cd build && \
    cmake .. && make install

#install freeglut
RUN wget http://prdownloads.sourceforge.net/freeglut/freeglut-3.0.0.tar.gz?download && \
    tar -zxvf freeglut-3.0.0.tar.gz?download && \
    rm -r freeglut-3.0.0.tar.gz?download && \
    cd freeglut-3.0.0 && \
    cmake . && make && make install

#install glew
RUN wget https://sourceforge.net/projects/glew/files/glew/2.1.0/glew-2.1.0.tgz && \
    tar -zxvf glew-2.1.0.tgz && \
    rm -r glew-2.1.0.tgz && \
    cd glew-2.1.0 && \
    make && make install && make clean

#install swig
RUN wget http://prdownloads.sourceforge.net/swig/swig-4.0.2.tar.gz && \
    tar -zxvf swig-4.0.2.tar.gz && \
    rm -r swig-4.0.2.tar.gz &&\
    cd swig-4.0.2 && \
    ./configure --without-pcre && \
    make && make install

#setting
RUN apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN ln /usr/lib64/libGLEW.so.2.1 /usr/lib/x86_64-linux-gnu/libGLEW.so.2.1 && \
    ln /usr/lib64/libGLEW.so.2.1.0 /usr/lib/x86_64-linux-gnu/libGLEW.so.2.1.0

ENV LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH

# Since uid and gid will change at entrypoint, anything can be used
ARG USER_ID=1000
ARG GROUP_ID=1000
ENV USER_NAME=jovyan
RUN groupadd -g ${GROUP_ID} ${USER_NAME} && \
    useradd -d /home/${USER_NAME} -m -s /bin/bash -u ${USER_ID} -g ${GROUP_ID} ${USER_NAME}
WORKDIR /home/${USER_NAME}

USER ${USER_NAME}
ENV HOME /home/${USER_NAME}

USER root

ENV NB_PREFIX /
ENV DISPLAY=":1.0"

#copy files from host
#ENV DIR=home
#WORKDIR ${DIR}
#COPY khi_robot_description/ /${DIR}/khi_robot_description/
#COPY Kaleido/ /${DIR}/Kaleido/

CMD ["sh","-c", "jupyter lab --notebook-dir=/home/jovyan --ip=0.0.0.0 --no-browser --allow-root --port=8888 --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*' --NotebookApp.base_url=${NB_PREFIX}"]
#CMD ["bash"]