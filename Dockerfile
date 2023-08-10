FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04
# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.6    (conda)
# pytorch       1.8.1    (pip)
# torchvision   0.9.1    (pip)
# ==================================================================

COPY ./Miniconda3-latest-Linux-x86_64.sh ./miniconda.sh
COPY ./apex ./apex
COPY ./requirements.txt ./requirements.txt

ENV PATH /opt/conda/bin:$PATH

RUN sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && \
    APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade -i https://pypi.mirrors.ustc.edu.cn/simple/" && \
    GIT_CLONE="git clone --depth 10" && \
    CONDA_INSTALL="conda install -y" && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
    apt-get -y upgrade && \
# ==================================================================
# tools
# ------------------------------------------------------------------
    $APT_INSTALL \
        apt-utils \
        ca-certificates \
        cmake \
        wget \
        git \
        vim \
        openssh-client \
        openssh-server \
        libboost-dev \
        libboost-thread-dev \
        libboost-filesystem-dev \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libjpeg-dev \
        libturbojpeg \
        ffmpeg \
        ninja-build \
        && \
# ==================================================================
# Miniconda3
# ------------------------------------------------------------------
    /bin/bash ./miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    #echo "export PATH=/opt/conda/bin:$PATH" >> ~/.bashrc && \
# ==================================================================
# conda
# ------------------------------------------------------------------
    $CONDA_INSTALL \
        python=3.6.9 && \
    pip install --upgrade pip && \
    $PIP_INSTALL \
        torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html  && \
    $PIP_INSTALL -r ./requirements.txt && \
    rm ./requirements.txt && \
    $PIP_INSTALL -v ./apex && \
    rm -r ./apex && \
    conda clean -y --all && \
# ==================================================================
# config & cleanup
# ------------------------------------------------------------------
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*
