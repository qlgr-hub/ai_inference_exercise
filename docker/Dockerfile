# ref:https://github.com/NVIDIA/TensorRT/blob/release/10.0/docker/ubuntu-22.04.Dockerfile
ARG CUDA_VERSION=12.2.2
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04
# Required to build Ubuntu 20.04 without user prompts with DLFW container
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai


# Install cudnn
ENV NV_CUDNN_VERSION 8.9.6.50
ENV NV_CUDNN_PACKAGE_NAME "libcudnn8"
ENV CUDA_VERSION_MAJOR_MINOR=12.2
ENV NV_CUDNN_PACKAGE "libcudnn8=$NV_CUDNN_VERSION-1+cuda${CUDA_VERSION_MAJOR_MINOR}"
ENV NV_CUDNN_PACKAGE_DEV "libcudnn8-dev=$NV_CUDNN_VERSION-1+cuda${CUDA_VERSION_MAJOR_MINOR}"
ENV TRT_VERSION 10.0.1.6
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    ${NV_CUDNN_PACKAGE} \
    ${NV_CUDNN_PACKAGE_DEV} \
    && apt-mark hold ${NV_CUDNN_PACKAGE_NAME} \
    && rm -rf /var/lib/apt/lists/*


# Update CUDA signing key
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub


# Install requried libraries
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    wget \
    git \
    pkg-config \
    sudo \
    ssh \
    libssl-dev \
    pbzip2 \
    pv \
    bzip2 \
    unzip \
    devscripts \
    lintian \
    fakeroot \
    dh-make \
    build-essential \
    net-tools \
    openssh-server \
    aria2 \
    vim \
    default-jre \
    graphviz \
    nasm


# Add default non-root sudoer user and start ssh service
RUN groupadd -r -g 1000 qlgr && useradd -rm -d /home/qlgr -s /bin/bash -g qlgr -G sudo -u 1000 qlgr
RUN echo 'qlgr:123456' | chpasswd && echo "qlgr ALL=(ALL) ALL" >> /etc/sudoers
RUN echo "PermitUserEnvironment yes" >> /etc/ssh/sshd_config && service ssh start


# Install python3
RUN apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-wheel &&\
    cd /usr/local/bin &&\
    ln -s /usr/bin/python3 python && \
    ln -s /usr/bin/pip3 pip


ENV http_proxy=http://192.168.3.81:7890
ENV https_proxy=http://192.168.3.81:7890
USER root
WORKDIR /tmp


# Install TensorRT
RUN wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.0.1/tars/TensorRT-10.0.1.6.Linux.x86_64-gnu.cuda-12.4.tar.gz \
    && tar -xf TensorRT-10.0.1.6.Linux.x86_64-gnu.cuda-12.4.tar.gz \
    && cp -a TensorRT-10.0.1.6/lib/*.so* /usr/lib/x86_64-linux-gnu \
    && pip install TensorRT-10.0.1.6/python/tensorrt-10.0.1-cp310-none-linux_x86_64.whl


# Download NGC client
RUN cd /usr/local/bin && wget https://ngc.nvidia.com/downloads/ngccli_cat_linux.zip && unzip ngccli_cat_linux.zip && chmod u+x ngc-cli/ngc && rm ngccli_cat_linux.zip ngc-cli.md5 && echo "no-apikey\nascii\n" | ngc-cli/ngc config set


# Install cmake
RUN aria2c https://github.com/Kitware/CMake/releases/download/v3.30.0/cmake-3.30.0-linux-x86_64.sh && \
    bash cmake-3.30.0-linux-x86_64.sh --skip-license --prefix=/usr/local


# Install PyPI packages
RUN pip3 install --upgrade pip
RUN pip3 install setuptools>=41.0.0
ADD requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
RUN pip3 install jupyter jupyterlab
# Workaround to remove numpy installed with tensorflow
RUN pip3 install --upgrade numpy


# Set environment and working directory
ENV TRT_LIBPATH /usr/lib/x86_64-linux-gnu
ENV TRT_OSSPATH /workspace/opensource/TensorRT
ENV PATH="${PATH}:/usr/local/bin/ngc-cli"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${TRT_OSSPATH}/build/out:${TRT_LIBPATH}"


# Clean up
RUN apt autopurge && rm -rf /tmp/*


EXPOSE 22
EXPOSE 8888
WORKDIR /home/qlgr
ADD startup.sh startup.sh
ENTRYPOINT ["bash", "startup.sh"]
