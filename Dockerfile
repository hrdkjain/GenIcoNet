from pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel

RUN apt-get update && apt-get install -y --no-install-recommends python3.8

RUN rm -f /usr/bin/python3 && ln -s /usr/bin/python3.8 /usr/bin/python3
RUN rm -f /opt/conda/bin/python3 && ln -s /usr/bin/python3.8 /opt/conda/bin/python3

RUN apt-get update && apt-get install -y --no-install-recommends \
python3-pip \
python3-setuptools \
python3.8-dev \
python3-wheel \
emacs \
git \
graphviz \
nano \
wget

RUN apt-get install -y --no-install-recommends \
libfreetype6-dev \
libzmq3-dev

RUN python3 -m pip install \
Cython==0.29.24 \
dash==1.13.4 \
kaleido==0.0.3.post1 \
matplotlib==3.4.3 \
natsort==7.0.1 \
numpy==1.21.3 \
packaging \
pillow==8.4.0 \
plotly==4.10.0 \
psutil==5.7.2 \
plyfile==0.7.2 \
requests==2.23.0 \
scikit-image==0.16.2 \
scikit-learn==0.23.2 \
scipy==1.4.1 \
setuptools==46.1.3 \
tensorboard==2.0.0 \
torch==1.7.0 \
torchvision==0.8.0 \
torchviz==0.0.1 \
tqdm==4.45.0 \
visdom==0.1.8.9 \
wheel==0.26.0

RUN python3 -m pip install pandas==1.3.4

RUN git clone https://github.com/NVIDIAGameWorks/kaolin.git /kaolin
WORKDIR /kaolin
RUN git checkout v0.9.1
ENV KAOLIN_HOME "/kaolin"
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5"
RUN python3 setup.py develop

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get install -y --no-install-recommends python3.8-tk
ENV MPLBACKEND "Agg"

RUN apt-get purge -y --no-install-recommends cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.17.2/cmake-3.17.2-Linux-x86_64.sh \
      -q -O /tmp/cmake-install.sh \
      && chmod u+x /tmp/cmake-install.sh \
      && mkdir /usr/bin/cmake \
      && /tmp/cmake-install.sh --skip-license --prefix=/usr/bin/cmake \
      && rm /tmp/cmake-install.sh
ENV PATH="/usr/bin/cmake/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libeigen3-dev \
    libgmp-dev \
    libgmpxx4ldbl \
    libmpfr-dev \
    libboost-dev \
    libboost-thread-dev \
    libtbb-dev \
    python3-dev
RUN git clone https://github.com/PyMesh/PyMesh.git /PyMesh
WORKDIR /PyMesh
RUN git checkout 384ba882
RUN pip install -r python/requirements.txt
RUN git submodule update --init
RUN python3 ./setup.py build
RUN python3 ./setup.py install

ENV DASH_DEBUG_MODE True

# ugly hack for python cache
RUN mkdir /.cache && chmod 777 /.cache

ARG USER
ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID $USER
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER
