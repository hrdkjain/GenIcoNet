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

RUN apt-get install -y --no-install-recommends libfreetype6-dev

RUN python3 -m pip install \
matplotlib==3.4.3 \
natsort==7.0.1 \
numpy==1.21.3 \
pillow==8.4.0 \
plotly==4.6.0 \
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
wheel==0.26.0

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

# ugly hack for python cache
RUN mkdir /.cache && chmod 777 /.cache

ARG USER
ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID $USER
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER
