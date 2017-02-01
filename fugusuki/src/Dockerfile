FROM nvidia/cuda:8.0-cudnn5-devel

# from continuumio/anaconda3 ###########################################
# https://hub.docker.com/r/continuumio/anaconda3/
#
RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/archive/Anaconda3-4.0.0-Linux-x86_64.sh && \
    /bin/bash /Anaconda3-4.0.0-Linux-x86_64.sh -b -p /opt/conda && \
    rm /Anaconda3-4.0.0-Linux-x86_64.sh

RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean

ENV PATH /opt/conda/bin:$PATH

RUN conda install pip
RUN conda install nomkl numpy scipy scikit-learn numexpr

# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
ENV LANG C.UTF-8

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]

#######################################################################
#######################################################################

RUN apt-get update && \
apt-get install -y libopenblas-dev libopenblas-base && \
apt-get install -y git

# from kaixhin/cuda-theano ############################################
# https://hub.docker.com/r/kaixhin/cuda-theano/
#
#RUN apt-get update && apt-get install -y \ 
# git
# libopenblas-dev \
# python-dev \
# python-pip \
# python-nose \
# python-numpy \
# python-scipy

# Set CUDA_ROOT
ENV CUDA_ROOT /usr/local/cuda/bin 
# Install bleeding-edge Theano 
RUN pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git 

###########################################################################
###########################################################################

RUN ln -s /usr/local/nvidia/lib64/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so

RUN pip install keras
ENV KERAS_BACKEND theano
RUN python -c "import keras"

# theano
# Set up .theanorc for CUDA 
RUN echo "[global]\ndevice=gpu\nfloatX=float32\noptimizer_including=cudnn\n[lib]\ncnmem=0.94\n[nvcc]\nfastmath=True" > /root/.theanorc

########################################################################################

RUN conda install -y shapely gdal
RUN pip install xgboost
RUN echo "umask 0000" >> /root/.bashrc

COPY program.py /root
COPY model /root/model

WORKDIR /root