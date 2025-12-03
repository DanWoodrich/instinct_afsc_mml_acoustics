#since it includes contrib dependences, put in contrib.
#however, should be built from the INSTINCT root. This dockerfile will assume relative paths to the INSTINCT root
#also, dockerfile will assume that build user has pulled in instinct_afsc_mml_acoustics into contrib

#how to build example:
#docker build -f Dockerfile -t instinct-afsc ../..

#other pre-build steps: define the following /etc/instinct.cfg:
#blah blah, make sure keypath is predictable to can provide

#at runtime: assume that on instance (like in batch), populate the cache locally if working off a previous step, and populate the local .nt file.

#ubuntu image with nvidia drivers for gcp server gpus. Look below to see specific libraries requested for GPU but ignore any that shouldn't affect reproducibility of tensorflow/keras inference
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

WORKDIR /app 

#COPY . . # move this later so can cache more effectively

ENV DEBIAN_FRONTEND=noninteractive

# --- Install R 4.1.2 ---
# Add the CRAN PPA (c2d4u) for Ubuntu 20.04 (focal)
# This key is specific to the R 4.x packages on Ubuntu 20.04
#RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
#RUN add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu focal-cran40/'

# --- Install Python 3.11 ---
# Add the 'deadsnakes' PPA for newer Python versions
#RUN add-apt-repository ppa:deadsnakes/ppa -y

# 2. SYSTEM DEPENDENCIES
# Install base utilities, wget, and build-essential.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    wget \
    ca-certificates \
    build-essential \
    software-properties-common \
    gnupg \
    dirmngr \
    unzip \
    xz-utils \
    locales \
    libxext6 \
    libxrender1 \
    libxtst6 \
    libxrandr2 \
    libxi6 \
    libglib2.0-0 \
    libfontconfig1 \
    libfreetype6 \
    libice6 \
    libsm6 \
    libxfixes3 \
    libxcursor1 \
    libxinerama1 \
    libglu1-mesa \
    libtiff5-dev \
    libpng-dev \
    libjpeg-turbo8-dev \
    libnss3 zlib1g && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa -y

# --- Run the installations ---
# We must run 'apt-get update' again after adding new repos.
# Note: PPAs give us python3.11, not 3.11.4 exactly, but this is the
# standard way to get 3.11 on this base image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    # This specific version string targets R 4.1.2 on the focal PPA
#    r-base=4.1.2-1.2004.0 \
#    r-base-dev=4.1.2-1.2004.0 \
#    r-recommended=4.1.2-1.2004.0 \
    r-base \
    r-base-dev \
    r-recommended \
    # Install pip for python 3.11
    && wget https://bootstrap.pypa.io/get-pip.py \
    && python3.11 get-pip.py \
    && rm get-pip.py \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

#install R

COPY ./lib/user/Installe.R ./lib/user/

RUN Rscript ./lib/user/Installe.R

# Set python3.11 as the default 'python3'
#RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2

#install instinct and instinct_afsc_mml_acoustics requirements.txt
COPY ./etc/requirements.txt ./etc/
COPY ./lib/user/requirements.txt ./lib/user/

RUN python3.11 -m pip install -r ./etc/requirements.txt
RUN python3.11 -m pip install -r ./lib/user/requirements.txt

#install matlab runtime R2025b, ensure ENV variables set to be able to run matlab executible later
#use Matlab MCR as 'donor' container

FROM containers.mathworks.com/matlab-runtime:r2025b AS mcr-donor

FROM base

COPY --from=mcr-donor /opt/matlabruntime /opt/matlabruntime

# auto-detect version subdir (v9xx or R20xxa/b)
RUN set -eux pipefail \
&& mkdir -p /tmp/mcr_cache && chmod 777 /tmp/mcr_cache

#hardcode instead of attempting to detect vers dir to avoid shell expansion issues
ENV LD_LIBRARY_PATH="/opt/matlabruntime/R2025b/runtime/glnxa64:/opt/matlabruntime/R2025b/bin/glnxa64:\
/opt/matlabruntime/R2025b/sys/os/glnxa64:${LD_LIBRARY_PATH}"

ENV XAPPLRESDIR="/opt/matlabruntime/R2025b/X11/app-defaults"
ENV MCR_CACHE_ROOT=/tmp/mcr_cache

#install conda
ENV CONDA_DIR=/opt/conda
ENV PATH=$PATH:$CONDA_DIR/bin

# 5. Download, install, and initialize Conda in one layer
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    # Run the installer in batch mode (-b) to a specific prefix (-p)
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    # Clean up the installer
    rm ~/miniconda.sh && \
    # Clean up all package caches to keep the image small
    conda clean -afy

# 6. *** THIS IS THE CRITICAL STEP ***
# Tell Docker to use bash as its default shell *from now on*.
# The "-l" flag makes it a "login" shell, which is required
# to source the .bashrc file where conda init put its scripts.
#SHELL ["/bin/bash", "-l", "-c"]

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

#init two conda environments: tf-gpu4 and mel-spec. see below for how they are to  be installed

#tf-gpu4 conda: init with whatever of these are essential to ensure reprocibility (leave floating to latest any that are not, and/or any that are covered by base image omit)
#python=3.10, cuda-nvcc, cudnn=8.1.0, cudatoolkit=11.2

COPY ./lib/user/tf_gpu4_requirements.txt ./lib/user/

RUN conda create -n tf-gpu4 python=3.10 -y && \
    conda clean -afy
SHELL ["conda", "run", "-n", "tf-gpu4", "/bin/bash", "-c"]
RUN pip install -r ./lib/user/tf_gpu4_requirements.txt
SHELL ["/bin/bash", "-c"]

COPY ./lib/user/mel-spec.yml ./lib/user/

#mel-spec conda: init with mel-spec.yml
RUN conda env create -f ./lib/user/mel-spec.yml && \
    conda clean -afy

#alleged matlab runtime deps. move back here to speed tshooting
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        libx11-6 libxext6 libxmu6 libxt6 \
        libxpm4 libxft2 libxtst6 \
        libglib2.0-0 libsm6 libice6 \
        libncurses5 libstdc++6 libgcc-s1 \
        libgfortran5 libffi8 libxml2 \
        libxrandr2 libxrender1 \
        libxfixes3 libxcursor1 libcups2 \
        libxi6 libxinerama1 libatk1.0-0 libatk-bridge2.0-0 \
        #python3 python3-pip \
        libdrm2 libxcomposite1 libxdamage1 libgbm1 \
        libasound2 libgl1 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

ENV AGREE_TO_MATLAB_RUNTIME_LICENSE=yes

COPY . . 

RUN chmod +x ./bin/instinct
RUN chmod +x ./lib/user/instinct-entrypoint.sh
RUN chmod +x ./lib/user/methods/FormatFG/matlabdecimate/matlabdecimateV1s1

ENTRYPOINT ["./lib/user/instinct-entrypoint.sh"]

CMD [""]

#define command at run time, job and parameters .nt gcs path. 