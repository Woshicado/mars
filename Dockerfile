#
# based on
# https://github.com/benjamin-heasly/mitsuba-docker/blob/master/rgb/Dockerfile
#

# builder
FROM ubuntu:20.04 AS builder
LABEL authors="Alexander Rath <rath@cg.uni-saarland.de>, Joshua Meyer <joshua.meyer@cs.uni-saarland.de>"

ENV TZ=Europe
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && \
    apt-get install -y \
        build-essential \
        scons \
        python3 \
        python3-pip \
        libgl1-mesa-dev \
        libglu1-mesa-dev \
        libxxf86vm-dev \
        libpng-dev \
        libjpeg-dev \
        libilmbase-dev \
        libxerces-c-dev \
        libboost-all-dev \
        libopenexr-dev \
        libglewmx-dev \
        libpcrecpp0v5 \
        libeigen3-dev \
        libfftw3-dev \
        wget \
        unzip && \
    apt-get clean && \
    apt-get autoclean && \
    apt-get autoremove

RUN wget https://github.com/OpenImageDenoise/oidn/releases/download/v1.4.1/oidn-1.4.1.x86_64.linux.tar.gz && \
    tar -xvzf oidn* --strip-components=1 -C /usr/local/ && \
    rm oidn*

# We need a newer scons version
RUN /usr/bin/env pip3 --no-cache-dir install scons

WORKDIR /mitsuba
COPY mitsuba .

COPY scenes ./scenes
WORKDIR /mitsuba/scenes
RUN /bin/sh get_scenes.sh
WORKDIR /mitsuba

RUN cp build/config-linux-gcc.py config.py && \
    /usr/bin/env python3 $(which scons) -j $(nproc)

# mitsuba
FROM ubuntu:20.04 AS mitsuba
LABEL authors="Alexander Rath <rath@cg.uni-saarland.de>, Joshua Meyer <joshua.meyer@cs.uni-saarland.de>"

ENV TZ=Europe
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && \
    apt-get install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y \
        libgl1-mesa-dev \
        libglu1-mesa-dev \
        libxxf86vm-dev \
        libpng-dev \
        libjpeg-dev \
        libilmbase-dev \
        libxerces-c-dev \
        libboost-all-dev \
        libopenexr-dev \
        libglewmx-dev \
        libpcrecpp0v5 \
        libeigen3-dev \
        python3-pip \
        python3.10 \
        python3.10-distutils \
        curl \
        libfftw3-dev && \
    apt-get clean && \
    apt-get autoclean && \
    apt-get autoremove

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    python3.10 -m pip --no-cache-dir install --upgrade \
    pip \
    setuptools \
    wheel

RUN python3.10 -m pip --no-cache-dir install \
    argcomplete==3.4.0 \
    PyYAML==6.0.1

WORKDIR /mitsuba
ENV MITSUBA_DIR=/mitsuba
ENV PYTHONPATH=/mitsuba/dist/python:/mitsuba/dist/python/3.10:
ENV PATH=/mitsuba/wrapper:/mitsuba/dist:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/mitsuba/dist:/usr/local/lib:

COPY --from=builder /usr/local /usr/local
COPY --from=builder /mitsuba .

COPY config.yaml .
COPY render.py .

ENTRYPOINT [ "python3.10", "render.py", "--scene-dir=/scenes" ]
