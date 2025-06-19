FROM python:3.11
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git
RUN  apt-get update && apt-get install ffmpeg libsm6 libxext6 openssh-client git-lfs -y
RUN apt-get install -y wget
WORKDIR /workspace/service
COPY . .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir uv
RUN uv pip install -e . --system
RUN uv pip install numba
ENV PYTHONPATH=$PYTHONPATH:/workspace/service