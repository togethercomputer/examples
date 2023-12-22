FROM --platform=linux/amd64 nvidia/cuda:12.1.1-devel-ubuntu20.04

USER root

# Install system stuff as root
RUN apt update && \
    apt install --no-install-recommends -y git git-lfs python3-pip gcc g++ python3-dev && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir /app && \
    mkdir /config && \
    mkdir /work

WORKDIR /app

RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

ENV HF_DATASETS_CACHE=/data/datasets
ENV HF_HOME=/data/model
ENV TRANSFORMERS_CACHE=/data/model
ENV TOKENIZERS_PARALLELISM=false

EXPOSE 29500

# Setup the entrypoint script
#ENTRYPOINT ["/app/entrypoint.sh"]
