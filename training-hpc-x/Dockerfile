FROM nvcr.io/nvidia/pytorch:23.10-py3
# For contents, see https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-10.html

COPY . /app
WORKDIR /app

# Add some CUDA paths
ENV CUDA_HOME="/usr/local/cuda-12"

ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64/stubs:${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 

# Install our PyPi requirements, including flash attention
RUN pip install -r requirements.txt

