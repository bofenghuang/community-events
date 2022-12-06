#!/usr/bin/env bash
# Copyright 2022 Bofeng Huang

# check NVIDIA driver
nvidia-smi

# update the Unix package ffmpeg to version 4
sudo add-apt-repository -y ppa:jonathonf/ffmpeg-4
sudo apt update
sudo apt install -y ffmpeg

# nvtop
# sudo apt update
# sudo apt install nvtop

# install git-lfs
# git-lfs -v
sudo apt-get install git-lfs

# setup python env
env_name=venv
python3 -m venv $env_name
echo "source ~/$env_name/bin/activate" >> ~/.bashrc

# install cuda by system side
cudapath="/home/ubuntu/cuda-11.7.0"
# install
wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
chmod +x cuda_11.7.0_515.43.04_linux.run 
./cuda_11.7.0_515.43.04_linux.run --silent --toolkit --installpath=$cudapath --no-opengl-libs --no-drm --no-man-page
# set env var
export CUDA_HOME=$cudapath
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
# msc
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export CUDA_TOOLKIT_ROOT=$CUDA_HOME
export CUDA_BIN_PATH=$CUDA_HOME
export CUDA_PATH=$CUDA_HOME
export CUDA_INC_PATH=$CUDA_HOME/targets/x86_64-linux
export CFLAGS=-I$CUDA_HOME/targets/x86_64-linux/include
# test
nvcc -V
# cudnn
# wget https://huggingface.co/csukuangfj/cudnn/resolve/main/cudnn-10.2-linux-x64-v8.0.2.39.tgz
# tar xvf cudnn-10.2-linux-x64-v8.0.2.39.tgz --strip-components=1 -C /ceph-data4/fangjun/software/cuda-10.2.89

# anaconda
# a little bir heavy
# wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
# bash Anaconda3-2022.10-Linux-x86_64.sh
# # create env
# conda create -y -n asr python=3.8
# conda activate asr
# # install cudatoolkit by conda side
# conda install -y cuda -c nvidia/label/cuda-11.7.0

# requirements
# git clone https://github.com/huggingface/community-events.git
pip install -r community-events/whisper-fine-tuning-event/requirements.txt
# test
python -c "import torch; print(torch.cuda.is_available())"

# need to install cudatoolkit by system side
pip install bitsandbytes

# git account
git config user.name bofenghuang
git config user.email bofenghuang7@gmail.com

# link hf account
git config --global credential.helper store
# https://huggingface.co/settings/tokens
huggingface-cli login

# wandb
# https://wandb.ai/authorize
wandb login

# create hf repo
# huggingface-cli repo create whisper-small-es
# git lfs install
# git clone https://huggingface.co/sanchit-gandhi/whisper-small-es

# cd community-events/whisper-fine-tuning-event
# 


# watch -n0.1 nvidia-smi