#/bin/bash

# For non-Slurm servers.

# Assumption: you have already cloned this repository
export ENV_NAME=relaug
export INSTALL_DIR=/scratch/pct4et
export DATASETS_DIR=/scratch/pct4et/datasets
export PROJECT_DIR=/scratch/pct4et/relaug

# Download pycocotools
cd ${INSTALL_DIR}
git clone https://github.com/cocodataset/cocoapi.git

# Download apex
git clone git@github.com:zhanwenchen/apex.git && cd apex

# TODO: assert that checkpoints and log directories don't exist.

singularity shell --nv docker://pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

export ENV_NAME=relaug
export INSTALL_DIR=/scratch/pct4et
export DATASETS_DIR=/scratch/pct4et/datasets
export PROJECT_DIR=/scratch/pct4et/relaug
export ENV_NAME=relaug
export DATASET_URL=https://sgg-zhanwen.s3.amazonaws.com/datasets.zip
export DATASETS_DIR=/project/sds-rise/zhanwen/datasets
export PROJECT_DIR=/scratch/pct4et/relaug
export INSTALL_DIR=/scratch/pct4et
export ENVS_DIR=/scratch/pct4et/envs

mkdir -p ${ENVS_DIR}

echo "Step 1: Installing dependencies (binaries)"
conda create -p ${ENVS_DIR}/${ENV_NAME} python=3.8 ipython scipy h5py pandas -y
conda config --add pkgs_dirs ${ENVS_DIR}
# conda config --set env_prompt '({name})' # Unnecessary
source activate ${ENV_NAME}

# quantization depends on pytorch=1.10 and above
# torchvision: https://github.com/pytorch/vision/releases/tag/v0.11.3
# torchaudio: https://github.com/pytorch/audio/releases/tag/v0.10.2
# The maximum cudatoolkit for 1.10.2 is 11.3
# 1.11.0 changes C++ API so DeviceUtils will be removed and csrc will need an update
conda install pytorch==1.10.2 torchvision==0.11.3 torchaudio==0.10.2 cudatoolkit=11.3 -c pytorch -y


# scene_graph_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib tqdm opencv-python-headless overrides tensorboard setuptools==59.5.0


# install pycocotools
cd ${INSTALL_DIR}/cocoapi/PythonAPI
python setup.py build_ext install
cd ${INSTALL_DIR}
rm -rf cocoapi

# install apex
# NOTE: you must have access to the target GPU for CUDA architecture detection (hence ijob).
cd ${INSTALL_DIR}/apex
exit # from singularity
ijob -A sds-rise -p gpu --gres=gpu:a100:1 -c 16 --mem=32000

singularity shell --nv docker://pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

export INSTALL_DIR=/scratch/pct4et
export ENV_NAME=relaug
export PROJECT_DIR=/scratch/pct4et/relaug

source activate ${ENV_NAME}
python setup.py install --cuda_ext --cpp_ext
cd ${INSTALL_DIR}
rm -rf apex

# install project code
# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
cd ${PROJECT_DIR}
python setup.py build develop
exit


# echo "Step 2: Downloading Data"
# parentdir="$(dirname "${DATASETS_DIR}")"
# cd ${parentdir}
# wget ${DATASET_URL}
# unzip datasets.zip


echo "Step 3: Test Training"

cd ${PROJECT_DIR}
ln -s ${DATASETS_DIR}/pretrained_faster_rcnn ${PROJECT_DIR}/checkpoints
