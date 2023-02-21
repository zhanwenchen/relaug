#/bin/bash

# For non-Slurm servers.

# Assumption: you have already cloned this repository

export ENV_NAME=relaug
export INSTALL_DIR=${HOME}
export DATASET_URL=https://sgg-zhanwen.s3.amazonaws.com/datasets.zip
export DATASETS_DIR=${HOME}/datasets
export PROJECT_DIR=${HOME}/relaug
export LOGDIR=${PROJECT_DIR}/log


# TODO: assert that checkpoints and log directories don't exist.


echo "Step 1: Installing dependencies (binaries)"
conda create --name ${ENV_NAME} python=3.8 ipython scipy h5py -y
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
cd ${INSTALL_DIR}
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install apex
# NOTE: you must have access to the target GPU for CUDA architecture detection.
cd ${INSTALL_DIR}
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout 0c8400aa04f4279b1384ae0633e73d6daf9fead7
sed '171' setup.py
# WARNING if you use older Versions of Pytorch (anything below 1.7), you will need a hard reset,
# as the newer version of apex does require newer pytorch versions. Ignore the hard reset otherwise.

python setup.py install --cuda_ext --cpp_ext


# install project code
cd ${PROJECT_DIR}

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop

unset INSTALL_DIR


echo "Step 2: Downloading Data"
mkdir -p ${DATASETS_DIR}
cd ${DATASETS_DIR}

wget ${DATASET_URL}
unzip datasets.zip


echo "Step 3: Test Training"

cd ${PROJECT_DIR}
mkdir ${PROJECT_DIR}/checkpoints
ln -s ${DATASETS_DIR}/pretrained_faster_rcnn ${PROJECT_DIR}/checkpoints
mkdir ${LOGDIR}

source activate ${ENV_NAME}
${PROJECT_DIR}/scripts/install_test.sh
