#!/bin/bash

#SBATCH -A sds-rise
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1 # need to match number of gpus
#SBATCH -t 4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB # need to match batch size.
#SBATCH -J test # TODO: CHANGE THIS
#SBATCH -o /scratch/pct4et/relaug/log/%x-%A.out
#SBATCH -e /scratch/pct4et/relaug/log/%x-%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pct4et@virginia.edu

timestamp() {
  date +"%Y-%m-%d%H%M%S"
}

error_exit()
{
#   ----------------------------------------------------------------
#   Function for exit due to fatal program error
#       Accepts 1 argument:
#           string containing descriptive error message
#   Source: http://linuxcommand.org/lc3_wss0140.php
#   ----------------------------------------------------------------
    echo "$(timestamp) ERROR ${PROGNAME}: ${1:-"Unknown Error"}" 1>&2
    echo "$(timestamp) ERROR ${PROGNAME}: Exiting Early."
    exit 1
}


export MODEL_NAME="47611844_transformer_none_semantic_4GPU_riv_1_predcls/"
export ITERATION=0012000
export PROJECT_DIR=/scratch/pct4et/relaug
export DATASETS_DIR=/scratch/pct4et/datasets
export MODEL_DIRPATH=${PROJECT_DIR}/checkpoints/${MODEL_NAME}/
export LOGDIR=${PROJECT_DIR}/log
export ALL_EDGES_FPATH=${DATASETS_DIR}/visual_genome/gbnet/all_edges.pkl
export CONDA_ENVS_DIR=/scratch/pct4et/envs
export CONDA_ENV_NAME=relaug
export SINGULARITYENV_PREPEND_PATH="${CONDA_ENVS_DIR}/${CONDA_ENV_NAME}/bin:/opt/conda/condabin"


if [ -d "${MODEL_DIRPATH}" ]; then
  singularity exec --nv --env LD_LIBRARY_PATH="\$LD_LIBRARY_PATH:${CONDA_ENVS_DIR}/${CONDA_ENV_NAME}/lib" docker://pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel ${PROJECT_DIR}/scripts/test.sh
else
  error_exit "Aborted: ${MODEL_DIRPATH} does not exist." 2>&1 | tee -a ${LOGDIR}/${MODEL_NAME}_${ITERATION}.log
fi
