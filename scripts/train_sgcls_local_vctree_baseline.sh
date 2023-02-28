#!/bin/bash

timestamp() {
  date +"%Y%m%d%H%M%S"
}

SLURM_JOB_NAME=vctree_none_none_sgcls_4GPU_lab2_1e3
SLURM_JOB_ID=$(timestamp)

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

error_check()
{
#   ----------------------------------------------------------------
#   This function simply checks a passed return code and if it is
#   non-zero it returns 1 (error) to the calling script.  This was
#   really only created because I needed a method to store STDERR in
#   other scripts and also check $? but also leave open the ability
#   to add in other stuff, too.
#
#   Accepts 1 arguments:
#       return code from prior command, usually $?
#  ----------------------------------------------------------------
    TO_CHECK=${1:-0}

    if [ "$TO_CHECK" != '0' ]; then
        return 1
    fi

}

export PROJECT_DIR=${HOME}/relaug
export MODEL_NAME="${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
export LOGDIR=${PROJECT_DIR}/log
MODEL_DIRNAME=${PROJECT_DIR}/checkpoints/${MODEL_NAME}/

if [ -d "$MODEL_DIRNAME" ]; then
  error_exit "Aborted: ${MODEL_DIRNAME} exists." 2>&1 | tee -a ${LOGDIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log
else
  # Experiment variables
  export PREDICTOR=VCTreePredictor
  export USE_GRAFT=False
  export USE_SEMANTIC=False
  export CONFIG_FILE=configs/e2e_relation_X_101_32_8_FPN_1x_vctree.yaml

  # Experiment class variables
  export USE_GT_BOX=True
  export USE_GT_OBJECT_LABEL=False
  export PRE_VAL=False

  # Experiment hyperparams
  export BATCH_SIZE=128
  export MAX_ITER=50000
  export LR=1e-3
  export SEED=1234

  # Paths and configss
  export WEIGHT="''"
  export ALL_EDGES_FPATH=/home/zhanwen/gbnet/graphs/005/all_edges.pkl
  export DATASETS_DIR=${HOME}/datasets

  # System variables
  export CUDA_VISIBLE_DEVICES=1,2,3,4
  export NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c); ((NUM_GPUS++))
  export PORT=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

  ${PROJECT_DIR}/scripts/train.sh
fi
