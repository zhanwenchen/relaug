#!/bin/bash


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


export MODEL_NAME="47611844_transformer_none_semantic_4GPU_riv_1_predcls"
export ITERATION=0012000
export CUDA_VISIBLE_DEVICES=0
export PROJECT_DIR=/home/zhanwen/relaug
export DATASETS_DIR=/home/zhanwen/datasets
export MODEL_DIRPATH=${PROJECT_DIR}/checkpoints/${MODEL_NAME}/
export LOGDIR=${PROJECT_DIR}/log
export ALL_EDGES_FPATH=${DATASETS_DIR}/visual_genome/gbnet/all_edges.pkl


if [ -d "${MODEL_DIRPATH}" ]; then
  ${PROJECT_DIR}/scripts/test.sh
else
  error_exit "Aborted: ${MODEL_DIRPATH} does not exist." 2>&1 | tee -a ${LOGDIR}/${MODEL_NAME}_${ITERATION}.log
fi
