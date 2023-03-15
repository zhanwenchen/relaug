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


export MODEL_NAME="20230228070836_vctree_none_semantic_sggen_4GPU_labx_1e3"
export ITERATION=0008000
export CUDA_VISIBLE_DEVICES=0
export PROJECT_DIR=/localtmp/pct4et/relaug
export DATASETS_DIR=/localtmp/pct4et/datasets
export LOGDIR=${PROJECT_DIR}/log
export ALL_EDGES_FPATH=${DATASETS_DIR}/visual_genome/gbnet/all_edges.pkl


if [ -d "${MODEL_DIRPATH}" ]; then
  ${PROJECT_DIR}/scripts/train.sh
else
  error_exit "Aborted: ${MODEL_DIRPATH} does not exist." 2>&1 | tee -a ${LOGDIR}/${MODEL_NAME}_${ITERATION}.log
fi
