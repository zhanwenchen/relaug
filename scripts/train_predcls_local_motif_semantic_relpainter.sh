#!/bin/bash

export GRAFT_ALPHA=0.5
export LR=1e-2
export MAX_BATCHSIZE_AUG=24

SLURM_JOB_NAME="motif_none_visual_predcls_1GPU_mini_${LR}_semantic_inpainter"
export CUDA_VISIBLE_DEVICES=1
export NUM_GPUS=$(echo ${CUDA_VISIBLE_DEVICES} | tr -cd , | wc -c); ((NUM_GPUS++))
export OMP_NUM_THREADS=$(($(nproc) / ${NUM_GPUS}))

export PROJECT_DIR=${HOME}/relaug
source ${PROJECT_DIR}/scripts/shared_functions/utils.sh
SLURM_JOB_ID=$(timestamp)
export MODEL_NAME="${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
export LOGDIR=${PROJECT_DIR}/log
MODEL_DIRNAME=${PROJECT_DIR}/checkpoints/${MODEL_NAME}/
export DATASETS_DIR=${HOME}/datasets

if [ -d "$MODEL_DIRNAME" ]; then
  error_exit "Aborted: ${MODEL_DIRNAME} exists." 2>&1 | tee -a ${LOGDIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log
else
  # Experiment variables
  export PREDICTOR=MotifPredictor
  export CONFIG_FILE=configs/e2e_relation_X_101_32_8_FPN_1x_motif.yaml
  export USE_GRAFT=False
  export USE_SEMANTIC=True
  export USE_RELPAINTER=True
  export STRATEGY='cooccurrence-pred_cov'
  export BOTTOM_K=30
  export NUM2AUG=4
  if [ "${USE_SEMANTIC}" = True ]; then
      export BATCH_SIZE_PER_GPU=$((${MAX_BATCHSIZE_AUG} / 2))
  else
      export BATCH_SIZE_PER_GPU=${MAX_BATCHSIZE_AUG}
  fi

  # Experiment class variables
  export WITH_CLEAN_CLASSIFIER=False
  export WITH_TRANSFER_CLASSIFIER=False
  export USE_GT_BOX=True
  export USE_GT_OBJECT_LABEL=True
  export PRE_VAL=False

  # Experiment hyperparams
  export MAX_ITER=50000
  export SEED=1234

  # Paths and configss
  export WEIGHT="''"
  export ALL_EDGES_FPATH=${DATASETS_DIR}/visual_genome/gbnet/all_edges.pkl

  # System variables
  export BATCH_SIZE=$((${NUM_GPUS} * ${BATCH_SIZE_PER_GPU}))
  export PORT=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

  ${PROJECT_DIR}/scripts/train.sh
fi