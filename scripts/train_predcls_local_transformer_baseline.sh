#!/bin/bash

SLURM_JOB_NAME=transformer_none_none_predcls_2GPU_lab1_0.01

export PROJECT_DIR=/localtmp/pct4et/relaug
source ${PROJECT_DIR}/scripts/shared_functions/utils.sh
SLURM_JOB_ID=$(timestamp)
export MODEL_NAME="${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
export LOGDIR=${PROJECT_DIR}/log
MODEL_DIRNAME=${PROJECT_DIR}/checkpoints/${MODEL_NAME}/

if [ -d "$MODEL_DIRNAME" ]; then
  error_exit "Aborted: ${MODEL_DIRNAME} exists." 2>&1 | tee -a ${LOGDIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log
else
  # Experiment variables
  export PREDICTOR=TransformerPredictor
  export USE_GRAFT=False
  export USE_SEMANTIC=False
  export CONFIG_FILE=configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml
  export STRATEGY="''"
  export BOTTOM_K=-1
  export NUM2AUG=-1
  export MAX_BATCHSIZE_AUG=4
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
  export MAX_ITER=32000
  export LR=0.01
  export SEED=1234

  # Paths and configss
  export WEIGHT="''"
  export DATASETS_DIR=/localtmp/pct4et/datasets
  export ALL_EDGES_FPATH=${DATASETS_DIR}/visual_genome/gbnet/all_edges.pkl

  # System variables
  export CUDA_VISIBLE_DEVICES=1,2
  export NUM_GPUS=$(echo ${CUDA_VISIBLE_DEVICES} | tr -cd , | wc -c); ((NUM_GPUS++))
  export BATCH_SIZE=$((${NUM_GPUS} * ${BATCH_SIZE_PER_GPU}))
  export PORT=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

  ${PROJECT_DIR}/scripts/train.sh
fi
