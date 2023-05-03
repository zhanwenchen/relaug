#!/bin/bash

# TODO: change this block. ITERATION needs 7 digits
MODEL_NAME_BASE=2023-03-26182152_transformer_none_semantic_visual_predcls_4GPU_labx_1e-4
ITERATION=0010000
USE_CONFIG_AUGS=True
BASE_LR=1e-3
export CUDA_VISIBLE_DEVICES=8,9

export MODEL_NAME="${MODEL_NAME_BASE}_${ITERATION}_bpl_sa"
export PROJECT_DIR=/localtmp/pct4et/relaug
export PRETRAINED_MODEL_CKPT=${PROJECT_DIR}/checkpoints/${MODEL_NAME_BASE}/model_${ITERATION}.pth
source ${PROJECT_DIR}/scripts/shared_functions/utils.sh
export LOGDIR=${PROJECT_DIR}/log
MODEL_DIRNAME=${PROJECT_DIR}/checkpoints/${MODEL_NAME}/
MODEL_DIRNAME_BASE=${PROJECT_DIR}/checkpoints/${MODEL_NAME_BASE}/
export SEED=1234

if [ -d "${MODEL_DIRNAME_BASE}" ]; then
  if [ -d "${MODEL_DIRNAME}" ]; then
    error_exit "Aborted: ${MODEL_DIRNAME} exists." 2>&1 | tee -a ${LOGDIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log
  else
    # Experiment variables
    export NUM_GPUS=$(echo ${CUDA_VISIBLE_DEVICES} | tr -cd , | wc -c); ((NUM_GPUS++))
    export DATASETS_DIR=/localtmp/pct4et/datasets
    export ALL_EDGES_FPATH=${DATASETS_DIR}/visual_genome/gbnet/all_edges.pkl
    export PORT=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
    export TORCH_DISTRIBUTED_DEBUG=INFO
    export TORCHELASTIC_MAX_RESTARTS=0
    cd ${PROJECT_DIR}
    mkdir ${MODEL_DIRNAME} &&
    cp -r ${PROJECT_DIR}/.git/ ${MODEL_DIRNAME} &&
    cp -r ${PROJECT_DIR}/tools/ ${MODEL_DIRNAME} &&
    cp -r ${PROJECT_DIR}/scripts/ ${MODEL_DIRNAME} &&
    cp -r ${PROJECT_DIR}/maskrcnn_benchmark/ ${MODEL_DIRNAME} ||
    echo "Failed to train BPL+SA model ${MODEL_NAME} at copying folders."

    echo "TRAINING BPL+SA model ${MODEL_NAME}"
    if [ "${USE_CONFIG_AUGS}" = True ]; then
      torchrun --master_port ${PORT} --nproc_per_node=${NUM_GPUS} \
              ${PROJECT_DIR}/tools/relation_train_net.py \
              --config-file "${MODEL_DIRNAME_BASE}/config.yml" \
              MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER True \
              MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER True  \
              SOLVER.PRE_VAL True \
              SOLVER.BASE_LR ${BASE_LR} \
              TEST.IMS_PER_BATCH ${NUM_GPUS} \
              MODEL.PRETRAINED_MODEL_CKPT ${PRETRAINED_MODEL_CKPT} \
              OUTPUT_DIR ./checkpoints/${MODEL_NAME} \
      2>&1 | tee ${MODEL_DIRNAME}/log_train.log &&
      echo "Finished training BPL+SA model ${MODEL_NAME}" ||
      echo "Failed to train BPL+SA model ${MODEL_NAME}"
    else
      torchrun --master_port ${PORT} --nproc_per_node=${NUM_GPUS} \
              ${PROJECT_DIR}/tools/relation_train_net.py \
              --config-file "${MODEL_DIRNAME_BASE}/config.yml" \
              SOLVER.AUGMENTATION.USE_GRAFT False \
              SOLVER.AUGMENTATION.USE_SEMANTIC False \
              MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER True \
              MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER True  \
              SOLVER.PRE_VAL True \
              SOLVER.BASE_LR ${BASE_LR} \
              TEST.IMS_PER_BATCH ${NUM_GPUS} \
              MODEL.PRETRAINED_MODEL_CKPT ${PRETRAINED_MODEL_CKPT} \
              OUTPUT_DIR ${MODEL_DIRNAME} \
      2>&1 | tee ${MODEL_DIRNAME}/log_train.log &&
      echo "Finished training BPL+SA model" ||
      echo "Failed to train BPL+SA model ${MODEL_NAME}"
    fi
  fi
else
  error_exit "Aborted: ${MODEL_DIRNAME_BASE} does not exist." 2>&1 | tee -a ${LOGDIR}/${MODEL_NAME_BASE}_${ITERATION}.log
fi
