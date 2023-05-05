#!/bin/bash

export PROJECT_DIR=/localtmp/pct4et/relaug
export MODEL_NAME="2023-04-10011025_transformer_none_none_predcls_2GPU_lab1_0.016_0016000_bpl_sa_semantic_visual"
ITERATION=00012000
export SEED=1234
MODEL_DIRPATH=${PROJECT_DIR}/checkpoints/${MODEL_NAME}/
export LOGDIR=${PROJECT_DIR}/log
if [ -d "$MODEL_DIRPATH" ]; then
  export CUDA_VISIBLE_DEVICES=0
  export NUM_GPUS=$(echo ${CUDA_VISIBLE_DEVICES} | tr -cd , | wc -c); ((NUM_GPUS++))
  export DATASETS_DIR=/localtmp/pct4et/datasets
  export ALL_EDGES_FPATH=${DATASETS_DIR}/visual_genome/gbnet/all_edges.pkl
  export PORT=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

  echo "Started testing model ${MODEL_NAME} at iteration ${ITERATION}"
  torchrun --master_port ${PORT} --nproc_per_node=${NUM_GPUS} \
          ${PROJECT_DIR}/tools/relation_test_net.py \
          --config-file "${MODEL_DIRPATH}/config.yml" \
          TEST.IMS_PER_BATCH ${NUM_GPUS} \
          DTYPE "float32" \
          GLOVE_DIR ${DATASETS_DIR}/glove \
          MODEL.PRETRAINED_DETECTOR_CKPT ${PROJECT_DIR}/checkpoints/pretrained_faster_rcnn/model_final.pth \
          MODEL.PRETRAINED_MODEL_CKPT ${MODEL_DIRPATH}/model_${ITERATION}.pth \
          MODEL.WEIGHT ${MODEL_DIRPATH}/model_${ITERATION}.pth \
          OUTPUT_DIR ./checkpoints/${MODEL_NAME} \
          TEST.ALLOW_LOAD_FROM_CACHE False \
  2>&1 | tee ${MODEL_DIRPATH}/log_test_${ITERATION}.log &&
  echo "Finished testing model ${MODEL_NAME} at iteration ${ITERATION}" ||
  echo "Failed to test model ${MODEL_NAME} at iteration ${ITERATION}"
else
  error_exit "Aborted: ${MODEL_DIRPATH} does not exist." 2>&1 | tee -a ${LOGDIR}/${MODEL_NAME}_${ITERATION}.log
fi
