#!/bin/bash

#SBATCH -A sds-rise
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH -C a100_80gb
#SBATCH --ntasks-per-node=4 # need to match number of gpus
#SBATCH -t 48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB # need to match batch size.
#SBATCH -J imp_none_semantic_4GPU_riv_1_sggen # TODO: CHANGE THIS
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

export PROJECT_DIR=/scratch/pct4et/relaug
export MODEL_NAME="${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
export LOGDIR=${PROJECT_DIR}/log
MODEL_DIRNAME=${PROJECT_DIR}/checkpoints/${MODEL_NAME}/

if [ -d "$MODEL_DIRNAME" ]; then
  error_exit "Aborted: ${MODEL_DIRNAME} exists." 2>&1 | tee -a ${LOGDIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log
else
  module purge
  module load singularity

  # Experiment variables
  export PREDICTOR=IMPPredictor
  export USE_GRAFT=False
  export USE_SEMANTIC=True
  export CONFIG_FILE=configs/e2e_relation_X_101_32_8_FPN_1x_imp.yaml
  export STRATEGY='cooccurrence-pred_cov'
  export BOTTOM_K=30
  export NUM2AUG=4
  export MAX_BATCHSIZE_AUG=64
  if [ "${USE_SEMANTIC}" = True ]; then
      export BATCH_SIZE_PER_GPU=$((${MAX_BATCHSIZE_AUG} / 2))
  else
      export BATCH_SIZE_PER_GPU=${MAX_BATCHSIZE_AUG}
  fi

  # Experiment class variables
  export USE_GT_BOX=False
  export USE_GT_OBJECT_LABEL=False
  export PRE_VAL=False

  # Experiment hyperparams
  export MAX_ITER=50000
  export LR=1e-3
  export SEED=1234

  # Paths and configss
  export WEIGHT="''"
  export DATASETS_DIR=/scratch/pct4et/datasets
  export ALL_EDGES_FPATH=${DATASETS_DIR}/visual_genome/gbnet/all_edges.pkl

  # System variables
  export CONDA_ENVS_DIR=/scratch/pct4et/envs
  export CONDA_ENV_NAME=relaug
  export SINGULARITYENV_PREPEND_PATH="${CONDA_ENVS_DIR}/${CONDA_ENV_NAME}/bin:/opt/conda/condabin"
  export NUM_GPUS=$(echo ${CUDA_VISIBLE_DEVICES} | tr -cd , | wc -c); ((NUM_GPUS++))
  export BATCH_SIZE=$((${NUM_GPUS} * ${BATCH_SIZE_PER_GPU}))
  export PORT=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

  # Slurm Variables
  singularity exec --nv --env LD_LIBRARY_PATH="\$LD_LIBRARY_PATH:${CONDA_ENVS_DIR}/${CONDA_ENV_NAME}/lib" docker://pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel ${PROJECT_DIR}/scripts/train.sh
fi