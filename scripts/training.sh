#!/bin/bash

DATASET=$1
MODEL=$2
DEVICES=$3

CONFIG_PATH=configs/${MODEL}/${DATASET}/
## Logging
TEST_SAVE_DIR=${DATASET}_training_$(date +"%F-%T")
LOG_PATH="log/${TEST_SAVE_DIR}/"
if [ ! -d ${LOG_PATH} ]; then
    mkdir -p ${LOG_PATH};
    touch ${LOG_PATH}/log.txt;
fi

CUDA_VISIBLE_DEVICES=${DEVICES} python -W ignore -u training.py \
											--config_path $CONFIG_PATH \
											--log_path $LOG_PATH \
											--dataset $DATASET \
											--model $MODEL \
											2>&1 | tee -a ${LOG_PATH}log.txt
