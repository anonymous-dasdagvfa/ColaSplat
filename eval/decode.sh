#!/bin/bash
CASE_NAME=$1
root_path="../"
python decode.py \
        --dataset_name ${CASE_NAME} \
        --feat_dir ${root_path}/output \
        --ae_ckpt_dir ${root_path}/autoencoder/ckpt \
        --output_dir ${root_path}/eval_result \
        --mask_thresh 0.4 \
        --encoder_dims 256 128 64 32 3 \
        --decoder_dims 16 32 64 128 256 256 512 \
        --dataset_path "~/LangSplat/data/3dovs/" 