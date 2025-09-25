#!/bin/bash
CASE_NAME=$1

# path to lerf/label

root_path="../"

python eval_3dovs.py \
        --dataset_name ${CASE_NAME} \
        --feat_dir ${root_path}/output \
        --ae_ckpt_dir ${root_path}/autoencoder/ckpt \
        --output_dir ${root_path}/eval_result \
        --mask_thresh 0.4 \
        --encoder_dims 256 128 64 32 3 \
        --decoder_dims 16 32 64 128 256 256 512 \
        --dataset_path "../data/3dovs/"  