#!/bin/bash
CASE_NAME=$1

# path to lerf_ovs/label

root_path="../"
gt_folder="../data/lerf_ovs/label"

python eval_lerf.py \
        --dataset_name ${CASE_NAME} \
        --feat_dir ${root_path}/output_admm_quant \
        --ae_ckpt_dir ${root_path}/autoencoder/ckpt \
        --output_dir ${root_path}/eval_result_admm_quant \
        --mask_thresh 0.4 \
        --encoder_dims 256 128 64 32 3 \
        --decoder_dims 16 32 64 128 256 256 512 \
        --json_folder ${gt_folder}