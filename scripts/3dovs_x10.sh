#!/bin/bash
#casenames=("bed" "bench" "lawn" "room" "sofa")
casenames=("room")

N=100
for id in $(seq 1 $N); do
    RANDOM_STR=$(date +"%m%d%H%M%S")_
    echo "Time-based string: $RANDOM_STR"

    for casename in "${casenames[@]}"; do
        echo "开始处理 ${casename}"
        mkdir -p output_temp/${RANDOM_STR}/nohup_logs/${casename}/train_ae
        mkdir -p output_temp/${RANDOM_STR}/nohup_logs/${casename}/render_ae
        mkdir -p output_temp/${RANDOM_STR}/nohup_logs/${casename}/train
        mkdir -p output_temp/${RANDOM_STR}/nohup_logs/${casename}/render
        mkdir -p output_temp/${RANDOM_STR}/nohup_logs/${casename}/eval
        mkdir -p output_temp/${RANDOM_STR}/autoencoder/${casename}
        # # ---------- 训练autoencoder ----------
    
        cd autoencoder

        CUDA_VISIBLE_DEVICES=0 nohup python train.py \
            --dataset_path ../data/3dovs/${casename} \
            --dataset_name ${casename} \
            --encoder_dims 256 128 64 32 3 \
            --decoder_dims 16 32 64 128 256 256 512 \
            --lr 0.0007 \
            > ../output_temp/${RANDOM_STR}/nohup_logs/${casename}/train_ae/log.txt 2>&1
        wait 

        CUDA_VISIBLE_DEVICES=0 nohup python test.py \
            --dataset_path ../data/3dovs/${casename} \
            --dataset_name ${casename} \
            > ../output_temp/${RANDOM_STR}/nohup_logs/${casename}/render_ae/log.txt 2>&1

        wait
        cd ..
        # # ---------- 训练 ----------
        for level in 1 2 3; do
            CUDA_VISIBLE_DEVICES=$((level-1)) nohup python train.py \
                -s data/3dovs/${casename} \
                -m output_temp/${RANDOM_STR}/${casename} \
                --start_checkpoint data/3dovs/${casename}/output/${casename}/chkpnt30000.pth \
                --port 600${level} \
                --feature_level ${level} \
                > output_temp/${RANDOM_STR}/nohup_logs/${casename}/train/lv_${level}.log 2>&1 &
        done

        wait  

        # ---------- 渲染 ----------
        for level in 1 2 3; do
            CUDA_VISIBLE_DEVICES=$((level-1)) nohup python render.py \
                -m output_temp/${RANDOM_STR}/${casename}_${level} \
                --include_feature \
                > output_temp/${RANDOM_STR}/nohup_logs/${casename}/render/lv_${level}.log 2>&1 &
        done

        wait  

        # ---------- 评估 ----------
        CUDA_VISIBLE_DEVICES=0 nohup python eval/eval_3dovs.py \
                --dataset_name ${casename} \
                --feat_dir output_temp/${RANDOM_STR} \
                --ae_ckpt_dir autoencoder/ckpt \
                --output_dir  output_temp/${RANDOM_STR}/eval_result \
                --mask_thresh 0.4 \
                --encoder_dims 256 128 64 32 3 \
                --dataset_path "data/3dovs/" \
                > output_temp/${RANDOM_STR}/nohup_logs/${casename}/eval/lv_${level}.log 2>&1 &
        
        cp -r autoencoder/ckpt/${casename}/best_ckpt.pth output_temp/${RANDOM_STR}/autoencoder/${casename}/best_ckpt.pth

    done

done
