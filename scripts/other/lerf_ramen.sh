#!/bin/bash
#casenames=("figurines" "ramen" "teatime" "waldo_kitchen")
casenames=("ramen")
#gpu_pool=(1 2 3 4)
gpu_pool=(1)
N=10

for id in $(seq 1 $N); do
    RANDOM_STR=$(date +"%m%d%H%M%S")_
    echo "Time-based string: $RANDOM_STR"

    for i in "${!casenames[@]}"; do
        casename=${casenames[$i]}
        gpu_id=${gpu_pool[$i]}
        mkdir -p output_temp/${RANDOM_STR}/nohup_logs/${casename}
        mkdir -p output_temp/${RANDOM_STR}/autoencoder/${casename}

        (
            echo "[${casename}] Starting on GPU ${gpu_id} with ID ${id}"

            # ---------- 训练 AutoEncoder ----------
            cd autoencoder
            CUDA_VISIBLE_DEVICES=${gpu_id} nohup python train.py \
                --dataset_path ../data/lerf_ovs/${casename} \
                --dataset_name ${casename} \
                --encoder_dims 256 128 64 32 3 \
                --decoder_dims 16 32 64 128 256 256 512 \
                --lr 0.0007 \
                > ../output_temp/${RANDOM_STR}/nohup_logs/${casename}/train_ae.log 2>&1

            CUDA_VISIBLE_DEVICES=${gpu_id} nohup python test.py \
                --dataset_path ../data/lerf_ovs/${casename} \
                --dataset_name ${casename} \
                > ../output_temp/${RANDOM_STR}/nohup_logs/${casename}/render_ae.log 2>&1
            cd ..

            # ---------- 训练主模型 ----------
            CUDA_VISIBLE_DEVICES=${gpu_id} nohup python train.py \
                -s data/lerf_ovs/${casename} \
                -m output_temp/${RANDOM_STR}/${casename} \
                --start_checkpoint data/lerf_ovs/${casename}/output/${casename}/chkpnt30000.pth \
                --port 600${gpu_id} \
                > output_temp/${RANDOM_STR}/nohup_logs/${casename}/train.log 2>&1

            # ---------- 渲染 ----------
            CUDA_VISIBLE_DEVICES=${gpu_id} nohup python render.py \
                -m output_temp/${RANDOM_STR}/${casename}\
                --include_feature \
                > output_temp/${RANDOM_STR}/nohup_logs/${casename}/render.log 2>&1

            # ---------- 评估 ----------
            gt_folder="data/lerf_ovs/label"
            CUDA_VISIBLE_DEVICES=${gpu_id} nohup python eval/eval_lerf.py \
                --dataset_name ${casename} \
                --feat_dir output_temp/${RANDOM_STR} \
                --ae_ckpt_dir autoencoder/ckpt \
                --output_dir output_temp/${RANDOM_STR}/eval_result \
                --mask_thresh 0.4 \
                --encoder_dims 256 128 64 32 3 \
                --json_folder ${gt_folder} \
                > output_temp/${RANDOM_STR}/nohup_logs/${casename}/eval.log 2>&1

            # ---------- 拷贝AE模型 ----------
            cp -r autoencoder/ckpt/${casename}/best_ckpt.pth output_temp/${RANDOM_STR}/autoencoder/${casename}/
        ) &
    done

    wait  # 等待4个GPU都完成后，再进入下一轮
done
