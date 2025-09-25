#!/bin/bash

casenames=("bed" "bench" "lawn" "room" "sofa")

gpu_pool=(0 1 2  4 5 6)  # 手动指定的GPU列表

mkdir -p logs/nohup_logs/train_ae
mkdir -p logs/nohup_logs/render_ae
mkdir -p logs/nohup_logs/train
mkdir -p logs/nohup_logs/render
mkdir -p logs/nohup_logs/eval

for i in "${!casenames[@]}"
do
    casename=${casenames[$i]}
    gpu_id=${gpu_pool[$i]}  # 从池中取GPU

    echo "Launching ${casename} on GPU ${gpu_id}"

    (
        # 场景内部训练+渲染+评估串行执行，场景之间并行
        # cd autoencoder
        # CUDA_VISIBLE_DEVICES=${gpu_id} nohup python train.py \
        #     --dataset_path ../data/3dovs/${casename} \
        #     --dataset_name ${casename} \
        #     --encoder_dims 256 128 64 32 3 \
        #     --decoder_dims 16 32 64 128 256 256 512 \
        #     --lr 0.0007 \
        #     > ../logs/nohup_logs/train_ae/${casename}.log 2>&1
        # CUDA_VISIBLE_DEVICES=${gpu_id} nohup python test.py \
        #     --dataset_path ../data/3dovs/${casename} \
        #     --dataset_name ${casename} \
        #     > ../logs/nohup_logs/render_ae/${casename}.log 2>&1
        # cd ..

        # CUDA_VISIBLE_DEVICES=${gpu_id} nohup python train.py \
        #     -s data/3dovs/${casename} \
        #     -m output/${casename} \
        #     --start_checkpoint data/3dovs/${casename}/output/${casename}/chkpnt30000.pth \
        #     --port 600${i} \
        #     > logs/nohup_logs/train/${casename}.log 2>&1

        CUDA_VISIBLE_DEVICES=${gpu_id} nohup python render.py \
            -m output/${casename} \
            --include_feature \
            > logs/nohup_logs/render/${casename}.log 2>&1

        # CUDA_VISIBLE_DEVICES=${gpu_id} bash -c "
        #     cd eval
        #     nohup sh eval_3dovs.sh ${casename} > ../logs/nohup_logs/eval/${casename}.log 2>&1
        # "
    ) &
done

wait  # 等待所有后台任务结束

echo "所有任务完成"
