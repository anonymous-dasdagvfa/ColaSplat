#!/bin/bash
casenames=("bed" "bench" "lawn" "room" "sofa")
gpu_pool=(0 1 2 3 4 5)  # 手动指定的GPU列表

iter=3000
mkdir -p logs/nohup_logs/train_admm
mkdir -p logs/nohup_logs/render_admm
mkdir -p logs/nohup_logs/eval_admm

# ---------- 阶段 1：训练和渲染 ----------
for i in "${!casenames[@]}"
do
    {
        casename=${casenames[$i]}
        gpu_id=${gpu_pool[$i]}  # 从池中取GPU
        echo "Launching ${casename} on GPU ${gpu_id}"

        # admm 训练阶段
        CUDA_VISIBLE_DEVICES=${gpu_id} nohup python train_admm.py \
            -s data/3dovs/${casename} \
            -m output_admm/${casename} \
            --start_checkpoint output/${casename}/chkpnt30000.pth \
            --port 610${i} \
            --include_feature \
            > logs/nohup_logs/train_admm/${casename}_nohup.log 2>&1

        # 渲染阶段
        CUDA_VISIBLE_DEVICES=${gpu_id} nohup python render_admm.py \
            -m output_admm/${casename} \
            --include_feature \
            > logs/nohup_logs/render_admm/${casename}_nohup.log 2>&1

        # 评估阶段
        CUDA_VISIBLE_DEVICES=${gpu_id} bash -c "
            cd eval
            nohup sh eval_3dovs_admm.sh ${casename} > ../logs/nohup_logs/eval_admm/${casename}_nohup.log 2>&1
        "
    } &
done
wait  # 等待所有流程完成
echo "所有任务完成"
