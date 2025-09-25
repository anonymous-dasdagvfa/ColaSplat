#!/bin/bash
#casenames=("bed" "bench" "lawn" "room" "sofa")
casenames=("bed")
# casenames=("room")
gpu_pool=(0 1 2)  # 手动指定的GPU列表
# render_admm_quant
# eval_admm_quant_3dovs
mkdir -p logs/nohup_logs/train_admm_quant
mkdir -p logs/nohup_logs/render_admm_quant
mkdir -p logs/nohup_logs/eval_admm_quant


# ---------- 阶段 1：训练和渲染 ----------
for i in "${!casenames[@]}"
do
    {
        casename=${casenames[$i]}
        gpu_id=${gpu_pool[$i]}  # 从池中取GPU
        echo "Launching ${casename} on GPU ${gpu_id}"

        # # quant 训练阶段
        # CUDA_VISIBLE_DEVICES=${gpu_id} nohup python train_admm_quant.py \
        #     -s data/3dovs/${casename} \
        #     -m output_admm_quant/${casename} \
        #     --start_checkpoint output/${casename}/chkpnt30000.pth \
        #     --port 630${i} \
        #     --include_feature \
        #     > logs/nohup_logs/train_admm_quant/${casename}_nohup.log 2>&1

        # 渲染阶段
        CUDA_VISIBLE_DEVICES=${gpu_id} nohup python render_admm_quant.py \
            -m output_admm_quant/${casename} \
            --dataset 3dovs \
            --include_feature \
            > logs/nohup_logs/render_admm_quant/${casename}_nohup.log 2>&1

        # CUDA_VISIBLE_DEVICES=${gpu_id} bash -c "
        #     cd eval
        #     nohup sh eval_3dovs_admm_quant.sh ${casename} > ../logs/nohup_logs/eval_admm_quant/${casename}_nohup.log 2>&1
        # "
    } &
done
wait  # 等待所有流程完成
echo "所有任务完成"
