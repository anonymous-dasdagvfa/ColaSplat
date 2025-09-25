#!/bin/bash
casenames=("bed")
gpu_pool=(0 1 2)  
mkdir -p logs/nohup_logs/train_admm_quant
mkdir -p logs/nohup_logs/render_admm_quant
mkdir -p logs/nohup_logs/eval_admm_quant
for i in "${!casenames[@]}"
do
    {
        casename=${casenames[$i]}
        gpu_id=${gpu_pool[$i]}  # 
        echo "Launching ${casename} on GPU ${gpu_id}"

        CUDA_VISIBLE_DEVICES=${gpu_id} nohup python train_admm_quant.py \
            -s data/3dovs/${casename} \
            -m output_admm_quant/${casename} \
            --start_checkpoint output/${casename}/chkpnt30000.pth \
            --port 630${i} \
            --include_feature \
            > logs/nohup_logs/train_admm_quant/${casename}_nohup.log 2>&1

        CUDA_VISIBLE_DEVICES=${gpu_id} nohup python render_admm_quant.py \
            -m output_admm_quant/${casename} \
            --dataset 3dovs \
            --include_feature \
            > logs/nohup_logs/render_admm_quant/${casename}_nohup.log 2>&1

        CUDA_VISIBLE_DEVICES=${gpu_id} bash -c "
            cd eval
            nohup sh eval_3dovs_admm_quant.sh ${casename} > ../logs/nohup_logs/eval_admm_quant/${casename}_nohup.log 2>&1
        "
    } &
done
wait  
echo "completed all casenames"
