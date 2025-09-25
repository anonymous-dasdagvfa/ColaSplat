#!/bin/bash
casenames=("figurines")
gpu_pool=(0 1 2 3 4 5)  
# render_admm_quant
mkdir -p logs/nohup_logs/train_admm_quant
mkdir -p logs/nohup_logs/render_admm_quant
mkdir -p logs/nohup_logs/eval_admm_quant



for i in "${!casenames[@]}"
do
    {
        casename=${casenames[$i]}
        gpu_id=${gpu_pool[$i]}  
        echo "Launching ${casename} on GPU ${gpu_id}"

 
        CUDA_VISIBLE_DEVICES=${gpu_id} nohup python train_admm_quant.py \
            -s data/lerf_ovs/${casename} \
            -m output_admm_quant/${casename} \
            --start_checkpoint output_admm_quant/${casename}/chkpnt5000.pth \
            --port 630${i} \
            --include_feature \
            > logs/nohup_logs/train_admm_quant/${casename}_nohup.log 2>&1


        CUDA_VISIBLE_DEVICES=${gpu_id} nohup python render_admm_quant.py \
            -m output_admm_quant/${casename} \
            --dataset lerf \
            --include_feature \
            > logs/nohup_logs/render_admm_quant/${casename}_nohup.log 2>&1


        CUDA_VISIBLE_DEVICES=${gpu_id} bash -c "
            cd eval
            nohup sh eval_lerf_admm_quant.sh ${casename} > ../logs/nohup_logs/eval_admm_quant/${casename}_eval.log 2>&1
        "
    } &
done
wait 
echo "end"
