# -*- coding:utf-8 -*-
# 
# Author: 
# Time: 

import torch
import fnmatch
import numpy as np
import os
from scene_admm import GaussianModel
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
#  交替方向乘子法（ADMM） 优化算法，通常用于解决带有约束和正则化的优化问题。
class ADMM:
    def __init__(self, gsmodel: GaussianModel, rho1, rho2, device):
        self.gsmodel = gsmodel 
        self.device = device # 指定张量存储的位置（CPU 或 GPU）
        self.init_rho1 = rho1 # 标量，用于 ADMM 中的惩罚项
        self.init_rho2 = rho2 #

        # u和z是ADMM算法中的辅助变量（对偶变量）
        self.u1 = {}
        self.u2 = {}
        self.z2 = {}
        self.z1 = {}

        self.rho1 = self.init_rho1
        self.rho2 = self.init_rho2

        opacity = self.gsmodel.get_opacity
        sh = self.gsmodel._features_rest

        self.u1 = torch.zeros(opacity.shape).to(device)
        self.z1 = torch.Tensor(opacity.data.cpu().clone().detach()).to(device)
        self.u2 = torch.zeros(sh.shape).to(device)
        self.z2 = torch.Tensor(sh.data.cpu().clone().detach()).to(device)

    # 用拉格朗日乘子 u 逐步逼近 x 和 z 的一致性
    def update1(self, threshold, update_u = True):
        # 1. 计算 z 的输入，等价于原始变量 + 拉格朗日乘子 λ
        z = self.gsmodel.get_opacity + self.u1

        # 2. 更新 z，使其满足约束 (这里是裁剪/稀疏化等操作)  
        self.z1 = torch.Tensor(self.prune_z(z, threshold)).to(self.device)
        
        # 3. 更新拉格朗日乘子 λ（u1） 逐步惩罚原始变量和辅助变量的差距，促进两者收敛一致
        if update_u: 
            with torch.no_grad():
                diff =  self.gsmodel.get_opacity - self.z1
                self.u1 += diff

    def update2(self, kmeans_sh_q, update_u=True, assign = True, update_center = True):

        if kmeans_sh_q.vec_dim == 0:
            kmeans_sh_q.vec_dim = self.gsmodel._features_rest.shape[1] * self.gsmodel._features_rest.shape[2]


        # 计算x + u
        z_input = self.gsmodel._features_rest + self.u2
        feat = z_input.reshape(-1, kmeans_sh_q.vec_dim)  # 形状 (N, D)

        # Step 1: 对每个样本点找到最近的聚类中心索引（聚类分配）
        kmeans_sh_q.update_centers(self.gsmodel._features_rest)
        kmeans_sh_q.cluster_assign(feat)  # 这里更新 kmeans_sh_q.nn_index
        
        # Step 2: 根据索引得到量化后的中心向量（投影操作）
        indices = kmeans_sh_q.nn_index.long()
        centers = kmeans_sh_q.centers  # 码本中心
        quantized_values = centers[indices]  # (N, vec_dim)

        # Step 3: 重新 reshape 回去，更新辅助变量 z
        self.z2 = quantized_values.reshape(z_input.shape).to(self.device)

        # Step 4: 更新对偶变量 u
        if update_u:
            with torch.no_grad():
                diff = self.gsmodel._features_rest - self.z2
                self.u2 += diff



    #  该方法根据不同的策略（由 opt 参数控制）来更新 z：
    def prune_z(self, z, threshold):
        z_update = self.metrics_sort(z, threshold)  
        return z_update
    

    # 使opacity和z1之间的差异最小化 
    def get_admm_loss_1(self): 
        return 0.5 * self.rho1 * (torch.norm(self.gsmodel.get_opacity - self.z1 + self.u1, p=2)) ** 2
    
    # 使features_rest和z2之间的差异最小化
    def get_admm_loss_2(self): 
        return 0.5 * self.rho2 * (torch.norm(self.gsmodel._features_rest - self.z2 + self.u2, p=2)) ** 2

    def adjust_rho(self, epoch, epochs, factor=5): # 根据训练的当前进度（epoch 和 epochs）调整 rho 值。通常，rho 会随着训练的进展而增大，从而增加对约束的惩罚
        if epoch > int(0.85 * epochs):
            self.rho1 = factor * self.init_rho1
    
    def metrics_sort(self, z, threshold): # 根据透明度的排序值来更新 z。它通过将透明度按升序排序并应用一个阈值来选择透明度值
        index = int(threshold * len(z))
        z_sort = {}
        z_update = torch.zeros(z.shape)
        z_sort, _ = torch.sort(z, 0)
        z_threshold = z_sort[index-1]
        z_update= ((z > z_threshold) * z)  
        return z_update
    
    def metrics_sample(self, z, opt): # 该方法根据透明度值的相对权重进行随机采样。首先，将透明度值归一化为概率分布，并按此分布随机选择样本。
        index = int((1 - opt.pruning_threshold) * len(z))
        prob = z / torch.sum(z)
        prob = prob.reshape(-1).cpu().numpy()
        indices = torch.tensor(np.random.choice(len(z), index, p = prob, replace=False))
        expand_indices = torch.zeros(z.shape[0] - len(indices)).int()
        indices = torch.cat((indices, expand_indices),0).to(self.device)
        z_update = torch.zeros(z.shape).to(self.device)
        z_update[indices] = z[indices]
        return z_update

    def metrics_imp_score(self, z, imp_score, opt): # 该方法基于重要性分数（imp_score）更新 z。重要性分数低于某个阈值的透明度值会被置为 0，从而对重要性较低的部分进行稀疏化。
        index = int(opt.pruning_threshold * len(z))
        imp_score_sort = {}
        imp_score_sort, _ = torch.sort(imp_score, 0)
        imp_score_threshold = imp_score_sort[index-1]
        indices = imp_score < imp_score_threshold 
        z[indices == 1] = 0  
        return z        



def get_unactivate_opacity(gaussians):
    opacity = gaussians._opacity[:, 0]
    scores = opacity
    return scores

def get_pruning_mask(scores, threshold):        
    scores_sorted, _ = torch.sort(scores, 0)
    threshold_idx = int(threshold * len(scores_sorted))
    abs_threshold = scores_sorted[threshold_idx - 1]
    mask = (scores <= abs_threshold).squeeze()
    return mask

def check_grad_leakage(model, optimizer=None):
    """
    适用于非 nn.Module 的模型，自行遍历属性检查参数梯度状态。
    """
    print("\n=== 🚨 检查梯度泄露和优化器参数（自定义模型） ===")
    leak_found = False

    for name in dir(model):
        param = getattr(model, name)
        if isinstance(param, torch.Tensor) and param.requires_grad is not None:
            if param.grad is not None and not param.requires_grad:
                print(f"⚠️ 参数 '{name}' 有 grad，但 requires_grad=False！可能泄露。")
                leak_found = True
            elif param.requires_grad:
                print(f"✅ 参数 '{name}' 设置为可训练 (requires_grad=True)。")
            else:
                print(f"🔒 参数 '{name}' 不可训练 (requires_grad=False)。")

    # 检查 optimizer 参数组
    if optimizer is not None:
        print("\n=== 🔍 检查 Optimizer 中的参数 ===")
        for i, group in enumerate(optimizer.param_groups):
            for p in group['params']:
                if not p.requires_grad:
                    print(f"⚠️ Optimizer param_group[{i}] 中包含 requires_grad=False 的参数")
                    leak_found = True

    if not leak_found:
        print("✅ 没发现泄露或异常参数。")
    raise("=== 检查完成 ===\n")
