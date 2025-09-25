
import torch
import fnmatch
import numpy as np
import os
from scene_admm import GaussianModel
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
#  # Alternating Direction Method of Multipliers (ADMM) optimization algorithm, usually used to solve optimization problems with constraints and regularization.
class ADMM:
    def __init__(self, gsmodel: GaussianModel, rho1, rho2, device):
        self.gsmodel = gsmodel 
        self.device = device # Specify where tensors are stored (CPU or GPU)
        self.init_rho1 = rho1 # Scalar, used for the penalty term in ADMM
        self.init_rho2 = rho2 #

        # u and z are auxiliary variables (dual variables) in the ADMM algorithm
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

    # Use Lagrange multipliers u to gradually approximate the consistency of x and z
    def update1(self, threshold, update_u = True):
        # 1. The input to compute z is equivalent to the original variable + the Lagrange multiplier λ
        z = self.gsmodel.get_opacity + self.u1

        # 2. Update z to satisfy the constraints (this is cropping/sparse operations)  
        self.z1 = torch.Tensor(self.prune_z(z, threshold)).to(self.device)
        
        # 3. Update the Lagrange multiplier λ(u1) to gradually penalize the gap between the original variable and the auxiliary variable, promoting the convergence of the two
        if update_u: 
            with torch.no_grad():
                diff =  self.gsmodel.get_opacity - self.z1
                self.u1 += diff

    def update2(self, kmeans_sh_q, update_u=True, assign = True, update_center = True):

        if kmeans_sh_q.vec_dim == 0:
            kmeans_sh_q.vec_dim = self.gsmodel._features_rest.shape[1] * self.gsmodel._features_rest.shape[2]


        # x + u
        z_input = self.gsmodel._features_rest + self.u2
        feat = z_input.reshape(-1, kmeans_sh_q.vec_dim)  #  (N, D)

        # Step 1: Find the nearest cluster center index for each sample point (cluster assignment
        kmeans_sh_q.update_centers(self.gsmodel._features_rest)
        kmeans_sh_q.cluster_assign(feat)  # update kmeans_sh_q.nn_index
        
        # Step 2: Get the quantized center vector according to the index (projection operation)
        indices = kmeans_sh_q.nn_index.long()
        centers = kmeans_sh_q.centers  # Codebook Center
        quantized_values = centers[indices]  # (N, vec_dim)

        # Step 3: Reshape it back and update the auxiliary variable z
        self.z2 = quantized_values.reshape(z_input.shape).to(self.device)

        # Step 4: Update the dual variable 
        if update_u:
            with torch.no_grad():
                diff = self.gsmodel._features_rest - self.z2
                self.u2 += diff



    #  
    def prune_z(self, z, threshold):
        z_update = self.metrics_sort(z, threshold)  
        return z_update
    

    # 
    def get_admm_loss_1(self): 
        return 0.5 * self.rho1 * (torch.norm(self.gsmodel.get_opacity - self.z1 + self.u1, p=2)) ** 2
    
    #
    def get_admm_loss_2(self): 
        return 0.5 * self.rho2 * (torch.norm(self.gsmodel._features_rest - self.z2 + self.u2, p=2)) ** 2

    # Adjust the rho value based on the current progress of training (epoch and epochs). 
    # Typically, rho will increase as training progresses, increasing the penalty on the constraint.
    def adjust_rho(self, epoch, epochs, factor=5): 
        if epoch > int(0.85 * epochs):
            self.rho1 = factor * self.init_rho1
    
    def metrics_sort(self, z, threshold): 
        index = int(threshold * len(z))
        z_sort = {}
        z_update = torch.zeros(z.shape)
        z_sort, _ = torch.sort(z, 0)
        z_threshold = z_sort[index-1]
        z_update= ((z > z_threshold) * z)  
        return z_update
    
    def metrics_sample(self, z, opt): 
        index = int((1 - opt.pruning_threshold) * len(z))
        prob = z / torch.sum(z)
        prob = prob.reshape(-1).cpu().numpy()
        indices = torch.tensor(np.random.choice(len(z), index, p = prob, replace=False))
        expand_indices = torch.zeros(z.shape[0] - len(indices)).int()
        indices = torch.cat((indices, expand_indices),0).to(self.device)
        z_update = torch.zeros(z.shape).to(self.device)
        z_update[indices] = z[indices]
        return z_update

    def metrics_imp_score(self, z, imp_score, opt): 
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

