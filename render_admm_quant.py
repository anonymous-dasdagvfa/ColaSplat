#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np
import torch
from scene_admm import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from arguments_admm_quant import OptimizationParams
from gaussian_renderer_admm import GaussianModel
import re
from glob import glob
import time

import time
from omegaconf import OmegaConf
import pprint

def render_set(model_path, source_path, name, iteration, views, gaussians, pipeline, background, args):
    render_path_1 = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_1")
    render_path_2 = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_2")
    render_path_3 = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_3")
    makedirs(render_path_1, exist_ok=True)
    makedirs(render_path_2, exist_ok=True)
    makedirs(render_path_3, exist_ok=True)
    

    img_render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "img_renders")

    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    gts_path2 = os.path.join(model_path, name, "ours_{}".format(iteration), "gt2")
    gts_path3 = os.path.join(model_path, name, "ours_{}".format(iteration), "gt3")

    img_gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "img_gt")
    mask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "mask")

    render_npy_path1 = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_npy1")
    render_npy_path2 = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_npy2")
    render_npy_path3 = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_npy3")
    
    gts_npy_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_npy")
    mask_npy_path = os.path.join(model_path, name, "ours_{}".format(iteration), "mask_npy")
    makedirs(render_npy_path1, exist_ok=True)
    makedirs(render_npy_path2, exist_ok=True)
    makedirs(render_npy_path3, exist_ok=True)
    makedirs(gts_npy_path, exist_ok=True)

    makedirs(gts_path, exist_ok=True)
    makedirs(gts_path2, exist_ok=True)
    makedirs(gts_path3, exist_ok=True)

    makedirs(mask_path, exist_ok=True)
    makedirs(mask_npy_path, exist_ok=True)

    makedirs(img_render_path, exist_ok=True)
    makedirs(img_gts_path, exist_ok=True)

    print(f"Point Nums (from opacity): {len(gaussians._opacity)}")
    print(f"Point Nums (from xyz): {len(gaussians._xyz)}")

    time_spend = 0
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        start_time = time.time()

        output = render(view, gaussians, pipeline, background, args)

        end_time = time.time()
        render_latence = end_time - start_time
        time_spend += render_latence


        img_rendering = output["render"]

        language_feature_image1 = output["language_feature_image1"]
        language_feature_image2 = output["language_feature_image2"]
        language_feature_image3 = output["language_feature_image3"]            

        img_gt = view.original_image[0:3, :, :]
        gt1, mask1 = view.get_language_feature(os.path.join(source_path, args.language_features_name), feature_level=1)
        gt2, mask2 = view.get_language_feature(os.path.join(source_path, args.language_features_name), feature_level=2)
        gt3, mask3 = view.get_language_feature(os.path.join(source_path, args.language_features_name), feature_level=3)
        
 
        np.save(os.path.join(render_npy_path1, '{0:05d}'.format(idx) + ".npy"),language_feature_image1.permute(1,2,0).cpu().numpy())
        np.save(os.path.join(render_npy_path2, '{0:05d}'.format(idx) + ".npy"),language_feature_image2.permute(1,2,0).cpu().numpy())
        np.save(os.path.join(render_npy_path3, '{0:05d}'.format(idx) + ".npy"),language_feature_image3.permute(1,2,0).cpu().numpy())
        
        np.save(os.path.join(gts_npy_path, '{0:05d}'.format(idx) + ".npy"),gt1.permute(1,2,0).cpu().numpy())
        #np.save(os.path.join(mask_npy_path, '{0:05d}'.format(idx) + ".npy"),mask.permute(1,2,0).cpu().numpy())


        # 
        torchvision.utils.save_image(language_feature_image1, os.path.join(render_path_1, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(language_feature_image2, os.path.join(render_path_2, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(language_feature_image3, os.path.join(render_path_3, '{0:05d}'.format(idx) + ".png"))

        torchvision.utils.save_image(gt1, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt2, os.path.join(gts_path2, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt3, os.path.join(gts_path3, '{0:05d}'.format(idx) + ".png"))

        torchvision.utils.save_image(img_rendering, os.path.join(img_render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(img_gt, os.path.join(img_gts_path, '{0:05d}'.format(idx) + ".png"))
        #torchvision.utils.save_image(mask.float(), os.path.join(mask_path, '{0:05d}'.format(idx) + ".png"))

    print(f"Total rendering time for {len(views)} views: {time_spend:.2f} seconds")
    return time_spend

def get_latest_checkpoint(model_path, ckpt):
    target_ckpt = os.path.join(model_path, f'chkpnt{ckpt}.pth')
    
    if os.path.isfile(target_ckpt):
        print(f"Found specified checkpoint: {target_ckpt}")
        return target_ckpt
    else:
        print(f"Specified checkpoint not found, searching for latest checkpoint in {model_path}...")
        pth_files = glob(os.path.join(model_path, 'chkpnt*.pth'))

        ckpt_pattern = re.compile(r'chkpnt(\d+)\.pth')
        valid_ckpts = []
        for f in pth_files:
            match = ckpt_pattern.search(os.path.basename(f))
            if match:
                valid_ckpts.append((int(match.group(1)), f))

        if not valid_ckpts:
            raise FileNotFoundError(f"No valid checkpoint found in {model_path}.")

        latest_ckpt = max(valid_ckpts, key=lambda x: x[0])[1]
        print(f"Using latest checkpoint: {latest_ckpt}")
        return latest_ckpt

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        #quant =getattr(args, 'quant', False)
        
        scene = Scene(dataset, gaussians, shuffle=False, load_iteration=iteration, quant=args.quant)
        
        #checkpoint = get_latest_checkpoint(args.model_path, iteration)

        #checkpoint = os.path.join(args.model_path, f'chkpnt{args.ckpt}.pth')
        
        
        #(model_params, first_iter) = torch.load(checkpoint)
        #gaussians.restore(model_params, args, mode='test')
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        #iteration = scene.loaded_iter 
        iteration = None
        if not skip_train:
             print("Rendering train set")
             time_spend_train = render_set(dataset.model_path, dataset.source_path, "train", iteration, scene.getTrainCameras(), gaussians, pipeline, background, args)
             print(f"Time spent on train set: {time_spend_train} seconds")

        if not skip_test:
             print("Rendering test set")
             time_spend_text = render_set(dataset.model_path, dataset.source_path, "test", iteration, scene.getTestCameras(), gaussians, pipeline, background, args)
             print(f"Time spent on test set: {time_spend_text} seconds")

        print(f"Rendering completed. Total time spent: {time_spend_train + time_spend_text} seconds")

def load_config_with_scene(dataset_name, scene_name):
    base_path = f"configs/{dataset_name}"
    base_config_path = os.path.join(base_path, "base.yaml")
    scene_config_path = os.path.join(base_path, f"{scene_name}.yaml")

    base_cfg = OmegaConf.load(base_config_path)
    
    if os.path.exists(scene_config_path):
        scene_cfg = OmegaConf.load(scene_config_path)
        config = OmegaConf.merge(base_cfg, scene_cfg)
        print(f"[INFO] Loaded base.yaml + {scene_name}.yaml (scene config overrides base)")
    else:
        config = base_cfg
        print(f"[INFO] Loaded base.yaml (no scene-specific config found)")

    return config

def inject_config_into_group(config_dict, opt):
    for k, v in config_dict.items():
        setattr(opt, k, v)
    return opt


if __name__ == "__main__":
    # Set up command line argument parser
    
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    # parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--dataset', type=str, default="", help='Dataset name, e.g., lerf, 3dovs')
    #parser.add_argument("--include_feature", action="store_true")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    scene_name = os.path.basename(args.source_path.rstrip("/"))
    # config = load_config_with_scene(args.dataset, scene_name)
    # args = inject_config_into_group(config, args)
    # args_dict = vars(args)  # 
    # print(f"args:\n{pprint.pformat(args_dict)}")

    safe_state(args.quiet)
    render_sets(model.extract(args), op.extract(args).iterations, pipeline.extract(args), args.skip_train, args.skip_test, args)