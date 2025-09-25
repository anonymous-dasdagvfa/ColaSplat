import numpy as np
import torch
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import mediapy as media
import cv2
import colormaps
from pathlib import Path


def show_points(coords, labels, ax, marker_size=30):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='firebrick', marker='o',
               s=marker_size, edgecolor='black', linewidth=2.5, alpha=1)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='o',
               s=marker_size, edgecolor='black', linewidth=1.5, alpha=1)   


def show_box(boxes, ax, color=None):
    if type(color) == str and color == 'random':
        color = np.random.random(3)
    elif color is None:
        color = 'black'
    for box in boxes.reshape(-1, 4):
        color = np.random.random(3)
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0,0,0,0), lw=2, 
                                   capstyle='round', joinstyle='round', linestyle='dotted')) 


def show_result(image, point, bbox, save_path):
    plt.figure()
    plt.imshow(image)
    rect = patches.Rectangle((0, 0), image.shape[1]-1, image.shape[0]-1, linewidth=0, edgecolor='none', facecolor='white', alpha=0.3)
    plt.gca().add_patch(rect)
    input_point = point.reshape(1,-1)
    input_label = np.array([1])
    show_points(input_point, input_label, plt.gca())
    show_box(bbox, plt.gca())
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=200)
    plt.close()

def show_result_2(image_np, coord_list, bboxes, save_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image_np)

    # 画预测框（绿色）
    for coord in coord_list:
        x, y = coord
        # 设置一个小的正方形框，作为预测框
        size = 10  # 可根据需要调整
        rect = patches.Rectangle((x - size/2, y - size/2), size, size,
                                 linewidth=2, edgecolor='green', facecolor='none', label='Prediction')
        ax.add_patch(rect)

    # 画 Ground Truth 框（红色）
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle((x1, y1), width, height,
                                 linewidth=2, edgecolor='red', facecolor='none', label='GT')
        ax.add_patch(rect)

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def smooth(mask):
    h, w = mask.shape[:2]
    im_smooth = mask.copy()
    scale = 3
    for i in range(h):
        for j in range(w):
            square = mask[max(0, i-scale) : min(i+scale+1, h-1),
                          max(0, j-scale) : min(j+scale+1, w-1)]
            im_smooth[i, j] = np.argmax(np.bincount(square.reshape(-1)))
    return im_smooth

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
from pathlib import Path
from PIL import Image

def colormap_saving(image: torch.Tensor, colormap_options, save_path):
    """
    应用颜色映射，并将colorbar嵌入图像角落保存为PNG。
    """
    output_image = (
        colormaps.apply_colormap(
            image=image,
            colormap_options=colormap_options,
        ).cpu().numpy()
    )  # (H, W, 3), numpy array, float32 or uint8

    if save_path is not None:
        save_path = Path(save_path).with_suffix(".png")
        output_image = np.clip(output_image*255, 0, 255).astype(np.uint8)

        plt.imsave(save_path, output_image)

        # # 绘制带嵌入 colorbar 的图像
        # fig, ax = plt.subplots(figsize=(6, 6))
        # ax.imshow(output_image)
        # ax.axis("off")

        # # 嵌入 colorbar
        # cbar_ax = fig.add_axes([0.65, 0.05, 0.3, 0.03])  # [left, bottom, width, height]
        # cmap = mpl.cm.get_cmap(colormap_options.colormap)
        # norm = mpl.colors.Normalize(vmin=colormap_options.colormap_min, vmax=colormap_options.colormap_max)
        # cb = mpl.colorbar.ColorbarBase(
        #     cbar_ax, cmap=cmap, norm=norm, orientation='horizontal'
        # )
        # cb.outline.set_visible(False)
        # cb.ax.tick_params(labelsize=8)

        # # 保存图像（带legend）
        # fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        # plt.close(fig)

    return output_image


# def colormap_saving(image: torch.Tensor, colormap_options, save_path):
#     """
#     if image's shape is (h, w, 1): draw colored relevance map;
#     if image's shape is (h, w, 3): return directively;
#     if image's shape is (h, w, c): execute PCA and transform it into (h, w, 3).
#     """
#     output_image = (
#         colormaps.apply_colormap(
#             image=image,
#             colormap_options=colormap_options,
#         ).cpu().numpy()
#     )
#     if save_path is not None:
#         media.write_image(save_path.with_suffix(".png"), output_image, fmt="png")
#     return output_image


def vis_mask_save(mask, save_path: Path = None):
    mask_save = mask.copy()
    mask_save[mask == 1] = 255
    save_path.parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(save_path), mask_save)


def polygon_to_mask(img_shape, points_list):
    points = np.asarray(points_list, dtype=np.int32)
    mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    return mask


def stack_mask(mask_base, mask_add):
    mask = mask_base.copy()
    mask[mask_add != 0] = 1
    return mask