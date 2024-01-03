import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



# convert arg line to args
def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)


# save args
def save_args(args, filename):
    with open(filename, 'w') as f:
        for arg in vars(args):
            f.write('{}: {}\n'.format(arg, getattr(args, arg)))


# concatenate images
def concat_image(image_path_list, concat_image_path):
    imgs = [Image.open(i).convert("RGB").resize((640, 480), resample=Image.BILINEAR) for i in image_path_list]
    imgs_list = []
    for i in range(len(imgs)):
        img = imgs[i]
        imgs_list.append(np.asarray(img))

        H, W, _ = np.asarray(img).shape
        imgs_list.append(255 * np.ones((H, 20, 3)).astype('uint8'))

    imgs_comb = np.hstack(imgs_list[:-1])
    imgs_comb = Image.fromarray(imgs_comb)
    imgs_comb.save(concat_image_path)


# load model
def load_checkpoint(fpath, model):
    ckpt = torch.load(fpath, map_location='cpu')['model']

    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v

    model.load_state_dict(load_dict)
    return model


# compute normal errors
def compute_normal_errors(total_normal_errors):
    metrics = {
        'mean': np.average(total_normal_errors),
        'median': np.median(total_normal_errors),
        'rmse': np.sqrt(np.sum(total_normal_errors * total_normal_errors) / total_normal_errors.shape),
        'a1': 100.0 * (np.sum(total_normal_errors < 5) / total_normal_errors.shape[0]),
        'a2': 100.0 * (np.sum(total_normal_errors < 7.5) / total_normal_errors.shape[0]),
        'a3': 100.0 * (np.sum(total_normal_errors < 11.25) / total_normal_errors.shape[0]),
        'a4': 100.0 * (np.sum(total_normal_errors < 22.5) / total_normal_errors.shape[0]),
        'a5': 100.0 * (np.sum(total_normal_errors < 30) / total_normal_errors.shape[0])
    }
    return metrics


# log normal errors
def log_normal_errors(metrics, where_to_write, first_line):
    print(first_line)
    print("mean median rmse 5 7.5 11.25 22.5 30")
    print("%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f" % (
        metrics['mean'], metrics['median'], metrics['rmse'],
        metrics['a1'], metrics['a2'], metrics['a3'], metrics['a4'], metrics['a5']))

    with open(where_to_write, 'a') as f:
        f.write('%s\n' % first_line)
        f.write("mean median rmse 5 7.5 11.25 22.5 30\n")
        f.write("%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n\n" % (
            metrics['mean'], metrics['median'], metrics['rmse'],
            metrics['a1'], metrics['a2'], metrics['a3'], metrics['a4'], metrics['a5']))


# makedir
def makedir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


# makedir from list
def make_dir_from_list(dirpath_list):
    for dirpath in dirpath_list:
        makedir(dirpath)



########################################################################################################################
# Visualization
########################################################################################################################


# unnormalize image
__imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
def unnormalize(img_in):
    img_out = np.zeros(img_in.shape)
    for ich in range(3):
        img_out[:, :, ich] = img_in[:, :, ich] * __imagenet_stats['std'][ich]
        img_out[:, :, ich] += __imagenet_stats['mean'][ich]
    img_out = (img_out * 255).astype(np.uint8)
    return img_out


# kappa to exp error (only applicable to AngMF distribution)
def kappa_to_alpha(pred_kappa):
    alpha = ((2 * pred_kappa) / ((pred_kappa ** 2.0) + 1)) \
            + ((np.exp(- pred_kappa * np.pi) * np.pi) / (1 + np.exp(- pred_kappa * np.pi)))
    alpha = np.degrees(alpha)
    return alpha


# normal vector to rgb values
def norm_to_rgb(norm):
    # norm: (B, H, W, 3)
    norm_rgb = ((norm[0, ...] + 1) * 0.5) * 255
    norm_rgb = np.clip(norm_rgb, a_min=0, a_max=255)
    norm_rgb = norm_rgb.astype(np.uint8)
    return norm_rgb


# visualize during training
def visualize(args, img, gt_norm, gt_norm_mask, norm_out_list, total_iter):
    B, _, H, W = gt_norm.shape

    pred_norm_list = []
    pred_kappa_list = []
    for norm_out in norm_out_list:
        norm_out = F.interpolate(norm_out, size=[gt_norm.size(2), gt_norm.size(3)], mode='nearest')
        pred_norm = norm_out[:, :3, :, :]  # (B, 3, H, W)
        pred_norm = pred_norm.detach().cpu().permute(0, 2, 3, 1).numpy()  # (B, H, W, 3)
        pred_norm_list.append(pred_norm)

        pred_kappa = norm_out[:, 3:, :, :]  # (B, 1, H, W)
        pred_kappa = pred_kappa.detach().cpu().permute(0, 2, 3, 1).numpy()  # (B, H, W, 1)
        pred_kappa_list.append(pred_kappa)

    # to numpy arrays
    img = img.detach().cpu().permute(0, 2, 3, 1).numpy()                     # (B, H, W, 3)
    gt_norm = gt_norm.detach().cpu().permute(0, 2, 3, 1).numpy()             # (B, H, W, 3)
    gt_norm_mask = gt_norm_mask.detach().cpu().permute(0, 2, 3, 1).numpy()   # (B, H, W, 1)

    # input image
    target_path = '%s/%08d_img.jpg' % (args.exp_vis_dir, total_iter)
    img = unnormalize(img[0, ...])
    plt.imsave(target_path, img)

    # gt norm
    gt_norm_rgb = ((gt_norm[0, ...] + 1) * 0.5) * 255
    gt_norm_rgb = np.clip(gt_norm_rgb, a_min=0, a_max=255)
    gt_norm_rgb = gt_norm_rgb.astype(np.uint8)

    target_path = '%s/%08d_gt_norm.jpg' % (args.exp_vis_dir, total_iter)
    plt.imsave(target_path, gt_norm_rgb * gt_norm_mask[0, ...])

    # pred_norm
    for i in range(len(pred_norm_list)):
        pred_norm = pred_norm_list[i]
        pred_norm_rgb = norm_to_rgb(pred_norm)
        target_path = '%s/%08d_pred_norm_%d.jpg' % (args.exp_vis_dir, total_iter, i)
        plt.imsave(target_path, pred_norm_rgb)

        pred_kappa = pred_kappa_list[i]
        pred_alpha = kappa_to_alpha(pred_kappa)
        target_path = '%s/%08d_pred_alpha_%d.jpg' % (args.exp_vis_dir, total_iter, i)
        plt.imsave(target_path, pred_alpha[0, :, :, 0], vmin=0, vmax=60, cmap='jet')

        # error in angles
        DP = np.sum(gt_norm * pred_norm, axis=3, keepdims=True)  # (B, H, W, 1)
        DP = np.clip(DP, -1, 1)
        E = np.degrees(np.arccos(DP))  # (B, H, W, 1)
        E = E * gt_norm_mask
        target_path = '%s/%08d_pred_error_%d.jpg' % (args.exp_vis_dir, total_iter, i)
        plt.imsave(target_path, E[0, :, :, 0], vmin=0, vmax=60, cmap='jet')
