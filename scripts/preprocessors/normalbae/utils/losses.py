import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


# compute loss
class compute_loss(nn.Module):
    def __init__(self, args):
        """args.loss_fn can be one of following:
            - L1            - L1 loss (no uncertainty)
            - L2            - L2 loss (no uncertainty)
            - AL            - Angular loss (no uncertainty)
            - NLL_vMF       - NLL of vonMF distribution
            - NLL_ours      - NLL of Angular vonMF distribution
            - UG_NLL_vMF    - NLL of vonMF distribution (+ pixel-wise MLP + uncertainty-guided sampling)
            - UG_NLL_ours   - NLL of Angular vonMF distribution (+ pixel-wise MLP + uncertainty-guided sampling)
        """
        super(compute_loss, self).__init__()
        self.loss_type = args.loss_fn
        if self.loss_type in ['L1', 'L2', 'AL', 'NLL_vMF', 'NLL_ours']:
            self.loss_fn = self.forward_R
        elif self.loss_type in ['UG_NLL_vMF', 'UG_NLL_ours']:
            self.loss_fn = self.forward_UG
        else:
            raise Exception('invalid loss type')

    def forward(self, *args):
        return self.loss_fn(*args)

    def forward_R(self, norm_out, gt_norm, gt_norm_mask):
        pred_norm, pred_kappa = norm_out[:, 0:3, :, :], norm_out[:, 3:, :, :]

        if self.loss_type == 'L1':
            l1 = torch.sum(torch.abs(gt_norm - pred_norm), dim=1, keepdim=True)
            loss = torch.mean(l1[gt_norm_mask])

        elif self.loss_type == 'L2':
            l2 = torch.sum(torch.square(gt_norm - pred_norm), dim=1, keepdim=True)
            loss = torch.mean(l2[gt_norm_mask])

        elif self.loss_type == 'AL':
            dot = torch.cosine_similarity(pred_norm, gt_norm, dim=1)

            valid_mask = gt_norm_mask[:, 0, :, :].float() \
                         * (dot.detach() < 0.999).float() \
                         * (dot.detach() > -0.999).float()
            valid_mask = valid_mask > 0.0

            al = torch.acos(dot[valid_mask])
            loss = torch.mean(al)

        elif self.loss_type == 'NLL_vMF':
            dot = torch.cosine_similarity(pred_norm, gt_norm, dim=1)

            valid_mask = gt_norm_mask[:, 0, :, :].float() \
                         * (dot.detach() < 0.999).float() \
                         * (dot.detach() > -0.999).float()
            valid_mask = valid_mask > 0.0

            dot = dot[valid_mask]
            kappa = pred_kappa[:, 0, :, :][valid_mask]

            loss_pixelwise = - torch.log(kappa) \
                             - (kappa * (dot - 1)) \
                             + torch.log(1 - torch.exp(- 2 * kappa))
            loss = torch.mean(loss_pixelwise)

        elif self.loss_type == 'NLL_ours':
            dot = torch.cosine_similarity(pred_norm, gt_norm, dim=1)

            valid_mask = gt_norm_mask[:, 0, :, :].float() \
                         * (dot.detach() < 0.999).float() \
                         * (dot.detach() > -0.999).float()
            valid_mask = valid_mask > 0.0

            dot = dot[valid_mask]
            kappa = pred_kappa[:, 0, :, :][valid_mask]

            loss_pixelwise = - torch.log(torch.square(kappa) + 1) \
                             + kappa * torch.acos(dot) \
                             + torch.log(1 + torch.exp(-kappa * np.pi))
            loss = torch.mean(loss_pixelwise)

        else:
            raise Exception('invalid loss type')

        return loss


    def forward_UG(self, pred_list, coord_list, gt_norm, gt_norm_mask):
        loss = 0.0
        for (pred, coord) in zip(pred_list, coord_list):
            if coord is None:
                pred = F.interpolate(pred, size=[gt_norm.size(2), gt_norm.size(3)], mode='bilinear', align_corners=True)
                pred_norm, pred_kappa = pred[:, 0:3, :, :], pred[:, 3:, :, :]

                if self.loss_type == 'UG_NLL_vMF':
                    dot = torch.cosine_similarity(pred_norm, gt_norm, dim=1)

                    valid_mask = gt_norm_mask[:, 0, :, :].float() \
                                * (dot.detach() < 0.999).float() \
                                * (dot.detach() > -0.999).float()
                    valid_mask = valid_mask > 0.5

                    # mask
                    dot = dot[valid_mask]
                    kappa = pred_kappa[:, 0, :, :][valid_mask]

                    loss_pixelwise = - torch.log(kappa) \
                                     - (kappa * (dot - 1)) \
                                     + torch.log(1 - torch.exp(- 2 * kappa))
                    loss = loss + torch.mean(loss_pixelwise)

                elif self.loss_type == 'UG_NLL_ours':
                    dot = torch.cosine_similarity(pred_norm, gt_norm, dim=1)

                    valid_mask = gt_norm_mask[:, 0, :, :].float() \
                                * (dot.detach() < 0.999).float() \
                                * (dot.detach() > -0.999).float()
                    valid_mask = valid_mask > 0.5

                    dot = dot[valid_mask]
                    kappa = pred_kappa[:, 0, :, :][valid_mask]

                    loss_pixelwise = - torch.log(torch.square(kappa) + 1) \
                                     + kappa * torch.acos(dot) \
                                     + torch.log(1 + torch.exp(-kappa * np.pi))
                    loss = loss + torch.mean(loss_pixelwise)

                else:
                    raise Exception

            else:
                # coord: B, 1, N, 2
                # pred: B, 4, N
                gt_norm_ = F.grid_sample(gt_norm, coord, mode='nearest', align_corners=True)  # (B, 3, 1, N)
                gt_norm_mask_ = F.grid_sample(gt_norm_mask.float(), coord, mode='nearest', align_corners=True)  # (B, 1, 1, N)
                gt_norm_ = gt_norm_[:, :, 0, :]  # (B, 3, N)
                gt_norm_mask_ = gt_norm_mask_[:, :, 0, :] > 0.5  # (B, 1, N)

                pred_norm, pred_kappa = pred[:, 0:3, :], pred[:, 3:, :]

                if self.loss_type == 'UG_NLL_vMF':
                    dot = torch.cosine_similarity(pred_norm, gt_norm_, dim=1)  # (B, N)

                    valid_mask = gt_norm_mask_[:, 0, :].float() \
                                 * (dot.detach() < 0.999).float() \
                                 * (dot.detach() > -0.999).float()
                    valid_mask = valid_mask > 0.5

                    dot = dot[valid_mask]
                    kappa = pred_kappa[:, 0, :][valid_mask]

                    loss_pixelwise = - torch.log(kappa) \
                                     - (kappa * (dot - 1)) \
                                     + torch.log(1 - torch.exp(- 2 * kappa))
                    loss = loss + torch.mean(loss_pixelwise)

                elif self.loss_type == 'UG_NLL_ours':
                    dot = torch.cosine_similarity(pred_norm, gt_norm_, dim=1)  # (B, N)

                    valid_mask = gt_norm_mask_[:, 0, :].float() \
                                 * (dot.detach() < 0.999).float() \
                                 * (dot.detach() > -0.999).float()
                    valid_mask = valid_mask > 0.5

                    dot = dot[valid_mask]
                    kappa = pred_kappa[:, 0, :][valid_mask]

                    loss_pixelwise = - torch.log(torch.square(kappa) + 1) \
                                     + kappa * torch.acos(dot) \
                                     + torch.log(1 + torch.exp(-kappa * np.pi))
                    loss = loss + torch.mean(loss_pixelwise)

                else:
                    raise Exception
        return loss
