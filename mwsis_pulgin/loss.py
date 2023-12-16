from pkg_resources import working_set
import torch
from torch import nn as nn
from torch.autograd import Variable
from mmdet.models import LOSSES
from mmdet.models.losses.utils import weight_reduce_loss
import torch.distributed as dist
from mmcv.runner.dist_utils import get_dist_info
import torch.nn.functional as F


@LOSSES.register_module()
class LovaszLoss_(nn.Module):
    def __init__(self,
                 ignore=0,
                 per_image=False,
                 reduction='mean',
                 loss_weight=1.0):
        super().__init__()
        self.ignore = ignore
        self.per_image = per_image
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                classes='present',
                per_image=False,
                ignore_override=None):
        """
        Multi-class Lovasz-Softmax loss
        pred: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
                Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
        target: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
        classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        per_image: compute the loss per image instead of per batch
        ignore: void class labels
        """
        reduction = (
            reduction_override if reduction_override else self.reduction)
        per_image = (per_image if per_image else self.per_image)
        ignore = (ignore_override if ignore_override else self.ignore)

        # 默认不运行, 此处self,mean(x)中的x为二维列表, 不同batch&不同类别下所有点的损失函数, 若运行该分支，可能需要修改代码
        if per_image:
            raise NotImplementedError
            # loss = self.mean(self.lovasz_softmax_flat(*self.flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
            #                  for prob, lab in zip(pred, target))
        if pred.dim() == 2:
            flatten_probas, flatten_labels = pred, target
        else:
            flatten_probas, flatten_labels = self.flatten_probas(
                pred, target, ignore)
        loss, class_presence = self.lovasz_softmax_flat(
            flatten_probas, flatten_labels)
        if classes == 'present':
            weight = class_presence.float()
            avg_factor = weight.sum()
        elif isinstance(classes, (list, tuple)):
            weight = torch.zeros_like(loss)
            weight[classes] = 1
            avg_factor = weight.sum()
        else:
            weight = torch.ones_like(loss)
        loss = self.loss_weight * \
            weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    def lovasz_softmax_flat(self, probas, labels):
        """
        Multi-class Lovasz-Softmax loss
        probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
        labels: [P] Tensor, ground truth labels (between 0 and C - 1)
        classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        """
        if probas.numel() == 0:
            # only void pixels, the gradients should be 0
            return probas * 0.
        P, C = probas.shape
        labels_onehot = F.one_hot(labels, C).float()
        class_errors = (labels_onehot - probas).abs()
        with torch.no_grad():
            rank, world_size = get_dist_info()
            # if world_size == 1:
            #     return
            P = torch.tensor(P, device='cuda')
            P_list = [P.clone() for _ in range(world_size)]
            if world_size > 1:
                dist.all_gather(P_list, P)
                P_max = torch.tensor(P_list).max()
                send = class_errors.new_empty([P_max, 2, C])
                send[:P, 0] = labels_onehot
                send[:P, 1] = class_errors
                recv_list = [
                    torch.empty_like(send) for _ in range(world_size)
                ]
                # gather all result part
                dist.all_gather(recv_list, send)
                recv_list = [r[:p] for r, p in zip(recv_list, P_list)]
                recv = torch.cat(recv_list)
                global_labels_onehot = recv[:, 0]
                global_class_errors = recv[:, 1]
            else:
                global_labels_onehot = labels_onehot
                global_class_errors = class_errors

            global_class_presence = global_labels_onehot.sum(dim=0) > 0
            global_class_errors_presence = global_class_errors[:,
                                                               global_class_presence]
            global_labels_onehot_presence = global_labels_onehot[:,
                                                                 global_class_presence]
            global_argsort = global_class_errors_presence.argsort(
                dim=0, descending=True)
            global_labels_onehot_sorted = global_labels_onehot_presence.gather(
                dim=0, index=global_argsort)
            global_lovasz_grad_sorted = self.lovasz_grad(
                global_labels_onehot_sorted)
            global_lovasz_grad = torch.zeros_like(global_lovasz_grad_sorted)
            global_lovasz_grad.scatter_(
                dim=0, index=global_argsort, src=global_lovasz_grad_sorted)
            local_slice_begin = sum(P_list[:rank])
            local_slice_end = local_slice_begin + P_list[rank]
            lovasz_grad = global_lovasz_grad[local_slice_begin: local_slice_end]
        losses = class_errors.new_zeros([C])
        losses[global_class_presence] = (
            class_errors[:, global_class_presence] * lovasz_grad * world_size).sum(0)
        return losses, global_class_presence

    def flatten_probas(self, probas, labels, ignore=None):
        B, C = probas.shape[:2]
        probas = probas.view(B, C, -1).permute(
            0, 2, 1).reshape(-1, C)
        labels = labels.view(-1)
        if ignore is None:
            return probas, labels
        valid = (labels != ignore)
        vprobas = probas[valid]
        vlabels = labels[valid]
        return vprobas, vlabels

    def lovasz_grad(self, gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = gt_sorted.shape[0]
        gts = gt_sorted.sum(dim=0, keepdim=True)
        intersection = gts - gt_sorted.float().cumsum(dim=0)
        union = gts + (1 - gt_sorted).float().cumsum(dim=0)
        jaccard = 1. - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[:-1]
        return jaccard
