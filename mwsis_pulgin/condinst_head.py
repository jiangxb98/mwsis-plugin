import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import ConvModule, Scale
from mmcv.runner import BaseModule, force_fp32
from mmcv.image import tensor2imgs
from skimage import color
from mmcv.ops.ball_query import ball_query
from mmdet.core import distance2bbox, multi_apply, reduce_mean
from mmcv.ops.nms import batched_nms
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.builder import HEADS, build_loss
from .ops import pairwise_nlog
import mmcv

INF = 1e8

def computer_sim_loss(teatch_logits, student_logits):
    log_t_fg_prob = F.logsigmoid(teatch_logits)
    log_s_fg_prob = F.logsigmoid(student_logits)
    log_t_bg_prob = F.logsigmoid(-teatch_logits)
    log_s_bg_prob = F.logsigmoid(-student_logits)
    log_same_fg_prob = log_t_fg_prob + log_s_fg_prob
    log_same_bg_prob = log_t_bg_prob + log_s_bg_prob
    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
    log_same_prob = torch.log(
                    torch.exp(log_same_fg_prob - max_) +
                    torch.exp(log_same_bg_prob - max_)
                    ) + max_
    return -log_same_prob

def nms_with_others(multi_bboxes,
                    multi_scores,
                    score_thr,
                    nms_cfg,
                    max_num=-1,
                    score_factors=None,
                    others=None):
    num_pos = multi_scores.size(0)
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4)

    scores = multi_scores[:, :-1]

    positions = torch.arange(num_pos, dtype=torch.long, device=scores.device)
    positions = positions.view(-1, 1).expand_as(scores)

    labels = torch.arange(num_classes, dtype=torch.long, device=scores.device)
    labels = labels.view(1, -1).expand_as(scores)

    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    positions = positions.reshape(-1)
    labels = labels.reshape(-1)

    if torch.onnx.is_in_onnx_export():
        raise NotImplementedError

    valid_mask = scores > score_thr
    # multiply score_factor after threshold to preserve more bboxes, improve
    # mAP by 1% for YOLOv3
    if score_factors is not None:
        # expand the shape to match original shape of score
        score_factors = score_factors.view(-1, 1).expand(
            multi_scores.size(0), num_classes)
        score_factors = score_factors.reshape(-1)
        scores = scores * score_factors

    # NonZero not supported  in TensorRT
    inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
    bboxes = bboxes[inds]
    scores = scores[inds]
    positions = positions[inds]
    labels = labels[inds]

    if bboxes.numel() == 0:
        dets = torch.cat([bboxes, scores[:, None]], -1)
        if others is not None:
            others = [item[positions] for item in others]
        return dets, labels, others

    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    if others is not None:
        _others = []
        for item in others:
            assert item.size(0) == num_pos
            _others.append(item[positions][keep])
        others = _others

    return dets, labels[keep], others


def compute_pairwise_term(mask_logits, pairwise_size, pairwise_dilation):
    assert mask_logits.dim() == 4

    log_fg_prob = F.logsigmoid(mask_logits)
    log_bg_prob = F.logsigmoid(-mask_logits)

    log_fg_prob_unfold = unfold_wo_center(
        log_fg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )
    log_bg_prob_unfold = unfold_wo_center(
        log_bg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )

    # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
    # we compute the the probability in log space to avoid numerical instability
    log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold
    log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold

    # this equation is equal to log(p_i * p_j + (1 - p_i) * (1 - p_j))
    # max is used to prevent overflow
    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)  #
    log_same_prob = torch.log(
        torch.exp(log_same_fg_prob - max_) +
        torch.exp(log_same_bg_prob - max_)
    ) + max_

    return -log_same_prob[:, 0]


def dice_coefficient(x, target):
    """
    Dice Loss: 1 - 2 * (intersection(A, B) / (A^2 + B^2))
    :param x:
    :param target:
    :return:
    """
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss


def compute_project_term(mask_scores, gt_bitmasks):
    mask_losses_y = dice_coefficient(
        mask_scores.max(dim=2, keepdim=True)[0],
        gt_bitmasks.max(dim=2, keepdim=True)[0]
    )
    mask_losses_x = dice_coefficient(
        mask_scores.max(dim=3, keepdim=True)[0],
        gt_bitmasks.max(dim=3, keepdim=True)[0]
    )
    return (mask_losses_x + mask_losses_y).mean()


def aligned_bilinear(tensor, factor, align_corners=True):
    assert tensor.dim() == 4
    assert factor >= 1
    assert isinstance(factor, int)

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=align_corners
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )
    return tensor[:, :, :oh - 1, :ow - 1]


def get_original_image(img, img_meta):
    """

    :param img(Tensor):  the image with pading [3, h, w]
    :param img_meta(dict): information about the image
    :return: original_img(Tensor)
    """
    original_shape = img_meta["img_shape"]
    original_shape_img = img[:, :original_shape[0], :original_shape[1]]
    img_norm_cfg = img_meta["img_norm_cfg"]
    original_img = tensor2imgs(original_shape_img.unsqueeze(0), mean=img_norm_cfg["mean"], std=img_norm_cfg["std"],
                               to_rgb=img_norm_cfg["to_rgb"])[0]  # in BGR format [h w c]
    original_img = torch.tensor(original_img[:, :, ::-1].copy()).permute(2, 0, 1)  # BGR[h w c] to RGB tensor [c h w]
    if False:
        import cv2
        cv2.imwrite('img.jpg', original_img)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,10))
        plt.axis('off') # 隐藏坐标轴
        plt.imshow(original_img[:, :, ::-1])  # 当前的图表fig和子图ax可以使用plt.gcf()和plt.gca()获得
        plt.savefig('savefig_example.png')
    original_img = original_img.float().to(img.device)

    return original_img

    # cv2.imwrite("show/cv_{}".format(img_meta["filename"].split("/")[-1]), original_img)
    # Image.fromarray(original_img).save("show/pil_{}".format(img_meta["filename"].split("/")[-1]))


def unfold_wo_center(x, kernel_size, dilation):
    """
    :param x: [N, C, H, W]
    :param kernel_size: k
    :param dilation:
    :return: [N, C, K^2-1, H, W]
    """
    assert x.dim() == 4
    assert kernel_size % 2 == 1

    # using SAME padding
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(
        x, kernel_size=kernel_size,
        padding=padding,
        dilation=dilation
    )

    unfolded_x = unfolded_x.reshape(
        x.size(0), x.size(1), -1, x.size(2), x.size(3)
    )

    # remove the center pixels
    size = kernel_size ** 2
    unfolded_x = torch.cat((unfolded_x[:, :, :size // 2], unfolded_x[:, :, size // 2 + 1:]), dim=2)

    return unfolded_x


def get_image_color_similarity(image, mask, pairwise_size, pairwise_dilation, points_flag=False):
    """
    \
    :param self:
    :param image: [1, 3, H, W]
    :param mask: [H, W]
    :param pairwise_size: k
    :param pairwise_dilation: d
    :return:[1, 8, H, W]
    """
    assert image.dim() == 4
    assert image.size(0) == 1

    unfolded_image = unfold_wo_center(
        image, kernel_size=pairwise_size, dilation=pairwise_dilation
    )

    diff = image.unsqueeze(2) - unfolded_image  # (1,3,8,320,480)

    if points_flag:
        similarity = torch.exp(-torch.norm(diff, dim=1, p=2))  # (1,8,320,480)
    else:
        similarity = torch.exp(-torch.norm(diff, dim=1) * 0.5)  # (1,8,320,480)

    unfolded_weight = unfold_wo_center(
        mask.unsqueeze(0).unsqueeze(0),
        kernel_size=pairwise_size, dilation=pairwise_dilation
    )[:, 0, :, :, :]

    return similarity * unfolded_weight


@HEADS.register_module()
class CondInstBoxHead(AnchorFreeHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=True,
                 center_sample_radius=1.5,
                 norm_on_bbox=True,
                 centerness_on_reg=False,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='conv_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        self.loss_centerness = build_loss(loss_centerness)

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def forward(self, feats, top_module):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            top_module (nn.Module): Generate dynamic parameters from FCOS
                regression branch.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level, \
                    each is a 4D-tensor, the channel number is \
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each \
                    scale level, each is a 4D-tensor, the channel number is \
                    num_points * 4.
                centernesses (list[Tensor]): centerness for each scale level, \
                    each is a 4D-tensor, the channel number is num_points * 1.
                param_preds (list[Tensor]): dynamic parameters generated from \
                    each scale level, each is a 4-D-tensor, the channel number \
                    is decided by top_module.
        """
        return multi_apply(self.forward_single, feats, self.scales,
                           self.strides, top_module=top_module)

    def forward_single(self, x, scale, stride, top_module):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.
            top_module (nn.Module): Exteral input module. #---------------

        Returns:
            tuple: scores for each class, bbox predictions and centerness \
                predictions of input feature maps.
        """
        cls_score, bbox_pred, cls_feat, reg_feat = \
            super(CondInstBoxHead, self).forward_single(x)
        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            bbox_pred = F.relu(bbox_pred)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        param_pred = top_module(reg_feat)
        return cls_score, bbox_pred, centerness, param_pred

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None,
             gt_masks=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            centernesses (list[Tensor]): centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes, bbox_preds[0].dtype, bbox_preds[0].device)  # image pixel cooresponding to feature map
        labels, bbox_targets, gt_inds = \
            self.get_targets(all_level_points, gt_bboxes, gt_labels, gt_masks)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        flatten_gt_inds = torch.cat(gt_inds)
        flatten_img_inds = []  # image index
        flatten_level_inds = []  # fpn level index
        for i, featmap_size in enumerate(featmap_sizes):
            H, W = featmap_size
            img_inds = torch.arange(num_imgs, device=bbox_preds[0].device)
            flatten_img_inds.append(img_inds.repeat_interleave(H * W))
            flatten_level_inds.append(torch.full(
                (num_imgs * H * W,), i, device=bbox_preds[0].device).long())
        flatten_img_inds = torch.cat(flatten_img_inds)
        flatten_level_inds = torch.cat(flatten_level_inds)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes  # bg_class是background
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)  # pos_inds pos sample index
        assert (((flatten_gt_inds!=-1).nonzero().reshape(-1))==pos_inds).all()
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        # if cls_scores[0].requires_grad:
        # reduce_mean_num_pos = reduce_mean(num_pos.double())
        num_pos = max(reduce_mean(num_pos.double()), 1.0)
        # num_pos = num_pos.type(flatten_cls_scores.type())
        # num_pos = max(num_pos, 1.0)
        loss_cls = self.loss_cls(flatten_cls_scores, 
            flatten_labels.long(), avg_factor=num_pos)
        
        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        # bug
        # reduce_mean_nums = pos_centerness_targets.sum().detach()
        # if cls_scores[0].requires_grad:
        #     centerness_denorm = max(reduce_mean(reduce_mean_nums.double()), 1e-6)
        # else:
        #     centerness_denorm = reduce_mean_nums
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach().double()), 1e-6)
        # centerness_denorm = centerness_denorm.type(flatten_cls_scores.type())

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()
            pos_decoded_bbox_preds = None
            pos_decoded_target_preds = None

        losses = dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness)
        return (losses, flatten_points, flatten_level_inds, flatten_img_inds,
                flatten_gt_inds, pos_decoded_bbox_preds, pos_decoded_target_preds)

    def get_targets(self, points, gt_bboxes_list, gt_labels_list, gt_masks=None):
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image.
        if gt_masks is None:
            labels_list, bbox_targets_list, gt_inds_list = multi_apply(
                self._get_target_single,
                gt_bboxes_list,
                gt_labels_list,
                points=concat_points,
                regress_ranges=concat_regress_ranges,
                num_points_per_lvl=num_points)
        else:
            labels_list, bbox_targets_list, gt_inds_list = multi_apply(
                self._get_target_single,
                gt_bboxes_list,
                gt_labels_list,
                gt_masks,
                points=concat_points,
                regress_ranges=concat_regress_ranges,
                num_points_per_lvl=num_points,
                )
        cum = 0
        for gt_inds, gt_bboxes in zip(gt_inds_list, gt_bboxes_list):
            gt_inds[gt_inds != -1] += cum
            cum += gt_bboxes.size(0)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        gt_inds_list = [gt_inds.split(num_points, 0) 
                        for gt_inds in gt_inds_list]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_gt_inds = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
            concat_lvl_gt_inds.append(
                torch.cat([gt_inds[i] for gt_inds in gt_inds_list]))
        return (concat_lvl_labels, concat_lvl_bbox_targets,
                concat_lvl_gt_inds)

    def _get_target_single(self, gt_bboxes, gt_labels, gt_masks=None, points=None, regress_ranges=None,
                           num_points_per_lvl=None):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes).long(), \
                   gt_bboxes.new_zeros((num_points, 4)) , gt_labels.new_full((num_points,), -1).long()

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
                gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            if gt_masks is None:
                center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
                center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            elif gt_masks is not None:
                h, w = gt_masks.height, gt_masks.width
                masks = gt_masks.to_tensor(
                    dtype=torch.bool, device=gt_bboxes.device)
                yys = torch.arange(
                    0, h, dtype=torch.float32, device=masks.device)
                xxs = torch.arange(
                    0, w, dtype=torch.float32, device=masks.device)
                # m00/m10/m01 represent the moments of a contour
                # centroid is computed by m00/m10 and m00/m01
                m00 = masks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
                m10 = (masks * xxs).sum(dim=-1).sum(dim=-1)
                m01 = (masks * yys[:, None]).sum(dim=-1).sum(dim=-1)
                center_xs = m10 / m00
                center_ys = m01 / m00

                center_xs = center_xs[None].expand(num_points, num_gts)
                center_ys = center_ys[None].expand(num_points, num_gts)
            
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
                (max_regress_distance >= regress_ranges[..., 0])
                & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        min_area_inds[min_area == INF] = -1

        return labels, bbox_targets, min_area_inds

    def simple_test(self, feats, top_module, img_metas, rescale=False):
        outs = self.forward(feats, top_module)
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale)
        return results_list

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   param_preds,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_points * 1, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.prior_generator.grid_priors(
            featmap_sizes, bbox_preds[0].dtype, bbox_preds[0].device)

        cls_score_list = [cls_scores[i].detach() for i in range(num_levels)]
        bbox_pred_list = [bbox_preds[i].detach() for i in range(num_levels)]
        centerness_pred_list = [
            centernesses[i].detach() for i in range(num_levels)
        ]
        if torch.onnx.is_in_onnx_export():
            assert len(
                img_metas
            ) == 1, 'Only support one input image while in exporting to ONNX'
            img_shapes = img_metas[0]['img_shape_for_onnx']
        else:
            img_shapes = [
                img_metas[i]['img_shape']
                for i in range(cls_scores[0].shape[0])
            ]
        scale_factors = [
            img_metas[i]['scale_factor'] for i in range(cls_scores[0].shape[0])
        ]
        result_list = self._get_bboxes(cls_score_list, bbox_pred_list,
                                       centerness_pred_list, param_preds,
                                       mlvl_points, img_shapes, scale_factors,
                                       cfg, rescale, with_nms)
        return result_list

    def _get_bboxes(self,
                    cls_scores,
                    bbox_preds,
                    centernesses,
                    param_preds,
                    mlvl_points,
                    img_shapes,
                    scale_factors,
                    cfg,
                    rescale=False,
                    with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (N, num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (N, num_points, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shapes (list[tuple[int]]): Shape of the input image,
                list[(height, width, 3)].
            scale_factors (list[ndarray]): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        device = cls_scores[0].device
        batch_size = cls_scores[0].shape[0]
        # convert to tensor to keep tracing
        nms_pre_tensor = torch.tensor(
            cfg.get('nms_pre', -1), device=device, dtype=torch.long)
        mlvl_coors = []
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        mlvl_param_pred = []

        for cls_score, bbox_pred, centerness, param_pred, points in zip(
                cls_scores, bbox_preds, centernesses, param_preds, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(0, 2, 3, 1).reshape(
                batch_size, -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(0, 2, 3,
                                            1).reshape(batch_size,
                                                       -1).sigmoid()

            bbox_pred = bbox_pred.permute(0, 2, 3,
                                          1).reshape(batch_size, -1, 4)
            param_num = param_pred.size(1)
            param_pred = param_pred.permute(0, 2, 3, 1).reshape(batch_size,
                                                                -1, param_num)
            points = points.expand(batch_size, -1, 2)
            # Get top-k prediction
            from mmdet.core.export import get_k_for_topk
            nms_pre = get_k_for_topk(nms_pre_tensor, bbox_pred.shape[1])
            if nms_pre > 0:
                max_scores, _ = (scores * centerness[..., None]).max(-1)
                _, topk_inds = max_scores.topk(nms_pre)
                batch_inds = torch.arange(batch_size).view(
                    -1, 1).expand_as(topk_inds).long()
                # Avoid onnx2tensorrt issue in https://github.com/NVIDIA/TensorRT/issues/1134 # noqa: E501
                if torch.onnx.is_in_onnx_export():
                    raise NotImplementedError(
                        "CondInst doesn't support ONNX currently")
                else:
                    points = points[batch_inds, topk_inds, :]
                    bbox_pred = bbox_pred[batch_inds, topk_inds, :]
                    scores = scores[batch_inds, topk_inds, :]
                    centerness = centerness[batch_inds, topk_inds]
                    param_pred = param_pred[batch_inds, topk_inds, :]

            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shapes)
            mlvl_coors.append(points)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
            mlvl_param_pred.append(param_pred)

        batch_lvl_inds = torch.cat(
            [torch.full_like(ctr, i).long()
             for i, ctr in enumerate(mlvl_centerness)], dim=1)
        batch_mlvl_coors = torch.cat(mlvl_coors, dim=1)
        batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
        if rescale:
            batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(
                np.array(scale_factors)).unsqueeze(1)
        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
        batch_mlvl_centerness = torch.cat(mlvl_centerness, dim=1)
        batch_mlvl_param_pred = torch.cat(mlvl_param_pred, dim=1)

        # Replace multiclass_nms with ONNX::NonMaxSuppression in deployment
        if torch.onnx.is_in_onnx_export() and with_nms:
            raise NotImplementedError(
                "CondInst doesn't support ONNX currently")
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = batch_mlvl_scores.new_zeros(batch_size,
                                              batch_mlvl_scores.shape[1], 1)
        batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)

        if with_nms:
            det_results = []
            for (lvl_inds, mlvl_coors, mlvl_bboxes, mlvl_scores, mlvl_centerness,
                 mlvl_param_pred) in zip(batch_lvl_inds, batch_mlvl_coors,
                                         batch_mlvl_bboxes, batch_mlvl_scores,
                                         batch_mlvl_centerness, batch_mlvl_param_pred):
                det_bbox, det_label, others = nms_with_others(
                    mlvl_bboxes,
                    mlvl_scores,
                    cfg.score_thr,
                    cfg.nms,
                    cfg.max_per_img,
                    score_factors=mlvl_centerness,
                    others=[mlvl_param_pred,
                            mlvl_coors,
                            lvl_inds]
                )
                outputs = (det_bbox, det_label) + tuple(others)
                det_results.append(outputs)
        else:
            det_results = [
                tuple(mlvl_bs)
                for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores,
                                   batch_mlvl_centerness,
                                   batch_mlvl_param_pred,
                                   batch_mlvl_coors,
                                   batch_lvl_inds)
            ]
        return det_results

    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = (
                                         left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                                         top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)


@HEADS.register_module()
class CondInstSegmHead(BaseModule):

    def __init__(self,
                 num_classes,
                 in_channels=256,
                 in_stride=8,
                 stacked_convs=2,
                 feat_channels=128,
                 loss_segm=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 norm_cfg=dict(
                     type='BN',
                     requires_grad=True),
                 init_cfg=dict(
                     type='Kaiming',
                     layer="Conv2d",
                     distribution='uniform',
                     a=1,
                     mode='fan_in',
                     nonlinearity='leaky_relu',
                     override=dict(
                         type='Kaiming',
                         name='segm_conv',
                         bias_prob=0.01))):
        super(CondInstSegmHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.in_stride = in_stride
        self.stacked_convs = stacked_convs
        self.feat_channels = feat_channels
        self.loss_segm = build_loss(loss_segm)
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self._init_layers()

    def _init_layers(self):
        segm_branch = []
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            segm_branch.append(ConvModule(
                chn,
                self.feat_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=self.norm_cfg))
        self.segm_branch = nn.Sequential(*segm_branch)
        self.segm_conv = nn.Conv2d(
            self.feat_channels, self.num_classes, kernel_size=1)

    def forward(self, x):
        return self.segm_conv(self.segm_branch(x))

    @force_fp32(apply_to=('segm_pred',))
    def loss(self, segm_pred, gt_masks, gt_labels):
        semantic_targets = self.get_targets(gt_masks, gt_labels)
        semantic_targets = semantic_targets.flatten()
        num_pos = (semantic_targets != self.num_classes).sum().float()
        num_pos = num_pos.clamp(min=1.)

        segm_pred = segm_pred.permute(0, 2, 3, 1)
        segm_pred = segm_pred.flatten(end_dim=2)
        loss_segm = self.loss_segm(
            segm_pred,
            semantic_targets,
            avg_factor=num_pos)
        return dict(loss_segm=loss_segm)

    def get_targets(self, gt_masks, gt_labels):
        semantic_targets = []
        for cur_gt_masks, cur_gt_labels in zip(gt_masks, gt_labels):
            h, w = cur_gt_masks.size()[-2:]
            areas = torch.sum(cur_gt_masks, dim=(1, 2), keepdim=True)
            areas = areas.repeat(1, h, w)
            areas[cur_gt_masks == 0] = INF
            min_areas, inds = torch.min(areas, dim=0, keepdim=True)

            cur_gt_labels = cur_gt_labels[:, None, None].repeat(1, h, w)
            per_img_targets = torch.gather(cur_gt_labels, 0, inds)
            per_img_targets[min_areas == INF] = self.num_classes
            semantic_targets.append(per_img_targets)

        stride = self.in_stride
        semantic_targets = torch.cat(semantic_targets, dim=0)
        semantic_targets = semantic_targets[:,
                                            stride // 2::stride, stride // 2::stride]
        return semantic_targets


@HEADS.register_module()
class CondInstMaskBranch(BaseModule):

    def __init__(self,
                 in_channels=256,
                 in_indices=[0, 1, 2],
                 strides=[8, 16, 32],
                 branch_convs=4,
                 branch_channels=128,
                 branch_out_channels=8,
                 norm_cfg=dict(
                     type='BN',
                     requires_grad=True),
                 init_cfg=dict(
                     type='Kaiming',
                     layer="Conv2d",
                     distribution='uniform',
                     a=1,
                     mode='fan_in',
                     nonlinearity='leaky_relu')):
        super(CondInstMaskBranch, self).__init__(init_cfg)
        self.in_channels = in_channels
        assert len(in_indices) == len(strides)
        assert in_indices[0] == 0
        self.in_indices = in_indices
        self.strides = strides
        self.branch_convs = branch_convs
        self.branch_channels = branch_channels
        self.branch_out_channels = branch_out_channels
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.refines = nn.ModuleList()
        for _ in self.in_indices:
            self.refines.append(ConvModule(
                self.in_channels,
                self.branch_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=self.norm_cfg))

        mask_branch = []
        for _ in range(self.branch_convs):
            mask_branch.append(ConvModule(
                self.branch_channels,
                self.branch_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=self.norm_cfg))
        mask_branch.append(
            nn.Conv2d(self.branch_channels, self.branch_out_channels, 1))
        self.mask_branch = nn.Sequential(*mask_branch)

    def forward(self, x):
        mask_stride = self.strides[self.in_indices[0]]
        mask_x = self.refines[0](x[self.in_indices[0]])
        for i in range(1, len(self.in_indices)):
            stride = self.strides[i]
            assert stride % mask_stride == 0
            p_x = self.refines[i](x[self.in_indices[i]])
            p_x = aligned_bilinear(p_x, stride // mask_stride)
            mask_x = mask_x + p_x
        return self.mask_branch(mask_x)


@HEADS.register_module()
class CondInstMaskHead(BaseModule):

    def __init__(self,
                 in_channels=8,
                 in_stride=8,
                 out_stride=4,
                 dynamic_convs=3,
                 dynamic_channels=8,
                 disable_rel_coors=False,
                 bbox_head_channels=256,
                 sizes_of_interest=[64, 128, 256, 512, 1024],
                 max_proposals=500,
                 topk_per_img=-1,
                 boxinst_enabled=False,
                 run_seg=False,
                 bottom_pixels_removed=10,
                 pairwise_size=3,
                 pairwise_dilation=2,
                 pairwise_color_thresh=0.3,
                 pairwise_warmup=10000,
                 start_kd_loss_iters=10000,
                 points_enabled=False,
                 use_enlarge_bbox=False,
                 pairwise_distance_thresh=0.9,  # exp(-0.1)=0.9048
                 num_classes=3,
                 norm_cfg=dict(
                     type='BN',
                     requires_grad=True),
                 init_cfg=dict(
                     type='Normal',
                     layer="Conv2d",
                     std=0.01,
                     bias=0),
                 geometry_loss_enabled=False,
                 use_weight_loss=False,
                 kd_loss_weight=0.2,
                 pairwise_geo_thresh=0.2,
                 geometry_loss_weight=0.2):
        super(CondInstMaskHead, self).__init__(init_cfg)
        self.in_channels = in_channels
        assert in_stride >= out_stride
        assert in_stride % out_stride == 0
        self.in_stride = in_stride
        self.out_stride = out_stride
        assert dynamic_channels > 1
        self.dynamic_convs = dynamic_convs
        self.dynamic_channels = dynamic_channels
        self.disable_rel_coors = disable_rel_coors
        self.use_weight_loss = use_weight_loss
        self.kd_loss_weight = kd_loss_weight
        self.pairwise_geo_thresh = pairwise_geo_thresh
        self.geometry_loss_enabled = geometry_loss_enabled
        self.geometry_loss_weight = geometry_loss_weight
        dy_weights, dy_biases = [], []
        dynamic_in_channels = in_channels if disable_rel_coors else in_channels + 2  #16是mask branch的输出，16+2，2是坐标x,y
        for i in range(dynamic_convs):
            in_chn = dynamic_in_channels if i == 0 else dynamic_channels
            out_chn = 1 if i == dynamic_convs - 1 else dynamic_channels
            dy_weights.append(in_chn * out_chn)
            dy_biases.append(out_chn)
        self.dy_weights = dy_weights  # 18*8=144,8*8=64,8*1=8
        self.dy_biases = dy_biases  # 8,8,1
        self.num_gen_params = sum(dy_weights) + sum(dy_biases)
        self.bbox_head_channels = bbox_head_channels

        self.register_buffer("sizes_of_interest",
                             torch.tensor(sizes_of_interest))
        assert max_proposals == -1 or topk_per_img == -1, \
            'max_proposals and topk_per_img cannot be used at the same time'
        self.max_proposals = max_proposals
        self.topk_per_img = topk_per_img

        self.boxinst_enabled = boxinst_enabled
        self.bottom_pixels_removed = bottom_pixels_removed
        self.pairwise_size = pairwise_size
        self.pairwise_dilation = pairwise_dilation
        self.pairwise_color_thresh = pairwise_color_thresh
        self.points_enabled = points_enabled
        self.run_seg = run_seg
        self.use_enlarge_bbox = use_enlarge_bbox
        self.pairwise_distance_thresh = pairwise_distance_thresh
        self.register_buffer("_iter", torch.zeros([1]))
        self._warmup_iters = pairwise_warmup
        self.start_kd_loss_iters = start_kd_loss_iters
        self.num_classes = num_classes
        self.norm_cfg = norm_cfg
        self.fp16_enable = False
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.param_conv = nn.Conv2d(
            self.bbox_head_channels,
            self.num_gen_params,
            3,
            stride=1,
            padding=1)

    def parse_dynamic_params(self, params):
        num_insts = params.size(0)
        params_list = list(torch.split_with_sizes(
            params, self.dy_weights + self.dy_biases, dim=1))
        weights_list = params_list[:self.dynamic_convs]
        biases_list = params_list[self.dynamic_convs:]

        for i in range(self.dynamic_convs):
            if i < self.dynamic_convs - 1:
                weights_list[i] = weights_list[i].reshape(
                    num_insts * self.dynamic_channels, -1, 1, 1)
                biases_list[i] = biases_list[i].reshape(
                    num_insts * self.dynamic_channels)
            else:
                weights_list[i] = weights_list[i].reshape(
                    num_insts * 1, -1, 1, 1)
                biases_list[i] = biases_list[i].reshape(num_insts * 1)
        return weights_list, biases_list

    def forward(self, feat, params, coors, level_inds, img_inds):
        B, C, H, W = feat.size()
        mask_feat = feat[img_inds].reshape((-1, C, H, W))  # [1,16,160,240]-->[N个mask,16,160,240]
        N, _, H, W = mask_feat.size()
        if not self.disable_rel_coors:
            shift_x = torch.arange(0, W * self.in_stride, step=self.in_stride,
                                   dtype=mask_feat.dtype, device=mask_feat.device)
            shift_y = torch.arange(0, H * self.in_stride, step=self.in_stride,
                                   dtype=mask_feat.dtype, device=mask_feat.device)
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            locations = torch.stack(
                [shift_x, shift_y], dim=0) + self.in_stride // 2

            rel_coors = coors[..., None, None] - locations[None]
            soi = self.sizes_of_interest.float()[level_inds]
            rel_coors = rel_coors / soi[..., None, None, None]
            mask_feat = torch.cat([rel_coors, mask_feat], dim=1)  # [N_inst,16+2,160,240]

        weights, biases = self.parse_dynamic_params(params)  # params (N, 233)
        x = mask_feat.reshape(1, -1, H, W)
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(x, w, bias=b, stride=1, padding=0, groups=N)
            if i < self.dynamic_convs - 1:
                x = F.relu(x)
        x = x.permute(1, 0, 2, 3)
        x = aligned_bilinear(x, self.in_stride // self.out_stride)  # 64x1xhxw --> 64x1x2hx2w(2h = ori_img.shape/4)
        return x

    def training_sample(self,
                        cls_scores,
                        centernesses,
                        param_preds,  # list([B, 233, h, w]) len(list)=5
                        coors,
                        level_inds,
                        img_inds,     # N
                        gt_inds,      # N -1 is no assign gt box
                        bbox_preds,
                        bbox_targets):
        num_imgs = param_preds[0].size(0)
        param_preds = torch.cat([
            param_pred.permute(0, 2, 3, 1).flatten(end_dim=2)
            for param_pred in param_preds], dim=0)
        # debug
        param_preds_zero = (param_preds[0] * 0).reshape(1, -1)
        coors_zero = (coors[0] * 0).reshape(1, -1)
        level_inds_zero = (level_inds[0] * 0).reshape(1)
        img_inds_zero = (img_inds[0] * 0).reshape(1)
        gt_inds_zero = (gt_inds[0] * 0).reshape(1)
        # get in gt boxes pixel 
        pos_mask = gt_inds != -1
        param_preds = param_preds[pos_mask]
        coors = coors[pos_mask]
        level_inds = level_inds[pos_mask]
        img_inds = img_inds[pos_mask]
        gt_inds = gt_inds[pos_mask]

        if self.max_proposals != -1:
            num_proposals = min(self.max_proposals, param_preds.size(0))
            sampled_inds = torch.randperm(
                num_proposals, device=param_preds.device).long()
        elif self.topk_per_img != -1:
            cls_scores = torch.cat([
                cls_score.permute(0, 2, 3, 1).flatten(end_dim=2)
                for cls_score in cls_scores], dim=0)  # 将所有采样点的得分展开
            cls_scores = cls_scores[pos_mask]         # 获得正样本点的cls分数
            centerness = torch.cat([
                centerness.permute(0, 2, 3, 1).reshape(-1)
                for centerness in centernesses], dim=0)
            centerness = centerness[pos_mask]

            sampled_inds = []
            inst_inds = torch.arange(
                param_preds.size(0), device=param_preds.device)
            for img_id in range(num_imgs):
                img_mask = img_inds == img_id
                if not img_mask.any():
                    continue
                img_gt_inds = gt_inds[img_mask]
                img_inst_inds = inst_inds[img_mask]
                unique_gt_inds = img_gt_inds.unique()
                inst_per_gt = max(
                    int(self.topk_per_img / unique_gt_inds.size(0)), 1)

                for gt_ind in unique_gt_inds:
                    gt_mask = img_gt_inds == gt_ind  # get instance=gt_ind's mask
                    img_gt_inst_inds = img_inst_inds[gt_mask]  # image instance id(local instance id)
                    if img_gt_inst_inds.size(0) > inst_per_gt:
                        cls_scores_ = cls_scores[img_mask][gt_mask]
                        cls_scores_ = cls_scores_.sigmoid().max(dim=1)[0]
                        centerness_ = centerness[img_mask][gt_mask]
                        centerness_ = centerness_.sigmoid()
                        inds = (cls_scores_ *
                                centerness_).topk(inst_per_gt, dim=0)[1]
                        img_gt_inst_inds = img_gt_inst_inds[inds]
                    sampled_inds.append(img_gt_inst_inds)
            nums = 0
            for i in range(len(sampled_inds)):
                nums += len(sampled_inds[i])
            if nums != 0:
                sampled_inds = torch.cat(sampled_inds, dim=0)
            else:
                sampled_inds = None
        if sampled_inds is not None:
            cls_scores = cls_scores[sampled_inds]
            param_preds = param_preds[sampled_inds]
            coors = coors[sampled_inds]
            level_inds = level_inds[sampled_inds]
            img_inds = img_inds[sampled_inds]
            gt_inds = gt_inds[sampled_inds]
            if bbox_preds is not None and bbox_targets is not None:
                bbox_preds = bbox_preds[sampled_inds]
                bbox_targets = bbox_targets[sampled_inds]
            return param_preds, coors, level_inds, img_inds, gt_inds, bbox_preds, bbox_targets, 1, cls_scores
        else:
            return param_preds_zero, coors_zero, level_inds_zero, img_inds_zero, gt_inds_zero, None, None, 0, cls_scores

    def simple_test(self,
                    mask_feat,
                    det_labels,
                    det_params,     # (N,233)
                    det_coors,      # (N,2)
                    det_level_inds, # 表示检测出来instance的再fpn的第几层 (N,)
                    img_metas,
                    num_classes,
                    rescale=False):
        num_imgs = len(img_metas)
        num_inst_list = [param.size(0) for param in det_params]
        det_img_inds = [
            torch.full((num,), i, dtype=torch.long, device=mask_feat.device)
            for i, num in enumerate(num_inst_list)
        ]

        det_params = torch.cat(det_params, dim=0)
        det_coors = torch.cat(det_coors, dim=0)
        det_level_inds = torch.cat(det_level_inds, dim=0)
        det_img_inds = torch.cat(det_img_inds, dim=0)
        if det_params.size(0) == 0:
            segm_results = [[[] for _ in range(num_classes)]
                            for _ in range(num_imgs)]
            return segm_results

        mask_preds = self.forward(mask_feat, det_params, det_coors, det_level_inds,
                                  det_img_inds)
        mask_preds = mask_preds.sigmoid()
        mask_preds = aligned_bilinear(mask_preds, self.out_stride)

        segm_results = []
        mask_preds = torch.split(mask_preds, num_inst_list, dim=0)
        for cur_mask_preds, cur_labels, img_meta in zip(
                mask_preds, det_labels, img_metas):
            if cur_mask_preds.size(0) == 0:
                segm_results.append([[] for _ in range(num_classes)])

            input_h, input_w = img_meta['img_shape'][:2]
            cur_mask_preds = cur_mask_preds[:, :, :input_h, :input_w]

            if rescale:
                ori_h, ori_w = img_meta['ori_shape'][:2]
                cur_mask_preds = F.interpolate(
                    cur_mask_preds, (ori_h, ori_w),
                    mode='bilinear',
                    align_corners=False)

            cur_mask_preds = cur_mask_preds.squeeze(1) > 0.5
            cur_mask_preds = cur_mask_preds.cpu().numpy().astype(np.uint8)
            cur_labels = cur_labels.detach().cpu().numpy()
            segm_results.append([cur_mask_preds[cur_labels == i]
                                 for i in range(num_classes)])
        return segm_results

    def loss_multi(self,
                    imgs,
                    img_metas,
                    mask_logits,   # pred mask N_samp x 1 x 2fpn_h x 2fpn_w
                    gt_inds,       # N_samp,1
                    gt_bboxes,     # gt_bboxes_nums, 4
                    gt_masks,
                    gt_labels,
                    points,        # 
                    pseudo_labels, # 
                    seg_logits=None,
                    img_inds=None,
                    num_classes=3,
                    flag=1,
                    seg_lables=None):
        self._iter += 1
        # similarities matrix=(N_gt, 160, 240), gt_bitmasks=(N_gt, 160, 240),
        # similarities的结果是将每张图片的相似度矩阵重复了N_gt次，即(similarities[1][0]==similarities[1][1]).all()=True
        # 每个2D Box对应一张bitmask_full,只有2D box内的mask值为1
        # bitmask_full是用来过滤pad的操作产生的不相关像素，pad是从bottom和right pad的,gt_bitmasks是bitmask_full按照间隔取值降采样程mask大小的
        similarities, gt_bitmasks, bitmasks_full, en_bitmasks, en_bitmasks_full = \
            self.get_targets(gt_bboxes, gt_masks, imgs, img_metas, None, None, None)
        
        mask_scores = mask_logits.sigmoid()          # (N_samp,1,320,480)
        gt_bitmasks = torch.cat(gt_bitmasks, dim=0)  # (N_gt,160,240)
        gt_bitmasks = gt_bitmasks[gt_inds].unsqueeze(1).to(mask_scores.dtype)  # (N_gt, 160, 240)-->(N_samp,1,160,240)复制几次mask，多个mask对应一个gt_mask
        en_bitmasks = torch.cat(en_bitmasks, dim=0)  # (N_gt,160,240)
        en_bitmasks = en_bitmasks[gt_inds].unsqueeze(1).to(mask_scores.dtype)  # (N_gt, 160, 240)-->(N_samp,1,160,240)复制几次mask，多个mask对应一个gt_mask
        
        if not (mask_scores.size()==gt_bitmasks.size()):
            mask_scores = mask_scores.reshape((gt_bitmasks.size()))
        assert mask_scores.dim()==4 and gt_bitmasks.dim()==4
        
        losses = {}

        if len(mask_scores) == 0 or flag == 0:  # there is no instances detected
            dummy_loss = 0 * mask_scores.sum()
            if not self.boxinst_enabled:
                losses["loss_mask"] = dummy_loss
            else:
                losses["loss_prj"] = dummy_loss
                losses["loss_pairwise"] = dummy_loss
                if self.points_enabled:
                    losses["loss_3d_to_2d"] = dummy_loss
                if self.geometry_loss_enabled:
                    losses["loss_geometry"] = dummy_loss
                return losses

        if self.boxinst_enabled:
            img_color_similarity = torch.cat(similarities, dim=0)  # [(N_gt_,8,160,240),(N_gt_,8,160,240)]
            img_color_similarity = img_color_similarity[gt_inds].to(dtype=mask_scores.dtype)  # [N_samp,8,320,480]

            # 1. projection loss
            loss_prj_term = compute_project_term(mask_scores, gt_bitmasks)

            # 2. pairwise loss
            # all pixels log, [64,8,160,240]
            pairwise_losses = pairwise_nlog(mask_logits, self.pairwise_size, self.pairwise_dilation)
            # (> color sim threshold points and in gt_boxes points) intersection
            # weights, [64,8,160,240]
            if self.use_enlarge_bbox:
                weights = (img_color_similarity >= self.pairwise_color_thresh).float() * en_bitmasks.float()
            else:
                weights = (img_color_similarity >= self.pairwise_color_thresh).float() * gt_bitmasks.float()
            loss_pairwise = (pairwise_losses * weights).sum() / weights.sum().clamp(min=1.0)

            warmup_factor = min(self._iter.item() / float(self._warmup_iters), 1.0)

            loss_pairwise = loss_pairwise * warmup_factor

            losses.update({
                "loss_prj": loss_prj_term,
                "loss_pairwise": loss_pairwise,
            })
            # warmup_factor_run = min(self._iter.item() / float(self._warmup_iters * 2.0), 1.0)

        if self.points_enabled:
            assert mask_scores.size(0) == gt_inds.size(0) == img_inds.size(0)
            # self._iter = torch.tensor(30000, device=imgs.device) # debug
            if self._iter >= self.start_kd_loss_iters:
                warmup_factor = min((self._iter.item() - self.start_kd_loss_iters) / 1000, 1.0)
                if len(imgs) != len(img_metas):
                    loss_3d_to_2d = self.multi_view_3d_to_2d(mask_scores,  # mask_scores or mask_logits
                                                            imgs,
                                                            gt_labels,
                                                            gt_bboxes,
                                                            num_classes,
                                                            img_inds,
                                                            gt_inds,
                                                            points,
                                                            seg_logits,  # seg_logits
                                                            warmup_factor,
                                                            img_metas,
                                                            seg_lables)
                else:
                    loss_3d_to_2d = self.front_view_3d_to_2d(mask_scores,
                                                            imgs,
                                                            gt_labels,
                                                            gt_bboxes,
                                                            num_classes,
                                                            img_inds,
                                                            gt_inds,
                                                            points,
                                                            seg_logits,
                                                            warmup_factor,
                                                            img_metas)
            else:
                loss_3d_to_2d = 0 * mask_scores.sum()
            losses.update({
                "loss_3d_to_2d": loss_3d_to_2d,
            })

            if self.run_seg:
                grid_targets, points_targets, fg_len, seg_scores_ = self.get_points_targets_v2(points, gt_bboxes, gt_labels, img_metas, seg_logits)
                gt_labels_ = torch.cat(gt_labels)
                gt_bboxes_ = torch.cat(gt_bboxes, dim=0)
                loss_2d_run_seg = torch.tensor(0, device=imgs.device, dtype=torch.float)
                # 采样 将归一化坐标[-1,+1]重复gt_labels数量,然后依据gt_inds来分配
                grid_targets = torch.repeat_interleave(grid_targets.unsqueeze(0), len(gt_labels_), 0).unsqueeze(1)[gt_inds]
                # (N, 1, 160, 240) (N, 1, (N1+N2+...), 2)-->(N, 1, 1, (N1+N2+...))
                grid_sample_scores = F.grid_sample(mask_scores, grid_targets, align_corners=False)
                grid_sample_scores = grid_sample_scores.squeeze(1).squeeze(1)
                grid_sample_scores = torch.clamp(grid_sample_scores, 1e-5, 1-1e-5)  # clamp，防止出现inf值
                # 获取点对应的mask scores和seg id
                ori_fg_len = fg_len
                fg_len = fg_len[gt_inds]
                # scores_with_id = torch.ones((int(fg_len.sum()), 2), device=imgs.device) * -1
                # nums = 0
                counts_nums = 0  # 统计多少mask参与计算
                for i in range(len(gt_inds)):
                    gt_ind = gt_inds[i]
                    gt_label = gt_labels_[gt_ind]
                    # scores_with_id[:,0][nums: nums+fg_len[i]] = grid_sample_scores[i][ori_fg_len[:gt_ind].sum(): ori_fg_len[:gt_ind+1].sum()]
                    # scores_with_id[:,1][nums: nums+fg_len[i]] = points_targets[gt_ind][:,14]  # get seg id
                    range_sets, inv_inds, counts = torch.unique(points_targets[gt_ind][:,14], return_inverse=True, return_counts=True)
                    if len(range_sets) == 0:
                        continue
                    scores = grid_sample_scores[i][ori_fg_len[:gt_ind].sum(): ori_fg_len[:gt_ind+1].sum()]

                    # nums += fg_len[i]

                    out = torch.zeros((range_sets.shape), device=imgs.device)
                    out.scatter_add_(0, inv_inds, scores)
                    ex = out/counts
                    out2 = torch.zeros((range_sets.shape), device=imgs.device)
                    out2.scatter_add_(0, inv_inds, scores**2)
                    ex2 = out2/counts
                    var = ex2 - ex**2

                    entropy = -(scores*scores.log()+(1-scores)*(1-scores).log())
                    out3 = torch.zeros((range_sets.shape), device=imgs.device)
                    out3.scatter_add_(0, inv_inds, entropy)
                    out3 = out3/counts

                    filter_mask = range_sets != -1
                    if (counts[filter_mask]>3).sum() == 0:
                        continue
                    counts_nums += (counts[filter_mask]>3).sum()
                    loss_2d_run_seg += (((out3[filter_mask]+var[filter_mask])*(counts[filter_mask]>3)).sum())

                if counts_nums != 0:
                    loss_2d_run_seg = loss_2d_run_seg * warmup_factor * 0.1 / counts_nums
                else:
                    loss_2d_run_seg = 0 * mask_scores.sum()
                losses.update({
                    "loss_2d_run_seg": loss_2d_run_seg,
                })
                # 可视化代码
                visual = False
                if visual:
                    import cv2
                    for i in range(int((img_inds==0).sum())):
                        aimg = aligned_bilinear(imgs[0].unsqueeze(0), 2)  # 1280，1920，3
                        aimg = tensor2imgs(aimg, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],to_rgb=False)[0]
                        aimg = cv2.cvtColor(aimg, cv2.COLOR_RGB2BGR)
                        gt_box_ind = gt_inds[i]  # 选择可视化哪个box对应的点和mask分数
                        gt_mask_id = torch.where(gt_inds==gt_box_ind)[0]
                        if len(gt_mask_id) == 0:
                            continue
                        gt_mask_id = gt_mask_id[0]
                        gt_mask = mask_scores[gt_mask_id] # 同一个2dbox对应多个mask_pred, 只选择第一个mask就好
                        # 可视化mask
                        gt_mask = aligned_bilinear(gt_mask.unsqueeze(0), 8).squeeze().squeeze()
                        gt_mask = gt_mask.detach().cpu().numpy()
                        cv2.imwrite('./work_dirs/out_images/pred_mask_{}.jpeg'.format(gt_box_ind), gt_mask/gt_mask.max()*255)
                        # 可视化扩大gt对应的点云
                        gt_bbox = gt_bboxes_[gt_box_ind]*2
                        cv2.rectangle(aimg, (int(gt_bbox[0]), int(gt_bbox[1])), (int(gt_bbox[2]), int(gt_bbox[3])), (0, 255, 0), 2)
                        for j, p in enumerate(points_targets[gt_box_ind]):
                            cv2.circle(aimg, (int(p[12]*2),int(p[13]*2)),1,(255,255,255),-1)
                        cv2.imwrite('./work_dirs/out_images/img_pts_{}.jpeg'.format(gt_box_ind), aimg)
                        # 可视化点云自身的mask值
                        max_score = seg_scores_[gt_box_ind][:, gt_label].max()
                        gt_label = gt_labels_[gt_box_ind]
                        for j, p in enumerate(points_targets[gt_box_ind]):
                            rgb = int(255 * seg_scores_[gt_box_ind][j][gt_label]/max_score)
                            cv2.circle(aimg, (int(p[12]*2),int(p[13]*2)), 1, (rgb, rgb, rgb), -1)
                        cv2.imwrite('./work_dirs/out_images/img_pts_mask{}.jpeg'.format(gt_box_ind), aimg)
                        # 可视化点云投到图像位置 图像的mask
                        grid_sample_scores_ind = grid_sample_scores[gt_mask_id]
                        grid_sample_scores_ind = grid_sample_scores_ind[ori_fg_len[:gt_box_ind].sum(): ori_fg_len[:gt_box_ind+1].sum()]
                        max_score2 = grid_sample_scores_ind.max()
                        for j, p in enumerate(points_targets[gt_box_ind]):
                            rgb = int(255 * grid_sample_scores_ind[j]/max_score2)
                            cv2.circle(aimg, (int(p[12]*2),int(p[13]*2)), 1, (rgb, rgb, rgb), -1)
                        cv2.imwrite('./work_dirs/out_images/img_mask{}.jpeg'.format(gt_box_ind), aimg)

                if False:
                    import cv2
                    aimg = aligned_bilinear(imgs[0].unsqueeze(0), 2)  # 1280，1920，3
                    aimg = tensor2imgs(aimg, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],to_rgb=False)[0]
                    aimg = cv2.cvtColor(aimg, cv2.COLOR_RGB2BGR)
                    pp = torch.cat(points_targets, dim=0)
                    max_run = points[0][:14].max()
                    for i in range(len(range_sets)):
                        rgb = np.array([np.random.randint(255) for _ in range(3)])
                        p = pp[pp[:,14] == range_sets[i]]
                        if range_sets[i]>max_run:
                            continue
                        if counts[i] < 4:
                            continue
                        for j in p:
                            x = int(j[12]*2)
                            y = int(j[13]*2)
                            cv2.circle(aimg, (x,y),2,rgb.tolist(),-1)
                    cv2.imwrite('run_img.jpeg', aimg)

                if False:
                    import cv2
                    aimg = aligned_bilinear(imgs[0].unsqueeze(0), 2)  # 1280，1920，3
                    aimg = tensor2imgs(aimg, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],to_rgb=False)[0]
                    aimg = cv2.cvtColor(aimg, cv2.COLOR_RGB2BGR)
                    range_sets_ = range_sets[counts>3]
                    counts_ = counts[counts>3]
                    topk = range_sets_[out3[counts>3].topk(100)[1]]
                    cur_img = topk <= points[0][:,14].max()
                    topk = topk[cur_img]
                    for i in range(len(topk)):
                        rgb = np.array([np.random.randint(255) for _ in range(3)])
                        pts = points[0][points[0][:,14]==topk[i]]
                        for p in pts:
                            cv2.circle(aimg, (int(p[12]*2),int(p[13]*2)), 1, rgb.tolist(), -1)
                    cv2.imwrite('run_img.jpeg', aimg)
        
        if self.geometry_loss_enabled:
            fg_points_targets, bg_points_targets = self.get_multi_view_points_targets(points, gt_bboxes, img_metas, imgs.shape)
            if False:
                import cv2
                aimg = imgs[0].permute(1, 2, 0).cpu().numpy()
                aimg = np.ascontiguousarray(aimg)
                vis_boxes = gt_bboxes[0]
                for v, box in enumerate(vis_boxes.cpu().numpy()):
                    cv2.rectangle(aimg, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                for v, pts in enumerate(fg_points_targets[0:len(vis_boxes)]):
                    # if v != 0:
                    #     continue
                    for vv, ptss in enumerate(pts):
                        cv2.circle(aimg, (int(ptss[3]), int(ptss[4])), 1, (255,0,0),1)
                for v, pts in enumerate(bg_points_targets[0:len(vis_boxes)]):
                    # if v != 0:
                    #     continue
                    for vv, ptss in enumerate(pts):
                        cv2.circle(aimg, (int(ptss[3]), int(ptss[4])), 2, (0,0,255),2)
                cv2.imwrite('img.jpg', aimg)
            
            # cal geometry sim
            geo_sim, weights, coords = self.cal_geometry_sim(fg_points_targets, bg_points_targets, imgs, img_metas)
            geo_sim = geo_sim[gt_inds]
            weights = weights[gt_inds]
            coords = coords[gt_inds]
            weights = (geo_sim >= self.pairwise_geo_thresh).float()
            # cal loss
            loss_geometry = self.cal_geometry_loss(mask_logits, coords, img_metas)
            weights_mean = max(reduce_mean(weights.sum().deatch()), 1.0)
            weights_mean = weights_mean.type(torch.float32)
            loss_geometry = (weights * loss_geometry).sum() / weights_mean.sum().clamp(min=1.0)
            losses['loss_geometry'] = loss_geometry * warmup_factor * self.geometry_loss_weight
            
        return losses

    def cal_geometry_loss(self, mask_logits, coords, img_metas):
        assert mask_logits.dim() == 4
        # sample
        # input(N,1,H,W) grid(N,1,N_pts,2) out(N,1,1,N_pts)
        coords[:, :, 0] = coords[:, :, 0] / img_metas[0]['pad_shape'][0][1] * 2 -1
        coords[:, :, 1] = coords[:, :, 1] / img_metas[0]['pad_shape'][0][0] * 2 -1
        points_logits = F.grid_sample(mask_logits, coords.unsqueeze(1), align_corners=False)
        # 1
        # points_logits = points_logits.squeeze()
        # loss = (points_logits.sigmoid().unsqueeze(2) * points_logits.sigmoid().unsqueeze(1) + \
        #         (1 - points_logits.sigmoid()).unsqueeze(2) * (1 - points_logits.sigmoid()).unsqueeze(1))
        # loss = torch.clamp(loss, 1e-5, 1)
        # loss = -loss.log()
        # 2
        log_fg_prob = F.logsigmoid(points_logits)
        log_bg_prob = F.logsigmoid(-points_logits)
        # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
        # we compute the the probability in log space to avoid numerical instability
        N = coords.shape[-2]
        log_fg_prob_unfold = log_fg_prob.expand(-1, -1, N, -1)
        log_bg_prob_unfold = log_bg_prob.expand(-1, -1, N, -1)
        log_same_fg_prob = log_fg_prob + log_fg_prob_unfold
        log_same_bg_prob = log_bg_prob + log_bg_prob_unfold

        max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
        log_same_prob = torch.log(
            torch.exp(log_same_fg_prob - max_) +
            torch.exp(log_same_bg_prob - max_)
        ) + max_
        #loss = -log(prob)
        loss = -log_same_prob.squeeze(1)
        return loss

    def cal_geometry_sim(self, fg_points_targets, bg_points_targets, imgs, img_metas):
        # points x,y,z,u,v
        assert len(fg_points_targets) == len(bg_points_targets)
        sim_list = []
        weight_list = []
        coords_list = []
        for i in range(len(fg_points_targets)):
            N_f = fg_points_targets[i][:, 0].nonzero().shape[0]
            N_b = bg_points_targets[i][:, 0].nonzero().shape[0]
            # # test
            # N_f = (fg_points_targets[i][:, 0]!=0).sum()
            # N_b = (bg_points_targets[i][:, 0]!=0).sum()
            assert (fg_points_targets[i][:, 0]!=0).sum() == N_f
            assert (bg_points_targets[i][:, 0]!=0).sum() == N_b
            N = N_f + N_b
            points = torch.zeros_like(torch.cat((fg_points_targets[i], bg_points_targets[i]), dim=0))
            points[:N_f+N_b, :] = torch.cat((fg_points_targets[i][:N_f, :], bg_points_targets[i][:N_b, :]), dim=0)
            dis_matrix = self.cal_dist(points[:, :3], points[:, :3])
            weight = torch.zeros_like(dis_matrix)
            sim1 = torch.zeros_like(dis_matrix)
            # sim2 = torch.zeros_like(dis_matrix)
            # sim3 = torch.zeros_like(dis_matrix)
            # sim4 = torch.zeros_like(dis_matrix)
            if N_f >=2 and N_b >=1:
                # cal sim
                # how to norm? m is ?
                # exp[-(dis_matrix / m - 1)]

                # min --> median
                # if there is an outlier, then the median is used, mean not use
                sort_dis_matrix, _ = torch.sort(dis_matrix[:N_f, :N_f], dim=1)  
                min = torch.median(sort_dis_matrix[:, 1])
                # sim1 = torch.exp(-(dis_matrix / min - 1)*0.5)  # 相对距离还是绝对距离？ min=0.1m
                # min_ = torch.mean(sort_dis_matrix[:, 1])
                sim1 = torch.exp(-(dis_matrix / min - 1))  # threshold 0.2

                # quantization vote
                # size = 0.05
                # mode, _ = torch.mode((dis_matrix[:N, :N][dis_matrix[:N, :N]!=0] / size).long())  # torch.diag()
                # mode = (mode * size)
                # sim2 = torch.exp(-(dis_matrix / mode - 1))

                # # mean?
                # mean = torch.mean(dis_matrix[:N_f, :N_f])
                # sim3 = torch.exp(-(dis_matrix / mean - 1))

                weight[:N, :N] = 1
                tri = torch.arange(0, N)
                weight[tri, tri] = 0
                
                # import matplotlib.pyplot as plt
                # # fig, ax = plt.subplots(1, 1, figsize=(10,10))
                # # ax.set_title('min_median')
                # # ax.imshow((sim1.cpu().numpy() > 0.3) * weight.cpu().numpy())

                # fig, ax = plt.subplots(2, 2, figsize=(10,10))
                # threshold = 0.5
                # ax[0][0].set_title('min_median')
                # ax[0][0].imshow((sim1.cpu().numpy() > threshold) * weight.cpu().numpy())
                # ax[0][1].set_title('mode_dis')
                # ax[0][1].imshow((sim1.cpu().numpy() > 0.3) * weight.cpu().numpy())
                # ax[1][1].set_title('mean')
                # ax[1][1].imshow((sim4.cpu().numpy() > 0.2) * weight.cpu().numpy())
                # ax[1][0].set_title('min_mean')
                # ax[1][0].imshow((sim4.cpu().numpy() > 0.5) * weight.cpu().numpy())                
                # plt.show()

            sim1 = sim1 * weight
            sim_list.append(sim1)
            weight_list.append(weight)
            coords_list.append(points[:, 3:5])

        return torch.stack(sim_list), torch.stack(weight_list), torch.stack(coords_list)

    def cal_dist(self, pts1, pts2):
        pts1 = pts1.double()  # why double()?
        pts2 = pts2.double()
        pts1_ = pts1 ** 2
        pts2_ = pts1 ** 2
        sum_pts_1 = torch.sum(pts1_, dim=1).unsqueeze(1)  # N,1
        sum_pts_2 = torch.sum(pts2_, dim=1).unsqueeze(0)  # 1,N
        matrix = torch.sqrt(sum_pts_1 + sum_pts_2 - 2 * pts1.mm(pts2.T))  # x**2 + y**2 - 2xy
        matrix[torch.isnan(matrix)] = 0  # why have nan? torch.where(torch.isnan(matrix))
        return matrix.type(torch.float32)

    def front_view_3d_to_2d(self, 
                            mask_scores,
                            imgs,
                            gt_labels,
                            gt_bboxes,
                            num_classes,
                            img_inds,
                            gt_inds,
                            points,
                            seg_scores,
                            warmup_factor,
                            img_metas):
        gt_labels_ = torch.cat(gt_labels)
        # gt_bboxes_ = torch.cat(gt_bboxes, dim=0)
        self.num_classes = num_classes
        loss_3d_to_2d = torch.tensor(0, device=imgs.device, dtype=torch.float)
        counts_nums = 0
        for i in range(len(imgs)):
            # 获得对应图片的mask_pred和label
            mask_pred_per_img = mask_scores[img_inds==i]
            gt_inds_per_img = gt_inds[img_inds==i]
            gt_labels_per_img = gt_labels_[gt_inds_per_img]
            # 分配 label
            mask_preds_cls = []
            gt_inds_cls = []
            for j in range(self.num_classes):
                mask_cls = gt_labels_per_img == j
                gt_inds_per_img_cls = gt_inds_per_img[mask_cls]
                mask_pred_per_img_cls = mask_pred_per_img[mask_cls]
                mask_preds_cls.append(mask_pred_per_img_cls)
                gt_inds_cls.append(gt_inds_per_img_cls)

            per_batch_points = points[i]

            for j in range(self.num_classes):
                if mask_preds_cls[j].size(0) != 0:
                    mask_pred_cls = mask_preds_cls[j].max(0)[0]
                    grid = per_batch_points[:, 12:14]
                    grid[:,0] = (grid[:,0]/img_metas[i]['pad_shape'][1]) * 2 - 1  # x
                    grid[:,1] = (grid[:,1]/img_metas[i]['pad_shape'][0]) * 2 - 1  # y
                    # input(1,1,H,W) grid(1,1,N_pts,2) out(1,1,1,N_pts)
                    out = F.grid_sample(mask_pred_cls.unsqueeze(0), grid.unsqueeze(0).unsqueeze(0), align_corners=False)
                    out = torch.clamp(out, 1e-5, 1-1e-5)
                    out = out.squeeze(0).squeeze(0).squeeze(0)
                    # 0.纯软标签
                    # -plog(q)
                    # counts_nums += out.size()[0]
                    # loss_3d_to_2d += -((seg_scores[i][:,j] * out.log()).sum() + \
                    #                 ((1-seg_scores[i][:,j]) * (1-out).log()).sum())  # 为什么x2？为了不改变样本数，软标签监督推拉是计算了2倍的样本数
                    # 1. 背景点强置0
                    bg_points_mask = per_batch_points[:,11] == 0
                    counts_nums += out.size()[0]
                    loss_3d_to_2d += -((out[~bg_points_mask].log() * seg_scores[i][:,j][~bg_points_mask]).sum() + \
                            ((1-out[~bg_points_mask]).log() * (1-seg_scores[i][:,j][~bg_points_mask])).sum() + \
                            2*(1-seg_scores[i][:,j][bg_points_mask]).log().sum())
                    
                    # 1. 强制将>0.7的置1，<0.3置0
                    # 1.1 bce loss
                    # targets = seg_scores[i][:,j]
                    # mask_pos = targets > 0.7
                    # mask_neg = targets < 0.3
                    # targets[mask_pos] = 1
                    # targets[mask_neg] = 0
                    # pos_sample_mask = out[mask_pos]
                    # neg_sample_mask = (1-out[mask_neg])
                    # counts_nums += (mask_pos.sum() + mask_neg.sum())
                    # # loss_3d_to_2d += -(pos_sample_mask.log().sum() + neg_sample_mask.log().sum())
                    # # 1.2 Focal loss 计算1.
                    # alpha = 0.8  # 0.25
                    # gamma = 3.0  # 2.0
                    # loss_3d_to_2d += -((alpha*(1-pos_sample_mask).pow(gamma)*pos_sample_mask.log()).sum() + \
                    #                   ((1-alpha)*(1-neg_sample_mask).pow(gamma)*(1-neg_sample_mask).log()).sum())
                    # 2. 利用run的信息将一些点强制？？我认为不应该加，本身就不太准
                    if False:
                        import cv2
                        aimg = aligned_bilinear(imgs[i].unsqueeze(0), 2)  # 1280，1920，3
                        aimg = tensor2imgs(aimg, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],to_rgb=False)[0]
                        aimg = cv2.cvtColor(aimg, cv2.COLOR_RGB2BGR)

                        # 可视化image mask
                        gt_mask = aligned_bilinear(mask_pred_cls.unsqueeze(0), 8).squeeze().squeeze()
                        gt_mask = gt_mask.detach().cpu().numpy()
                        cv2.imwrite('./work_dirs/out_images/pred_mask_cls_id_{}.jpeg'.format(j), gt_mask/gt_mask.max()*255)
                        # 可视化点云自身的mask
                        max_score = seg_scores[i][:,j].max()
                        for k, p in enumerate(points[i]):
                            rgb = int(255 * seg_scores[i][k][j]/max_score)
                            cv2.circle(aimg, (int(p[12]*2),int(p[13]*2)), 1, (rgb, rgb, rgb), -1)
                        cv2.imwrite('./work_dirs/out_images/img_pts_mask_cls_id_{}.jpeg'.format(j), aimg)
                        # 可视化点云投影到图像位置图像的mask
                        max_score = out.max()
                        for k, p in enumerate(points[i]):
                            rgb = int(255 * out[k]/max_score)
                            cv2.circle(aimg, (int(p[12]*2),int(p[13]*2)), 1, (rgb, rgb, rgb), -1)
                        cv2.imwrite('./work_dirs/out_images/img_mask_cls_id_{}.jpeg'.format(j), aimg)

                        ascore = (-(seg_scores[i][:,j] * out.log()+((1-seg_scores[i][:,j]) * (1-out).log())))
                        topk_val, topk_inds = torch.topk(ascore, 5)
                        max_loss = topk_val[0]
                        for k, p in enumerate(points[i][topk_inds]):
                            rgb = int(255 * topk_val[k]/max_loss)
                            cv2.circle(aimg, (int(p[12]*2),int(p[13]*2)), 3, (rgb, rgb, rgb), -1)
                        cv2.imwrite('./work_dirs/out_images/0cls_id{}.jpeg'.format(j), aimg)
        if counts_nums != 0:
            loss_3d_to_2d = loss_3d_to_2d * warmup_factor * 0.5 / counts_nums
        else:
            loss_3d_to_2d = 0 * mask_scores.sum()
        return loss_3d_to_2d

    def multi_view_3d_to_2d(self, 
                            mask_logits,
                            imgs,
                            gt_labels,
                            gt_bboxes,
                            num_classes,
                            img_inds,
                            gt_inds,
                            points,
                            seg_logits,
                            warmup_factor,
                            img_metas,
                            seg_lables):
        gt_labels_ = torch.cat(gt_labels)
        self.num_classes = num_classes
        loss_3d_to_2d = torch.tensor(0., device=imgs.device, dtype=torch.float)
        counts_nums = torch.tensor(0., device=imgs.device, dtype=torch.float)
        for i in range(imgs.shape[0]):
            batch_img_nums = int(len(imgs) / len(img_metas))
            batch_id = int(i // batch_img_nums)
            mask_pred_per_img = mask_logits[img_inds==i]
            gt_inds_per_img = gt_inds[img_inds==i]
            gt_labels_per_img = gt_labels_[gt_inds_per_img]
            # assign label
            mask_preds_cls_list = []
            gt_inds_cls = []
            for j in range(self.num_classes):
                mask_cls = gt_labels_per_img == j
                gt_inds_per_img_cls = gt_inds_per_img[mask_cls]
                mask_pred_per_img_cls = mask_pred_per_img[mask_cls]
                mask_preds_cls_list.append(mask_pred_per_img_cls)
                gt_inds_cls.append(gt_inds_per_img_cls)

            per_batch_points = points[batch_id]
            cls_weight = torch.ones((self.num_classes), device=per_batch_points.device)
            if self.use_weight_loss:
                # 按照频率计算loss的加权值，会导致不收敛，出现nan值
                # per_batch_labels = seg_lables[batch_id]
                # for j in range(self.num_classes):
                #     cls_weight[j] = (per_batch_labels == j).sum()
                # # assert (cls_weight==0).sum() != self.num_classes
                # cls_weight_norm = (cls_weight / cls_weight.sum())
                # cls_weight = torch.clamp(cls_weight_norm**0.2, 1, 3).type(torch.int32)
                cls_weight[2] = 5
            for j in range(self.num_classes):
                if mask_preds_cls_list[j].size(0) != 0:
                    mask_pred_cls = mask_preds_cls_list[j].max(0)[0]
                    # get points
                    in_img_mask1 = per_batch_points[:,10]==int(i%batch_img_nums)
                    x1, y1 = per_batch_points[in_img_mask1][:, 12], per_batch_points[in_img_mask1][:, 14]
                    in_img_mask2 = per_batch_points[:,11]==int(i%batch_img_nums)
                    x2, y2 = per_batch_points[in_img_mask2][:, 13], per_batch_points[in_img_mask2][:, 15]
                    if (in_img_mask1 | in_img_mask2).sum()==0:
                        continue
                    x, y = torch.cat((x1, x2)), torch.cat((y1, y2))
                    grid = torch.stack((x, y), dim=0).T
                    grid[:,0] = (grid[:,0]/img_metas[batch_id]['pad_shape'][int(i%batch_img_nums)][1]) * 2 - 1  # x
                    grid[:,1] = (grid[:,1]/img_metas[batch_id]['pad_shape'][int(i%batch_img_nums)][0]) * 2 - 1  # y
                    # input(1,1,H,W) grid(1,1,N_pts,2) out(1,1,1,N_pts)
                    out = F.grid_sample(mask_pred_cls.unsqueeze(0), grid.unsqueeze(0).unsqueeze(0), align_corners=False)
                    out = torch.clamp(out, 1e-5, 1-1e-5)
                    out = out.squeeze(0).squeeze(0).squeeze(0)
                    # 0.纯软标签
                    # -plog(q)
                    counts_nums += out.size()[0]
                    seg_scores1 = seg_logits[batch_id][:,j][in_img_mask1].sigmoid()
                    seg_scores2 = seg_logits[batch_id][:,j][in_img_mask2].sigmoid()
                    seg_scores_ = torch.cat((seg_scores1, seg_scores2))
                    weight = torch.ones_like(out)
                    # sample hard label
                    mask1 = (seg_scores_ > 0.7) | (seg_scores_ < 0.3)
                    mask2 = (out > 0.7) | (out < 0.3)
                    mask = mask1 | mask2
                    weight[~mask] = 0.
                    loss_3d_to_2d += -((seg_scores_ * out.log() * weight).sum() + ((1-seg_scores_) * (1-out).log() * weight).sum()) * cls_weight[j]
                    # loss_3d_to_2d += computer_sim_loss(seg_scores_, out).sum()
                    # neg_sample
                    # neg_flag1 = per_batch_points[in_img_mask1][:, [17, 19]]
                    # neg_flag2 = per_batch_points[in_img_mask2][:, [17, 19]]
                    # neg_flag = torch.cat((neg_flag1, neg_flag2), dim=0)
                    # neg_masks = (neg_flag[:, 0]==0) & (neg_flag[:, 1]==1)
                    # loss_3d_to_2d += -((seg_scores_[~neg_masks] * out[~neg_masks].log()).sum() + \
                    #                    ((1-seg_scores_[~neg_masks]) * (1-out[~neg_masks]).log()).sum() + \
                    #                     2 * (1-seg_scores_[neg_masks]).log().sum())
                    # 1. 强制将>0.7的置1，<0.3置0
                    # 1.1 bce loss
                    # targets = seg_scores[i][:,j]
                    # mask_pos = targets > 0.7
                    # mask_neg = targets < 0.3
                    # targets[mask_pos] = 1
                    # targets[mask_neg] = 0
                    # pos_sample_mask = out[mask_pos]
                    # neg_sample_mask = (1-out[mask_neg])
                    # counts_nums += (mask_pos.sum() + mask_neg.sum())
                    # # loss_3d_to_2d += -(pos_sample_mask.log().sum() + neg_sample_mask.log().sum())
                    # # 1.2 Focal loss 计算1.
                    # alpha = 0.8  # 0.25
                    # gamma = 3.0  # 2.0
                    # loss_3d_to_2d += -((alpha*(1-pos_sample_mask).pow(gamma)*pos_sample_mask.log()).sum() + \
                    #                   ((1-alpha)*(1-neg_sample_mask).pow(gamma)*(1-neg_sample_mask).log()).sum())
                    if False:
                        import cv2
                        aimg = aligned_bilinear(imgs[i].unsqueeze(0), 2)  # 1280，1920，3
                        aimg = tensor2imgs(aimg, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],to_rgb=False)[0]
                        aimg = cv2.cvtColor(aimg, cv2.COLOR_RGB2BGR)

                        # 可视化image mask
                        gt_mask = aligned_bilinear(mask_pred_cls.unsqueeze(0), 8).squeeze().squeeze()
                        gt_mask = gt_mask.detach().cpu().numpy()
                        cv2.imwrite('./work_dirs/out_images/pred_mask_cls_id_{}.jpeg'.format(j), gt_mask/gt_mask.max()*255)
                        # 可视化点云自身的mask
                        max_score = seg_scores[i][:,j].max()
                        for k, p in enumerate(points[i]):
                            rgb = int(255 * seg_scores[i][k][j]/max_score)
                            cv2.circle(aimg, (int(p[12]*2),int(p[13]*2)), 1, (rgb, rgb, rgb), -1)
                        cv2.imwrite('./work_dirs/out_images/img_pts_mask_cls_id_{}.jpeg'.format(j), aimg)
                        # 可视化点云投影到图像位置图像的mask
                        max_score = out.max()
                        for k, p in enumerate(points[i]):
                            rgb = int(255 * out[k]/max_score)
                            cv2.circle(aimg, (int(p[12]*2),int(p[13]*2)), 1, (rgb, rgb, rgb), -1)
                        cv2.imwrite('./work_dirs/out_images/img_mask_cls_id_{}.jpeg'.format(j), aimg)

                        ascore = (-(seg_scores[i][:,j] * out.log()+((1-seg_scores[i][:,j]) * (1-out).log())))
                        topk_val, topk_inds = torch.topk(ascore, 5)
                        max_loss = topk_val[0]
                        for k, p in enumerate(points[i][topk_inds]):
                            rgb = int(255 * topk_val[k]/max_loss)
                            cv2.circle(aimg, (int(p[12]*2),int(p[13]*2)), 3, (rgb, rgb, rgb), -1)
                        cv2.imwrite('./work_dirs/out_images/0cls_id{}.jpeg'.format(j), aimg)
        # counts_nums_mean = max(reduce_mean(counts_nums.detach().double()), 1)
        # counts_nums_mean = counts_nums_mean.type(torch.float32)
        if counts_nums != 0:
            loss_3d_to_2d = loss_3d_to_2d * warmup_factor * self.kd_loss_weight / counts_nums
        else:
            loss_3d_to_2d = 0 * mask_logits.sum()
        return loss_3d_to_2d

    def get_points_targets(self, points, roi_points, gt_bboxes, gt_labels, img_metas):
        # points[:, 5]=[0,1]表示是否是背景点，通过3d分支给每个点得分(N,3)，如果类别得分小于阈值，那么就是背景点，
        # points[:, 14]ccl后每个2d box内的点云，这个可以保证绝大多数点都是前景点, ==box.index()获取在roi内的点
        # points[:, 11]=记录在2d box内的点，但一点可以在多个2d box内,这个是记录的点云的索引 >=0获取在2d box内的点
        sample_nums = 100
        neg_sample_nums = 50
        batch_size = len(points)
        points_batch_target = [] # len is batch_size
        points_batch_target_bg = []
        len_fg, len_fg_3d, len_bg, len_bg_3d = [], [], [], []
        for i in range(batch_size):
            points_target = []
            points_target_bg = []
            roi_nums = len(gt_bboxes[i])
            len_fg_, len_fg_3d_, len_bg_, len_bg_3d_ = [], [], [], []
            for j in range(roi_nums):
                # 这里返回两次回波的数据
                in_box_mask = ((points[i][:,12]>=gt_bboxes[i][j][0]) & (points[i][:,12]<gt_bboxes[i][j][2]) &
                            (points[i][:,13]>=gt_bboxes[i][j][1]) & (points[i][:,13]<gt_bboxes[i][j][3]))  # & (points[i][:,5]==0)
                in_box_points = points[i][in_box_mask]
                fg_points = roi_points[i][j]  # (1, 14)
                bg_points_mask = (np.isin(in_box_points[:,0:3].cpu().numpy(),fg_points[:,0:3].cpu().numpy(),assume_unique=True)).sum(1)!=3
                bg_points = in_box_points[bg_points_mask]
                # fg_point_3d = in_box_points[in_box_points[:, 5]==1]  # 3d mask中分为前景点的点
                fg_point_3d = torch.zeros((0,points[i].size()[1]),device=fg_points.device)
                # bg_points_3d = in_box_points[in_box_points[:, 5]==0] # 这个一开始不准，随着训练才准
                bg_points_3d = torch.zeros((0,points[i].size()[1]),device=fg_points.device)

                # 采样
                if len(fg_points) != 0:
                    fg_points = fg_points[torch.randint(low=0,high=len(fg_points),size=(1,min(len(fg_points),sample_nums))).long().squeeze(0)]
                if len(bg_points) != 0:
                    bg_points = bg_points[torch.randint(low=0,high=len(bg_points),size=(1,min(len(bg_points),neg_sample_nums))).long().squeeze(0)]
                if len(fg_point_3d) != 0:
                    fg_point_3d = fg_point_3d[torch.randint(low=0,high=len(fg_point_3d),size=(1,min(len(fg_point_3d),sample_nums))).long().squeeze(0)]
                if len(bg_points_3d) != 0:
                    bg_points_3d = bg_points_3d[torch.randint(low=0,high=len(bg_points_3d),size=(1,min(len(bg_points_3d),neg_sample_nums))).long().squeeze(0)]
                points_target.append(torch.cat((fg_points, fg_point_3d),dim=0))
                points_target_bg.append(torch.cat((bg_points, bg_points_3d),dim=0))
                len_fg_.append(len(fg_points))
                len_fg_3d_.append(len(fg_point_3d))
                len_bg_.append(len(bg_points))
                len_bg_3d_.append(len(bg_points_3d))
                
            points_batch_target.extend(points_target)
            points_batch_target_bg.extend(points_target_bg)
            len_fg.extend(len_fg_)
            len_fg_3d.extend(len_fg_3d_)
            len_bg.extend(len_bg_)
            len_bg_3d.extend(len_bg_3d_)

        return points_batch_target, points_batch_target_bg, len_fg, len_fg_3d, len_bg, len_bg_3d

    def get_points_targets_v2(self, points, gt_bboxes, gt_labels, img_metas, seg_scores):
        batch_size = len(points)
        batch_points_targets = [] # len is batch_size
        batch_grid_targets = []
        batch_seg_scores = []
        fg_len = []
        run_id_count = 0
        for i in range(batch_size):
            points_targets = []
            seg_scores_targets = []
            roi_nums = len(gt_bboxes[i])
            fg_len_ = []
            grid_targets = []
            # 区分不同batch的run id
            max_run_id = points[i][:,14].max() + 1
            relevant_mask = points[i][:,14] >= 0
            points[i][:,14][relevant_mask] = points[i][:,14][relevant_mask] + run_id_count
            run_id_count += max_run_id

            grid = points[i][:,12:14]/img_metas[i]['scale_factor'][0]
            grid[:,0] = (grid[:,0]/img_metas[i]['pad_shape'][1]) - 1  # x
            grid[:,1] = (grid[:,1]/img_metas[i]['pad_shape'][0]) - 1  # y
            for j in range(roi_nums):
                w = gt_bboxes[i][j][2] - gt_bboxes[i][j][0]
                h = gt_bboxes[i][j][3] - gt_bboxes[i][j][1]
                x1,y1 = gt_bboxes[i][j][0]-w*0.05, gt_bboxes[i][j][1]-h*0.05
                x2,y2 = gt_bboxes[i][j][2]+w*0.05, gt_bboxes[i][j][3]+h*0.05
                # 选择主雷达的点云（两次回波都要）
                in_box_mask = ((points[i][:,12]>=x1) & (points[i][:,12]<x2) &
                               (points[i][:,13]>=y1) & (points[i][:,13]<y2) &
                               (points[i][:,6]==0))
                if in_box_mask.sum()==0:
                    pass
                in_box_points = points[i][in_box_mask]
                in_box_grid = grid[in_box_mask]
                in_box_scores = seg_scores[i][in_box_mask]
                # assert len(in_box_grid) != 0 如果上面获取的仅仅是主雷达的结果，那么就会出现2D框内没有点
                fg_len_.append(int(in_box_mask.sum()))

                points_targets.append(in_box_points)
                grid_targets.append(in_box_grid)
                seg_scores_targets.append(in_box_scores)

            batch_points_targets.extend(points_targets)
            batch_grid_targets.extend(grid_targets)
            fg_len.extend(fg_len_)
            batch_seg_scores.extend(seg_scores_targets)

        return torch.cat(batch_grid_targets, dim=0), batch_points_targets, torch.tensor(fg_len, device=points[0].device), batch_seg_scores

    @force_fp32(apply_to=('mask_logits',))  # TODO add the sem_loss
    def loss(self,
             imgs,
             img_metas,
             mask_logits,   # pred mask N_samp x 1 x 2fpn_h x 2fpn_w
             gt_inds,       # N_samp,1
             gt_bboxes,     # gt_bboxes_nums, 4
             gt_masks,
             gt_labels,
             points,        # 
             pseudo_labels, # 
             seg_scores=None,
             img_inds=None,
             num_classes=3,
             flag=1):
        self._iter += 1
        # similarities matrix=(N_gt, 160, 240), gt_bitmasks=(N_gt, 160, 240),
        # similarities的结果是将每张图片的相似度矩阵重复了N_gt次，即(similarities[1][0]==similarities[1][1]).all()=True
        # 每个2D Box对应一张bitmask_full,只有2D box内的mask值为1
        # bitmask_full是用来过滤pad的操作产生的不相关像素，pad是从bottom和right pad的,gt_bitmasks是bitmask_full按照间隔取值降采样程mask大小的
        similarities, gt_bitmasks, bitmasks_full, en_bitmasks, en_bitmasks_full = \
            self.get_targets(gt_bboxes, gt_masks, imgs, img_metas, None, None, None)
        
        mask_scores = mask_logits.sigmoid()          # (N_samp,1,320,480)
        gt_bitmasks = torch.cat(gt_bitmasks, dim=0)  # (N_gt,160,240)
        gt_bitmasks = gt_bitmasks[gt_inds].unsqueeze(1).to(mask_scores.dtype)  # (N_gt, 160, 240)-->(N_samp,1,160,240)复制几次mask，多个mask对应一个gt_mask
        if en_bitmasks is not None:
            en_bitmasks = torch.cat(en_bitmasks, dim=0)  # (N_gt,160,240)
            en_bitmasks = en_bitmasks[gt_inds].unsqueeze(1).to(mask_scores.dtype)  # (N_gt, 160, 240)-->(N_samp,1,160,240)复制几次mask，多个mask对应一个gt_mask
        
        if not (mask_scores.size()==gt_bitmasks.size()):
            mask_scores = mask_scores.reshape((gt_bitmasks.size()))
        assert mask_scores.dim()==4 and gt_bitmasks.dim()==4
        
        losses = {}

        if len(mask_scores) == 0 or flag == 0:  # there is no instances detected
            dummy_loss = 0 * mask_scores.sum()
            if not self.boxinst_enabled:
                losses["loss_mask"] = dummy_loss
                return losses
            else:
                losses["loss_prj"] = dummy_loss
                losses["loss_pairwise"] = dummy_loss
                if self.points_enabled:
                    losses["loss_3d_to_2d"] = dummy_loss
                return losses

        if self.boxinst_enabled:
            img_color_similarity = torch.cat(similarities, dim=0)  # [(14,8,320,480),(N_gt,8,320,480)]
            img_color_similarity = img_color_similarity[gt_inds].to(dtype=mask_scores.dtype)  # [64,8,320,480]

            # 1. projection loss
            loss_prj_term = compute_project_term(mask_scores, gt_bitmasks)

            # 2. pairwise loss
            # all pixels log, [64,8,320,480]
            pairwise_losses = pairwise_nlog(mask_logits, self.pairwise_size, self.pairwise_dilation)
            # (> color sim threshold points and in gt_boxes points) intersection
            # weights, [64,8,320,480]
            weights = (img_color_similarity >= self.pairwise_color_thresh).float() * gt_bitmasks.float()

            loss_pairwise = (pairwise_losses * weights).sum() / weights.sum().clamp(min=1.0)

            warmup_factor = min(self._iter.item() / float(self._warmup_iters), 1.0)
            loss_pairwise = loss_pairwise * warmup_factor

            losses.update({
                "loss_prj": loss_prj_term,
                "loss_pairwise": loss_pairwise,
            })

        # if use real gt_mask
        else:
            mask_losses = dice_coefficient(mask_scores, gt_bitmasks)
            loss_mask = mask_losses.mean()
            losses["loss_mask"] = loss_mask

        return losses

    def get_targets(self, gt_bboxes, gt_masks, img, img_metas, points, points2img, points2img_idx):
        """get targets, inputs: list,list,list,list,list
        """
        if self.boxinst_enabled:
            if len(img_metas) != len(img):
                batch_nums = int(len(gt_bboxes)/len(img_metas))
                padded_image_masks = []
                padded_images = []
                new_img_metas = [dict() for _ in range(len(img))]
                for bsz in range(len(img_metas)):
                    for key, value in img_metas[bsz].items():
                        for i in range(batch_nums):
                            if isinstance(value, list) and len(value)==batch_nums:
                                new_img_metas[int(bsz*batch_nums+i)][key] = value[i]
                            else:
                                new_img_metas[int(bsz*batch_nums+i)][key] = value

                for i in range(len(new_img_metas)):
                    original_image_masks = torch.ones(new_img_metas[i]['img_shape'][:2], dtype=torch.float32, device=img.device)
                    # if 'crop_shape' in new_img_metas[i].keys():
                    #     im_h = new_img_metas[i]['crop_shape'][0]
                    # else:
                    im_h = new_img_metas[i]['ori_shape'][0]  #1280 or 886
                    pixels_removed = int(
                        self.bottom_pixels_removed * float(new_img_metas[i]['img_shape'][0]) / float(im_h)
                    )
                    if pixels_removed > 0:
                        original_image_masks[-pixels_removed:, :] = 0  # 最后几行为0

                    padding = (0, img.shape[3] - new_img_metas[i]['img_shape'][1],
                            0, img.shape[2] - new_img_metas[i]['img_shape'][0])  # [left, right, top, bottom]

                    padded_image_mask = F.pad(original_image_masks, pad=padding)
                    padded_image_masks.append(padded_image_mask)

                    original_image = get_original_image(img[i], new_img_metas[i])  # get RGB image tensor, ori_image already rgb, no need to swap dim
                    # original_image = original_image.to(img.device)
                    if False:
                        import cv2
                        aimg = original_image.permute(1,2,0).cpu().numpy().astype(np.uint8)
                        aimg = cv2.cvtColor(aimg, cv2.COLOR_RGB2BGR)
                        for b, gt_bbox in enumerate(gt_bboxes[i]):
                            cv2.rectangle(aimg, (int(gt_bbox[0]), int(gt_bbox[1])), (int(gt_bbox[2]), int(gt_bbox[3])), (0, 255, 0), 1)
                        # cv2.imwrite('ori_img.jpg',aimg)
                        # for b, gt_bbox in enumerate(img_metas[0]['ann_info']['gt_bboxes'][i]):
                            # cv2.rectangle(aimg, (int(gt_bbox[0]/2), int(gt_bbox[1]/2)), (int(gt_bbox[2]/2), int(gt_bbox[3]/2)), (255, 0, 0), 1)
                        cv2.imwrite('ori_img.jpg',aimg)
                    padded_image = F.pad(original_image, pad=padding)
                    padded_images.append(padded_image)

                padded_image_masks = torch.stack(padded_image_masks, dim=0)  # (N,h,w)
                padded_images = torch.stack(padded_images, dim=0)  # (N,3,h,w)
                # get color similarities and mask. output is list,list,list
                similarities, bitmasks, bitmasks_full, en_bitmasks, en_bitmasks_full = self.get_bitmasks_from_boxes(gt_bboxes, padded_images,
                                                                        padded_image_masks, None)                   

            else:
                padded_image_masks = []
                padded_images = []

                for i in range(len(img_metas)):
                    original_image_masks = torch.ones(img_metas[i]['img_shape'][:2], dtype=torch.float32, device=img.device)  # 640,960

                    im_h = img_metas[i]['ori_shape'][0]  #1280 or 886
                    pixels_removed = int(
                        self.bottom_pixels_removed * float(img_metas[i]['img_shape'][0]) / float(im_h)
                    )
                    if pixels_removed > 0:
                        original_image_masks[-pixels_removed:, :] = 0

                    padding = (0, img.shape[3] - img_metas[i]['img_shape'][1],
                            0, img.shape[2] - img_metas[i]['img_shape'][0])  # [left, right, top, bottom]

                    padded_image_mask = F.pad(original_image_masks, pad=padding)
                    padded_image_masks.append(padded_image_mask)

                    original_image = get_original_image(img[i], img_metas[i])  # get RGB image tensor, ori_image already rgb, no need to swap dim
                    # original_image = original_image.to(img.device)
                    padded_image = F.pad(original_image, pad=padding)
                    padded_images.append(padded_image)

                padded_image_masks = torch.stack(padded_image_masks, dim=0)  # (N,h,w)
                padded_images = torch.stack(padded_images, dim=0)  # (N,3,h,w)
                # get color similarities and mask. output is list,list,list
                similarities, bitmasks, bitmasks_full, en_bitmasks, en_bitmasks_full = self.get_bitmasks_from_boxes(gt_bboxes, padded_images,
                                                                        padded_image_masks, None)       
                # if use points info
                if self.points_enabled and points is not None:
                    
                    # v1
                    # gt_points_image, gt_points_image_masks, gt_points_ind = self.get_gt_points_image(points, gt_bboxes, img_metas)
                    # gt_points_image = gt_points_image.to(points[0].device)
                    # gt_points_image_masks = gt_points_image_masks.to(points[0].device)

                    # v2
                    gt_points_image, gt_points_image_masks, gt_points_ind = self.get_gt_points_image_v2(gt_bboxes, points2img, points2img_idx)
                    dis_simlarities, pt_img_bitmasks, pt_img_bitmasks_full = \
                        self.get_bitmasks_from_boxes(gt_bboxes, 
                                                    gt_points_image,           # torch (B.3.640.960)
                                                    gt_points_image_masks,     # torch(B.640.960)
                                                    points)

                    return similarities, bitmasks, bitmasks_full, dis_simlarities, pt_img_bitmasks, pt_img_bitmasks_full, gt_points_ind         
        else:
            start = int(self.out_stride // 2)
            bitmasks = []
            for i, mask in enumerate(gt_masks):
                if len(gt_bboxes[i]) == 0:
                    continue
                mask = mask.to_tensor(dtype=torch.long, device=img.device)
                bitmasks.append(mask[:, start::self.out_stride, start::self.out_stride])
                
            similarities = None
            bitmasks_full = gt_masks
            return similarities, bitmasks, bitmasks_full, None, None

        return similarities, bitmasks, bitmasks_full, en_bitmasks, en_bitmasks_full

    def get_bitmasks_from_boxes(self, gt_bboxes, padded_images, padded_image_masks, points):
        h, w = padded_images.shape[2:]
        stride = self.out_stride
        start = int(stride // 2)

        assert padded_images.size(2) % stride == 0
        assert padded_images.size(3) % stride == 0

        if points is not None:
            # downsampled_images = F.max_pool2d(padded_images.float(), kernel_size=stride, stride=stride, padding=0)
            downsampled_images = self.mean_pool(padded_images, padded_image_masks, kernel_size=stride, stride=stride)
            downsampled_image_masks = F.max_pool2d(padded_image_masks.float(), kernel_size=stride, stride=stride, padding=0)
        else:
            downsampled_images = F.avg_pool2d(padded_images.float(), kernel_size=stride, stride=stride, padding=0)
            downsampled_image_masks = padded_image_masks[:, start::stride, start::stride]

        similarities = []
        bitmasks = []
        bitmasks_full = []
        en_bitmasks = []
        en_bitmasks_full = []
        for i, per_img_gt_bboxes in enumerate(gt_bboxes):
            if points is not None:
                points_image = downsampled_images[i]
                points_image = torch.as_tensor(points_image, device=padded_image_masks.device, dtype=torch.float32)
                points_image = points_image.unsqueeze(0)  # (1,3,320,480)
                # 得到一个点与周围几个点的距离(exp(-d))
                image_color_similarity = get_image_color_similarity(
                                                                points_image, 
                                                                downsampled_image_masks[i], 
                                                                self.pairwise_size,
                                                                1,
                                                                self.points_enabled)  # [1,8,320,480]
            else:
                image_lab = color.rgb2lab(downsampled_images[i].byte().permute(1, 2, 0).cpu().numpy())
                image_lab = torch.as_tensor(image_lab, device=padded_image_masks.device, dtype=torch.float32)
                image_lab = image_lab.permute(2, 0, 1)[None]  # [160,240,3]-->[1,3,160,240]
                image_color_similarity = get_image_color_similarity(
                                                                image_lab,
                                                                downsampled_image_masks[i],
                                                                self.pairwise_size,
                                                                self.pairwise_dilation)  # [N,8,320,480]

            per_im_bitmasks = []
            per_im_bitmasks_full = []
            per_im_en_bitmasks = []
            per_im_en_bitmasks_full = []
            for per_box in per_img_gt_bboxes:  # [x1, y1, x2, y2]
                bitmask_full = torch.zeros((h, w), device=per_box.device).float()  # 640,960
                enlarge_bitmask_full = torch.zeros((h, w), device=per_box.device).float()  # 640,960
                bitmask_full[int(per_box[1]): int(per_box[3]) + 1, int(per_box[0]):int(per_box[2]) + 1] = 1.0
                # downsample
                bitmask = bitmask_full[start::stride, start::stride]
                # enlarge_bbox
                x1, y1, x2, y2 = per_box
                w_, h_ = (x2 - x1), (y2 - y1)
                x1, y1 = (per_box[0] - w_ * 0.05).clamp(min=0.), (per_box[1] - h_ * 0.05).clamp(min=0.)
                x2, y2 = (per_box[2] + w_ * 0.05).clamp(max=w), (per_box[3] + h_ * 0.05).clamp(max=h)
                enlarge_bitmask_full[int(y1): int(y2) + 1, int(x1):int(x2) + 1] = 1.0
                enlarge_bitmask = enlarge_bitmask_full[start::stride, start::stride]
                assert bitmask.size(0) * stride == h
                assert bitmask.size(1) * stride == w

                per_im_bitmasks.append(bitmask)
                per_im_bitmasks_full.append(bitmask_full)
                per_im_en_bitmasks.append(enlarge_bitmask)
                per_im_en_bitmasks_full.append(enlarge_bitmask_full)
            if len(per_im_bitmasks)==0:
                continue
            per_im_bitmasks = torch.stack(per_im_bitmasks, dim=0)
            per_im_bitmasks_full = torch.stack(per_im_bitmasks_full, dim=0)
            per_im_en_bitmasks = torch.stack(per_im_en_bitmasks, dim=0)
            per_im_en_bitmasks_full = torch.stack(per_im_en_bitmasks_full, dim=0)

            similarities.append(torch.cat([image_color_similarity for _ in range(len(per_img_gt_bboxes))], dim=0))
            bitmasks.append(per_im_bitmasks)
            bitmasks_full.append(per_im_bitmasks_full)
            en_bitmasks.append(per_im_en_bitmasks)
            en_bitmasks_full.append(per_im_en_bitmasks_full)
        return similarities, bitmasks, bitmasks_full, en_bitmasks, en_bitmasks_full


    def get_gt_points_image(self, points_, gt_bboxes_, img_metas):
        """ 
        Return: torch,torch 
        """

        gt_points_images = []
        gt_points_image_masks = []
        gt_points_inds = []

        for i, per_img_metas in enumerate(img_metas):
            sample_img_id = per_img_metas['sample_img_id']
            points = points_[i]
            gt_bboxes = gt_bboxes_[i]  # /per_img_metas['scale_factor'][0]

            # 1. 过滤掉没有投影到相机的点, 这个写在了主模型里
            in_img_points = points
            # mask = (points[:, 6] == sample_img_id) | (points[:, 7] == sample_img_id)  # 真值列表
            mask_id = torch.tensor([i for i in range(len(points))], device=points.device) # 全局索引值
            # in_img_points = points[mask]

            # 2. 得到在2D bboxes内的点
            gt_mask = torch.tensor([False for _ in range(points.shape[0])]).to(points.device)
            # 使用所有的gt bbox进行筛选
            for gt_bbox in gt_bboxes:
                # if 0 cam 8,10（列，行）
                gt_mask_0 = (((in_img_points[:, 8] > gt_bbox[0]) & (in_img_points[:, 8] < gt_bbox[2])) &
                            ((in_img_points[:, 10] > gt_bbox[1]) & (in_img_points[:, 10] < gt_bbox[3])) &
                            (in_img_points[:, 6] == sample_img_id))
                # if 1 cam 9,11
                gt_mask_1 = (((in_img_points[:, 9] > gt_bbox[0]) & (in_img_points[:, 9] < gt_bbox[2])) &
                            ((in_img_points[:, 11] > gt_bbox[1]) & (in_img_points[:, 11] < gt_bbox[3])) &
                            (in_img_points[:, 7] == sample_img_id))
                gt_mask = gt_mask_0 | gt_mask_1 | gt_mask
            # 得到id全局索引值
            gt_points_ind = mask_id[gt_mask]
            # 得到所有投射到2D框内的点的值(N,12)
            in_gt_bboxes_points = in_img_points[gt_mask]

            # 3. 将得到的点云映射到图片
            ori_points_image = torch.zeros(per_img_metas['img_shape'], dtype=torch.float)
            ori_points_image_mask = torch.zeros(per_img_metas['img_shape'][:2])  # 是否有值的mask
            # 将点投影到原始图片
            for point in in_gt_bboxes_points:
                if point[6] == sample_img_id:
                    x_0 = point[8]
                    y_0 = point[10]
                    ori_points_image[int(y_0), int(x_0)] = torch.tensor([point[0], point[1], point[2]])  # int(a) == torch.floor(a)
                    ori_points_image_mask[int(y_0), int(x_0)] = 1
                if point[7] == sample_img_id:
                    x_1 = point[9]
                    y_1 = point[11]
                    ori_points_image[int(y_1), int(x_1)] = torch.tensor([point[0], point[1], point[2]])
                    ori_points_image_mask[int(y_1), int(x_1)] = 1
            
            ori_points_image = ori_points_image.permute(2,0,1)
            ori_points_image_mask = ori_points_image_mask

            gt_points_images.append(ori_points_image)
            gt_points_image_masks.append(ori_points_image_mask)
            gt_points_inds.append(gt_points_ind)

        gt_points_images = torch.stack(gt_points_images, dim=0)  # ()
        gt_points_image_masks = torch.stack(gt_points_image_masks, dim=0)
        gt_points_inds = gt_points_inds  # 点云长度不同，所以只能是列表
        return gt_points_images, gt_points_image_masks, gt_points_inds

    def get_gt_points_image_v2(self, gt_bboxes_, points2img, points2img_idx):
        """ 
        Return: torch,torch
        """
        points2img = torch.stack(points2img).permute([0,3,1,2])   # torch (B.3.640.960)
        points2img_idx = (torch.stack(points2img_idx)>=0).type(torch.int)
        points2img_mask = (points2img_idx * 0).type(torch.int)  # torch(B.640.960)

        for i, gt_bboxs in enumerate(gt_bboxes_):
            gt_bboxs = gt_bboxs.type(torch.int)
            for j, gt_bbox in enumerate(gt_bboxs):
                points2img_mask[i][gt_bbox[1]:gt_bbox[3], gt_bbox[0]:gt_bbox[2]] = 1
            for j in range(points2img.shape[1]):
                points2img[i][j] = points2img[i][j] * points2img_mask[i]

            points2img_idx[i] = points2img_idx[i] * points2img_mask[i]

        return points2img, points2img_idx, None
    
    def mean_pool(self, points_image, points_iamge_mask, kernel_size=4, stride=4):
        # 需要添加过滤部分，参考lwsis网络中过滤点的操作
        n, h, w = points_iamge_mask.shape
        mean_pools = []
        for i, per_img in enumerate(points_image):  # 表示几张图片
            mean_pool = []
            unfold_pt_img_mask = F.unfold(points_iamge_mask[i].reshape((1, 1, h, w)).float(), kernel_size=kernel_size, stride=stride)
            unfold_mask = unfold_pt_img_mask.sum(dim=1)
            # X,Y,Z
            for j in range(len(per_img)):
                unfold_pt_img = F.unfold(per_img[j].reshape((1, 1, h, w)).float(), kernel_size=kernel_size, stride=stride)
                unfold_sum = unfold_pt_img.sum(dim=1)
                # 如果分母为0，也就是没有点，那么池化后的结果是nan
                unfold_mean = unfold_sum / unfold_mask
                # 0替换nan
                unfold_mean = torch.nan_to_num(unfold_mean, nan=0)
                mean_pool.append(unfold_mean.reshape(int(h/kernel_size), int(w/kernel_size)))
            mean_pools.append(torch.stack(mean_pool, dim=0))
        return torch.stack(mean_pools, dim=0)

    def copyloss(self,img_metas,imgs,points,roi_points,gt_inds,img_inds,seg_scores,losses,warmup_factor):
        if self.points_enabled:
            use_aligned_bilinear = True
            loss_dis_pairwise = torch.tensor(0, device=imgs.device, dtype=torch.float)
            loss_3d_to_2d = torch.tensor(0, device=imgs.device, dtype=torch.float)
            # 1. 获取targets，长度为2dbox的数量，N_gt,采样的点数
            # points_batch_target_3d是从3d mask获取的，这个一开始是不准的，所以loss就warmup形式
            batch_points_targets, batch_points_targets_bg, len_fg, len_fg_3d, \
                len_bg, len_bg_3d = self.get_points_targets(points, roi_points, gt_bboxes, gt_labels, img_metas)
            dist = [4.73, 0.91, 1.81]  # 参考mdet3d second的anchor配置 l最长边的大小
            gt_labels = torch.cat(gt_labels)
            gt_bboxes = torch.cat(gt_bboxes, dim=0)
            scale = img_metas[0]['scale_factor'][0]
            if use_aligned_bilinear:
                mask_scores = aligned_bilinear(mask_scores, int(self.out_stride/scale))  # 双线性差值提高分辨率
            mask_scores = mask_scores.squeeze()
            
            for i in range(len(mask_scores)):  # 长度是mask_pred，因为len(mask_pred)与len(torch.cat(gt_box))是不一致的
                gt_inds_ = gt_inds[i]
                gt_label_ = gt_labels[gt_inds_]
                gt_bbox = gt_bboxes[gt_inds_]
                l1, l2, l3, l4 = len_fg[gt_inds_], len_fg_3d[gt_inds_], len_bg[gt_inds_], len_bg_3d[gt_inds_]

                # 得到每个点对应的mask值
                points_targets = batch_points_targets[gt_inds_]
                x = (points_targets[:,12]/scale).long()
                y = (points_targets[:,13]/scale).long()
                mask_scores2points = mask_scores[i][y,x]  # 每个points对应2d mask scores[0,1]
                points_targets_bg = batch_points_targets_bg[gt_inds_]
                x = (points_targets_bg[:,12]/scale).long()
                y = (points_targets_bg[:,13]/scale).long()
                mask_scores2points_bg = mask_scores[i][y,x]  # 每个points对应2d mask scores[0,1]
                mask_scores2points_all = torch.cat((mask_scores2points, mask_scores2points_bg),dim=0)
                points_targets_all = torch.cat((points_targets[:,0:2], points_targets_bg[:,0:2]),dim=0)
                len_xy = 2
                matrix_size = l1 + l2 + l3 + l4
                # 1. 计算点与点之间的p(y=1)矩阵(size,size)
                # 计算P(ye=1)=m*m+(1-m)*(1-m) -log(P) 当两者颜色相近，那么概率值就会拉到一块，背景和前景就区分开了
                # torch.mul是对应元素相乘，利用广播的性质进行运算
                pred_prob = torch.mul(mask_scores2points_all.repeat(matrix_size,1).T, mask_scores2points_all) + \
                            torch.mul((1-mask_scores2points_all.repeat(matrix_size,1).T), (1-mask_scores2points_all))
                pred_prob = torch.clamp(pred_prob, 1e-5, 1-1e-5)  # clamp，防止出现inf值
                # pos_prob = - torch.log(pred_prob)
                neg_prob = - torch.log(1 - pred_prob)
                mask_all = torch.ones_like(pred_prob, device=imgs.device, dtype=torch.float32)
                # 取上三角
                mask_all = torch.triu(mask_all)
                # mask掉负-负位置
                mask_all[(l1+l2):matrix_size,(l1+l2):matrix_size]=0
                # mask自身
                eye_mask = (~ torch.eye(matrix_size, device=imgs.device, dtype=torch.bool)).type(torch.int)
                mask_all = mask_all*eye_mask
                if mask_all.sum() == 0:
                    continue
                # 2. 进行每个点的x,y距离计算
                dis2matrix = torch.zeros((matrix_size, matrix_size, len_xy),device=imgs.device)
                for j in range(len_xy):
                    points_coor = points_targets_all[:, j]
                    dis2matrix[:,:,j] = points_coor.repeat(matrix_size,1).T - points_targets_all[:, j]
                dis_matrix = torch.sum(dis2matrix**2, dim=2).sqrt()

                threshold_mask = dis_matrix <= dist[gt_label_]

                # 既拉又推
                # loss_dis_pairwise += ((pos_prob*threshold_mask*mask_all).sum() + \
                #     (neg_prob*(~threshold_mask)*mask_all*10).sum()) / mask_all.sum()

                # 认为拉的状态会破坏掉基于rgb的pairwise loss，所以只采用推即可
                if ((~threshold_mask)*mask_all).sum() != 0:
                    loss_dis_pairwise += (neg_prob*(~threshold_mask)*mask_all).sum() / ((~threshold_mask)*mask_all).sum()

                # 全监督 mask_logits seg_logits 软目标 使用了两次回波数据
                img_id = img_inds[i]
                in_box_points_mask = ((points[img_id][:, 12] >= gt_bbox[0]) & (points[img_id][:, 12] < gt_bbox[2]) &
                                      (points[img_id][:, 13] >= gt_bbox[1]) & (points[img_id][:, 13] < gt_bbox[3]))
                in_box_points = points[img_id][in_box_points_mask]
                in_box_points_scores = seg_scores[img_id][in_box_points_mask][:, gt_label_]

                x = (in_box_points[:,12]/scale).long()
                y = (in_box_points[:,13]/scale).long()
                mask_scores2complete = mask_scores[i][y, x]
                mask_scores2complete = torch.clamp(mask_scores2complete, 1e-5, 1-1e-5)
                loss_3d_to_2d += -(((in_box_points_scores * mask_scores2complete.log()).sum()) + \
                    (((1-in_box_points_scores) * (1-mask_scores2complete).log()).sum())) / (in_box_points_mask.sum())
                # 利用range来监督，我们只使用第一次回波和主雷达的数据
                # 利用range image的射线深度关系来监督

            warmup_factor_ground = min(self._iter.item() / float(10000.), 1.0)
            
            losses.update({
                "loss_dis_pairwise": loss_dis_pairwise * warmup_factor * 0.0015,  # * 
                "loss_3d_to_2d": loss_3d_to_2d * warmup_factor_ground * 0.00125,      # * 0.005
            })

            # 备份
            grid_targets, points_targets, fg_len, seg_scores_ = self.get_points_targets_v2(points, gt_bboxes, gt_labels, img_metas, seg_scores)
            gt_labels = torch.cat(gt_labels)
            gt_bboxes = torch.cat(gt_bboxes, dim=0)
            loss_2d_run_seg = torch.tensor(0, device=imgs.device, dtype=torch.float)
            # loss_3d_to_2d = torch.tensor(0, device=imgs.device, dtype=torch.float)
            # 采样 将归一化坐标[-1,+1]重复gt_labels数量,然后依据gt_inds来分配
            grid_targets = torch.repeat_interleave(grid_targets.unsqueeze(0), len(gt_labels), 0).unsqueeze(1)[gt_inds]
            # (N, 1, 160, 240) (N, 1, (N1+N2+...), 2)-->(N, 1, 1, (N1+N2+...))
            grid_sample_scores = F.grid_sample(mask_scores, grid_targets, align_corners=False)
            grid_sample_scores = grid_sample_scores.squeeze(1).squeeze(1)
            grid_sample_scores = torch.clamp(grid_sample_scores, 1e-5, 1-1e-5)  # clamp，防止出现inf值
            # 获取点对应的mask scores和seg id
            ori_fg_len = fg_len
            fg_len = fg_len[gt_inds]
            scores_with_id = torch.ones((int(fg_len.sum()), 2), device=imgs.device) * -1
            nums = 0
            for i in range(len(gt_inds)):
                gt_ind = gt_inds[i]
                gt_label = gt_labels[gt_ind]

                if gt_ind == 0:
                    scores_with_id[:,0][nums: nums+fg_len[i]] = grid_sample_scores[i][0: ori_fg_len[gt_ind]]
                    scores_with_id[:,1][nums: nums+fg_len[i]] = points_targets[gt_ind][:,14]  # get seg id
                else:
                    scores_with_id[:,0][nums: nums+fg_len[i]] = grid_sample_scores[i][ori_fg_len[:gt_ind].sum(): ori_fg_len[:gt_ind+1].sum()]
                    scores_with_id[:,1][nums: nums+fg_len[i]] = points_targets[gt_ind][:,14]  # get seg id

                nums += fg_len[i]
        # 备份
        if self.points_enabled:
            grid_targets, points_targets, fg_len, seg_scores_ = self.get_points_targets_v2(points, gt_bboxes, gt_labels, img_metas, seg_scores)
            gt_labels = torch.cat(gt_labels)
            gt_bboxes = torch.cat(gt_bboxes, dim=0)
            loss_2d_run_seg = torch.tensor(0, device=imgs.device, dtype=torch.float)
            # 采样 将归一化坐标[-1,+1]重复gt_labels数量,然后依据gt_inds来分配
            grid_targets = torch.repeat_interleave(grid_targets.unsqueeze(0), len(gt_labels), 0).unsqueeze(1)[gt_inds]
            # (N, 1, 160, 240) (N, 1, (N1+N2+...), 2)-->(N, 1, 1, (N1+N2+...))
            grid_sample_scores = F.grid_sample(mask_scores, grid_targets, align_corners=False)
            grid_sample_scores = grid_sample_scores.squeeze(1).squeeze(1)
            grid_sample_scores = torch.clamp(grid_sample_scores, 1e-5, 1-1e-5)  # clamp，防止出现inf值
            # 获取点对应的mask scores和seg id
            ori_fg_len = fg_len
            fg_len = fg_len[gt_inds]
            scores_with_id = torch.ones((int(fg_len.sum()), 2), device=imgs.device) * -1
            nums = 0
            for i in range(len(gt_inds)):
                gt_ind = gt_inds[i]
                gt_label = gt_labels[gt_ind]

                if gt_ind == 0:
                    scores_with_id[:,0][nums: nums+fg_len[i]] = grid_sample_scores[i][0: ori_fg_len[gt_ind]]
                    scores_with_id[:,1][nums: nums+fg_len[i]] = points_targets[gt_ind][:,14]  # get seg id
                else:
                    scores_with_id[:,0][nums: nums+fg_len[i]] = grid_sample_scores[i][ori_fg_len[:gt_ind].sum(): ori_fg_len[:gt_ind+1].sum()]
                    scores_with_id[:,1][nums: nums+fg_len[i]] = points_targets[gt_ind][:,14]  # get seg id

                nums += fg_len[i]
            # 可视化代码
            visual = False
            if visual:
                import cv2
                for i in range(int((img_inds==0).sum())):
                    aimg = aligned_bilinear(imgs[0].unsqueeze(0), 2)  # 1280，1920，3
                    aimg = tensor2imgs(aimg, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],to_rgb=False)[0]
                    aimg = cv2.cvtColor(aimg, cv2.COLOR_RGB2BGR)
                    gt_box_ind = gt_inds[i]  # 选择可视化哪个box对应的点和mask分数
                    gt_mask_id = torch.where(gt_inds==gt_box_ind)[0]
                    if len(gt_mask_id) == 0:
                        continue
                    gt_mask_id = gt_mask_id[0]
                    gt_mask = mask_scores[gt_mask_id] # 同一个2dbox对应多个mask_pred, 只选择第一个mask就好
                    # 可视化mask
                    gt_mask = aligned_bilinear(gt_mask.unsqueeze(0), 8).squeeze().squeeze()
                    gt_mask = gt_mask.detach().cpu().numpy()
                    cv2.imwrite('./work_dirs/out_images/pred_mask_{}.jpeg'.format(gt_box_ind), gt_mask/gt_mask.max()*255)
                    # 可视化扩大gt对应的点云
                    gt_bbox = gt_bboxes[gt_box_ind]*2
                    cv2.rectangle(aimg, (int(gt_bbox[0]), int(gt_bbox[1])), (int(gt_bbox[2]), int(gt_bbox[3])), (0, 255, 0), 2)
                    for j, p in enumerate(points_targets[gt_box_ind]):
                        cv2.circle(aimg, (int(p[12]*2),int(p[13]*2)),1,(255,255,255),-1)
                    cv2.imwrite('./work_dirs/out_images/img_pts_{}.jpeg'.format(gt_box_ind), aimg)
                    # 可视化点云自身的mask值
                    max_score = seg_scores_[gt_box_ind][:, gt_label].max()
                    gt_label = gt_labels[gt_box_ind]
                    for j, p in enumerate(points_targets[gt_box_ind]):
                        rgb = int(255 * seg_scores_[gt_box_ind][j][gt_label]/max_score)
                        cv2.circle(aimg, (int(p[12]*2),int(p[13]*2)), 1, (rgb, rgb, rgb), -1)
                    cv2.imwrite('./work_dirs/out_images/img_pts_mask{}.jpeg'.format(gt_box_ind), aimg)
                    # 可视化点云投到图像位置 图像的mask
                    grid_sample_scores_ind = grid_sample_scores[gt_mask_id]
                    grid_sample_scores_ind = grid_sample_scores_ind[ori_fg_len[:gt_box_ind].sum(): ori_fg_len[:gt_box_ind+1].sum()]
                    max_score2 = grid_sample_scores_ind.max()
                    for j, p in enumerate(points_targets[gt_box_ind]):
                        rgb = int(255 * grid_sample_scores_ind[j]/max_score2)
                        cv2.circle(aimg, (int(p[12]*2),int(p[13]*2)), 1, (rgb, rgb, rgb), -1)
                    cv2.imwrite('./work_dirs/out_images/img_mask{}.jpeg'.format(gt_box_ind), aimg)

            range_sets, inv_inds, counts = torch.unique(scores_with_id[:,1], return_inverse=True, return_counts=True)
            if False:
                import cv2
                aimg = aligned_bilinear(imgs[0].unsqueeze(0), 2)  # 1280，1920，3
                aimg = tensor2imgs(aimg, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],to_rgb=False)[0]
                aimg = cv2.cvtColor(aimg, cv2.COLOR_RGB2BGR)
                pp = torch.cat(points_targets, dim=0)
                max_run = points[0][:14].max()
                for i in range(len(range_sets)):
                    rgb = np.array([np.random.randint(255) for _ in range(3)])
                    p = pp[pp[:,14] == range_sets[i]]
                    if range_sets[i]>max_run:
                        continue
                    if counts[i] < 4:
                        continue
                    for j in p:
                        x = int(j[12]*2)
                        y = int(j[13]*2)
                        cv2.circle(aimg, (x,y),2,rgb.tolist(),-1)
                cv2.imwrite('run_img.jpeg', aimg)
            out = torch.zeros((range_sets.shape), device=imgs.device)
            out.scatter_add_(0, inv_inds, scores_with_id[:,0])
            ex = out/counts
            out2 = torch.zeros((range_sets.shape), device=imgs.device)
            out2.scatter_add_(0, inv_inds, scores_with_id[:,0]**2)
            ex2 = out2/counts
            var = ex2 - ex**2
            
            out3 = torch.zeros((range_sets.shape), device=imgs.device)
            out3.scatter_add_(0, inv_inds, (scores_with_id[:,0]-0.5)**2)
            # 避免nan值
            out3 = torch.clamp((out3/counts/0.5**2), 1e-5, 1-1e-5)
            out3 = -out3.log()
            if False:
                import cv2
                aimg = aligned_bilinear(imgs[0].unsqueeze(0), 2)  # 1280，1920，3
                aimg = tensor2imgs(aimg, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],to_rgb=False)[0]
                aimg = cv2.cvtColor(aimg, cv2.COLOR_RGB2BGR)
                range_sets_ = range_sets[counts>3]
                counts_ = counts[counts>3]
                topk = range_sets_[out3[counts>3].topk(100)[1]]
                cur_img = topk <= points[0][:,14].max()
                topk = topk[cur_img]
                for i in range(len(topk)):
                    rgb = np.array([np.random.randint(255) for _ in range(3)])
                    pts = points[0][points[0][:,14]==topk[i]]
                    for p in pts:
                        cv2.circle(aimg, (int(p[12]*2),int(p[13]*2)), 1, rgb.tolist(), -1)
                cv2.imwrite('run_img.jpeg', aimg)
            if (counts[1:]>3).sum() == 0:
                pass
            # 过滤掉分段id为-1的无关点(非主雷达点)
            elif -1 in range_sets:
                loss_2d_run_seg = ((var[1:]*(counts[1:]>3)).sum())/(counts[1:]>3).sum()
            else:
                loss_2d_run_seg = ((var*(counts>3)).sum())/(counts>3).sum()
            losses.update({
                "loss_2d_run_seg": loss_2d_run_seg * warmup_factor * 0.15,      #
            })

    def get_multi_view_points_targets(self, points, gt_bboxes, img_metas, img_shape):
        # points[:, 17]=[0, 1, -1]表示是否是背景点，通过3d分支给每个点得分(N,3)，如果类别得分小于阈值，那么就是背景点，
        # points[:, [20, 21]]ccl后每个2d box内的点云，这个可以保证绝大多数点都是前景点, ==box.index()获取在roi内的点
        sample_nums = 20
        neg_sample_nums = 20
        fake_batch_size = img_shape[0]
        batch_size = len(points)
        # batch_nums = int(fake_batch_size / batch_size)
        points_batch_target_fg = []  # x,y,z,u,v
        points_batch_target_bg = []
        for i in range(fake_batch_size):
            in_img_gt_boxes = gt_bboxes[i]
            roi_nums = len(in_img_gt_boxes)
            batch_points = points[int(i//5)]
            for j in range(roi_nums):
                # in_box_mask = ((batch_points[:, 12]>=in_img_gt_boxes[j][0]) & (batch_points[:, 12]<in_img_gt_boxes[j][2]) &
                #                (batch_points[:, 14]>=in_img_gt_boxes[j][1]) & (batch_points[:, 14]<in_img_gt_boxes[j][3]) &
                #                (batch_points[:, 10]==int(i%5)))
                in_box_mask = batch_points[:, 20]==int(i%5)*1000+j
                in_box_points = batch_points[in_box_mask]
                enlarge_gt_boxes = self.enlarge_bbox(in_img_gt_boxes[j])
                out_box_mask = (((enlarge_gt_boxes[0]<=batch_points[:, 12])&(batch_points[:, 12]<enlarge_gt_boxes[2]) & 
                                (enlarge_gt_boxes[1]<=batch_points[:, 14])&(batch_points[:, 14]<in_img_gt_boxes[j][1])) |
                               ((enlarge_gt_boxes[0]<=batch_points[:, 12])&(batch_points[:, 12]<enlarge_gt_boxes[2]) & 
                                (in_img_gt_boxes[j][3]<=batch_points[:, 14])&(batch_points[:, 14]<enlarge_gt_boxes[3])) |
                               ((enlarge_gt_boxes[0]<=batch_points[:, 12])&(batch_points[:, 12]<in_img_gt_boxes[j][0]) & 
                                (enlarge_gt_boxes[1]<=batch_points[:, 14])&(batch_points[:, 14]<enlarge_gt_boxes[3])) |
                               ((in_img_gt_boxes[j][2]<batch_points[:, 12])&(batch_points[:, 12]<enlarge_gt_boxes[2]) & 
                                (enlarge_gt_boxes[1]<=batch_points[:, 14])&(batch_points[:, 14]<enlarge_gt_boxes[3]))) & \
                                (batch_points[:, 10]==int(i%5))
                out_box_points = batch_points[out_box_mask]
                points_batch_target_fg.append(self.padding_points(in_box_points, sample_nums))
                points_batch_target_bg.append(self.padding_points(out_box_points, neg_sample_nums))

        return points_batch_target_fg, points_batch_target_bg

    def enlarge_bbox(self, boxes):
        x1, y1, x2, y2 = boxes
        w, h = (x2 - x1), (y2 - y1)
        x1, y1 = boxes[0] - w * 0.1, boxes[1] - h * 0.1
        x2, y2 = boxes[2] + w * 0.1, boxes[3] + h * 0.1
        return torch.tensor([x1, y1, x2, y2], device=boxes.device)

    def padding_points(self, points, max_nums=100):
        N = points.shape[0]
        dim = 5 # x, y, z, u, v
        if N >= max_nums:
            id = np.random.choice(N, max_nums, replace=False)
            tmp_points = points[id][:, [0, 1, 2, 12, 14]]
        else:
            tmp_points = torch.zeros((max_nums, dim), device=points.device)
            tmp_points[:N, ] = points[:, [0, 1, 2, 12, 14]]
        return tmp_points

    @staticmethod
    def plot_points(img, points_targets, points_targets_bg, gt_bbox, l1, l2, l3, l4):
        import cv2
        img = tensor2imgs(img.unsqueeze(0), mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],
                                to_rgb=False)[0]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.rectangle(img, (int(gt_bbox[0]), int(gt_bbox[1])), (int(gt_bbox[2]), int(gt_bbox[3])), (0, 255, 0), 2)
        # 分配的前景点
        for p in points_targets[0:l1]:
            cv2.circle(img, (int(p[12]), int(p[13])),1,(0,255,0),1)
        # 分配的背景点
        for p in points_targets_bg[0:l3]:
            cv2.circle(img, (int(p[12]), int(p[13])),1,(255,255,255),1)
        # 3d mask 前景点
        for p in points_targets[l1:l1+l2]:
            cv2.circle(img, (int(p[12]), int(p[13])),1,(255,0,0),1)
        # 3d mask 背景点
        for p in points_targets_bg[l3:l3+l4]:
            cv2.circle(img, (int(p[12]), int(p[13])),1,(0,0,255),1)
        cv2.imwrite('img.jpeg', img)
        # run seg max计算方式
        if False:
            grid_targets, points_targets, fg_len, seg_scores_ = self.get_points_targets_v2(points, gt_bboxes, gt_labels, img_metas, seg_scores)
            # gt_labels_ = torch.cat(gt_labels)
            # gt_bboxes_ = torch.cat(gt_bboxes, dim=0)
            loss_2d_run_seg = torch.tensor(0, device=imgs.device, dtype=torch.float)
            # (N, 1, 160, 240) --> (1, 1, 160, 240)
            # (1, 1, 160, 240) (1, 1, (N1+N2+...), 2)-->(1, 1, 1, (N1+N2+...))
            grid_sample_scores = F.grid_sample(mask_scores.max(0)[0].unsqueeze(0), grid_targets.unsqueeze(0).unsqueeze(0), align_corners=False)
            grid_sample_scores = grid_sample_scores.squeeze(0).squeeze(0).squeeze(0)
            grid_sample_scores = torch.clamp(grid_sample_scores, 1e-5, 1-1e-5)  # clamp，防止出现inf值

            range_sets, inv_inds, counts = torch.unique(torch.cat(points_targets)[:,14], return_inverse=True, return_counts=True)
            
            # cal loss
            out = torch.zeros((range_sets.shape), device=imgs.device)
            out.scatter_add_(0, inv_inds, grid_sample_scores)
            ex = out/counts
            out2 = torch.zeros((range_sets.shape), device=imgs.device)
            out2.scatter_add_(0, inv_inds, grid_sample_scores**2)
            ex2 = out2/counts
            var = ex2 - ex**2

            entropy = -(grid_sample_scores*grid_sample_scores.log()+(1-grid_sample_scores)*(1-grid_sample_scores).log())
            out3 = torch.zeros((range_sets.shape), device=imgs.device)
            out3.scatter_add_(0, inv_inds, entropy)
            out3 = out3/counts

            filter_mask = range_sets != -1
            if (counts[filter_mask]>3).sum() == 0:
                pass
            else:
                loss_2d_run_seg = (((out3[filter_mask]+var[filter_mask])*(counts[filter_mask]>3)).sum())/(counts[filter_mask]>3).sum()

            loss_2d_run_seg = loss_2d_run_seg * warmup_factor * 0.2
            losses.update({
                "loss_2d_run_seg": loss_2d_run_seg,
            })

            # 可视化代码
            if False:
                # 可视化loss最大的段，或者最小的段
                import cv2
                aimg = aligned_bilinear(imgs[0].unsqueeze(0), 2)  # 1280，1920，3
                aimg = tensor2imgs(aimg, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],to_rgb=False)[0]
                aimg = cv2.cvtColor(aimg, cv2.COLOR_RGB2BGR)
                range_sets_ = range_sets[counts>3]
                counts_ = counts[counts>3]
                topk = range_sets_[ex[counts>3].topk(100, largest=True)[1]]
                cur_img = topk <= points[0][:,14].max()  # 只可视化第一张图
                topk = topk[cur_img]
                for i in range(len(topk)):
                    rgb = np.array([np.random.randint(255) for _ in range(3)])
                    pts = torch.cat(points_targets)[torch.cat(points_targets)[:,14]==topk[i]]
                    for p in pts:
                        cv2.circle(aimg, (int(p[12]*2),int(p[13]*2)), 2, rgb.tolist(), -1)
                cv2.imwrite('ex.jpeg', aimg)
            if False:
                # 可视化image mask、pts mask
                import cv2
                aimg = aligned_bilinear(imgs[0].unsqueeze(0), 2)  # 1280，1920，3
                aimg = tensor2imgs(aimg, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=False)[0]
                aimg = cv2.cvtColor(aimg, cv2.COLOR_RGB2BGR)
                max_score = grid_sample_scores.max()
                for i, p in enumerate(torch.cat(points_targets[0:len(gt_labels_[0])], dim=0)):
                    rgb = int(255 * grid_sample_scores[i]/max_score)
                    cv2.circle(aimg, (int(p[12]*2),int(p[13]*2)), 1, (rgb, rgb, rgb), -1)
                cv2.imwrite('img_mask.jpeg', aimg)

                max_score = torch.cat(seg_scores_[0: len(gt_labels_[0])]).max()
                for i in range(len(gt_labels_[0])):
                    gt_label = gt_labels_[0][i]
                    for j, p in enumerate(points_targets[i]):
                        rgb = int(255 * seg_scores_[i][j][gt_label]/max_score)
                        cv2.circle(aimg, (int(p[12]*2),int(p[13]*2)), 1, (rgb, rgb, rgb), -1)
                    cv2.imwrite('img_pts_mask.jpeg', aimg)
                    
            if False:
                # 可视化run
                import cv2
                aimg = aligned_bilinear(imgs[0].unsqueeze(0), 2)  # 1280，1920，3
                aimg = tensor2imgs(aimg, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],to_rgb=False)[0]
                aimg = cv2.cvtColor(aimg, cv2.COLOR_RGB2BGR)
                pp = torch.cat(points_targets, dim=0)
                max_run = points[0][:14].max()
                for i in range(len(range_sets)):
                    rgb = np.array([np.random.randint(255) for _ in range(3)])
                    p = pp[pp[:,14] == range_sets[i]]
                    if range_sets[i]>max_run:
                        continue
                    if counts[i] < 4:
                        continue
                    for j in p:
                        x = int(j[12]*2)
                        y = int(j[13]*2)
                        cv2.circle(aimg, (x,y),2,rgb.tolist(),-1)
                cv2.imwrite('run_img.jpeg', aimg)

            assert mask_scores.size(0) == gt_inds.size(0) == img_inds.size(0)
            gt_labels_ = torch.cat(gt_labels)
            gt_bboxes_ = torch.cat(gt_bboxes, dim=0)
            self.num_classes = 3
            loss_3d_to_2d = torch.tensor(0, device=imgs.device, dtype=torch.float)
            for i in range(imgs.shape[0]):  # batch_size
                # 获得对应图片的mask_pred和label
                mask_pred_per_img = mask_scores[img_inds==i]
                gt_inds_per_img = gt_inds[img_inds==i]
                gt_labels_per_img = gt_labels_[gt_inds_per_img]
                # 分配 label
                mask_preds_cls = []
                gt_inds_cls = []
                for j in range(self.num_classes):
                    mask_cls = gt_labels_per_img == j
                    gt_inds_per_img_cls = gt_inds_per_img[mask_cls]
                    mask_pred_per_img_cls = mask_pred_per_img[mask_cls]
                    mask_preds_cls.append(mask_pred_per_img_cls)
                    gt_inds_cls.append(gt_inds_per_img_cls)

                mask_pred = mask_preds_cls
                per_batch_points = points[i]
                for j in range(self.num_classes):
                    mask_pred_cls = mask_pred[j]
                    if mask_pred_cls.size(0) != 0:
                        mask_pred_cls = mask_pred_cls.max(0)[0]
                        grid = per_batch_points[:, 12:14]/img_metas[i]['scale_factor'][0]
                        grid[:,0] = (grid[:,0]/img_metas[i]['pad_shape'][1]) - 1  # x
                        grid[:,1] = (grid[:,1]/img_metas[i]['pad_shape'][0]) - 1  # y
                        # input(1,1,H,W) grid(1,1,N_pts,2) out(1,1,1,N_pts)
                        out = F.grid_sample(mask_pred_cls.unsqueeze(0), grid.unsqueeze(0).unsqueeze(0), align_corners=False)
                        out = torch.clamp(out, 1e-5, 1-1e-5)
                        out = out.squeeze(0).squeeze(0).squeeze(0)
                        # -plog(q)
                        loss_3d_to_2d += -((seg_scores[i][:,j] * out.log()).sum() + \
                            ((1-seg_scores[i][:,j]) * (1-out).log()).sum()) / out.size()[0]
                
                        if False:
                            import cv2
                            aimg = aligned_bilinear(imgs[i].unsqueeze(0), 2)  # 1280，1920，3
                            aimg = tensor2imgs(aimg, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],to_rgb=False)[0]
                            aimg = cv2.cvtColor(aimg, cv2.COLOR_RGB2BGR)

                            # 可视化image mask
                            gt_mask = aligned_bilinear(mask_pred_cls.unsqueeze(0), 8).squeeze().squeeze()
                            gt_mask = gt_mask.detach().cpu().numpy()
                            cv2.imwrite('./work_dirs/out_images/pred_mask_cls_id_{}.jpeg'.format(j), gt_mask/gt_mask.max()*255)
                            # 可视化点云自身的mask
                            max_score = seg_scores[i][:,j].max()
                            for k, p in enumerate(points[i]):
                                rgb = int(255 * seg_scores[i][k][j]/max_score)
                                cv2.circle(aimg, (int(p[12]*2),int(p[13]*2)), 1, (rgb, rgb, rgb), -1)
                            cv2.imwrite('./work_dirs/out_images/img_pts_mask_cls_id_{}.jpeg'.format(j), aimg)
                            # 可视化点云投影到图像位置图像的mask
                            max_score = out.max()
                            for k, p in enumerate(points[i]):
                                rgb = int(255 * out[k]/max_score)
                                cv2.circle(aimg, (int(p[12]*2),int(p[13]*2)), 1, (rgb, rgb, rgb), -1)
                            cv2.imwrite('./work_dirs/out_images/img_mask_cls_id_{}.jpeg'.format(j), aimg)

                            ascore = (-(seg_scores[i][:,j] * out.log()+((1-seg_scores[i][:,j]) * (1-out).log())))
                            topk_val, topk_inds = torch.topk(ascore, 5)
                            max_loss = topk_val[0]
                            for k, p in enumerate(points[i][topk_inds]):
                                rgb = int(255 * topk_val[k]/max_loss)
                                cv2.circle(aimg, (int(p[12]*2),int(p[13]*2)), 3, (rgb, rgb, rgb), -1)
                            cv2.imwrite('./work_dirs/out_images/0cls_id{}.jpeg'.format(j), aimg)

            loss_3d_to_2d = loss_3d_to_2d * warmup_factor * 0.05
            losses.update({
                "loss_3d_to_2d": loss_3d_to_2d,
            })
