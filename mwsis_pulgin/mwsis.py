# Copyright (c) OpenMMLab. All rights reserved.
import time
import random
import numpy as np
from os import path as osp
import torch
from torch.nn import functional as F
from scipy.sparse.csgraph import connected_components  # CCL
import os
import mmcv
from mmcv.ops import Voxelization
from mmcv.parallel import DataContainer as DC
from mmcv.runner import force_fp32
from pycocotools import mask as mask_utils
from mmdet.core import encode_mask_results
from mmdet.core import multi_apply, bbox2result

from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result, merge_aug_bboxes_3d, show_result)
from mmdet3d.models import builder
from mmdet3d.models.builder import DETECTORS
from mmdet3d.models.detectors.base import Base3DDetector

from mmseg.models import SEGMENTORS
from mmdet3d.models.segmentors.base import Base3DSegmentor

from .fsd_ops import scatter_v2, get_inner_win_inds
from .utils import pts_semantic_confusion_matrix
from mmdet.core.bbox.iou_calculators import bbox_overlaps
import torch_scatter
from mmcv.image import tensor2imgs
from deploy3d.symfun.ops.ccl import VoxelSPCCL3D, voxel_spccl3d

def gen_shape(pc_range, voxel_size):
    voxel_size = np.array(voxel_size).reshape(-1, 3)
    ncls = len(voxel_size)
    spatial_shape = []
    for i in range(ncls):
        spatial_shape.append([(pc_range[3]-pc_range[0])/voxel_size[i][0],
                              (pc_range[4]-pc_range[1])/voxel_size[i][1],
                              (pc_range[5]-pc_range[2])/voxel_size[i][2]])
    return np.array(spatial_shape).astype(np.int32).reshape(-1).tolist()

def points_padding(x, num_out, padding_nb):
    padding_shape = (num_out, ) + tuple(x.shape[1:])
    x_padding = x.new_ones(padding_shape) * (padding_nb)
    x_padding[:x.shape[0]] = x
    return x_padding

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

def aligned_bilinear(tensor, factor):
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
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )
    return tensor[:, :, :oh - 1, :ow - 1]

def filter_almost_empty(coors, min_points):
    new_coors, unq_inv, unq_cnt = torch.unique(coors, return_inverse=True, return_counts=True, dim=0)
    cnt_per_point = unq_cnt[unq_inv]
    valid_mask = cnt_per_point >= min_points
    return valid_mask

def find_connected_componets(points, batch_idx, dist):

    device = points.device
    bsz = batch_idx.max().item() + 1
    base = 0
    components_inds = torch.zeros_like(batch_idx) - 1

    for i in range(bsz):
        batch_mask = batch_idx == i
        if batch_mask.any():
            this_points = points[batch_mask]
            dist_mat = this_points[:, None, :2] - this_points[None, :, :2] # only care about xy
            dist_mat = (dist_mat ** 2).sum(2) ** 0.5
            adj_mat = dist_mat < dist
            adj_mat = adj_mat.cpu().numpy()
            c_inds = connected_components(adj_mat, directed=False)[1]
            c_inds = torch.from_numpy(c_inds).to(device).int() + base
            base = c_inds.max().item() + 1
            components_inds[batch_mask] = c_inds

    assert len(torch.unique(components_inds)) == components_inds.max().item() + 1

    return components_inds

def find_connected_componets_single_batch(points, batch_idx, dist):

    device = points.device

    this_points = points
    dist_mat = this_points[:, None, :2] - this_points[None, :, :2] # only care about xy
    dist_mat = (dist_mat ** 2).sum(2) ** 0.5
    # dist_mat = torch.cdist(this_points[:, :2], this_points[:, :2], p=2)
    adj_mat = dist_mat < dist
    adj_mat = adj_mat.cpu().numpy()
    c_inds = connected_components(adj_mat, directed=False)[1]
    c_inds = torch.from_numpy(c_inds).to(device).int()

    return c_inds

def modify_cluster_by_class(cluster_inds_list):
    new_list = []
    for i, inds in enumerate(cluster_inds_list):
        cls_pad = inds.new_ones((len(inds),)) * i
        inds = torch.cat([cls_pad[:, None], inds], 1)
        # inds = F.pad(inds, (1, 0), 'constant', i)
        new_list.append(inds)
    return new_list

@DETECTORS.register_module()
class MWSIS(Base3DDetector):
    """Base class of Multi-modality autolabel."""

    def __init__(self,
                with_pts_branch=True,
                with_img_branch=True,
                img_backbone=None,  #
                img_neck=None,  #
                img_bbox_head=None,  #
                img_mask_branch=None,  #
                img_mask_head=None,  #
                img_segm_head= None,
                pretrained=None,
                img_roi_head=None,
                img_rpn_head=None,
                middle_encoder_pts=None,  # points completion 
                pts_segmentor=None, #
                pts_voxel_layer=None,
                pts_voxel_encoder=None,
                pts_middle_encoder=None,
                pts_backbone=None,  #
                pts_neck=None,
                pts_bbox_head=None, #
                pts_roi_head=None,  # 二阶段，暂时先不用
                pts_fusion_layer=None,
                train_cfg=None,  # 记住cfg是分img和pts的
                test_cfg=None,  #
                cluster_assigner=None,  #
                init_cfg=None,
                only_one_frame_label=True,
                sweeps_num=1,
                gt_box_type=1, # # 1 is 3d, 2 is 2d
                num_classes=3,
                use_2d_mask=True,
                use_ema=False,
                run_seg=False,
                use_dynamic=False,
                use_refine_pseudo_mask=False,
                warmup_iters=10000,  # ignore
                start_kd_loss_iters=10000,
                use_weight_loss=False,
                kd_loss_weight=0.2,
                kd_loss_weight_3d = 0.2,
                use_his_labels_iters=35544,
                pseudo_loss_weight=1.0,
                freeze_img_backbone=False):  
        super(MWSIS, self).__init__(init_cfg=init_cfg)
        print('start_kd_loss_iters: ',start_kd_loss_iters)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.start_kd_loss_iters = start_kd_loss_iters
        self.point_cloud_range = [-80, -80, -2, 80, 80, 4]
        self.voxel_size = [[0.15, 0.15, 6], [0.05, 0.05, 6], [0.1, 0.1, 6]]
        self.dist_size = [[0.6, 0.6, 0], [0.1, 0.1, 0], [0.4, 0.4, 0]]
        self.kernel_size_ = [[1, 9, 9], [1, 5, 5], [1, 9, 9]]  # [z, y, x]

        self.with_pts_branch = with_pts_branch
        self.with_img_branch = with_img_branch
        self.gt_box_type = gt_box_type
        self.num_classes = num_classes
        self.use_2d_mask = use_2d_mask
        self.use_dynamic = use_dynamic
        self.run_seg = run_seg
        self.use_refine_pseudo_mask = use_refine_pseudo_mask
        self.use_weight_loss = use_weight_loss
        self.kd_loss_weight = kd_loss_weight
        self.kd_loss_weight_3d = kd_loss_weight_3d
        self.use_his_labels_iters = use_his_labels_iters
        self.pseudo_loss_weight = pseudo_loss_weight
        # FSD
        if pts_segmentor is not None:  # 
            self.pts_segmentor = builder.build_detector(pts_segmentor)
        if pts_backbone is not None:  
            self.pts_backbone = builder.build_backbone(pts_backbone)  # 
        if pts_neck is not None:
            self.pts_neck = builder.build_neck(pts_neck)
        if pts_voxel_layer is not None:
            self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
        if pts_voxel_encoder is not None:
            self.pts_voxel_encoder = builder.build_voxel_encoder(pts_voxel_encoder)
        if pts_middle_encoder is not None:
            self.pts_middle_encoder = builder.build_middle_encoder(pts_middle_encoder)

        if pts_bbox_head is not None:  # 
            pts_train_cfg = train_cfg.pts if train_cfg.pts else None
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg.pts else None
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = builder.build_head(pts_bbox_head)
            self.num_classes = self.pts_bbox_head.num_classes

        if pts_roi_head is not None:
            rcnn_train_cfg = train_cfg.pts.rcnn if train_cfg.pts else None
            pts_roi_head.update(train_cfg=rcnn_train_cfg)
            pts_roi_head.update(test_cfg=test_cfg.pts.rcnn)
            pts_roi_head.pretrained = pretrained
            self.pts_roi_head = builder.build_head(pts_roi_head)
        # if pts_segmentor is not None:
        #     self.pts_cfg = self.train_cfg.pts if self.train_cfg else self.test_cfg.pts

        self.print_info = {}
        self.as_rpn = False

        self.runtime_info = dict()
        self.only_one_frame_label = only_one_frame_label
        self.sweeps_num = sweeps_num
        self.register_buffer("_iter", torch.tensor(0, dtype=torch.int))
        self._warmup_iters = warmup_iters
        self.ema_decay = 0.99

        # BoxInst
        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = builder.build_neck(img_neck)
        if img_bbox_head is not None:  # train_cfg.img
            img_train_cfg = train_cfg.img if train_cfg else None
            img_bbox_head.update(train_cfg=img_train_cfg)
            img_test_cfg = test_cfg.img if test_cfg else None
            img_bbox_head.update(test_cfg=img_test_cfg)
            self.img_bbox_head = builder.build_head(img_bbox_head)
        if img_mask_branch is not None:
            self.img_mask_branch = builder.build_head(img_mask_branch)
        if img_mask_head is not None:
            self.img_mask_head = builder.build_head(img_mask_head)
            self.img_mask_head.run_seg = run_seg
        if img_segm_head is not None:
            self.img_segm_head = builder.build_head(img_segm_head)
        else:
            self.img_segm_head = None

        self.use_ema = use_ema
        if self.use_ema:
            self.register_ema()

        self.freeze_img_backbone = freeze_img_backbone
        if self.freeze_img_backbone:
            print('\n freeze img backbone')
            for name, param in self.img_backbone.named_parameters():
                param.requires_grad = False
            # for name, param in self.img_neck.named_parameters():
            #     param.requires_grad = False

    @property
    def with_pts_bbox(self):
        """bool: Whether the detector has a 3D box head."""
        return hasattr(self, 'pts_bbox_head') and self.pts_bbox_head is not None

    @property
    def with_img_bbox(self):
        """bool: Whether the detector has a 2D image box head."""
        return hasattr(self, 'img_bbox_head') and self.img_bbox_head is not None

    @property
    def with_img_backbone(self):
        """bool: Whether the detector has a 2D image backbone."""
        return hasattr(self, 'img_backbone') and self.img_backbone is not None

    @property
    def with_pts_backbone(self):
        """bool: Whether the detector has a 3D backbone."""
        return hasattr(self, 'pts_backbone') and self.pts_backbone is not None

    @property
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'img_neck') and self.img_neck is not None

    @property
    def with_pts_neck(self):
        """bool: Whether the detector has a neck in 3D detector branch."""
        return hasattr(self, 'pts_neck') and self.pts_neck is not None

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            img_feats = self.img_backbone(img)
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats  # FPN 5layers, img_feats=5*[B,C,new_H,new_W]

    # single stage fsd extract pts feat
    def extract_pts_feat(self, points, pts_feats, pts_cluster_inds, img_metas, center_preds):
        """Extract features from points."""
        if not self.with_pts_backbone:
            return None
        # if not self.with_pts_bbox:
        #     return None      
        cluster_xyz, _, inv_inds = scatter_v2(center_preds, pts_cluster_inds, mode='avg', return_inv=True)

        f_cluster = points[:, :3] - cluster_xyz[inv_inds]
        # (N,128), (N_clusters,768), (N_clusters,3)
        out_pts_feats, cluster_feats, out_coors = self.pts_backbone(points, pts_feats, pts_cluster_inds, f_cluster)
        out_dict = dict(
            cluster_feats=cluster_feats,
            cluster_xyz=cluster_xyz,
            cluster_inds=out_coors,
            inv_inds = inv_inds,
        )
        if self.as_rpn:
            out_dict['cluster_pts_feats'] = out_pts_feats
            out_dict['cluster_pts_xyz'] = points

        return out_dict

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)  # FPN 5layers, img_feats=5*[B,C,new_H,new_W]
        # pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        return (img_feats, None)

    def forward_train(self,
                      points=None,  # need
                      img_metas=None,  # need
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,  # need
                      gt_bboxes=None,  # need
                      gt_masks=None,
                      img=None, # need
                      proposals=None,
                      gt_bboxes_ignore=None,
                      pts_semantic_mask=None,
                      pts_instance_mask=None,
                      gt_semantic_seg=None,
                      gt_yaw=None,
                      lidar_density=None,
                      pseudo_labels=None,
                      batch_roi_points=None,
                      points2img=None,
                      points2img_idx=None,
                      history_labels=None,
                      ):
        """Forward training function.
        Returns:
            dict: Losses of different branches.
        """
        self.use_multi_view = False
        if img is None:
            self.use_multi_view = True
        else:
            if img.dim() == 5:
                self.use_multi_view = True
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
        if gt_bboxes is not None:
            if isinstance(gt_bboxes[0], list):
                gt_bboxes_ = []
                gt_labels_ = []
                gt_masks_ = []
                for i in range(len(img_metas)):
                    gt_bboxes_.extend(gt_bboxes[i])
                    gt_labels_.extend(gt_labels[i])
                    if gt_masks is not None:
                        gt_masks_.extend(gt_masks[i])
                gt_bboxes = gt_bboxes_
                gt_labels = gt_labels_
                if gt_masks is not None:
                    gt_masks = gt_masks_
        
        losses = dict()
        if self.use_multi_view:
            multi_losses = self.multiview_forward_train(points, img_metas, gt_bboxes_3d, gt_labels_3d, gt_labels, gt_bboxes,
                                    gt_masks, img, proposals, gt_bboxes_ignore, self.runtime_info, pts_semantic_mask,
                                    pts_instance_mask, gt_semantic_seg, gt_yaw, lidar_density, pseudo_labels, history_labels
                                    )
            losses.update(multi_losses)
        else:
            raise ValueError("need multi-view images")

        return losses

    def register_ema(self):
        print("init ema params")
        with torch.no_grad():
            for name, param in self.state_dict().items():
                if not param.is_floating_point():
                    continue
                shadow_name = ("ema_" + name).replace(".","__")
                self.register_buffer(shadow_name, param.data.clone().detach())

    def update_ema(self):
        with torch.no_grad():
            for name, param in self.state_dict().items():
                if name.startswith('ema_') or (not param.is_floating_point()):
                    continue
                shadow_name = ("ema_" + name).replace(".","__")
                shadow_param = getattr(self, shadow_name, None)
                if shadow_param is not None:
                    shadow_param.mul_(self.ema_decay).add_(param.data, alpha=1.0 - self.ema_decay)

    def swap_ema(self):
        with torch.no_grad():
            for name, param in self.state_dict(keep_vars=True).items():
                if name.startswith('ema_') or (not param.is_floating_point()):
                    continue
                shadow_name = ("ema_" + name).replace(".","__")
                tmp_shadow_param = getattr(self, shadow_name, None)
                if tmp_shadow_param is not None:
                    self.__setattr__(shadow_name, param.data)
                    param.data = tmp_shadow_param

    def cal_pseudo_mask_loss(self, pseudo_mask, mask_pred, ema_gt_inds, ema_img_inds):
        # self._iter=torch.tensor(30000, device=mask_pred.device)
        if self._iter < self._warmup_iters:  # 
            loss_pseudo = mask_pred.sum() * 0
            return loss_pseudo
        _, invs = torch.unique(ema_gt_inds, return_inverse=True, return_counts=False)
        # >0.7 is 1, <0.3 is 0, [0.3,0.7] ignore
        pseudo_mask = pseudo_mask[invs]
        mask_pos = pseudo_mask > 0.7
        mask_neg = pseudo_mask < 0.3
        mask_ignore = ~(mask_pos | mask_neg)
        # pseudo_mask[mask_pos] == 1
        # pseudo_mask[mask_neg] == 0
        # N_gt --> N_sample(N_gt<len(gt_bboxes),because some gt_bboxes no assign postive sample)
        # cal BCE loss
        mask_pred = mask_pred.squeeze(1)
        mask_pred = torch.clamp(mask_pred.sigmoid(), 1e-5, 1-1e-5)
        if (mask_pos.sum() + mask_neg.sum()) != 0:
            loss_bce = -(mask_pred[mask_pos].log().sum() + (1-mask_pred[mask_neg]).log().sum())/(mask_pos.sum() + mask_neg.sum())
        else:
            loss_bce = mask_pred.sum() * 0.
     
        # Dice Loss: 1 - 2 * (intersection(A, B) / (A^2 + B^2))
        pseudo_mask[mask_pos] = 1
        pseudo_mask[mask_neg] = 0
        eps = 1e-5
        targets = pseudo_mask.flatten(1)[(~mask_ignore).flatten(1)]
        pred = mask_pred.flatten(1)[(~mask_ignore).flatten(1)]
        a = torch.sum(pred * targets, 0)
        b = torch.sum(pred ** 2.0, 0)
        c = torch.sum(targets ** 2.0, 0)
        d = (2 * a) / (b + c + eps)
        loss_dice = 1 - d
        return self.pseudo_loss_weight * (loss_bce + loss_dice)

    def eval_pseudo_mask(self, gt_semantic_seg, pseudo_mask, img, gt_labels, gt_inds, img_inds, img_metas):
        # gt_labels_ = torch.cat(gt_labels)
        sets, invs, counts = torch.unique(gt_inds, return_inverse=True, return_counts=True)
        assert (sets[invs]==gt_inds).all()
        gt_len = [len(gt_label) for gt_label in gt_labels]
        pseudo_mask_list = [[] for _ in range(len(img))]
        tmp_nums = 0
        for i in range(len(img)):
            for j in range(tmp_nums, len(pseudo_mask)):
                if sets[j] < sum(gt_len[:i+1]):
                    pseudo_mask_list[i].append(pseudo_mask[j])
                    tmp_nums += 1
        for i in range(len(gt_semantic_seg)):
            bsz_gt_semantic_seg = gt_semantic_seg[i].clone().detach()
            bsz_mask = torch.stack(pseudo_mask_list[i]).max(0)[0]
            bsz_mask = F.interpolate(bsz_mask.unsqueeze(0).unsqueeze(0),size=(640,960),mode='bilinear', align_corners=False).squeeze()
            pos_ = bsz_mask>0.8
            neg_ = bsz_mask<0.2
            bsz_mask[pos_]=1
            bsz_mask[neg_]=0
            bsz_mask[~(pos_|neg_)]=2
            bsz_gt_semantic_seg[bsz_gt_semantic_seg>=0]=1
            bsz_gt_semantic_seg[bsz_gt_semantic_seg<0]=0
            pos_nums = ((bsz_gt_semantic_seg==1)&(bsz_mask==1)).sum()
            # pos_nums/(bsz_gt_semantic_seg==1).sum()
            # print("正样本查准率：", pos_nums/pos_.sum())
            neg_nums = ((bsz_gt_semantic_seg==0)&(bsz_mask==0)).sum()
            # print("负样本查准率：", neg_nums/neg_.sum())
            # vis
            import cv2
            fp_mask = ((~bsz_gt_semantic_seg.type(torch.bool))&(bsz_mask==1)).type(torch.long)*255
            fp_mask[bsz_gt_semantic_seg==1]=100
            cv2.imwrite('FP_mask.jpg', fp_mask.cpu().numpy())
            cv2.imwrite('PT_mask.jpg', bsz_mask.cpu().numpy()*122)
            cv2.imwrite('GT_mask.jpg', bsz_gt_semantic_seg.cpu().numpy()*255)

    def gen_weight_pseudo_mask(self, img, gt_bboxes, gt_labels, ema_bbox_preds, ema_bbox_targets, ema_img_inds, ema_gt_inds, ema_mask_pred, ema_scores):
        gt_bboxes_ = torch.cat(gt_bboxes, dim=0)
        # n, c, h, w = ema_mask_pred.shape
        gt_bboxes_ = gt_bboxes_[ema_gt_inds]
        gt_labels = torch.cat(gt_labels, dim=0)[ema_gt_inds]
        # ious = bbox_overlaps(ema_bbox_preds, gt_bboxes_, mode='iou', is_aligned=True)
        ious = bbox_overlaps(ema_bbox_preds, ema_bbox_targets, is_aligned=True)
        # if u=0, is average mask
        u = 1
        weight = ema_scores.sigmoid()[torch.arange(len(gt_labels)), gt_labels.long()] * (u * ious).exp().type(torch.float32)
        sets, invs = torch.unique(ema_gt_inds, return_inverse=True, return_counts=False)
        sum = torch.zeros((sets.shape), device=ema_mask_pred.device)
        sum.scatter_add_(0, invs, weight)
        normal_weight = weight / sum[invs]
        ema_mask_scores = (normal_weight[:, None, None, None] * ema_mask_pred.sigmoid()).squeeze(1) # squeeze(1)不可省略
        pseudo_mask_scores = torch.zeros((sets.shape[0], ema_mask_pred.shape[2], ema_mask_pred.shape[3]), device=ema_mask_pred.device)
        for i in range(len(sets)):
            tmp_mask = invs==i
            pseudo_mask_scores[i] = ema_mask_scores[tmp_mask].sum(0)
        if False:
            import cv2
            cv2.imwrite('pseudo_masks.jpg', pseudo_mask_scores.max(0)[0].cpu().numpy() * 255)
        return pseudo_mask_scores

    def multiview_forward_train(self,
                            points=None,     # need N,21
                            img_metas=None,  # need
                            gt_bboxes_3d=None,
                            gt_labels_3d=None,
                            gt_labels=None,  # need
                            gt_bboxes=None,  # need
                            gt_masks=None,
                            img=None,        # need
                            proposals=None,
                            gt_bboxes_ignore=None,
                            runtime_info=dict(),
                            pts_semantic_mask=None,
                            pts_instance_mask=None,
                            gt_semantic_seg=None,
                            gt_yaw=None,     # need
                            lidar_density=None,
                            pseudo_labels=None, # need
                            history_labels=None,
                            ):
        self._iter += 1
        bsz = len(img_metas)
        self.ema_decay = torch.clamp(1 - (-1 * self._iter / (5000 / bsz)).exp(), 0, 0.999)
        warmup_factor = min((self._iter.item() - self.start_kd_loss_iters) / 1000, 1.0)

        losses = dict()
        # Point-based voting correction (PVC)
        if self._iter < self.use_his_labels_iters or history_labels is None:
            history_labels_ = None
        else:
            history_labels_ = history_labels

        if self.gt_box_type == 2 and points is not None:
            points = self.ccl(points, gt_bboxes, gt_labels, img_metas, pseudo_labels)  # [N, (C+2)]

        if self.gt_box_type == 2:
            gt_bboxes_3d = gt_bboxes
            gt_labels_3d = gt_labels

        if self.use_ema:
            if self._iter != 1:
                self.update_ema()
            self.swap_ema()
            with torch.no_grad():
                # img
                img_feats = self.extract_img_feat(img=img, img_metas=img_metas)
                ema_cls_score, ema_bbox_pred, ema_centerness, ema_param_pred = self.img_bbox_head(img_feats, self.img_mask_head.param_conv)
                img_bbox_head_loss_inputs = (ema_cls_score, ema_bbox_pred, ema_centerness) + (gt_bboxes, gt_labels, img_metas)
                _, ema_coors, ema_level_inds, ema_img_inds, ema_gt_inds, ema_bbox_preds, ema_bbox_targets = self.img_bbox_head.loss(
                    *img_bbox_head_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
                mask_feat = self.img_mask_branch(img_feats)  # Bx16xHxW
                inputs = (ema_cls_score, ema_centerness, ema_param_pred, ema_coors, 
                          ema_level_inds, ema_img_inds, ema_gt_inds, ema_bbox_preds, ema_bbox_targets)
                # if training_sample() has empty sample results then return None
                ema_param_pred, ema_coors, ema_level_inds, ema_img_inds, ema_gt_inds, \
                    ema_bbox_preds, ema_bbox_targets, ema_flag, ema_scores = self.img_mask_head.training_sample(*inputs)
                ema_mask_pred = None
                if ema_flag==1:
                    ema_mask_pred = self.img_mask_head(mask_feat, ema_param_pred, ema_coors, ema_level_inds, ema_img_inds)   
                pseudo_mask = None
                if self.use_refine_pseudo_mask and ema_mask_pred is not None:
                    pseudo_mask = self.gen_weight_pseudo_mask(img, gt_bboxes, gt_labels, ema_bbox_preds, ema_bbox_targets, 
                                                            ema_img_inds, ema_gt_inds, ema_mask_pred, ema_scores)
                if False:
                    import cv2
                    import matplotlib.pyplot as plt
                    vis_masks = ema_mask_pred[-4:].sigmoid().squeeze(1).cpu().numpy()
                    for i in range(5):
                        id =int(i+1)
                        fig = plt.figure()
                        a = ema_mask_pred[-id].sigmoid().squeeze().cpu().numpy()
                        # a[a>0.9]=1
                        # a[a<0.1]=0
                        plt.imshow(a, cmap='Blues')
                        plt.colorbar()
                        # plt.axes().get_xaxis().set_visible(False)
                        # plt.axes().get_yaxis().set_visible(False)
                        fig.savefig('map_{}.png'.format(id))
                        plt.show()
                    fig = plt.figure()
                    b = pseudo_mask[-1].sigmoid().squeeze().cpu().numpy()
                    # b[b>0.9]=1
                    # b[b<0.1]=0
                    plt.imshow(b, cmap='Blues')
                    plt.colorbar()
                    # plt.axes().get_xaxis().set_visible(False)
                    # plt.axes().get_yaxis().set_visible(False)
                    fig.savefig('map_{}.png'.format('pseudo_mask'))
                    plt.show()                    
                    ema_mask = ema_mask_pred[10:14].sigmoid().max(0)[0].squeeze()
                    # 对照ema mask和加权平均后的pseudo mask
                    cv2.imwrite('pseudo_mask.jpg',pseudo_mask.max(0)[0].sigmoid().cpu().numpy()*255)
                    cv2.imwrite('ema_mask.jpg',ema_mask.cpu().numpy()*255)
                    aimg = aligned_bilinear(img[0].unsqueeze(0), 2)  # 1280，1920，3
                    aimg = tensor2imgs(aimg, mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)[0]
                    # aimg = cv2.cvtColor(aimg, cv2.COLOR_RGB2BGR)
                    cv2.imwrite('ori_img.jpg', aimg)
                # use gt evaluate pseudo mask
                # if False:
                #     self.eval_pseudo_mask(gt_semantic_seg, pseudo_mask, img, gt_labels, ema_gt_inds, ema_img_inds, img_metas)
                # pts
                ema_x, ema_pts_coors, points = self.pts_segmentor.extract_feat(points, None)  # feats
                valid_pts_mask = ema_x[1]
                points = points[valid_pts_mask]  # # points is tensor not list
                ema_pts_coors = ema_pts_coors[valid_pts_mask]
                batch_idx = ema_pts_coors[:, 0]  # batch_idx
                ema_seg_logits, ema_vote_preds, _ = self.pts_segmentor.segmentation_head.forward(ema_x[0])  # seg_logits, vote_preds
                offsets = self.pts_segmentor.segmentation_head.decode_vote_targets(ema_vote_preds)  # offsets
            self.swap_ema()
        
        if self.with_img_branch:
            # 2D Box Branch
            img_feats = self.extract_img_feat(img=img, img_metas=img_metas)  # list [B, c, h, w]
            cls_score, bbox_pred, centerness, param_pred = self.img_bbox_head(img_feats, self.img_mask_head.param_conv)
            img_bbox_head_loss_inputs = (cls_score, bbox_pred, centerness) + (gt_bboxes, gt_labels, img_metas)
            
            losses, coors, level_inds, img_inds, gt_inds, bbox_preds, bbox_targets = self.img_bbox_head.loss(*img_bbox_head_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore,gt_masks=gt_masks)
            # 2D Mask Branch
            mask_feat = self.img_mask_branch(img_feats)  # Bx16xHxW
            inputs = (cls_score, centerness, param_pred, coors, level_inds, img_inds, gt_inds, bbox_preds, bbox_targets)
            # 2D Mask Head
            param_pred_, coors_, level_inds_, img_inds_, gt_inds_, _, _, flag, _ = self.img_mask_head.training_sample(*inputs)
            mask_pred = None
            if flag == 1:
                mask_pred = self.img_mask_head(mask_feat, param_pred_, coors_, level_inds_, img_inds_)

        if self.with_pts_branch:
            # 3D VoteSegmentor forward
            if not isinstance(points, list) and self.use_ema:
                batch_size = len(img_metas)
                points = [points[batch_idx==i] for i in range(batch_size)]
            seg_out_dict = self.pts_segmentor(points=points, img_metas=img_metas, gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d, 
                                            as_subsegmentor=True, pts_semantic_mask=pts_semantic_mask, pts_instance_mask=pts_instance_mask,
                                            gt_box_type=self.gt_box_type, history_labels=history_labels_)
            losses.update(seg_out_dict['losses'])
        
        if self.use_ema and self.with_img_branch:
            batch_size = len(img_metas)
            if not isinstance(points, list):
                points = [points[batch_idx==i] for i in range(batch_size)]
            ema_seg_logits = [ema_seg_logits[batch_idx==i] for i in range(batch_size)]
            seg_labels = [seg_out_dict['labels'][batch_idx==i] for i in range(batch_size)]
            # 2D mask Loss
            loss_mask = self.img_mask_head.loss_multi(img, img_metas, mask_pred, gt_inds_, gt_bboxes,
                                gt_masks, gt_labels, points, None, ema_seg_logits, img_inds_, 
                                self.num_classes, flag, seg_labels)
            losses.update(loss_mask)

            # self.use_refine_pseudo_mask = False # Test
            if self.use_refine_pseudo_mask:
                loss_pseudo = self.cal_pseudo_mask_loss(pseudo_mask, mask_pred, ema_gt_inds, ema_img_inds)
                losses.update(loss_pseudo=loss_pseudo)
        
        # 2D to 3D loss
        if self.use_ema and self.use_2d_mask:
            
            if self.use_refine_pseudo_mask and self.use_multi_view:
                if pseudo_mask is not None:
                    loss_2d_to_3d_refine = self.cal_multiview_2d_to_3d_refine_loss(img, pseudo_mask, gt_labels, ema_gt_inds,
                                                        ema_img_inds, img_metas, points, seg_out_dict,
                                                        warmup_factor)
                else:
                    loss_2d_to_3d_refine = seg_out_dict['seg_logits'].sum() * 0
                losses.update({'loss_2d_to_3d_refine': loss_2d_to_3d_refine})
        
        if not self.use_ema:
            batch_size = len(img_metas)
            if self.with_pts_branch and self.with_img_branch:
                points = seg_out_dict['seg_points']
                batch_idx = seg_out_dict['batch_idx']
                points = [points[batch_idx==i] for i in range(batch_size)]
                ema_seg_logits = seg_out_dict['seg_logits']#.clone().detach()  #.sigmoid()
                ema_seg_logits = [ema_seg_logits[batch_idx==i] for i in range(batch_size)]
                seg_labels = [seg_out_dict['labels'][batch_idx==i] for i in range(batch_size)]
                # 2D mask Loss
                loss_mask = self.img_mask_head.loss_multi(img, img_metas, mask_pred, gt_inds_, gt_bboxes,
                                    gt_masks, gt_labels, points, None, ema_seg_logits, img_inds_, 
                                    self.num_classes, flag, seg_labels)
                losses.update(loss_mask)
            elif ~self.with_pts_branch and self.with_img_branch:
                if gt_masks is None:
                    loss_mask = self.img_mask_head.loss_multi(img, img_metas, mask_pred, gt_inds_, gt_bboxes,
                                        gt_masks, gt_labels, None, None, None, img_inds_, 
                                        self.num_classes, flag, None)
                if gt_masks is not None:
                    loss_mask = self.img_mask_head.loss(img, img_metas, mask_pred, gt_inds_, gt_bboxes,
                                        gt_masks, gt_labels, None, None, None, img_inds_, 
                                        self.num_classes, flag)
                losses.update(loss_mask)

        if not self.use_ema and self.use_2d_mask:
            if self.use_refine_pseudo_mask and self.use_multi_view:
                if pseudo_mask is not None:
                    loss_2d_to_3d_refine = self.cal_multiview_2d_to_3d_refine_loss(img, pseudo_mask, gt_labels, gt_inds_,
                                                        img_inds_, img_metas, points, seg_out_dict,
                                                        warmup_factor)
                else:
                    loss_2d_to_3d_refine = seg_out_dict['seg_logits'].sum() * 0
                losses.update({'loss_2d_to_3d_refine': loss_2d_to_3d_refine})
            elif ~self.use_refine_pseudo_mask and self.use_multi_view:
                if mask_pred is not None:
                    loss_2d_to_3d = self.cal_multiview_2d_to_3d_loss(img, mask_pred, gt_bboxes,
                                                        gt_labels, gt_inds_, img_inds_, img_metas, 
                                                        seg_out_dict, warmup_factor)
                else:
                    loss_2d_to_3d = seg_out_dict['seg_logits'].sum() * 0
                losses.update({'loss_2d_to_3d': loss_2d_to_3d})
        # save pts predict results
        if history_labels is not None:
            if self.use_ema:
                self.save_pts_logits(img_metas, seg_out_dict['seg_points'], ema_seg_logits, batch_idx, history_labels)
            else:
                self.save_pts_logits(img_metas, seg_out_dict['seg_points'], seg_out_dict['seg_logits'].clone().detach(), seg_out_dict['batch_idx'], history_labels)

        return losses

    def cal_multiview_2d_to_3d_refine_loss(self, img, pseudo_mask, gt_labels, gt_inds, img_inds, img_metas, points, dict_to_sample, warmup_factor):
        # self._iter=torch.tensor(30000, device=img.device)  # debug
        if self._iter < self.start_kd_loss_iters or pseudo_mask is None:
            loss_2d_to_3d = dict_to_sample['seg_logits'].sum() * 0
            return loss_2d_to_3d
        gt_labels_ = torch.cat(gt_labels)
        sets, invs, _ = torch.unique(gt_inds, return_inverse=True, return_counts=True)
        assert (sets[invs]==gt_inds).all()
        gt_len = [len(gt_label) for gt_label in gt_labels]
        pseudo_mask_list = [[[] for _ in range(self.num_classes)] for _ in range(len(img))]
        tmp_nums = 0
        for i in range(len(img)):
            for j in range(tmp_nums, len(pseudo_mask)):
                if sets[j] < sum(gt_len[:i+1]):
                    pseudo_mask_list[i][int(gt_labels_[sets[j]])].append(pseudo_mask[j])
                    tmp_nums += 1

        device = img.device
        loss_2d_to_3d = torch.tensor(0., device=device)
        counts_nums = 0
        # labels = dict_to_sample['labels']
        mask_points = dict_to_sample['seg_points']
        mask_in = mask_points[:, 16] == 1  # on img points
        mask_points = mask_points[mask_in]
        mask_prob = dict_to_sample['seg_logits'].sigmoid()  # (N, 3)
        mask_prob = torch.clamp(mask_prob, 1e-5, 1-1e-5)
        mask_prob = mask_prob[mask_in]
        batch_idx = dict_to_sample['batch_idx']
        batch_idx = batch_idx[mask_in]
        batch_img_nums = int(len(img)/len(img_metas))
        for i in range(len(img)):
            batch_id = int(i//batch_img_nums)
            per_batch_points = mask_points[batch_idx == batch_id]
            cls_weight = torch.ones((self.num_classes), device=per_batch_points.device)
            if self.use_weight_loss:
                # per_batch_labels = labels[batch_idx == batch_id]
                # for j in range(self.num_classes):
                #     cls_weight[j] = (per_batch_labels == j).sum()
                # # assert (cls_weight==0).sum() != self.num_classes
                # cls_weight_norm = (cls_weight / cls_weight.sum())
                # cls_weight = torch.clamp(cls_weight_norm**0.2, 1, 3).type(torch.int32)
                cls_weight[2] = 5
            for j in range(self.num_classes):
                if len(pseudo_mask_list[i][j]) == 0:
                    continue
                mask_pred_cls = torch.stack(pseudo_mask_list[i][j])
                mask_pred_cls = mask_pred_cls.max(0)[0]
                in_img_mask1 = per_batch_points[:,10]==int(i%batch_img_nums)
                x1, y1 = per_batch_points[in_img_mask1][:, 12], per_batch_points[in_img_mask1][:, 14]
                in_img_mask2 = per_batch_points[:,11]==int(i%batch_img_nums)
                x2, y2 = per_batch_points[in_img_mask2][:, 13], per_batch_points[in_img_mask2][:, 15]
                if (in_img_mask1 | in_img_mask2).sum()==0:
                    continue
                x, y = torch.cat((x1, x2)), torch.cat((y1, y2))
                grid = torch.stack((x, y), dim=0).T
                # grid = grid/img_metas[batch_id]['scale_factor'][int(i%batch_img_nums)][0]
                grid[:,0] = (grid[:,0]/img_metas[batch_id]['pad_shape'][int(i%batch_img_nums)][1]) * 2 - 1  # x
                grid[:,1] = (grid[:,1]/img_metas[batch_id]['pad_shape'][int(i%batch_img_nums)][0]) * 2 - 1  # y
                # input(1,1,H,W) grid(1,1,N_pts,2) out(1,1,1,N_pts)
                out = F.grid_sample(mask_pred_cls.unsqueeze(0).unsqueeze(0), grid.unsqueeze(0).unsqueeze(0), align_corners=False)
                out = out.squeeze(0).squeeze(0).squeeze(0)
                
                # -plog(q)
                # counts_nums += out.size()[0]
                mask_prob1 = mask_prob[batch_idx == batch_id][:,j][in_img_mask1]
                mask_prob2 = mask_prob[batch_idx == batch_id][:,j][in_img_mask2]
                mask_prob_ = torch.cat((mask_prob1, mask_prob2))
                weight = torch.ones_like(out)
                mask1 = (mask_prob_ > 0.7) | (mask_prob_ < 0.3)
                mask2 = (out > 0.7) | (out < 0.3)
                mask = mask1 | mask2
                weight[~mask] = 0.
                counts_nums += weight.sum()
                loss_2d_to_3d += -((out * mask_prob_.log() * weight).sum() + ((1-out) * (1-mask_prob_).log() * weight).sum()) * cls_weight[j]

        if counts_nums != 0:
            loss_2d_to_3d = loss_2d_to_3d * warmup_factor * self.kd_loss_weight_3d / counts_nums
        else:
            loss_2d_to_3d = dict_to_sample['seg_logits'].sum() * 0

        return loss_2d_to_3d

    def cal_multiview_2d_to_3d_loss(self, img, mask_pred, 
                                    gt_bboxes, gt_labels, 
                                    gt_inds, img_inds, img_metas, 
                                    dict_to_sample, warmup_factor):
        # self._iter=torch.tensor(30000,device=img.device)  # debug
        if self._iter < self.start_kd_loss_iters:
            loss_2d_to_3d = dict_to_sample['seg_logits'].sum() * 0
            return loss_2d_to_3d

        gt_labels_ = torch.cat(gt_labels, dim=0)
        assert mask_pred.size(0) == gt_inds.size(0) == img_inds.size(0)
        mask_preds = []
        mask_pred_detach = mask_pred.sigmoid()          # N,1,H,W
        mask_pred_detach = mask_pred_detach.squeeze(1)  # 
        assert mask_pred_detach.dim() == 3
        for i in range(img.shape[0]):
            # get mask_pred and label
            if (img_inds==i).sum() == 0:
                mask_preds.append(dict(mask_preds=[], gt_inds=[]))
                continue
            mask_pred_per_img = mask_pred_detach[img_inds==i]
            gt_inds_per_img = gt_inds[img_inds==i]
            gt_labels_per_img = gt_labels_[gt_inds_per_img]
            # assign label
            mask_preds_cls = []
            gt_inds_cls = []
            for j in range(self.num_classes):
                mask_cls = gt_labels_per_img == j
                gt_inds_per_img_cls = gt_inds_per_img[mask_cls]
                mask_pred_per_img_cls = mask_pred_per_img[mask_cls]
                mask_preds_cls.append(mask_pred_per_img_cls)
                gt_inds_cls.append(gt_inds_per_img_cls)
            mask_preds.append(dict(mask_preds=mask_preds_cls, gt_inds=gt_inds_cls))

        device = img.device
        loss_2d_to_3d = torch.tensor(0., device=device)
        counts_nums = 0
        # labels = dict_to_sample['labels']
        mask_points = dict_to_sample['seg_points']
        mask_in = mask_points[:, 16] == 1  # on img points
        mask_points = mask_points[mask_in]
        mask_prob = dict_to_sample['seg_logits'].sigmoid()  # (N, 3)
        mask_prob = torch.clamp(mask_prob, 1e-5, 1-1e-5)
        mask_prob = mask_prob[mask_in]
        batch_idx = dict_to_sample['batch_idx']
        batch_idx = batch_idx[mask_in]
        batch_img_nums = int(len(img)/len(img_metas))

        for i, mask_pred_ in enumerate(mask_preds):
            batch_id = int(i//batch_img_nums)
            mask_pred = mask_pred_['mask_preds']
            if len(mask_pred) == 0:
                continue
            per_batch_points = mask_points[batch_idx == batch_id]
            cls_weight = torch.ones((self.num_classes), device=per_batch_points.device)
            if self.use_weight_loss:
                # per_batch_labels = seg_lables[batch_id]
                # for j in range(self.num_classes):
                #     cls_weight[j] = (per_batch_labels == j).sum()
                # # assert (cls_weight==0).sum() != self.num_classes
                # cls_weight_norm = (cls_weight / cls_weight.sum())
                # cls_weight = torch.clamp(cls_weight_norm**0.2, 1, 3).type(torch.int32)
                cls_weight[2] = 5
            for j in range(self.num_classes):
                if mask_pred[j].size(0) != 0:
                    mask_pred_cls = mask_pred[j].max(0)[0]
                    if False:
                        import cv2
                        aimg = aligned_bilinear(img[i].unsqueeze(0), 2)  # 1280，1920，3
                        aimg = tensor2imgs(aimg, mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)[0]
                        aimg = cv2.cvtColor(aimg, cv2.COLOR_RGB2BGR)
                        cv2.imwrite('img.jpg', aimg)
                        cv2.imwrite('masks.jpg', (mask_pred_cls / mask_pred_cls.max()).cpu().numpy() * 255)
                    in_img_mask1 = per_batch_points[:, 10]==int(i%batch_img_nums)
                    x1, y1 = per_batch_points[in_img_mask1][:, 12], per_batch_points[in_img_mask1][:, 14]
                    in_img_mask2 = per_batch_points[:, 11]==int(i%batch_img_nums)
                    x2, y2 = per_batch_points[in_img_mask2][:, 13], per_batch_points[in_img_mask2][:, 15]
                    if (in_img_mask1 | in_img_mask2).sum()==0:
                        continue
                    x, y = torch.cat((x1, x2)), torch.cat((y1, y2))
                    grid = torch.stack((x, y), dim=0).T
                    grid[:, 0] = (grid[:, 0]/img_metas[batch_id]['pad_shape'][int(i%batch_img_nums)][1]) * 2 - 1  # x
                    grid[:, 1] = (grid[:, 1]/img_metas[batch_id]['pad_shape'][int(i%batch_img_nums)][0]) * 2 - 1  # y
                    # input(1,1,H,W) grid(1,1,N_pts,2) out(1,1,1,N_pts)
                    out = F.grid_sample(mask_pred_cls.unsqueeze(0).unsqueeze(0), grid.unsqueeze(0).unsqueeze(0), align_corners=False)
                    out = out.squeeze(0).squeeze(0).squeeze(0)
                    # counts_nums += out.size()[0]
                    # -plog(q)
                    mask_prob1 = mask_prob[batch_idx == batch_id][:, j][in_img_mask1]
                    mask_prob2 = mask_prob[batch_idx == batch_id][:, j][in_img_mask2]
                    mask_prob_ = torch.cat((mask_prob1, mask_prob2))
                    weight = torch.ones_like(out)
                    # sample hard label
                    mask1 = (mask_prob_ > 0.7) | (mask_prob_ < 0.3)
                    mask2 = (out > 0.7) | (out < 0.3)
                    mask = mask1 | mask2
                    weight[~mask] = 0.
                    counts_nums += weight.sum()
                    loss_2d_to_3d += -((out * mask_prob_.log() * weight).sum() + ((1 - out) * (1 - mask_prob_).log() * weight).sum()) * cls_weight[j]
        if counts_nums != 0:
            loss_2d_to_3d = loss_2d_to_3d * warmup_factor * self.kd_loss_weight_3d / counts_nums
        else:
            loss_2d_to_3d = dict_to_sample['seg_logits'].sum() * 0
        return loss_2d_to_3d

    def save_pts_logits(self, img_metas, points, seg_logits, batch_idx, history_labels):
        batch_size = len(img_metas)
        if not isinstance(seg_logits, list):
            seg_logits = [seg_logits[batch_idx==i] for i in range(batch_size)]
        if not isinstance(points, list):
            points = [points[batch_idx==i] for i in range(batch_size)]
        for i in range(batch_size):
            n, t, c = history_labels[i].shape
            sample_idx = img_metas[i]['sample_idx']
            if not os.path.exists('./work_dirs/results/history_labels'):
                os.makedirs('./work_dirs/results/history_labels')
            path = './work_dirs/results/history_labels/{}.{}'.format(sample_idx, 'npy')
            index = (points[i][:, 5] // 10).long()
            history_labels[i][:, -1, :][index] = seg_logits[i].sigmoid()
            history_labels[i] = torch.cat((history_labels[i][:, -1, :][:, None, :], history_labels[i][:, 0:-1, :]), dim=1)
            history_labels_i = history_labels[i].cpu().numpy()
            np.save(path, history_labels_i)
        history_labels = None

    # Base3DDetector.forward_test()-->simple_test()
    def simple_test(self,
                    points=None,
                    img_metas=None,
                    img=None,
                    rescale=False,
                    gt_bboxes_3d=None,
                    gt_labels_3d=None,
                    pts_semantic_mask=None,
                    gt_bboxes=None,
                    gt_labels=None,
                    pts_instance_mask=None,
                    gt_semantic_seg=None,
                    gt_masks=None,
                    ori_gt_masks=None,
                    ccl_labels=None,
                    pseudo_labels=None):
        
        if pts_semantic_mask is not None:
            eval_2d = False
        else:
            eval_2d = True
        
        if eval_2d:
            single_eval_2d = True
            eval_3d_semantic = False
            eval_3d_mask = False        # weak 3D Box or 2D Box
        else:
            single_eval_2d = False
            eval_3d_semantic = True
            eval_3d_mask = True
            ring_segment_correct = False#

        is_local = False
        is_visul = False             # visualization
        eval_full_3d_mask = False    # overall scene point cloud (conatin the behind point clouds)
        pts_semantic_mask_ccl = False# use the ccl to cluster semantic gt to get the instance
        eval_coarse_3d_mask = False
        eval_2d_gt_mask = False      # test gen 2d gt mask is accurate (is yes)
        eval_all_points = False       # default: False, 只评估在图片上的点云

        if is_visul:
            eval_2d = False; eval_3d_semantic = True; eval_3d_mask = True; ring_segment_correct = False; self.use_ema=False; is_visul = True

        if self.use_ema:
            self.swap_ema()
        
        results_list = [dict() for i in range(len(img_metas))]

        if eval_2d_gt_mask:
            results_list = self.eval_gt_mask(results_list, img, img_metas, gt_masks, gt_semantic_seg, gt_labels, ori_gt_masks)
            return results_list
        
        if eval_full_3d_mask:
            # Input the point cloud of the full scene for evaluation, not point cloud onto multi-images
            results_list = self.simple_test_pts_inst_full(points, pts_semantic_mask, img_metas)
            return results_list

        if self.with_img_bbox and self.with_img_branch and eval_2d:

            self.use_multi_view = False
            if img.dim() == 5:
                self.use_multi_view = True
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)

            img_feats = self.extract_img_feat(img=img, img_metas=img_metas)
            batch_nums = int(len(img)/len(points))
            if batch_nums != 1:
                new_img_metas = [dict() for _ in range(len(img))]
                for b in range(len(img_metas)):
                    for key, value in img_metas[b].items():
                        for i in range(batch_nums):
                            if isinstance(value, list):
                                new_img_metas[b*batch_nums+i][key] = value[i]
                            else:
                                new_img_metas[b*batch_nums+i][key] = value
                # batch eval
                if is_local is False and is_visul is False and single_eval_2d is False:
                    bbox_results, mask_results = self.simple_test_img(img_feats, new_img_metas, rescale=rescale) # bbox_results, mask_results
                    for i in range(len(img_metas)):
                        results_list[i]['img_preds'] = [dict() for _ in range(batch_nums)]
                        for j in range(batch_nums):
                            results_list[i]['img_preds'][j]['img_bbox'] = bbox_results[i*batch_nums+j]
                            results_list[i]['img_preds'][j]['img_mask'] = encode_mask_results(mask_results[i*batch_nums+j])
                            results_list[i]['img_preds'][j]['img_metas'] = new_img_metas[i*batch_nums+j]
                # single image eval
                elif single_eval_2d is True and is_visul is False:
                    for i in range(len(img_metas)):
                        results_list[i]['img_preds'] = [dict() for _ in range(batch_nums)]
                        for j in range(batch_nums):
                            new_img_feats = []
                            for k in range(batch_nums):
                                new_img_feats.append(img_feats[k][j].unsqueeze(0))
                            new_img_feats = tuple(new_img_feats)
                            bbox_results, mask_results = self.simple_test_img(new_img_feats, [new_img_metas[int(i*batch_nums+j)]], rescale=rescale) # bbox_results, mask_results
                            if bbox_results is not None:
                                results_list[i]['img_preds'][j]['img_bbox'] = bbox_results[0]
                                results_list[i]['img_preds'][j]['img_mask'] = encode_mask_results(mask_results[0])
                                results_list[i]['img_preds'][j]['img_metas'] = new_img_metas[int(i*batch_nums+j)]
                
                elif is_local and is_visul is False:
                    results_list = [dict() for i in range(len(new_img_metas))]
                    for i in range(len(new_img_metas)):
                        new_img_feats = []
                        for j in range(batch_nums):
                            new_img_feats.append(img_feats[j][i].unsqueeze(0))
                        new_img_feats = tuple(new_img_feats)
                        bbox_results, mask_results = self.simple_test_img(new_img_feats, [new_img_metas[i]], rescale=rescale) # bbox_results, mask_results
                        if bbox_results is not None:
                            results_list[i]['img_bbox'] = bbox_results[0]     # 长度是类别数 x (N,5),左上角和右下角坐标+得分 (N,5)
                            results_list[i]['img_mask'] = encode_mask_results(mask_results[0])  # 转为rle格式
                            results_list[i]['img_metas'] = new_img_metas[i]
                        else:
                            import cv2
                            # aimg = aligned_bilinear(img[j].unsqueeze(0), 1)  # 1280，1920，3
                            # aimg = tensor2imgs(aimg, mean=[123.675, 116.28, 103.53], std=[1.0, 1.0, 1.0],to_rgb=False)[0]
                            # aimg = cv2.cvtColor(aimg, cv2.COLOR_RGB2BGR)
                            # cv2.imwrite('{}_{}.jpg'.format(new_img_metas[int(i*batch_nums+j)]['filename'][17:-2], j),aimg)
                            # print('\n No Object{}'.format(new_img_metas[int(i*batch_nums+j)]['filename']))
                # visualization
                elif is_visul:
                    results_list[0]['img_preds'] = [dict() for _ in range(batch_nums)]
                    for i in range(len(new_img_metas)):
                        new_img_feats = []
                        for j in range(batch_nums):
                            new_img_feats.append(img_feats[j][i].unsqueeze(0))
                        new_img_feats = tuple(new_img_feats)
                        bbox_results, mask_results = self.simple_test_img(new_img_feats, [new_img_metas[i]], rescale=rescale) # bbox_results, mask_results
                        if bbox_results is not None:
                            results_list[0]['img_preds'][i]['img_bbox'] = bbox_results[0]
                            results_list[0]['img_preds'][i]['img_mask'] = encode_mask_results(mask_results[0])
                            results_list[0]['img_preds'][i]['img_metas'] = new_img_metas[i]
            else:
                bbox_results, mask_results = self.simple_test_img(img_feats, img_metas, rescale=rescale) # bbox_results, mask_results
                if False:
                    import cv2
                    cv2.imwrite('masks.jpg', mask_results[0][0].max(0) * 255)
                if bbox_results is not None:
                    for i in range(len(img_metas)):
                        results_list[i]['img_bbox'] = bbox_results[i]     # 长度是类别数 x (N,5),左上角和右下角坐标+得分 (N,5)
                        # results_list[i]['img_mask'] = mask_results[i]   # 长度是类别数 x (1280,1920) eg(N, 1280, 1920)
                        results_list[i]['img_mask'] = encode_mask_results(mask_results[i])  # 转为rle格式
                        results_list[i]['img_metas'] = img_metas[i]

        # 3D semantic eval
        if eval_3d_semantic:
            pts_results = self.simple_test_pts_segmet(points, img_metas)

            for i in range(len(img_metas)):
                # only top liadr has semantic label
                top_lidar_mask = points[i][:, 6] == 0
                if eval_all_points:
                    top_lidar_mask = top_lidar_mask
                else:
                    top_lidar_mask = top_lidar_mask & ((points[i][:, 10]!=-1) | (points[i][:, 11]!=-1))
                results_list[i]['segment3d'] = pts_results[i]['segment_results'][top_lidar_mask]
                results_list[i]['offsets'] = pts_results[i]['offsets'][top_lidar_mask]
                results_list[i]['seg_scores'] = pts_results[i]['seg_logits'][top_lidar_mask].sigmoid()
                if pts_semantic_mask is not None:
                    pts_semantic_mask[0][i][pts_semantic_mask[0][i]==-1] = self.num_classes  # bg_label=self.num_classes
                    results_list[i]['pts_semantic_mask'] = pts_semantic_mask[0][i][top_lidar_mask].cpu().type(torch.float32)
                # RSC
                if ring_segment_correct:
                    results_list[i]['segment3d'] = self.rsc(results_list[i]['segment3d'], ccl_labels[0][i][:, 4][top_lidar_mask])
        
        # 3D instance eval
        if eval_3d_mask:
            results_list = self.simple_test_pts_inst_weak(results_list, img_metas, points)
        
        # ccl gt semantic label generate inst
        if pts_semantic_mask_ccl:
            results_list = self.eval_pts_semantic_mask_ccl(points, pts_semantic_mask, img_metas)
            return results_list
        
        # use the preprocess directly eval the 3d mask
        if eval_coarse_3d_mask:
            if ccl_labels is not None:
                results_list = self.simple_test_coarse_mask_3d(points, results_list, img_metas, gt_bboxes, gt_labels, pts_semantic_mask, ccl_labels)
            else:
                results_list = self.simple_test_coarse_mask_3d(points, results_list, img_metas, gt_bboxes, gt_labels, pts_semantic_mask, pseudo_labels)

        if self.use_ema:
            self.swap_ema()
        
        return results_list

    # 检查完成 Return：list(zip(bbox_results, mask_results))
    def simple_test_img(self, img_feats, img_metas, proposals=None, rescale=False):
        feat = img_feats
        outputs = self.img_bbox_head.simple_test(
            feat, self.img_mask_head.param_conv, img_metas, rescale=rescale)  # 注意这个outputs的输出是不是按照batch来的
        det_bboxes, det_labels, det_params, det_coors, det_level_inds = zip(*outputs)  # 框的两个角位置坐标+得分，所属类别，滤波器参数(N,233)，box在当前level的位置(N,2)，所属的layer索引
        # if len(det_bboxes) == 1 and len(det_bboxes[0]) == 0:
        #     return None, None
        boxes_nums = 0
        for i in range(len(det_bboxes)):
            boxes_nums += len(det_bboxes[i])
        if boxes_nums == 0:
            return None, None
        bbox_results = [
            bbox2result(det_bbox, det_label, self.img_bbox_head.num_classes)
            for det_bbox, det_label in zip(det_bboxes, det_labels)
        ]  # 转为list，长度是和num_classes一样的
        mask_feat = self.img_mask_branch(feat)
        mask_results = self.img_mask_head.simple_test(
            mask_feat,
            det_labels,
            det_params,
            det_coors,
            det_level_inds,
            img_metas,
            self.img_bbox_head.num_classes,
            rescale=rescale) # mask_results 是个列表
        # return list(zip(bbox_results, mask_results))
        return bbox_results, mask_results

    def simple_test_pts_segmet(self, points, img_metas):
        seg_out_dict = self.pts_segmentor.simple_test(points, img_metas, rescale=False)
        seg_logits = seg_out_dict['seg_logits']  # N,3
        seg_offsets = seg_out_dict['offsets'] # N,9
        batch_idx = seg_out_dict['batch_idx']
        out = []  # length is batch size
        threshold = torch.tensor([0.5, 0.5, 0.5], device=seg_logits.device)
        for i in range(len(img_metas)):
            seg_logits_idx = seg_logits[batch_idx==i].sigmoid()
            seg_offsets_idx = seg_logits[batch_idx==i]
            segment_results = torch.ones(seg_logits_idx.shape[0], device=seg_logits.device) * self.num_classes
            segment_offsets = torch.zeros((seg_offsets_idx.shape[0],3), device=seg_logits.device, dtype=torch.float32)
            assert len(seg_logits_idx) == len(seg_offsets_idx)
            # segment_mask = segment_results == 3
            max_inds = seg_logits_idx.max(1)[1]
            for j in range(self.num_classes):
                mask = (max_inds==j) & (seg_logits_idx[:,j]>threshold[j])
                segment_results[mask] = j
                segment_offsets[mask] = seg_offsets[batch_idx==i][mask][:,j*3:j*3+3]
            out.append(dict(segment_results=segment_results.cpu(), offsets=segment_offsets.cpu(), seg_logits=seg_logits_idx.cpu()))
        return out

    def simple_test_pts_inst_full(self, points, pts_semantic_mask, img_metas):
        x, pts_coors, points_ = self.pts_segmentor.extract_feat(points, img_metas)
        batch_idx=pts_coors[:, 0]
        feats = x[0]
        valid_pts_mask = x[1]
        points_ = points_[valid_pts_mask]
        pts_coors = pts_coors[valid_pts_mask]
        # segmentation head
        output = self.pts_segmentor.segmentation_head.pre_seg_conv(feats)  # (N, 67) --> (N, 128)
        seg_logits = self.pts_segmentor.segmentation_head.cls_seg(output)  # (N, 128) --> (N,2)
        vote_preds = self.pts_segmentor.segmentation_head.voting(output)
        offsets = self.pts_segmentor.segmentation_head.decode_vote_targets(vote_preds)
        results_list = [dict() for i in range(len(img_metas))]
        for i in range(len(img_metas)):
            results_list[i]['img_metas'] = img_metas[i]
        threshold = torch.tensor([0.5, 0.5, 0.5], device=seg_logits.device)
        for i in range(len(img_metas)):
            seg_logits_idx = seg_logits[batch_idx==i].sigmoid()
            seg_offsets_idx = offsets[batch_idx==i]
            segment_results = torch.ones(seg_logits_idx.shape[0], device=seg_logits.device) * self.num_classes
            segment_offsets = torch.zeros((seg_offsets_idx.shape[0],3), device=seg_logits.device, dtype=torch.float32)
            max_inds = seg_logits_idx.max(1)[1]
            for j in range(self.num_classes):
                mask = (max_inds==j) & (seg_logits_idx[:, j]>threshold[j])
                segment_results[mask] = j
                segment_offsets[mask] = offsets[batch_idx==i][mask][:,j*3:j*3+3]
            results_list[i]['segment3d'] = segment_results.cpu().type(torch.int32)
            results_list[i]['seg_scores'] = seg_logits_idx.cpu().type(torch.float32)
            results_list[i]['offsets'] = segment_offsets.cpu().type(torch.float32)
            if pts_semantic_mask is not None:
                pts_semantic_mask[0][i][pts_semantic_mask[0][i]==-1] = self.num_classes
                results_list[i]['pts_semantic_mask'] = pts_semantic_mask[0][i].cpu().type(torch.float32)
        
        self.point_cloud_range = [-80, -80, -2, 80, 80, 4]
        self.voxel_size = [[0.15, 0.15, 6], [0.05, 0.05, 6], [0.1, 0.1, 6]]
        self.dist_size = [[0.6, 0.6, 0], [0.1, 0.1, 0], [0.4, 0.4, 0]]
        self.kernel_size_ = [[1, 9, 9], [1, 5, 5], [1, 9, 9]]  # [z, y, x]
        for i in range(len(img_metas)):
            points_shape = results_list[i]['segment3d'].shape
            points_clusters = torch.zeros((points_shape[0]), dtype=torch.int)

            dist = [0.6, 0.1, 0.4]  # car ped cycl
            top_points = points[i][points[i][:,6]==0]
            tmp_inds = [0]
            mask_list = []
            mask_scores_list = []
            for j in range(self.num_classes):
                mask_cls = []
                mask_cls_scores = []
                seg_scores_ = results_list[i]['seg_scores']
                xyz_mask = results_list[i]['segment3d'] == j
                if xyz_mask.sum() == 0:
                    tmp_inds.append(0)
                    continue
                xyz = top_points[:,0:3][xyz_mask]#.cpu()
                offsets_ = results_list[i]['offsets'][xyz_mask].to(xyz.device)
                center_pred = xyz + offsets_
                # c_inds = find_connected_componets_single_batch(center_pred.cpu(), None, dist[j])

                # use spccl
                device = center_pred.device
                cls_id = j
                class_id = torch.zeros((xyz.shape[0]), device=device, dtype=torch.int32)
                batch_id = torch.zeros((xyz.shape[0]), device=device, dtype=torch.int32)
                nums = xyz_mask.sum().to(device)
                if nums < 100:
                    num_act_in = 100
                elif nums < 1000:
                    num_act_in = int((nums//100)*100+100)
                elif nums < 10000:
                    num_act_in = int((nums//1000)*1000+1000)
                elif nums < 100000:
                    num_act_in = int((nums//10000)*10000+10000)
                else:
                    num_act_in = 300000
                center_pred = points_padding(center_pred, num_act_in, 0).contiguous()
                class_id = points_padding(class_id, num_act_in, -1)
                batch_id = points_padding(batch_id, num_act_in, -1)
                spatial_shape = gen_shape(self.point_cloud_range, self.voxel_size[cls_id])
                cluster_inds, _, _, _ = voxel_spccl3d(center_pred, 
                                batch_id.type(torch.int32),
                                class_id.type(torch.int32),
                                nums.type(torch.int32),
                                self.kernel_size_[cls_id],
                                self.point_cloud_range,
                                [1, 1, 1],
                                self.voxel_size[cls_id],
                                self.dist_size[cls_id],
                                spatial_shape,
                                1,
                                1)
                c_inds = cluster_inds[0 : nums]

                c_inds = c_inds + 1 + tmp_inds[j]
                tmp_inds.append(c_inds.max())
                sets, _, counts = torch.unique(c_inds, return_counts=True, return_inverse=True)
                for ci in range(len(counts)):
                    if counts[ci] <= 0:
                        c_inds[c_inds==sets[ci]]=0
                    else:
                        points_id = torch.zeros((points_shape[0]), dtype=torch.int)
                        c_inds_copy = c_inds.clone()*0
                        c_inds_copy[c_inds==sets[ci]] = 1
                        points_id[xyz_mask] = c_inds_copy.cpu()
                        mask_cls.append(points_id.view(1,-1))
                        mask_cls_scores.append(seg_scores_[:,j][points_id==1].mean().cpu().numpy())
                if len(mask_cls)==0:
                    continue
                mask_cls = torch.stack(mask_cls)
                points_clusters[xyz_mask] = c_inds.cpu()
                mask_list.append(mask_cls)
                mask_scores_list.append(mask_cls_scores)
            results_list[i]['mask_3d_rle'] = encode_mask_results(mask_list)
            results_list[i]['mask_3d_scores'] = mask_scores_list
        
        return results_list

    def eval_pts_semantic_mask_ccl(self, points, pts_semantic_mask, img_metas):
        results_list = [dict() for i in range(len(img_metas))]
        for i in range(len(img_metas)):
            results_list[i]['img_metas'] = img_metas[i]
        self.point_cloud_range = [-80, -80, -2, 80, 80, 4]
        self.voxel_size = [[0.15, 0.15, 6], [0.05, 0.05, 6], [0.1, 0.1, 6]]
        self.dist_size = [[0.6, 0.6, 0], [0.1, 0.1, 0], [0.4, 0.4, 0]]
        self.kernel_size_ = [[1, 9, 9], [1, 5, 5], [1, 9, 9]]  # [z, y, x]
        for i in range(len(img_metas)):
            top_lidar_mask = points[i][:, 6] == 0
            semantic = pts_semantic_mask[0][i][top_lidar_mask].cpu().type(torch.float32)
            semantic[semantic==-1] = self.num_classes
            points_shape = semantic.shape
            points_clusters = torch.zeros((points_shape[0]), dtype=torch.int)
            dist = [0.6, 0.1, 0.4]  # car ped cycl
            top_points = points[i][points[i][:,6]==0]
            tmp_inds = [0]
            mask_list = []
            mask_scores_list = []
            for j in range(self.num_classes):
                mask_cls = []
                mask_cls_scores = []
                xyz_mask = semantic == j
                if xyz_mask.sum() == 0:
                    tmp_inds.append(0)
                    continue
                xyz = top_points[:,0:3][xyz_mask]#.cpu()
                center_pred = xyz
                # c_inds = find_connected_componets_single_batch(center_pred.cpu(), None, dist[j])

                # use spccl
                device = center_pred.device
                cls_id = j
                class_id = torch.zeros((xyz.shape[0]), device=device, dtype=torch.int32)
                batch_id = torch.zeros((xyz.shape[0]), device=device, dtype=torch.int32)
                nums = xyz_mask.sum().to(device)
                if nums < 100:
                    num_act_in = 100
                elif nums < 1000:
                    num_act_in = int((nums//100)*100+100)
                elif nums < 10000:
                    num_act_in = int((nums//1000)*1000+1000)
                elif nums < 100000:
                    num_act_in = int((nums//10000)*10000+10000)
                else:
                    num_act_in = 300000
                center_pred = points_padding(center_pred, num_act_in, 0).contiguous()
                class_id = points_padding(class_id, num_act_in, -1)
                batch_id = points_padding(batch_id, num_act_in, -1)
                spatial_shape = gen_shape(self.point_cloud_range, self.voxel_size[cls_id])
                cluster_inds, _, _, _ = voxel_spccl3d(center_pred, 
                                batch_id.type(torch.int32),
                                class_id.type(torch.int32),
                                nums.type(torch.int32),
                                self.kernel_size_[cls_id],
                                self.point_cloud_range,
                                self.voxel_size[cls_id],
                                self.dist_size[cls_id],
                                spatial_shape,
                                1,
                                1)
                c_inds = cluster_inds[0 : nums]
                c_inds = c_inds + 1 + tmp_inds[j]
                tmp_inds.append(c_inds.max())
                sets, _, counts = torch.unique(c_inds, return_counts=True, return_inverse=True)
                for ci in range(len(counts)):
                    if counts[ci] <= 0:
                        c_inds[c_inds==sets[ci]]=0
                    else:
                        points_id = torch.zeros((points_shape[0]), dtype=torch.int)
                        c_inds_copy = c_inds.clone()*0
                        c_inds_copy[c_inds==sets[ci]] = 1
                        points_id[xyz_mask] = c_inds_copy.cpu()
                        mask_cls.append(points_id.view(1,-1))
                        mask_cls_scores.append(np.array(1.0))
                if len(mask_cls)==0:
                    continue
                mask_cls = torch.stack(mask_cls)
                points_clusters[xyz_mask] = c_inds.cpu()
                mask_list.append(mask_cls)
                mask_scores_list.append(mask_cls_scores)
            # results_list[i]['3d_instance_id'] = points_clusters
            results_list[i]['mask_3d_rle'] = encode_mask_results(mask_list)
            results_list[i]['mask_3d_scores'] = mask_scores_list
            results_list[i]['segment3d'] = semantic.cpu().type(torch.int32)
            results_list[i]['pts_semantic_mask'] = semantic.cpu().type(torch.int32)
        return results_list

    def simple_test_pts_inst_weak(self, results_list, img_metas, points):
        self.point_cloud_range = [-80, -80, -2, 80, 80, 4]
        self.voxel_size = [[0.15, 0.15, 6], [0.05, 0.05, 6], [0.1, 0.1, 6]]
        self.dist_size = [[0.6, 0.6, 0], [0.1, 0.1, 0], [0.4, 0.4, 0]]
        self.kernel_size_ = [[1, 9, 9], [1, 5, 5], [1, 9, 9]]  # [z, y, x]
        for i in range(len(img_metas)):
            results_list[i]['img_metas'] = img_metas[i]
        for i in range(len(img_metas)):
            points_shape = results_list[i]['segment3d'].shape
            points_clusters = torch.zeros((points_shape[0]), dtype=torch.int)
            dist = [0.6, 0.1, 0.4]  # car ped cycl
            top_points = points[i][points[i][:,6]==0]
            tmp_inds = [0]
            mask_list = []
            mask_scores_list = []
            for j in range(self.num_classes):
                mask_cls = []
                mask_cls_scores = []
                seg_scores_ = results_list[i]['seg_scores']
                xyz_mask = results_list[i]['segment3d'] == j
                if xyz_mask.sum() == 0:
                    tmp_inds.append(0)
                    continue
                xyz = top_points[:,0:3][xyz_mask]
                offsets = results_list[i]['offsets'][xyz_mask].to(xyz.device) # cuda
                center_pred = xyz + offsets
                # c_inds = find_connected_componets_single_batch(center_pred.cpu(), None, dist[j])
                # use spccl
                device = center_pred.device
                cls_id = j
                class_id = torch.zeros((xyz.shape[0]), device=device, dtype=torch.int32)
                batch_id = torch.zeros((xyz.shape[0]), device=device, dtype=torch.int32)
                nums = xyz_mask.sum().to(device)
                if nums < 100:
                    num_act_in = 100
                elif nums < 1000:
                    num_act_in = int((nums//100)*100+100)
                elif nums < 10000:
                    num_act_in = int((nums//1000)*1000+1000)
                elif nums < 100000:
                    num_act_in = int((nums//10000)*10000+10000)
                else:
                    num_act_in = 300000
                center_pred = points_padding(center_pred, num_act_in, 0).contiguous()
                class_id = points_padding(class_id, num_act_in, -1)
                batch_id = points_padding(batch_id, num_act_in, -1)
                spatial_shape = gen_shape(self.point_cloud_range, self.voxel_size[cls_id])
                cluster_inds, _, _, _ = voxel_spccl3d(center_pred, 
                                batch_id.type(torch.int32),
                                class_id.type(torch.int32),
                                nums.type(torch.int32),
                                self.kernel_size_[cls_id],
                                self.point_cloud_range,
                                [1, 1, 1],
                                self.voxel_size[cls_id],
                                self.dist_size[cls_id],
                                spatial_shape,
                                1,
                                1)
                c_inds = cluster_inds[0 : nums]

                c_inds = c_inds + 1 + tmp_inds[j]
                tmp_inds.append(c_inds.max())
                sets, _, counts = torch.unique(c_inds, return_counts=True, return_inverse=True)
                for ci in range(len(counts)):
                    if counts[ci] <= 0:
                        c_inds[c_inds==sets[ci]]=0
                    else:
                        points_id = torch.zeros((points_shape[0]), dtype=torch.int)
                        c_inds_copy = c_inds.clone() * 0
                        c_inds_copy[c_inds==sets[ci]] = 1
                        points_id[xyz_mask] = c_inds_copy.cpu()
                        mask_cls.append(points_id.view(1, -1))
                        mask_cls_scores.append(seg_scores_[:,j][points_id==1].mean().cpu().numpy())
                if len(mask_cls)==0:
                    continue
                mask_cls = torch.stack(mask_cls)
                points_clusters[xyz_mask] = c_inds.cpu()
                mask_list.append(mask_cls)
                mask_scores_list.append(mask_cls_scores)
            results_list[i]['3d_instance_id'] = points_clusters # vis
            results_list[i]['mask_3d_rle'] = encode_mask_results(mask_list)
            results_list[i]['mask_3d_scores'] = mask_scores_list
        return results_list

    def simple_test_coarse_mask_3d(self, points, results_list, img_metas, gt_bboxes, gt_labels, pts_semantic_mask, pseudo_labels=None):

        if pseudo_labels is not None:
            pseudo_labels = pseudo_labels[0]

        gt_bboxes_ = []
        gt_labels_ = []
        if isinstance(gt_bboxes[0][0], list):
            gt_bboxes = gt_bboxes[0]
            if gt_labels is not None:
                gt_labels = gt_labels[0]
        for i in range(len(img_metas)):
            gt_bboxes_.extend(gt_bboxes[i])
            if gt_labels is not None:
                gt_labels_.extend(gt_labels[i])
        points = self.ccl(points, gt_bboxes_, gt_labels_, img_metas)  # [N, (C+2)]
        for i in range(len(img_metas)):
            results_list[i]['img_metas'] = img_metas[i]
        for i in range(len(img_metas)):
            top_lidar_mask = points[i][:, 6] == 0
            mask_list = [[] for _ in range(self.num_classes)]
            mask_scores_list = [[] for _ in range(self.num_classes)]
            semantic_segment = torch.ones((points[i].shape[0]), dtype=torch.int32) * self.num_classes
            assert len(semantic_segment) == top_lidar_mask.sum()
            instance = points[i][:, 20].cpu().long()
            sets = torch.unique(instance)
            for j in range(len(sets)):
                if sets[j]==-1:
                    continue
                tmp_mask = instance * 0
                tmp_mask[instance==sets[j]] = 1
                tmp_mask = tmp_mask.view(1, -1)
                # img_id*1000+box_inds
                img_id = int(sets[j] // 1000)
                box_id = int(sets[j] % 1000)
                cls_id = gt_labels[i][img_id][box_id]
                mask_list[int(cls_id)].append(tmp_mask)
                mask_scores_list[int(cls_id)].append(1.0)
                semantic_segment[instance==sets[j]] = int(cls_id)
            for j in range(self.num_classes):
                if len(mask_list[j]) != 0:
                    mask_list[j] = torch.stack(mask_list[j])
            results_list[i]['mask_3d_rle'] = encode_mask_results(mask_list)
            results_list[i]['mask_3d_scores'] = mask_scores_list
            results_list[i]['segment3d'] = semantic_segment
            if pts_semantic_mask is not None:
                pts_semantic_mask[0][i][pts_semantic_mask[0][i]==-1] = self.num_classes  # bg_label=self.num_classes
                results_list[i]['pts_semantic_mask'] = pts_semantic_mask[0][i][top_lidar_mask].cpu().type(torch.float32)
        
        return results_list

    def eval_gt_mask(self, results_list, img, img_metas, gt_masks, gt_semantic_seg, gt_labels, ori_gt_masks):
        self.use_multi_view = False
        if img.dim() == 5:
            self.use_multi_view = True
            B, N, C, H, W = img.size()
            img = img.view(B * N, C, H, W)
        if isinstance(gt_labels[0][0], list):
            gt_masks = gt_masks[0]
            gt_labels = gt_labels[0]
            if ori_gt_masks is not None:
                ori_gt_masks = ori_gt_masks[0]
        batch_nums = int(len(img)/len(img_metas))
        if batch_nums != 1:
            new_img_metas = [dict() for _ in range(len(img))]
            for b in range(len(img_metas)):
                for key, value in img_metas[b].items():
                    for i in range(batch_nums):
                        if isinstance(value, list):
                            new_img_metas[b*batch_nums+i][key] = value[i]
                        else:
                            new_img_metas[b*batch_nums+i][key] = value
        # batch eval
        for i in range(len(img_metas)):
            results_list[i]['img_preds'] = [dict() for _ in range(batch_nums)]
            for j in range(batch_nums):
                out_masks = []
                for c in range(self.num_classes):
                    ori_shape = new_img_metas[i*batch_nums+j]['ori_shape']
                    img_shape = new_img_metas[i*batch_nums+j]['img_shape']
                    pad_shape = new_img_metas[i*batch_nums+j]['pad_shape']
                    c_mask = gt_labels[i][j]==c
                    c_mask_ = gt_masks[i][j].crop(np.array([0, 0, img_shape[1], img_shape[0]])).rescale((ori_shape[0], ori_shape[1]), interpolation='bilinear').masks[c_mask.cpu().numpy()]
                    # c_mask_ = c_mask_[:, 0:ori_shape[0], 0:ori_shape[1]]
                    # ori_gt_masks_ = ori_gt_masks[i][j].masks[c_mask.cpu().numpy()]
                    out_masks.append(c_mask_)
                    if False:
                        import matplotlib.pyplot as plt
                        # plt.figure(figsize=(10, 10))
                        # plt.title("")
                        aimg = tensor2imgs(img[j][:, 0:img_shape[0], 0:img_shape[1]].unsqueeze(0), mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)[0]
                        # plt.imshow(aimg[:,:,::-1])
                        # vis_mask = gt_masks[i][j].masks[c_mask.cpu().numpy()].sum(0)[0:img_shape[0], 0:img_shape[1]]
                        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
                        # plt.gca().imshow(vis_mask.reshape(img_shape[0], img_shape[1], 1) * color.reshape(1,1,-1))
                        aimg = mmcv.imresize(aimg, ori_shape[0:2][::-1], interpolation='bilinear')
                        # plt.figure(figsize=(10, 10))
                        # plt.imshow(aimg[:,:,::-1])
                        # vis_mask = mmcv.imresize(vis_mask, ori_shape[0:2][::-1], interpolation='nearest')
                        # plt.gca().imshow(vis_mask.reshape(ori_shape[0], ori_shape[1], 1) * color.reshape(1,1,-1))
                        plt.figure(figsize=(10, 10))
                        plt.title("逆变换后的mask")
                        plt.imshow(aimg[:,:,::-1])
                        plt.gca().imshow(ori_gt_masks_[6][:,:,None]** color.reshape(1,1,-1))
                        for mm in range(len(c_mask_)):
                            if mm != 6:
                                continue
                            color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
                            plt.gca().imshow(c_mask_[mm].reshape(ori_shape[0], ori_shape[1], 1) * color.reshape(1,1,-1))
                            
                results_list[i]['img_preds'][j]['img_mask'] = encode_mask_results(out_masks)
                results_list[i]['img_preds'][j]['img_metas'] = new_img_metas[i*batch_nums+j]
        return results_list
    
    def rsc(self, segment_3d, run_id):
        new_segment_3d = segment_3d.clone()
        bg_mask = segment_3d == self.num_classes
        bg_run_id = run_id[bg_mask]
        for i in range(self.num_classes):
            cls_mask = segment_3d == i
            sets, counts = torch.unique(run_id[cls_mask], return_counts=True)
            for j in range(len(sets)):
                if sets[j] == -1 or i == 1:  # run 对于行人的效果有反作用
                    continue
                if (bg_run_id == sets[j]).sum() > counts[j]:  # (bg_run_id == sets[j]).sum() > counts[j] * 1.5
                    new_segment_3d[run_id == sets[j]] = self.num_classes
                elif (run_id == sets[j]).sum() * 0.7 <= counts[j]:  # (run_id == sets[j]).sum() * 0.8 <= counts[j]
                    new_segment_3d[run_id == sets[j]] = i
        return new_segment_3d
    
    # x
    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats, pts_feats = self.extract_feats(points, img_metas, imgs)

        bbox_list = dict()
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.aug_test_pts(pts_feats, img_metas, rescale)
            bbox_list.update(pts_bbox=bbox_pts)
        return [bbox_list]
    
    # x
    def extract_feats(self, points, img_metas, imgs=None):
        """Extract point and image features of multiple samples."""
        if imgs is None:
            imgs = [None] * len(img_metas)
        img_feats, pts_feats = multi_apply(self.extract_img_feat, points, imgs,
                                           img_metas)
        return img_feats, pts_feats
    
    # x
    def aug_test_pts(self, feats, img_metas, rescale=False):
        """Test function of point cloud branch with augmentaiton."""
        # only support aug_test for one sample
        aug_bboxes = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.pts_bbox_head(x)
            bbox_list = self.pts_bbox_head.get_bboxes(
                *outs, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.pts_bbox_head.test_cfg)
        return merged_bboxes
    
    # x
    def show_results(self, data, result, out_dir):
        """Results visualization.

        Args:
            data (dict): Input points and the information of the sample.
            result (dict): Prediction results.
            out_dir (str): Output directory of visualization result.
        """
        for batch_id in range(len(result)):
            if isinstance(data['points'][0], DC):
                points = data['points'][0]._data[0][batch_id].numpy()
            elif mmcv.is_list_of(data['points'][0], torch.Tensor):
                points = data['points'][0][batch_id]
            else:
                ValueError(f"Unsupported data type {type(data['points'][0])} "
                           f'for visualization!')
            if isinstance(data['img_metas'][0], DC):
                pts_filename = data['img_metas'][0]._data[0][batch_id][
                    'pts_filename']
                box_mode_3d = data['img_metas'][0]._data[0][batch_id][
                    'box_mode_3d']
            elif mmcv.is_list_of(data['img_metas'][0], dict):
                pts_filename = data['img_metas'][0][batch_id]['pts_filename']
                box_mode_3d = data['img_metas'][0][batch_id]['box_mode_3d']
            else:
                ValueError(
                    f"Unsupported data type {type(data['img_metas'][0])} "
                    f'for visualization!')
            file_name = osp.split(pts_filename)[-1].split('.')[0]

            assert out_dir is not None, 'Expect out_dir, got none.'
            inds = result[batch_id]['pts_bbox']['scores_3d'] > 0.1
            pred_bboxes = result[batch_id]['pts_bbox']['boxes_3d'][inds]

            # for now we convert points and bbox into depth mode
            if (box_mode_3d == Box3DMode.CAM) or (box_mode_3d
                                                  == Box3DMode.LIDAR):
                points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                                   Coord3DMode.DEPTH)
                pred_bboxes = Box3DMode.convert(pred_bboxes, box_mode_3d,
                                                Box3DMode.DEPTH)
            elif box_mode_3d != Box3DMode.DEPTH:
                ValueError(
                    f'Unsupported box_mode_3d {box_mode_3d} for conversion!')

            pred_bboxes = pred_bboxes.tensor.cpu().numpy()
            show_result(points, None, pred_bboxes, out_dir, file_name)
     
    def ccl(self, points, gt_bboxes, gt_labels, img_metas, pseudo_labels):
        if pseudo_labels is not None:
            if pseudo_labels[0].shape[1] > 5:
                for i in range(len(pseudo_labels)):
                    points[i] = torch.cat((points[i], pseudo_labels[i][:, 5:7]), dim=1)  # 20-->22
                if self._iter < 5:
                    print('load ccl results from minio')
                return points
        device = points[0].device
        batch_nums = int(len(gt_bboxes) / len(img_metas))
        for i in range(len(img_metas)):
            batch_points = points[i]
            points_index = torch.arange(0, len(batch_points), device=device, dtype=torch.int32)
            box_flag = torch.zeros((batch_points.shape[0], 3), device=device, dtype=torch.int32)
            for j in range(batch_nums):
                if len(gt_bboxes[(batch_nums * i + j)])==0:
                    continue
                for b, gt_bbox in enumerate(gt_bboxes[(batch_nums * i + j)]):
                    gt_mask1 = (((batch_points[:, 12] >= gt_bbox[0]) & (batch_points[:, 12] < gt_bbox[2])) &
                                ((batch_points[:, 14] >= gt_bbox[1]) & (batch_points[:, 14] < gt_bbox[3])  &
                                (batch_points[:, 10]==j)))
                    gt_mask2 = (((batch_points[:, 13] >= gt_bbox[0]) & (batch_points[:, 13] < gt_bbox[2])) &
                                ((batch_points[:, 15] >= gt_bbox[1]) & (batch_points[:, 13] < gt_bbox[3])  &
                                (batch_points[:, 11]==j)))
                    gt_mask = gt_mask1 | gt_mask2
                    in_box_mask = gt_mask & (batch_points[:, 17] > 0)
                    if in_box_mask.sum() == 0:
                        continue
                    cls_id = gt_labels[(batch_nums * i + j)][b].long()
                    # must class_id = 0
                    class_id = torch.zeros((in_box_mask.sum().long()), device=device, dtype=torch.int32)
                    xyz = batch_points[in_box_mask][:, 0:3].contiguous()
                    batch_id = torch.zeros((xyz.shape[0]), device=device, dtype=torch.int32)
                    nums = in_box_mask.sum()
                    spatial_shape = gen_shape(self.point_cloud_range, self.voxel_size[cls_id])
                    if nums < 100:
                        num_act_in = 100
                    elif nums < 1000:
                        num_act_in = int((nums//100)*100+100)
                    elif nums < 10000:
                        num_act_in = int((nums//1000)*1000+1000)
                    else:
                        num_act_in = int((nums//10000)*10000+10000)
                    xyz = points_padding(xyz, num_act_in, 0).contiguous()
                    class_id = points_padding(class_id, num_act_in, -1)
                    batch_id = points_padding(batch_id, num_act_in, -1)

                    # If cuda operator is not available, use this function
                    # dist = [0.6, 0.1, 0.4]
                    # cluster_inds = find_connected_componets_single_batch(xyz, batch_id, dist[cls_id])[0 : nums]

                    # cuda op
                    cluster_inds, _, _, _ = voxel_spccl3d(xyz,
                                                        batch_id.type(torch.int32),
                                                        class_id.type(torch.int32),
                                                        nums.type(torch.int32),
                                                        self.kernel_size_[cls_id],
                                                        self.point_cloud_range,
                                                        [1, 1, 1],
                                                        self.voxel_size[cls_id],
                                                        self.dist_size[cls_id],
                                                        spatial_shape,
                                                        1,
                                                        1)
                    cluster_inds = cluster_inds[0 : nums]
                    cluster_sets, cluster_invs, cluster_counts = torch.unique(cluster_inds, return_inverse=True, return_counts=True)
                    c_max_inds = cluster_counts.argmax()
                    c_mask = cluster_invs==cluster_sets[c_max_inds]
                    c_mask_ = box_flag[:, 2][in_box_mask] == 0
                    max_in_box_index = points_index[in_box_mask][c_mask & c_mask_].long()
                    box_flag[:, 2][max_in_box_index] = 1
                    box_flag[:, 0][max_in_box_index] = j*1000+b+1
                    max_in_box_index_2 = points_index[in_box_mask][c_mask & (~c_mask_)].long()
                    box_flag[:, 1][max_in_box_index_2] = j*1000+b+1                    
            box_flag[:, 0:2] = box_flag[:, 0:2] - 1
            fg_mask = (box_flag[:, 0] != -1) |  (box_flag[:, 1] != -1)
            bg_mask = ~(fg_mask | (points[i][:, 17]==-1))
            points[i][:, 17][bg_mask] = 0
            points[i] = torch.cat((points[i], box_flag[:, 0:2]), dim=1)
        return points
