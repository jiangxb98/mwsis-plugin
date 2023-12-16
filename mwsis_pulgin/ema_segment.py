# Copyright (c) OpenMMLab. All rights reserved.
import time
import numpy as np
from os import path as osp
import torch
from torch.nn import functional as F
from scipy.sparse.csgraph import connected_components  # CCL
import copy
import mmcv
from mmcv.ops import Voxelization
from mmcv.parallel import DataContainer as DC
from mmcv.runner import force_fp32

from mmdet.core import multi_apply, bbox2result

from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result, merge_aug_bboxes_3d, show_result)
from mmdet3d.models import builder
from mmdet3d.models.builder import DETECTORS
from mmdet3d.models.detectors.base import Base3DDetector

from mmseg.models import SEGMENTORS
from mmdet3d.models.segmentors.base import Base3DSegmentor

# from xxxx_common.ops.ccl.ccl_utils import spccl, voxel_spccl, voxelized_sampling, sample
from .fsd_ops import scatter_v2, get_inner_win_inds
from .utils import pts_semantic_confusion_matrix

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
class EMASegment(Base3DDetector):
    """Base class of Multi-modality autolabel."""

    def __init__(self,
                with_pts_branch=True,
                with_img_branch=True,
                img_backbone=None,  #
                img_neck=None,      #
                img_bbox_head=None, #
                img_mask_branch=None,  #
                img_mask_head=None,    #
                img_segm_head= None,
                pretrained=None,
                img_roi_head=None,
                img_rpn_head=None,
                middle_encoder_pts=None,  # points completion 
                pts_segmentor=None,       #
                pts_voxel_layer=None,
                pts_voxel_encoder=None,
                pts_middle_encoder=None,
                pts_backbone=None,  #
                pts_neck=None,
                pts_bbox_head=None, #
                pts_roi_head=None,  # 二阶段，暂时先不用
                pts_fusion_layer=None,            
                train_cfg=None,     # 记住cfg是分img和pts的
                test_cfg=None,      #
                cluster_assigner=None,
                init_cfg=None,
                only_one_frame_label=True,
                sweeps_num=1,
                gt_box_type=1,      # 1 is 3d, 2 is 2d
                num_classes=3,
                loss_mask_3d=None,
                use_2d_mask=True,):
        super(EMASegment, self).__init__(init_cfg=init_cfg)

        # 这里也是个全局的配置
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        self.with_pts_branch = with_pts_branch
        self.with_img_branch = with_img_branch
        self.gt_box_type = gt_box_type
        self.num_classes = num_classes
        self.use_2d_mask = use_2d_mask
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

        self.pts_roi_head = pts_roi_head  # 
        if pts_roi_head is not None:
            rcnn_train_cfg = train_cfg.pts.rcnn if train_cfg.pts else None
            pts_roi_head.update(train_cfg=rcnn_train_cfg)
            pts_roi_head.update(test_cfg=test_cfg.pts.rcnn)
            pts_roi_head.pretrained = pretrained
            self.roi_head = builder.build_head(pts_roi_head)
        # 这里有个pts_cfg配置
        self.pts_cfg = self.train_cfg.pts if self.train_cfg.pts else self.test_cfg.pts
        if 'radius' in cluster_assigner:
            raise NotImplementedError
            self.cluster_assigner = SSGAssigner(**cluster_assigner)
        elif 'hybrid' in cluster_assigner:
            raise NotImplementedError
            cluster_assigner.pop('hybrid')
            self.cluster_assigner = HybridAssigner(**cluster_assigner)
        else:
            self.cluster_assigner = ClusterAssigner(**cluster_assigner)

        # self.cluster_assigner.num_classes = self.num_classes
        self.print_info = {}
        self.as_rpn = False

        self.runtime_info = dict()
        self.only_one_frame_label = only_one_frame_label
        self.sweeps_num = sweeps_num
        self.register_buffer("_iter", torch.zeros([1]))
        self._warmup_iters = 2000
        self.loss_mask_3d = builder.build_loss(loss_mask_3d)

        # 点云补全
        if middle_encoder_pts:
            self.middle_encoder_pts = builder.build_middle_encoder(middle_encoder_pts)

        # BoxInst
        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = builder.build_neck(img_neck)
        if img_bbox_head is not None:  # train_cfg.img 送入到SingleStageDetector的bbox_head
            img_train_cfg = train_cfg.img if train_cfg else None
            img_bbox_head.update(train_cfg=img_train_cfg)
            img_test_cfg = test_cfg.img if test_cfg else None
            img_bbox_head.update(test_cfg=img_test_cfg)
            self.img_bbox_head = builder.build_head(img_bbox_head)
        if img_mask_branch is not None:
            self.img_mask_branch = builder.build_head(img_mask_branch)
        if img_mask_head is not None:
            self.img_mask_head = builder.build_head(img_mask_head)           
        if img_segm_head is not None:
            self.img_segm_head = builder.build_head(img_segm_head)
        else:
            self.img_segm_head = None
        
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

    @property
    def with_middle_encoder_pts(self):
        return hasattr(self, 'middle_encoder_pts') and self.middle_encoder_pts is not None

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
        if not self.with_pts_bbox:
            return None      
        cluster_xyz, _, inv_inds = scatter_v2(center_preds, pts_cluster_inds, mode='avg', return_inv=True)

        f_cluster = points[:, :3] - cluster_xyz[inv_inds]
        # (N,128), (N_clusters,768), (N_clusters,3)
        out_pts_feats, cluster_feats, out_coors = self.pts_backbone(points, pts_feats, pts_cluster_inds, f_cluster)
        out_dict = dict(
            cluster_feats=cluster_feats,
            cluster_xyz=cluster_xyz,
            cluster_inds=out_coors
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
                      roi_points=None,
                      batch_roi_points=None,
                      points2img=None,
                      points2img_idx=None,
                      ):
        """Forward training function.
        Returns:
            dict: Losses of different branches.
        """
        
        # 过滤掉 -1 label
        # gt_bboxes_3d = [b[l>=0] for b, l in zip(gt_bboxes_3d, gt_labels_3d)]
        # gt_labels_3d = [l[l>=0] for l in gt_labels_3d]
        losses = dict()
        multi_losses = self.multi_forward_train(points, img_metas, gt_bboxes_3d, gt_labels_3d, gt_labels, gt_bboxes,
                                gt_masks, img, proposals, gt_bboxes_ignore, self.runtime_info, pts_semantic_mask,
                                pts_instance_mask, gt_semantic_seg, gt_yaw, lidar_density, roi_points,
                                batch_roi_points, points2img, points2img_idx,
                                )
        losses.update(multi_losses)

        return losses

    def multi_forward_train(self,
                            points=None,     # need
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
                            roi_points=None, # need
                            batch_roi_points=None,
                            points2img=None,
                            points2img_idx=None,
                            ):
        # 2D Box Branch
        img_feats = self.extract_img_feat(img=img, img_metas=img_metas)
        cls_score, bbox_pred, centerness, param_pred = self.img_bbox_head(img_feats, self.img_mask_head.param_conv)
        img_bbox_head_loss_inputs = (cls_score, bbox_pred, centerness) + (gt_bboxes, gt_labels, img_metas)
        # gt_inds是指 torch.cat(gt_labels)后的索引值 返回的是所有采样点对应的gt_inds，如果没有分配GT，那么就是-1
        losses, coors, level_inds, img_inds, gt_inds = self.img_bbox_head.loss(
            *img_bbox_head_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        # 2D Mask Branch
        mask_feat = self.img_mask_branch(img_feats)  # Bx16xHxW
        inputs = (cls_score, centerness, param_pred, coors, level_inds, img_inds, gt_inds)
        # 如果没有gt，则直接返回
        pos_mask = gt_inds != -1
        if ~pos_mask.any():
            return losses
        
        # 3D VoteSegmentor Forward
        if self.gt_box_type == 2:
            if gt_yaw is not None:
                gt_bboxes = self.combine_yaw_info(gt_bboxes, gt_yaw)
            gt_bboxes_3d = gt_bboxes
            gt_labels_3d = gt_labels
            points = self.paint_rgb2pts(points, img, img_metas) # 这里采用的是grid_sample的形式
            roi_points =self.paint_rgb2pts(roi_points, img, img_metas)
        # seg_points=points(N,3/12)没变, seg_logits(N,3), seg_vote_preds(N,9), vote_preds decode后的偏置offsets(N,9) seg_feats(N,64+3)每个点的特征
        seg_out_dict = self.pts_segmentor(points=points, img_metas=img_metas, gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d, 
                                          as_subsegmentor=True, pts_semantic_mask=pts_semantic_mask, gt_box_type=self.gt_box_type)
        losses.update(seg_out_dict['losses'])

        # 计算动态卷积loss
        self._iter += 1
        warmup_factor = min(self._iter.item() / float(self._warmup_iters), 1.0)
        loss_dynamic = self.cal_dynamic_loss(img, points, gt_bboxes, gt_labels, img_metas, gt_inds, level_inds,
                                             img_inds, param_pred, seg_out_dict, mask_feat, warmup_factor)
        losses.update({'loss_dynamic': loss_dynamic})
      
        # 2D Mask Head
        param_pred, coors, level_inds, img_inds, gt_inds = self.img_mask_head.training_sample(*inputs)
        # (N, 1, 640/4, 960/4) （640，960）is input img shape
        mask_pred = self.img_mask_head(mask_feat, param_pred, coors, level_inds, img_inds)

        points = seg_out_dict['seg_points']
        seg_scores = seg_out_dict['seg_logits'].clone().detach().sigmoid()
        dict_to_sample = dict(
            seg_points=points,                      # points:(N, +rgb)
            seg_logits=seg_out_dict['seg_logits'],  # points_seg_logits:(N,3)
            seg_vote_preds=seg_out_dict['seg_vote_preds'],
            seg_feats=seg_out_dict['seg_feats'],                    # points_feats:(N,67)
            batch_idx=seg_out_dict['batch_idx'],
            vote_offsets=seg_out_dict['offsets'],
        )
        batch_idx = dict_to_sample['batch_idx']
        batch_size = len(img_metas)
        points = [points[batch_idx==i] for i in range(batch_size)]
        seg_scores = [seg_scores[batch_idx==i] for i in range(batch_size)]

        # 2D mask Loss
        # EMA seg_scores
        loss_mask, warmup_factor_img, warmup_factor_run = self.img_mask_head.loss_multi(img, img_metas, mask_pred, gt_inds, gt_bboxes,
                                            gt_masks, gt_labels, points, roi_points, seg_scores, img_inds)
        losses.update(loss_mask)

        # 2D to 3D loss
        # EMA mask_pred
        if self.use_2d_mask:
            loss_2d_to_3d = self.cal_2d_to_3d_loss(img, mask_pred, gt_labels, gt_inds, 
                                                   img_inds, img_metas, points, dict_to_sample, 
                                                   warmup_factor_img)
            losses.update({'loss_2d_to_3d': loss_2d_to_3d})

        # Run seg loss
        self.run_seg = False

        if self.run_seg:
            loss_3d_run_seg = self.cal_run_3d_loss(points, dict_to_sample, gt_bboxes, gt_labels, img_metas, img.device, warmup_factor_run)
            losses.update({'loss_3d_run_seg': loss_3d_run_seg})
        return losses

    def cal_dynamic_loss(self, img, points, gt_bboxes, gt_labels, img_metas, 
                         gt_inds, level_inds, img_inds, param_pred, seg_out_dict, 
                         mask_feat, warmup_factor):
        # 采样得到2D框内的前topk点
        dynamic_points = [[] for i in range(len(img_metas))]  # [points, flag]
        gt_inds_sets = torch.unique(gt_inds)
        gt_inds_sets = gt_inds_sets[gt_inds_sets!=-1]
        len_box = [len(gt_bbox) for gt_bbox in gt_bboxes]
        tmp_bboxes = torch.cat(gt_bboxes)
        tmp_labels = torch.cat(gt_labels)

        for i in range(len(gt_inds_sets)):
            gt_ind = gt_inds_sets[i]
            for b in range(len(img_metas)):
                if gt_inds_sets[i] < sum(len_box[:b+1]):
                    tmp_points = points[b]
                    break
            pos_mask = ((tmp_points[:,12]>=tmp_bboxes[gt_ind][0]) & (tmp_points[:,12]<tmp_bboxes[gt_ind][2]) &
                        (tmp_points[:,13]>=tmp_bboxes[gt_ind][1]) & (tmp_points[:,13]<tmp_bboxes[gt_ind][3]))
            # 扩大2D box，外面一层作为负样本点
            w = tmp_bboxes[gt_ind][2] - tmp_bboxes[gt_ind][0]
            h = tmp_bboxes[gt_ind][3] - tmp_bboxes[gt_ind][1]
            x1,y1 = tmp_bboxes[gt_ind][0]-w*0.05, tmp_bboxes[gt_ind][1]-h*0.05
            x2,y2 = tmp_bboxes[gt_ind][2]+w*0.05, tmp_bboxes[gt_ind][3]+h*0.05
            enlarge_mask = ((tmp_points[:,12]>=x1) & (tmp_points[:,12]<x2) &
                            (tmp_points[:,13]>=y1) & (tmp_points[:,13]<y2))
            neg_mask = enlarge_mask & (~pos_mask)
            sample_pos_nums = min(pos_mask.sum(), 80)
            sample_neg_nums = min(neg_mask.sum(), 20)
            sample_pos_indices = seg_out_dict['seg_logits'][(seg_out_dict['batch_idx']==b)][:, tmp_labels[gt_ind]][pos_mask].topk(sample_pos_nums)[1]
            
            sample_pos_points = tmp_points[pos_mask][sample_pos_indices]
            sample_neg_indices = seg_out_dict['seg_logits'][(seg_out_dict['batch_idx']==b)][neg_mask][:, tmp_labels[gt_ind]].topk(sample_neg_nums)[1]
            sample_neg_points = tmp_points[neg_mask][sample_neg_indices]
            sample_points_flag = torch.zeros(((sample_pos_nums+sample_neg_nums), 1), device=points[b].device)
            sample_points_flag[:sample_pos_nums] = 1
            sample_points_flag[sample_pos_nums:] = 0
            sample_points = torch.cat((sample_pos_points, sample_neg_points))
            dynamic_points[b].append(torch.cat((sample_points, sample_points_flag), dim=1))
        if False:
            import cv2
            from mmcv.image import tensor2imgs
            aimg = aligned_bilinear(img[0].unsqueeze(0), 2)  # 1280，1920，3
            aimg = tensor2imgs(aimg, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],to_rgb=False)[0]
            aimg = cv2.cvtColor(aimg, cv2.COLOR_RGB2BGR)
            dynamic_points_ = torch.cat(dynamic_points[0], dim=0)
            for i, pts in enumerate(dynamic_points_):
                cv2.circle(aimg, (int(pts[12]*2),int(pts[13]*2)),1,(255,0,0),-1)
            cv2.imwrite('aimg.jpeg', aimg)

        # 然后去2D分支里拿动态卷积参数，注意这个坐标是不是还原到原始大小的
        strides = self.img_bbox_head.strides
        dynamic_params = [[] for i in range(len(img_metas))]
        tmp_img_inds = []
        tmp_dynamic_points = []
        tmp_mean_coors = []
        for i in range(len(dynamic_points)):
            tmp_dynamic_points.extend(dynamic_points[i])
        for i in range(len(gt_inds_sets)):
            gt_ind = gt_inds_sets[i]
            for b in range(len(img_metas)):
                if gt_inds_sets[i] < sum(len_box[:b+1]):
                    break
            mean_coors = torch.mean(tmp_dynamic_points[i][:, 12:14][tmp_dynamic_points[i][:,-1]==1], dim=0)  # 这里的是*0.5的，这里不逆变换回去
            # 去哪一层（FPN层）采样？
            sets, counts = torch.unique(level_inds[gt_inds == gt_ind], return_inverse=False, return_counts=True)
            level_ind = sets[torch.argmax(counts)]      # 得到了去采样的层
            tmp_mean_coors.append(mean_coors)
            mean_coors = mean_coors/strides[level_ind]  # float
            dynamic_param = param_pred[level_ind][img_inds[gt_inds == gt_ind].min()][:, int(mean_coors[1]), int(mean_coors[0])]
            tmp_img_inds.append(img_inds[gt_inds == gt_ind].min())
            dynamic_params[b].append(dynamic_param)
        if False:
            import cv2
            from mmcv.image import tensor2imgs
            aimg = aligned_bilinear(img[0].unsqueeze(0), 2)  # 1280，1920，3
            aimg = tensor2imgs(aimg, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],to_rgb=False)[0]
            aimg = cv2.cvtColor(aimg, cv2.COLOR_RGB2BGR)
            # tmp_mean_coors = torch.cat(tmp_mean_coors)
            for j, gt_bbox in enumerate(gt_bboxes[0]):
                cv2.rectangle(aimg, (int(gt_bbox[0]*2), int(gt_bbox[1]*2)), (int(gt_bbox[2]*2), int(gt_bbox[3]*2)), (0, 255, 0), 1)
            tmp_mean_coors = torch.stack(tmp_mean_coors, dim=0)
            for j, p in enumerate(tmp_mean_coors):
                cv2.circle(aimg, (int(p[0]*2),int(p[1]*2)),4,(255,0,0),-1)
            cv2.imwrite('aimg.jpeg', aimg)
        # 动态卷积
        # sample_mask_feat = []
        counts_nums = 0
        loss_dynamic = torch.tensor(0., device=points[0].device)
        for i in range(len(img_metas)):
            out_list = []
            if len(dynamic_points[i]) == 0:
                continue
            tmp_coors = torch.cat(dynamic_points[i])[:,12:14]/img_metas[i]['scale_factor'][0]
            grid = torch.zeros_like(tmp_coors)
            grid[:,0] = (tmp_coors[:,0]/img_metas[i]['img_shape'][1]) - 1  # x
            grid[:,1] = (tmp_coors[:,1]/img_metas[i]['img_shape'][0]) - 1  # y
            out = F.grid_sample(mask_feat[i].unsqueeze(0), grid.unsqueeze(0).unsqueeze(0), align_corners=False)
            out = out.squeeze(0).squeeze(1).permute(1,0)  # (16,N)
            out = torch.cat((tmp_coors, out),dim=1)  # (18,N)
            # sample_mask_feat.append(out) # 18,N
            out_len = [len(pts) for pts in dynamic_points[i]]
            for j in range(len(dynamic_points[i])):
                H, W = 1, len(dynamic_points[i][j])
                weights, biases = self.img_mask_head.parse_dynamic_params(dynamic_params[i][j].reshape(1,-1))
                out_ = out[sum(out_len[:j]):sum(out_len[:j+1])].reshape(1, -1, H, W)
                for k, (w, b) in enumerate(zip(weights, biases)):
                    out_ = F.conv2d(out_, w, bias=b, stride=1, padding=0, groups=1)
                    if k < self.img_mask_head.dynamic_convs - 1:
                        out_ = F.relu(out_)
                out_ = out_.permute(1, 0, 2, 3).squeeze(0).squeeze(0).squeeze(0)
                out_ = out_.sigmoid()
                targets = dynamic_points[i][j][:,-1]
                # out_[targets==0] = 1 - out_[targets==0] 这样写反向传播会报错 因为存在inplace操作
                out_ = torch.clamp(out_, 1e-5, 1-1e-5)
                out_list.append(out_.clone().detach())
                if out_.shape[0] == 0:
                    continue
                counts_nums += 1
                loss_dynamic += -(out_[targets==1].log().sum()+(1-out_[targets==0]).log().sum())/out_.shape[0]
            if False:
                import cv2
                from mmcv.image import tensor2imgs
                aimg = aligned_bilinear(img[i].unsqueeze(0), 2)  # 1280，1920，3
                aimg = tensor2imgs(aimg, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],to_rgb=False)[0]
                aimg = cv2.cvtColor(aimg, cv2.COLOR_RGB2BGR)
                for j, gt_bbox in enumerate(gt_bboxes[i]):
                    cv2.rectangle(aimg, (int(gt_bbox[0]*2), int(gt_bbox[1]*2)), (int(gt_bbox[2]*2), int(gt_bbox[3]*2)), (0, 255, 0), 1)
                mean_coors = []
                for j in range(len(dynamic_points[i])):
                    mean_coors.append(torch.mean(dynamic_points[i][j][:, 12:14][dynamic_points[i][j][:,-1]==1]/img_metas[i]['scale_factor'][0], dim=0))
                mean_coors = torch.stack(mean_coors,dim=0)
                for j, p in enumerate(mean_coors):
                    cv2.circle(aimg, (int(p[0]),int(p[1])),4,(255,0,0),-1)
                out_list  # out_.sigmoid()
                for j in range(len(out_list)):
                    out_list[j][dynamic_points[i][j][:,-1]==0] = 1 - out_list[j][dynamic_points[i][j][:,-1]==0]
                    max_scores = (-1 * out_list[j].log()).max()
                    for p in range(len(out_list[j])):
                        pp = dynamic_points[i][j][p]
                        rgb = int(255*(-1*out_list[j][p].log())/max_scores)
                        cv2.circle(aimg, (int(pp[12]*2),int(pp[13]*2)), 2, (rgb,rgb,rgb), -1)
                # for j, pts in enumerate(torch.cat(dynamic_points[i], dim=0)):
                #     if pts[-1] == 0:
                #         rgb = int()
                #         cv2.circle(aimg, (int(pts[12]*2),int(pts[13]*2)),1,(255,255,255),-1)
                #     else:
                #         cv2.circle(aimg, (int(pts[12]*2),int(pts[13]*2)),1,(0,255,0),-1)
                cv2.imwrite('aimg.jpeg', aimg)
                pass
        loss_dynamic = warmup_factor * 0.1 * loss_dynamic / counts_nums
        return loss_dynamic

    def cal_2d_to_3d_loss(self, img, mask_pred, gt_labels, gt_inds, img_inds, img_metas, points, dict_to_sample, warmup_factor_img):
        mask_pred_detach = mask_pred.clone().detach()
        # 将mask改成batch形式
        gt_labels_ = torch.cat(gt_labels, dim=0)
        assert mask_pred.size(0) == gt_inds.size(0) == img_inds.size(0)
        # 将mask改成batch形式
        mask_preds = []
        mask_pred_detach = mask_pred_detach.sigmoid()
        # mask_pred_detach = aligned_bilinear(mask_pred_detach, 4)  # N,1,160,240 --> N,1,640,960
        mask_pred_detach = mask_pred_detach.squeeze()
        # mask_pred = mask_pred > 0.5
        # mask_pred = mask_pred.type(torch.int)
        for i in range(img.shape[0]):  # batch_size
            # 获得对应图片的mask_pred和label
            mask_pred_per_img = mask_pred_detach[img_inds==i]
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
            mask_preds.append(dict(mask_preds=mask_preds_cls,gt_inds=gt_inds_cls))

        device = points[0].device

        loss_2d_to_3d = torch.tensor(0.,device=device)
        counts_nums = 0
        mask_points = dict_to_sample['seg_points'] # points:(N, 11+rgb)
        mask_in = mask_points[:, 10] == 1
        mask_points = mask_points[mask_in]
        mask_prob = dict_to_sample['seg_logits'].sigmoid()  # (N, 3)
        mask_prob = torch.clamp(mask_prob, 1e-5, 1-1e-5)
        mask_prob = mask_prob[mask_in]
        batch_idx = dict_to_sample['batch_idx']
        batch_idx = batch_idx[mask_in]
        # mask_preds: list[dict(mask_preds[[car],[ped],[cyc]], gt_inds[[car],[ped],[cyc]])]
        for i, mask_pred_ in enumerate(mask_preds):  # B,640,960
            mask_pred = mask_pred_['mask_preds']
            per_batch_points = mask_points[batch_idx == i]
            for j in range(self.num_classes):
                mask_pred_cls = mask_pred[j]
                if mask_pred_cls.size(0) != 0:
                    mask_pred_cls = mask_pred_cls.max(0)[0]
                    # per_batch_points = mask_points[batch_idx == i]
                    grid = per_batch_points[:, 12:14]/img_metas[i]['scale_factor'][0]
                    grid[:,0] = (grid[:,0]/img_metas[i]['img_shape'][1]) - 1  # x
                    grid[:,1] = (grid[:,1]/img_metas[i]['img_shape'][0]) - 1  # y
                    # input(1,1,H,W) grid(1,1,N_pts,2) out(1,1,1,N_pts)
                    out = F.grid_sample(mask_pred_cls.unsqueeze(0).unsqueeze(0), grid.unsqueeze(0).unsqueeze(0), align_corners=False)
                    out = out.squeeze(0).squeeze(0).squeeze(0)
                    # -plog(q)
                    counts_nums += out.size()[0]
                    loss_2d_to_3d += -((out * mask_prob[batch_idx == i][:,j].log()).sum() + \
                        ((1-out) * (1-mask_prob[batch_idx == i][:,j]).log()).sum())
        if counts_nums != 0:
            loss_2d_to_3d = loss_2d_to_3d * warmup_factor_img * 0.2 / counts_nums  # 第一版调通参数是0.05
        return loss_2d_to_3d

    def cal_run_3d_loss(self, points, dict_to_sample, gt_bboxes, gt_labels, img_metas, device, warmup_factor_run):
        batch_size = len(img_metas)
        assert isinstance(points, list)
        loss_3d_run_seg = torch.tensor(0., device=device)
        mask_scores = dict_to_sample['seg_logits'].sigmoid() # N,3
        mask_scores = torch.clamp(mask_scores, 1e-5, 1-1e-5)
        batch_idx = dict_to_sample['batch_idx'] # N,1
        counts_nums = 0
        for i in range(batch_size):
            batch_points_run_id = []
            batch_scores = []
            # 1. 得到2D box内的点云
            for j, gt_bbox in enumerate(gt_bboxes[i]):
                # 扩大2D box
                w = gt_bbox[2] - gt_bbox[0]
                h = gt_bbox[3] - gt_bbox[1]
                x1, y1 = gt_bbox[0] - w * 0.05, gt_bbox[1] - h * 0.05
                x2, y2 = gt_bbox[2] + w * 0.05, gt_bbox[3] + h * 0.05
                # 选择主雷达的点云
                in_box_mask = ((points[i][:,12]>=x1) & (points[i][:,12]<x2) &
                                (points[i][:,13]>=y1) & (points[i][:,13]<y2) &
                                (points[i][:,6]==0))
                # 得到2D box内点云的mask scores 和 run seg id
                batch_points_run_id.append(points[i][in_box_mask][:,14])
                batch_scores.append(mask_scores[batch_idx==i][in_box_mask][:, gt_labels[i][j]]) # 这里只获取当前label的score
            batch_points_run_id = torch.cat(batch_points_run_id, dim=0)
            batch_scores = torch.cat(batch_scores, dim=0)
            # cal loss
            run_sets, inv_inds, counts = torch.unique(batch_points_run_id, return_inverse=True, return_counts=True)

            out = torch.zeros((run_sets.shape), device=batch_points_run_id.device)
            out.scatter_add_(0, inv_inds, batch_scores)
            ex = out/counts
            out2 = torch.zeros((run_sets.shape), device=batch_points_run_id.device)
            out2.scatter_add_(0, inv_inds, batch_scores**2)
            ex2 = out2/counts
            var = ex2 - ex**2

            entropy = -(batch_scores*batch_scores.log()+(1-batch_scores)*(1-batch_scores).log())
            out3 = torch.zeros((run_sets.shape), device=batch_points_run_id.device)
            out3.scatter_add_(0, inv_inds, entropy)
            out3 = out3/counts

            filter_mask = run_sets != -1
            # 过滤掉分段id为-1的无关点(非主雷达点)
            if (counts[filter_mask]>3).sum() == 0:
                continue
            else:
                counts_nums += (counts[filter_mask]>3).sum()
                loss_3d_run_seg += ((var[filter_mask]+out3[filter_mask]) * (counts[filter_mask]>3)).sum()
        if counts_nums != 0:
            loss_3d_run_seg = loss_3d_run_seg * warmup_factor_run * 0.1 / counts_nums
        return loss_3d_run_seg

    # Base3DDetector的forward_test()内进入simple_test函数
    def simple_test(self, points, img_metas, img=None, rescale=False,
                    gt_bboxes_3d=None, gt_labels_3d=None, pts_semantic_mask=None, gt_bboxes=None, gt_labels=None):
        """Test function without augmentaiton."""

        results_list = [dict() for i in range(len(img_metas))]
        pts_results = self.simple_test_pts_segmet(points,img_metas)
        for i in range(len(img_metas)):
            results_list[i]['segment3d'] = pts_results[i]['segment_results']
            results_list[i]['offsets'] = pts_results[i]['offsets']
            results_list[i]['seg_logits'] = pts_results[i]['seg_logits']

        # if self.with_pts_branch:
        #     # list[i]={'boxes_3d':LiDARInstance3DBoxes,'scores_3d':(N3d,1),'labels_3d':(N3d,1)}
        #     pts_results = self.simple_test_pts(
        #         points, img_metas, rescale, gt_bboxes_3d, gt_labels_3d, pts_semantic_mask)
        #     for i in range(len(img_metas)):
        #         results_list[i]['boxes_3d']=pts_results[i]['boxes_3d']    # (LiDARInstance3DBoxes)
        #         results_list[i]['scores_3d']=pts_results[i]['scores_3d']  # (N3d,1)
        #         results_list[i]['labels_3d']=pts_results[i]['labels_3d']  # (N3d,1)
        
        # finished
        # 这里需要注意的是输出是不是按照batch来的，我下面的赋值都是按照batch来索引赋值的
        if self.with_img_bbox and self.with_img_branch:
            img_feats = self.extract_img_feat(img=img, img_metas=img_metas)
            bbox_results, mask_results = self.simple_test_img(img_feats, img_metas, rescale=rescale) # bbox_results, mask_results
            for i in range(len(img_metas)):
                results_list[i]['img_bbox'] = bbox_results[i]   # 长度是类别数 x (N,5),左上角和右下角坐标+得分 (N,5)
                results_list[i]['img_mask'] = mask_results[i]   # 长度是类别数 x (1280,1920) eg(N, 1280, 1920)
        # 3D 实例分割
        for i in range(len(img_metas)):
            
            box_pred_2d = results_list[i]['img_bbox']
            box_pred_2d = torch.tensor(np.concatenate(box_pred_2d))
            instance_nums = len(box_pred_2d)

            points_shape = results_list[i]['segment3d'].shape
            points_inds = torch.ones((points_shape[0]), dtype=torch.int)*-1
            # 得到所有的投票点
            dist = [0.6, 0.1, 0.4]  # car ped cycl
            tmp_inds = [0]
            for j in range(self.num_classes):
                xyz_mask = results_list[i]['segment3d'] == j
                if xyz_mask.sum() == 0:
                    continue
                xyz = points[i][:,0:3][xyz_mask].cpu()
                offsets = results_list[i]['offsets'][xyz_mask]
                center_pred = xyz + offsets
                c_inds = find_connected_componets_single_batch(center_pred, None, dist[j])
                tmp_inds.append(c_inds.max()+1)
                points_inds[xyz_mask] = c_inds + tmp_inds[j]
            results_list[i]['3d_instance_id'] = points_inds
        return results_list

    # 检查完成 Return：list(zip(bbox_results, mask_results))
    def simple_test_img(self, img_feats, img_metas, proposals=None, rescale=False):
        feat = img_feats
        outputs = self.img_bbox_head.simple_test(
            feat, self.img_mask_head.param_conv, img_metas, rescale=rescale)  # 注意这个outputs的输出是不是按照batch来的
        det_bboxes, det_labels, det_params, det_coors, det_level_inds = zip(*outputs)  # 框的两个角位置坐标+得分，所属类别，滤波器参数(N,233)，box在当前level的位置(N,2)，所属的layer索引
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
            segment_results = torch.ones(seg_logits_idx.shape[0], device=seg_logits.device) * 3
            segment_offsets = torch.zeros((seg_offsets_idx.shape[0],3), device=seg_logits.device, dtype=torch.float32)
            assert len(seg_logits_idx) == len(seg_offsets_idx)
            # segment_mask = segment_results == 3
            max_inds = seg_logits_idx.max(1)[1]
            for j in range(3):
                mask = (max_inds==j) & (seg_logits_idx[:,j]>threshold[j])
                segment_results[mask] = j
                segment_offsets[mask] = seg_offsets[batch_idx==i][mask][:,j*3:j*3+3]
            out.append(dict(segment_results=segment_results.cpu(), offsets=segment_offsets.cpu(), seg_logits=seg_logits.cpu()))
        return out

    def simple_test_pts(self, points, img_metas, rescale, gt_bboxes_3d, gt_labels_3d, pts_semantic_mask):
        """Test function of point cloud branch."""
        out = []
        # first stage fsd output
        rpn_outs = self.simple_test_single_fsd(points=points,
                                               img_metas=img_metas,
                                               gt_bboxes_3d=gt_bboxes_3d,
                                               gt_labels_3d=gt_labels_3d,)  # proposal_list, seg_logits_full
        
        proposal_list = rpn_outs['proposal_list']  # LidarInstance3DBoxes, box_scores, box_label
        if self.test_cfg.pts.return_mode in [0, 2]:   # 0: both, 1: detection, 2: segmentation
            seg_logits_full = rpn_outs.get('seg_logits_full')
            assert isinstance(seg_logits_full, list)
            with_confusion_matrix = self.test_cfg.pts.get('with_confusion_matrix', False)
            for b_seg in seg_logits_full:
                if with_confusion_matrix and (pts_semantic_mask is not None):
                    assert len(pts_semantic_mask) == 1
                    if self.sweeps_num > 1 and self.only_one_frame_label:
                        pts_semantic_mask[0] = pts_semantic_mask[0][points[:, -1]==0]
                    b_pred = b_seg.argmax(1)
                    confusion_matrix = pts_semantic_confusion_matrix(
                        b_pred + 1,
                        pts_semantic_mask[0],
                        self.test_cfg.pts.get('num_seg_cls') + 1)    # add cls: unlabel
                    out.append(dict(seg3d_confusion_matrix=confusion_matrix))
                else:
                    out.append(dict(segmap_3d=F.softmax(b_seg, dim=1).argmax(1).cpu()))

        if self.test_cfg.pts.return_mode == 2:  # 只返回semantic segment结果
            return out

        if self.test_cfg.pts.get('skip_rcnn', False):
            bbox_results = [
                bbox3d2result(bboxes, scores, labels)
                for bboxes, scores, labels in proposal_list
            ]
            return bbox_results  # type(bbox_results)=list len(list)=batch_size list[i]={'boxes_3d':LiDARInstance3DBoxes,'scores_3d':(N3d,1),'labels_3d':(N3d,1)}

        if self.num_classes > 1 or self.test_cfg.pts.get('enable_multi_class_test', False):
            prepare_func = self.prepare_multi_class_roi_input
        else:
            prepare_func = self.prepare_roi_input

        pts_xyz, pts_feats, pts_batch_inds = prepare_func(
            rpn_outs['all_input_points'],
            rpn_outs['valid_pts_feats'],
            rpn_outs['seg_feats'],
            rpn_outs['pts_mask'],
            rpn_outs['pts_batch_inds'],
            rpn_outs['valid_pts_xyz']
        )

        results = self.pts_roi_head.simple_test(
            pts_xyz,
            pts_feats,
            pts_batch_inds,
            img_metas,
            proposal_list,
            gt_bboxes_3d,
            gt_labels_3d,
        )
        if self.test_cfg.pts.return_mode == 1: # 只返回detection结果
            return results
        assert len(out) == len(results)
        for idx in range(len(out)):
            out[idx].update(results[idx])
        return out
 
    def simple_test_single_fsd(self, points, img_metas, imgs=None, rescale=False, gt_bboxes_3d=None, gt_labels_3d=None):
        """Test function without augmentaiton."""
        if gt_bboxes_3d is not None:
            gt_bboxes_3d = gt_bboxes_3d[0]
            gt_labels_3d = gt_labels_3d[0]
            assert isinstance(gt_bboxes_3d, list)
            assert isinstance(gt_labels_3d, list)
            assert len(gt_bboxes_3d) == len(gt_labels_3d) == 1, 'assuming single sample testing'

        seg_out_dict = self.pts_segmentor.simple_test(points, img_metas, rescale=False)

        seg_feats = seg_out_dict['seg_feats']
        seg_logits_full = seg_out_dict.get('seg_logits_full', None)
        assert seg_out_dict['batch_idx'].max() == 0     # for inference, batch size = 0.

        dict_to_sample = dict(
            seg_points=seg_out_dict['seg_points'],
            seg_logits=seg_out_dict['seg_logits'],
            seg_vote_preds=seg_out_dict['seg_vote_preds'],
            seg_feats=seg_feats,
            batch_idx=seg_out_dict['batch_idx'],
            vote_offsets = seg_out_dict['offsets']
        )
        if self.pts_cfg.get('pre_voxelization_size', None) is not None:
            dict_to_sample = self.pre_voxelize(dict_to_sample)
        sampled_out = self.sample(dict_to_sample, dict_to_sample['vote_offsets'], gt_bboxes_3d, gt_labels_3d) # per cls list in sampled_out 返回每个点加上前面预测的offsets,得到的投票点

        # we filter almost empty voxel in clustering, so here is a valid_mask 比较慢，计算量大
        pts_cluster_inds, valid_mask_list = self.cluster_assigner(sampled_out['center_preds'], sampled_out['batch_idx'], gt_bboxes_3d, gt_labels_3d, origin_points=sampled_out['seg_points']) # per cls list

        if isinstance(pts_cluster_inds, list):
            pts_cluster_inds = torch.cat(pts_cluster_inds, dim=0) #[N, 3], (cls_id, batch_idx, cluster_id)

        sampled_out = self.update_sample_results_by_mask(sampled_out, valid_mask_list)

        combined_out = self.combine_classes(sampled_out, ['seg_points', 'seg_logits', 'seg_vote_preds', 'seg_feats', 'center_preds'])

        points = combined_out['seg_points']
        pts_feats = torch.cat([combined_out['seg_logits'], combined_out['seg_vote_preds'], combined_out['seg_feats']], dim=1)
        assert len(pts_cluster_inds) == len(points) == len(pts_feats)
        # 这个函数里面涉及到
        extracted_outs = self.extract_pts_feat(points, pts_feats, pts_cluster_inds, img_metas,  combined_out['center_preds'])
        cluster_feats = extracted_outs['cluster_feats']  # N_instance vector
        cluster_xyz = extracted_outs['cluster_xyz']
        cluster_inds = extracted_outs['cluster_inds']
        assert (cluster_inds[:, 1] == 0).all()

        outs = self.pts_bbox_head(cluster_feats, cluster_xyz, cluster_inds)  # cls_logits, reg_preds 一个类别一个tensor(N,1),(N,8)
        # 这里用到一些参数nms_pre等 bbox_list: LidarInstance3DBoxes box_scores box_label
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs['cls_logits'], outs['reg_preds'],
            cluster_xyz, cluster_inds, img_metas,
            rescale=rescale,
            iou_logits=outs.get('iou_logits', None))

        if self.as_rpn:
            output_dict = dict(
                all_input_points=dict_to_sample['seg_points'],
                valid_pts_feats=extracted_outs['cluster_pts_feats'],
                valid_pts_xyz=extracted_outs['cluster_pts_xyz'],
                seg_feats=dict_to_sample['seg_feats'],
                pts_mask=sampled_out['fg_mask_list'],
                pts_batch_inds=dict_to_sample['batch_idx'],
                proposal_list=bbox_list,
                seg_logits_full=[seg_logits_full]
            )
            return output_dict
        else:
            # bbox_results = [
            #     bbox3d2result(bboxes, scores, labels)
            #     for bboxes, scores, labels in bbox_list
            # ]
            # return bbox_results
            output_dict = dict(
                proposal_list=bbox_list,
                seg_logits_full=[seg_logits_full])
            return output_dict

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

    # GenPoints Points Completion
    def encoder_pts(self, points, img, img_metas, gt_bboxes, gt_labels):
        pts_dict = self.middle_encoder_pts(points, img, img_metas, gt_bboxes, gt_labels)
        return pts_dict

    # FSD_two_stage
    def prepare_roi_input(self, points, cluster_pts_feats, pts_seg_feats, pts_mask, pts_batch_inds, cluster_pts_xyz):
        assert isinstance(pts_mask, list)
        pts_mask = pts_mask[0]
        assert points.shape[0] == pts_seg_feats.shape[0] == pts_mask.shape[0] == pts_batch_inds.shape[0]

        if self.training and self.train_cfg.get('detach_seg_feats', False):
            pts_seg_feats = pts_seg_feats.detach()

        if self.training and self.train_cfg.get('detach_cluster_feats', False):
            cluster_pts_feats = cluster_pts_feats.detach()
        
        pad_feats = cluster_pts_feats.new_zeros(points.shape[0], cluster_pts_feats.shape[1])
        pad_feats[pts_mask] = cluster_pts_feats
        assert torch.isclose(points[pts_mask], cluster_pts_xyz).all()

        cat_feats = torch.cat([pad_feats, pts_seg_feats], dim=1)

        return points, cat_feats, pts_batch_inds

    def prepare_multi_class_roi_input(self, points, cluster_pts_feats, pts_seg_feats, pts_mask, pts_batch_inds, cluster_pts_xyz):
        assert isinstance(pts_mask, list)
        bg_mask = sum(pts_mask) == 0
        assert points.shape[0] == pts_seg_feats.shape[0] == bg_mask.shape[0] == pts_batch_inds.shape[0]

        if self.training and self.train_cfg.get('detach_seg_feats', False):
            pts_seg_feats = pts_seg_feats.detach()

        if self.training and self.train_cfg.get('detach_cluster_feats', False):
            cluster_pts_feats = cluster_pts_feats.detach()


        ##### prepare points for roi head
        fg_points_list = [points[m] for m in pts_mask]
        all_fg_points = torch.cat(fg_points_list, dim=0)

        assert torch.isclose(all_fg_points, cluster_pts_xyz).all()

        bg_pts_xyz = points[bg_mask]
        all_points = torch.cat([bg_pts_xyz, all_fg_points], dim=0)
        #####

        ##### prepare features for roi head
        fg_seg_feats_list = [pts_seg_feats[m] for m in pts_mask]
        all_fg_seg_feats = torch.cat(fg_seg_feats_list, dim=0)
        bg_seg_feats = pts_seg_feats[bg_mask]
        all_seg_feats = torch.cat([bg_seg_feats, all_fg_seg_feats], dim=0)

        num_out_points = len(all_points)
        assert num_out_points == len(all_seg_feats)

        pad_feats = cluster_pts_feats.new_zeros(bg_mask.sum(), cluster_pts_feats.shape[1])
        all_cluster_pts_feats = torch.cat([pad_feats, cluster_pts_feats], dim=0)
        #####

        ##### prepare batch inds for roi head
        bg_batch_inds = pts_batch_inds[bg_mask]
        fg_batch_inds_list = [pts_batch_inds[m] for m in pts_mask]
        fg_batch_inds = torch.cat(fg_batch_inds_list, dim=0)
        all_batch_inds = torch.cat([bg_batch_inds, fg_batch_inds], dim=0)


        # pad_feats[pts_mask] = cluster_pts_feats

        cat_feats = torch.cat([all_cluster_pts_feats, all_seg_feats], dim=1)

        # sort for roi extractor
        all_batch_inds, inds = all_batch_inds.sort()
        all_points = all_points[inds]
        cat_feats = cat_feats[inds]

        return all_points, cat_feats, all_batch_inds
    
    # 这个是fsd第二阶段的simple
    def simple_test_fsd_two_stage(self, points, img_metas, imgs=None, rescale=False,
                    gt_bboxes_3d=None, gt_labels_3d=None,
                    pts_semantic_mask=None):

        out = []

        rpn_outs = super().simple_test(
            points=points,
            img_metas=img_metas,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
        )

        proposal_list = rpn_outs['proposal_list']
        if self.test_cfg.return_mode in [0, 2]:   # 0: both, 1: detection, 2: segmentation
            seg_logits_full = rpn_outs.get('seg_logits_full')
            assert isinstance(seg_logits_full, list)
            with_confusion_matrix = self.test_cfg.get('with_confusion_matrix', False)
            for b_seg in seg_logits_full:
                if with_confusion_matrix and (pts_semantic_mask is not None):
                    assert len(pts_semantic_mask) == 1
                    if self.sweeps_num > 1 and self.only_one_frame_label:
                        pts_semantic_mask[0] = pts_semantic_mask[0][points[:, -1]==0]
                    b_pred = b_seg.argmax(1)
                    confusion_matrix = pts_semantic_confusion_matrix(
                        b_pred + 1,
                        pts_semantic_mask[0],
                        self.test_cfg.get('num_seg_cls') + 1)    # add cls: unlabel
                    out.append(dict(seg3d_confusion_matrix=confusion_matrix))
                else:
                    out.append(dict(segmap_3d=F.softmax(b_seg, dim=1).argmax(1).cpu()))
        if self.test_cfg.return_mode == 2:
            return out

        if self.test_cfg.get('skip_rcnn', False):
            bbox_results = [
                bbox3d2result(bboxes, scores, labels)
                for bboxes, scores, labels in proposal_list
            ]
            return bbox_results

        if self.num_classes > 1 or self.test_cfg.get('enable_multi_class_test', False):
            prepare_func = self.prepare_multi_class_roi_input
        else:
            prepare_func = self.prepare_roi_input

        pts_xyz, pts_feats, pts_batch_inds = prepare_func(
            rpn_outs['all_input_points'],
            rpn_outs['valid_pts_feats'],
            rpn_outs['seg_feats'],
            rpn_outs['pts_mask'],
            rpn_outs['pts_batch_inds'],
            rpn_outs['valid_pts_xyz']
        )


        results = self.roi_head.simple_test(
            pts_xyz,
            pts_feats,
            pts_batch_inds,
            img_metas,
            proposal_list,
            gt_bboxes_3d,
            gt_labels_3d,
        )
        if self.test_cfg.return_mode == 1:
            return results
        assert len(out) == len(results)
        for idx in range(len(out)):
            out[idx].update(results[idx])
        return out
    
    def extract_fg_by_gt(self, point_list, gt_bboxes_3d, gt_labels_3d, extra_width):
        if isinstance(gt_bboxes_3d[0], list):
            assert len(gt_bboxes_3d) == 1
            assert len(gt_labels_3d) == 1
            gt_bboxes_3d = gt_bboxes_3d[0]
            gt_labels_3d = gt_labels_3d[0]

        bsz = len(point_list)

        new_point_list = []
        for i in range(bsz):
            points = point_list[i]
            gts = gt_bboxes_3d[i].to(points.device)
            if len(gts) == 0:
                this_fg_mask = points.new_zeros(len(points), dtype=torch.bool)
                this_fg_mask[:min(1000, len(points))] = True
            else:
                if isinstance(extra_width, dict):
                    this_labels = gt_labels_3d[i]
                    enlarged_gts_list = []
                    for cls in range(self.num_classes):
                        cls_mask = this_labels == cls
                        if cls_mask.any():
                            this_enlarged_gts = gts[cls_mask].enlarged_box(extra_width[cls])
                            enlarged_gts_list.append(this_enlarged_gts)
                    enlarged_gts = gts.cat(enlarged_gts_list)
                else:
                    enlarged_gts = gts.enlarged_box(extra_width)
                pts_inds = enlarged_gts.points_in_boxes(points[:, :3])
                this_fg_mask = pts_inds > -1
                if not this_fg_mask.any():
                    this_fg_mask[:min(1000, len(points))] = True
            
            new_point_list.append(points[this_fg_mask])
        return new_point_list

    # SingleStageFSD
    def pre_voxelize(self, data_dict):
        batch_idx = data_dict['batch_idx']
        points = data_dict['seg_points']  # (N,3 or 12)

        voxel_size = torch.tensor(self.pts_cfg.pre_voxelization_size, device=batch_idx.device)
        pc_range = torch.tensor(self.cluster_assigner.point_cloud_range, device=points.device)
        coors = torch.div(points[:, :3] - pc_range[None, :3], voxel_size[None, :], rounding_mode='floor').long()  # 点体素化后对应的坐标(N,3)
        coors = coors[:, [2, 1, 0]] # to zyx order
        coors = torch.cat([batch_idx[:, None], coors], dim=1)
        # unq_inv表示points对应的体素索引
        new_coors, unq_inv  = torch.unique(coors, return_inverse=True, return_counts=False, dim=0)
        # 这里体素化，从这里可以获得原始点云对应的voxel
        voxelized_data_dict = {}
        for data_name in data_dict:
            data = data_dict[data_name]
            if data.dtype in (torch.float, torch.float16):
                voxelized_data, voxel_coors = scatter_v2(data, coors, mode='avg', return_inv=False, new_coors=new_coors, unq_inv=unq_inv)
                voxelized_data_dict[data_name] = voxelized_data  # (Nv,
        # voxel2points_inv这个参数是为了获得每个points对应的voxel，new_coors[unq_inv]即可 (new_coors[unq_inv]==coors).all() = True
        voxelized_data_dict['points2voxel_inv'] = unq_inv   # (N,1)
        voxelized_data_dict['voxel_coors'] = new_coors      # (Nv,4) (batch_idx,z,y,x) 这里只是体素坐标的索引，但实际体素的坐标点还是依据xyz来的
        voxelized_data_dict['points_voxel_coors'] = coors   # 这个就是直接保存了每个点对应的voxel坐标 (new_coors[unq_inv]==coors).all() = True
        voxelized_data_dict['batch_idx'] = voxel_coors[:, 0]
        return voxelized_data_dict

    def sample(self, dict_to_sample, offset, gt_bboxes_3d=None, gt_labels_3d=None):

        if self.pts_cfg.get('group_sample', False):
            return self.group_sample(dict_to_sample, offset)

        cfg = self.train_cfg.pts if self.training else self.test_cfg.pts

        seg_logits = dict_to_sample['seg_logits']
        # assert (seg_logits < 0).any() # make sure no sigmoid applied

        if seg_logits.size(1) == self.num_classes:
            seg_scores = seg_logits.sigmoid()
        else:
            raise NotImplementedError

        offset = offset.reshape(-1, self.num_classes, 3)
        seg_points = dict_to_sample['seg_points'][:, :3]  # (Nv, 3)
        fg_mask_list = [] # fg_mask of each cls list(bool): [(Nv,),(Nv,),(Nv,)]
        center_preds_list = [] # fg_mask of each cls

        batch_idx = dict_to_sample['batch_idx']
        batch_size = batch_idx.max().item() + 1
        for cls in range(self.num_classes):
            cls_score_thr = cfg['score_thresh'][cls]
            # 在 test 阶段返回的是 得分＞阈值的点 fg_mask(bool): (Nv,)
            fg_mask = self.get_fg_mask(seg_scores, seg_points, cls, batch_idx, gt_bboxes_3d, gt_labels_3d)

            if len(torch.unique(batch_idx[fg_mask])) < batch_size:
                one_random_pos_per_sample = self.get_sample_beg_position(batch_idx, fg_mask)
                fg_mask[one_random_pos_per_sample] = True # at least one point per sample

            fg_mask_list.append(fg_mask)

            this_offset = offset[fg_mask, cls, :]
            this_points = seg_points[fg_mask, :]
            this_centers = this_points + this_offset
            center_preds_list.append(this_centers)

        # fg_mask[dict_to_sample['voxel2points_inv]]
        output_dict = {}
        for data_name in dict_to_sample:
            data = dict_to_sample[data_name]
            cls_data_list = []
            if len(fg_mask) == len(data):
                for fg_mask in fg_mask_list:
                    # fg_mask(bool):(Nv,)
                    cls_data_list.append(data[fg_mask])
                output_dict[data_name] = cls_data_list
        output_dict['fg_mask_list'] = fg_mask_list  # list(bool) [(Nv,1),(Nv,1),(Nv,1)]
        output_dict['center_preds'] = center_preds_list  # list [(fg_mask=True,3),(fg_mask=True,3),(fg_mask=True,3)]

        return output_dict

    def update_sample_results_by_mask(self, sampled_out, valid_mask_list):
        for k in sampled_out:
            old_data = sampled_out[k]
            if len(old_data[0]) == len(valid_mask_list[0]) or 'fg_mask' in k:
                if 'fg_mask' in k:
                    new_data_list = []
                    for data, mask in zip(old_data, valid_mask_list):
                        new_data = data.clone()
                        new_data[data] = mask
                        assert new_data.sum() == mask.sum()
                        new_data_list.append(new_data)
                    sampled_out[k] = new_data_list
                else:
                    new_data_list = [data[mask] for data, mask in zip(old_data, valid_mask_list)]
                    sampled_out[k] = new_data_list
        return sampled_out

    def combine_classes(self, data_dict, name_list):
        out_dict = {}
        for name in data_dict:
            if name in name_list:
                out_dict[name] = torch.cat(data_dict[name], 0)
        return out_dict

    def get_fg_mask(self, seg_scores, seg_points, cls_id, batch_inds, gt_bboxes_3d, gt_labels_3d):
        if self.training and self.train_cfg.pts.get('disable_pretrain', False) and not self.runtime_info.get('enable_detection', False):
            seg_scores = seg_scores[:, cls_id]
            topks = self.train_cfg.pts.get('disable_pretrain_topks', [100, 100, 100])
            k = min(topks[cls_id], len(seg_scores))
            top_inds = torch.topk(seg_scores, k)[1]
            fg_mask = torch.zeros_like(seg_scores, dtype=torch.bool)
            fg_mask[top_inds] = True
        else:
            seg_scores = seg_scores[:, cls_id]
            cls_score_thr = self.pts_cfg['score_thresh'][cls_id]
            if self.training:
                buffer_thr = self.runtime_info.get('threshold_buffer', 0)
            else:
                buffer_thr = 0
            fg_mask = seg_scores > cls_score_thr + buffer_thr

        # add fg points
        cfg = self.train_cfg.pts if self.training else self.test_cfg.pts

        # 没修改
        if cfg.get('add_gt_fg_points', False):
            import pdb;pdb.set_trace()
            bsz = len(gt_bboxes_3d)
            assert len(seg_scores) == len(seg_points) == len(batch_inds)
            point_list = self.split_by_batch(seg_points, batch_inds, bsz)
            gt_fg_mask_list = []

            for i, points in enumerate(point_list):
                
                gt_mask = gt_labels_3d[i] == cls_id
                gts = gt_bboxes_3d[i][gt_mask]

                if not gt_mask.any() or len(points) == 0:
                    gt_fg_mask_list.append(gt_mask.new_zeros(len(points), dtype=torch.bool))
                    continue
                
                gt_fg_mask_list.append(gts.points_in_boxes(points) > -1)
            
            gt_fg_mask = self.combine_by_batch(gt_fg_mask_list, batch_inds, bsz)
            fg_mask = fg_mask | gt_fg_mask
            
        return fg_mask

    def split_by_batch(self, data, batch_idx, batch_size):
        assert batch_idx.max().item() + 1 <= batch_size
        data_list = []
        for i in range(batch_size):
            sample_mask = batch_idx == i
            data_list.append(data[sample_mask])
        return data_list

    def combine_by_batch(self, data_list, batch_idx, batch_size):
        assert len(data_list) == batch_size
        if data_list[0] is None:
            return None
        data_shape = (len(batch_idx),) + data_list[0].shape[1:]
        full_data = data_list[0].new_zeros(data_shape)
        for i, data in enumerate(data_list):
            sample_mask = batch_idx == i
            full_data[sample_mask] = data
        return full_data

    def get_sample_beg_position(self, batch_idx, fg_mask):
        assert batch_idx.shape == fg_mask.shape
        inner_inds = get_inner_win_inds(batch_idx.contiguous())
        pos = torch.where(inner_inds == 0)[0]
        return pos

    def group_sample(self, dict_to_sample, offset):

        """
        For argoverse 2 dataset, where the number of classes is large
        """

        bsz = dict_to_sample['batch_idx'].max().item() + 1
        assert bsz == 1, "Maybe some codes need to be modified if bsz > 1"
        # combine all classes as fg class.
        cfg = self.train_cfg if self.training else self.test_cfg

        seg_logits = dict_to_sample['seg_logits']
        assert (seg_logits < 0).any() # make sure no sigmoid applied

        assert seg_logits.size(1) == self.num_classes + 1 # we have background class
        seg_scores = seg_logits.softmax(1)

        offset = offset.reshape(-1, self.num_classes + 1, 3)
        seg_points = dict_to_sample['seg_points'][:, :3]
        fg_mask_list = [] # fg_mask of each cls
        center_preds_list = [] # fg_mask of each cls


        cls_score_thrs = cfg['score_thresh']
        group_lens = cfg['group_lens']
        num_groups = len(group_lens)
        assert num_groups == len(cls_score_thrs)
        assert isinstance(cls_score_thrs, (list, tuple))
        grouped_score = self.gather_group(seg_scores[:, :-1], group_lens) # without background score

        beg = 0
        for i, group_len in enumerate(group_lens):
            end = beg + group_len

            fg_mask = grouped_score[:, i] > cls_score_thrs[i]

            if not fg_mask.any():
                fg_mask[0] = True # at least one point

            fg_mask_list.append(fg_mask)

            this_offset = offset[fg_mask, beg:end, :] 
            offset_weight = self.get_offset_weight(seg_logits[fg_mask, beg:end])
            assert torch.isclose(offset_weight.sum(1), offset_weight.new_ones(len(offset_weight))).all()
            this_offset = (this_offset * offset_weight[:, :, None]).sum(dim=1)
            this_points = seg_points[fg_mask, :]
            this_centers = this_points + this_offset
            center_preds_list.append(this_centers)
            beg = end
        assert end == 26, 'for 26class argo'


        output_dict = {}
        for data_name in dict_to_sample:
            data = dict_to_sample[data_name]
            cls_data_list = []
            for fg_mask in fg_mask_list:
                cls_data_list.append(data[fg_mask])

            output_dict[data_name] = cls_data_list
        output_dict['fg_mask_list'] = fg_mask_list
        output_dict['center_preds'] = center_preds_list

        return output_dict
    
    def get_offset_weight(self, seg_logit):
        mode = self.cfg['offset_weight']
        if mode == 'max':
            weight = ((seg_logit - seg_logit.max(1)[0][:, None]).abs() < 1e-6).float()
            assert ((weight == 1).any(1)).all()
            weight = weight / weight.sum(1)[:, None] # in case of two max values
            return weight
        else:
            raise NotImplementedError
    
    def gather_group(self, scores, group_lens):
        assert (scores >= 0).all()
        score_per_group = []
        beg = 0
        for group_len in group_lens:
            end = beg + group_len
            score_this_g = scores[:, beg:end].sum(1)
            score_per_group.append(score_this_g)
            beg = end
        assert end == scores.size(1) == sum(group_lens)
        gathered_score = torch.stack(score_per_group, dim=1)
        assert gathered_score.size(1) == len(group_lens)
        return  gathered_score

    def combine_yaw_info(self, bboxes, gt_yaw):
        for i in range(len(bboxes)):
            bboxes[i] = torch.cat((bboxes[i], gt_yaw[i].unsqueeze(dim=1)), dim=1)
        return bboxes

    def paint_rgb2pts_v2(self, batch_points, batch_img, img_metas):
        
        batch_img = batch_img.permute(0,2,3,1)
        if isinstance(batch_points[0], list):
            for i, points in enumerate(batch_points):
                for j in range(len(points)):
                    x = points[j][:,12].type(torch.long)
                    y = points[j][:,13].type(torch.long)   
                    points_rgb = batch_img[i][y, x]
                    batch_points[i][j] = torch.cat((points[j], points_rgb), dim=1)
            return batch_points
        else:
            output_points = []
            for i, points in enumerate(batch_points):
                x = points[:,12].type(torch.long)
                y = points[:,13].type(torch.long)
                points_rgb = batch_img[i][y, x]
                output_points.append(torch.cat((points, points_rgb), dim=1))
            return output_points

    def paint_rgb2pts(self, batch_points, batch_img, img_metas):
        
        if isinstance(batch_points[0], list):
            for i, points in enumerate(batch_points):
                for j in range(len(points)):
                    # 放大到原分辨率
                    grid = points[j][:,12:14]/img_metas[i]['scale_factor'][0]
                    # 将 x,y 归一化到[-1,1]
                    grid[:,0] = (grid[:,0]/img_metas[i]['img_shape'][1]) - 1  # x
                    grid[:,1] = (grid[:,1]/img_metas[i]['img_shape'][0]) - 1  # y
                    out = F.grid_sample(batch_img[i].unsqueeze(0), grid.unsqueeze(0).unsqueeze(0), align_corners=False)
                    out = out.permute(0,2,3,1).squeeze(0).squeeze(0)
                    batch_points[i][j] = torch.cat((points[j], out), dim=1)
            return batch_points
        else:
            output_points = []
            for i, points in enumerate(batch_points):
                # 放大到原分辨率
                grid = points[:,12:14]/img_metas[i]['scale_factor'][0]
                # 将 x,y 归一化到[-1,1]
                grid[:,0] = (grid[:,0]/img_metas[i]['img_shape'][1]) - 1
                grid[:,1] = (grid[:,1]/img_metas[i]['img_shape'][0]) - 1
                # input(1,3,H,W) grid(1,1,N_pts,2) out (1,3,1,N_pts)
                out = F.grid_sample(batch_img[i].unsqueeze(0), grid.unsqueeze(0).unsqueeze(0), align_corners=False)
                out = out.permute(0,2,3,1).squeeze(0).squeeze(0)
                output_points.append(torch.cat((points, out), dim=1))
            return output_points


class EMA:
    def __init__(self, decay=0.99):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self, model):
        for name, param in model.named_parameters():
            self.shadow[name] = param.data.clone().detach()

    def update(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = param.data.clone()


class ClusterAssigner(torch.nn.Module):
    ''' Generating cluster centers for each class and assign each point to cluster centers
    '''

    def __init__(
        self,
        cluster_voxel_size,
        min_points,
        point_cloud_range,
        connected_dist,
        class_names=['Car', 'Cyclist', 'Pedestrian'],
    ):
        super().__init__()
        self.cluster_voxel_size = cluster_voxel_size
        self.min_points = min_points
        self.connected_dist = connected_dist
        self.point_cloud_range = point_cloud_range
        self.class_names = class_names

    @torch.no_grad()
    def forward(self, points_list, batch_idx_list, gt_bboxes_3d=None, gt_labels_3d=None, origin_points=None):
        gt_bboxes_3d = None
        gt_labels_3d = None
        assert self.num_classes == len(self.class_names)
        cluster_inds_list, valid_mask_list = \
            multi_apply(self.forward_single_class, points_list, batch_idx_list, self.class_names, origin_points)
        cluster_inds_list = modify_cluster_by_class(cluster_inds_list)
        return cluster_inds_list, valid_mask_list

    def forward_single_class(self, points, batch_idx, class_name, origin_points):
        batch_idx = batch_idx.int()

        if isinstance(self.cluster_voxel_size, dict):
            cluster_vsize = self.cluster_voxel_size[class_name]
        elif isinstance(self.cluster_voxel_size, list):
            cluster_vsize = self.cluster_voxel_size[self.class_names.index(class_name)]
        else:
            cluster_vsize = self.cluster_voxel_size

        voxel_size = torch.tensor(cluster_vsize, device=points.device)
        pc_range = torch.tensor(self.point_cloud_range, device=points.device)
        coors = torch.div(points - pc_range[None, :3], voxel_size[None, :], rounding_mode='floor').int()
        # coors = coors[:, [2, 1, 0]] # to zyx order
        coors = torch.cat([batch_idx[:, None], coors], dim=1)
        # 得到非空mask
        valid_mask = filter_almost_empty(coors, min_points=self.min_points)
        if not valid_mask.any():
            valid_mask = ~valid_mask
            # return coors.new_zeros((3,0)), valid_mask

        points = points[valid_mask]
        batch_idx = batch_idx[valid_mask]
        coors = coors[valid_mask]
        # elif len(points) 
        # 都是非空的voxel
        sampled_centers, voxel_coors, inv_inds = scatter_v2(points, coors, mode='avg', return_inv=True)

        if isinstance(self.connected_dist, dict):
            dist = self.connected_dist[class_name]
        elif isinstance(self.connected_dist, list):
            dist = self.connected_dist[self.class_names.index(class_name)]
        else:
            dist = self.connected_dist

        if self.training:
            cluster_inds = find_connected_componets(sampled_centers, voxel_coors[:, 0], dist)
        else:
            cluster_inds = find_connected_componets_single_batch(sampled_centers, voxel_coors[:, 0], dist)
        assert len(cluster_inds) == len(sampled_centers)

        cluster_inds_per_point = cluster_inds[inv_inds]  # 得到每个点对应的聚类id
        cluster_inds_per_point = torch.stack([batch_idx, cluster_inds_per_point], 1)
        return cluster_inds_per_point, valid_mask
