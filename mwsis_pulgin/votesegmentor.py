# Copyright (c) OpenMMLab. All rights reserved.
from os import path as osp
import torch
from torch.nn import functional as F
from mmcv.ops import Voxelization
from mmcv.parallel import DataContainer as DC
from mmcv.runner import force_fp32

from mmdet3d.models import builder
from mmdet3d.models.builder import DETECTORS

from mmseg.models import SEGMENTORS
from mmdet3d.models.segmentors.base import Base3DSegmentor

@SEGMENTORS.register_module()
@DETECTORS.register_module()
class VoteSegmentor(Base3DSegmentor):

    def __init__(self,
                 voxel_layer,
                 voxel_encoder,
                 middle_encoder,
                 backbone,
                 segmentation_head,
                 decode_neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None,
                 tanh_dims=None,
                 need_full_seg=False,
                 only_one_frame_label=True,
                 sweeps_num=1,
                 **extra_kwargs):
        super().__init__(init_cfg=init_cfg)

        self.voxel_layer = Voxelization(**voxel_layer)

        self.voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
        self.middle_encoder = builder.build_middle_encoder(middle_encoder)
        self.backbone = builder.build_backbone(backbone)
        self.segmentation_head = builder.build_head(segmentation_head)
        self.segmentation_head.train_cfg = train_cfg
        self.segmentation_head.test_cfg = test_cfg
        self.decode_neck = builder.build_neck(decode_neck)

        assert voxel_encoder['type'] == 'DynamicScatterVFE'


        self.print_info = {}
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.cfg = train_cfg if train_cfg is not None else test_cfg
        self.num_classes = segmentation_head['num_classes']
        self.save_list = []
        self.point_cloud_range = voxel_layer['point_cloud_range']
        self.voxel_size = voxel_layer['voxel_size']
        self.tanh_dims = tanh_dims
        self.need_full_seg = need_full_seg
        self.only_one_frame_label = only_one_frame_label
        self.sweeps_num = sweeps_num
    
    def encode_decode(self, ):
        return None
    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        return NotImplementedError

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.
        Args:
            points (list[torch.Tensor]): Points of each sample.
        Returns:
            tuple[torch.Tensor]: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provide a coors mapping
        for res in points:
            res_coors = self.voxel_layer(res)
            coors.append(res_coors)
        points = torch.cat(points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch

    def extract_feat(self, points, img_metas):
        """Extract features from points."""
        batch_points, coors = self.voxelize(points)
        coors = coors.long()
        
        voxel_features, voxel_coors, voxel2point_inds = self.voxel_encoder(batch_points, coors, return_inv=True)
        voxel_info = self.middle_encoder(voxel_features, voxel_coors)
        x = self.backbone(voxel_info)[0]
        padding = -1
        voxel_coors_dropped = x['voxel_feats'] # bug, leave it for feature modification
        if 'shuffle_inds' not in voxel_info:
            voxel_feats_reorder = x['voxel_feats']
        else:
            # this branch only used in SST-based FSD 
            voxel_feats_reorder = self.reorder(x['voxel_feats'], voxel_info['shuffle_inds'], voxel_info['voxel_keep_inds'], padding) #'not consistent with voxel_coors any more'
        # 注意这里out[1]这个索引的意义
        out = self.decode_neck(batch_points, coors, voxel_feats_reorder, voxel2point_inds, padding)

        return out, coors, batch_points
    
    
    def reorder(self, data, shuffle_inds, keep_inds, padding=-1):
        '''
        Padding dropped voxel and reorder voxels.  voxel length and order will be consistent with the output of voxel_encoder.
        '''
        num_voxel_no_drop = len(shuffle_inds)
        data_dim = data.size(1)

        temp_data = padding * data.new_ones((num_voxel_no_drop, data_dim))
        out_data = padding * data.new_ones((num_voxel_no_drop, data_dim))

        temp_data[keep_inds] = data
        out_data[shuffle_inds] = temp_data

        return out_data
    

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      as_subsegmentor=False,
                      pts_semantic_mask=None,
                      pts_instance_mask=None,
                      gt_box_type=1,
                      history_labels=None,
                      ):
        if pts_instance_mask is None:
            use_multiview = len(points) != len(gt_labels_3d)
            # labels: B*N,1 vote_targets: B*N,3 vote_mask(bool) B*N,1
            labels, vote_targets, vote_mask = self.segmentation_head.get_targets(points, gt_bboxes_3d, gt_labels_3d, gt_box_type, img_metas)
            # neck_out(list):[points_feature(N,64+3), bool(N,)]pts_coors: N,4(batch_id,vx,vy,vz) points=input points 没变
            neck_out, pts_coors, points = self.extract_feat(points, img_metas)  # 提取每个点的特征，这里提取特征是通过体素化提取，然后将体素特征再传播给对应的点

            losses = dict()

            feats = neck_out[0]
            valid_pts_mask = neck_out[1]
            points = points[valid_pts_mask]
            pts_coors = pts_coors[valid_pts_mask]
            labels = labels[valid_pts_mask]
            vote_targets = vote_targets[valid_pts_mask]
            vote_mask = vote_mask[valid_pts_mask]
            if pts_semantic_mask is not None:
                pts_semantic_mask = torch.cat(pts_semantic_mask, dim=0)
                pts_semantic_mask = pts_semantic_mask[valid_pts_mask]

            assert feats.size(0) == labels.size(0)
        else:
            neck_out, pts_coors, points = self.extract_feat(points, img_metas)
            losses = dict()
            feats = neck_out[0]
            valid_pts_mask = neck_out[1]
            points = points[valid_pts_mask]
            pts_coors = pts_coors[valid_pts_mask]
            pts_semantic_mask = torch.cat(pts_semantic_mask, dim=0)
            pts_semantic_mask = pts_semantic_mask[valid_pts_mask]
            pts_instance_mask = torch.cat(pts_instance_mask, dim=0)
            pts_instance_mask = pts_instance_mask[valid_pts_mask]
            # pts_instance_mask[pts_instance_mask!=0] = 1
            assert feats.size(0) == pts_semantic_mask.size(0) == pts_instance_mask.size(0)
            losses, seg_logits = self.segmentation_head.seg_forward_train(points, feats, img_metas, pts_semantic_mask, pts_instance_mask, pts_coors[:, 0], history_labels)
            output_dict = dict(
                    losses=losses,
                    seg_points=points,
                    seg_logits=seg_logits,
                    batch_idx=pts_coors[:,0],
            )
            return output_dict

        if as_subsegmentor and pts_semantic_mask is None:
            loss_decode, preds_dict, out = self.segmentation_head.forward_train(
                points, feats, img_metas, labels, vote_targets, vote_mask, 
                return_preds=True, pts_semantic_mask_full=None, use_multiview=use_multiview,
                history_labels=history_labels, batch_idx=pts_coors[:, 0])
            losses.update(loss_decode)

            seg_logits = preds_dict['seg_logits']
            vote_preds = preds_dict['vote_preds']

            offsets = self.segmentation_head.decode_vote_targets(vote_preds)

            output_dict = dict(
                seg_points=points,
                seg_logits=preds_dict['seg_logits'],
                seg_vote_preds=preds_dict['vote_preds'],
                offsets=offsets,
                seg_feats=feats,
                batch_idx=pts_coors[:, 0],
                losses=losses,
                vote_head_feats=out,
                labels=labels,
            )
        else:
            raise NotImplementedError
            loss_decode = self.segmentation_head.forward_train(feats, img_metas, labels, vote_targets, vote_mask, return_preds=False)
            losses.update(loss_decode)
            output_dict = losses

        return output_dict


    def simple_test(self, points, img_metas, gt_bboxes_3d=None, gt_labels_3d=None, rescale=False):

        # if self.tanh_dims is not None:
        #     for p in points:
        #         p[:, self.tanh_dims] = torch.tanh(p[:, self.tanh_dims])
        # elif points[0].size(1) in (4,5):
        #     points = [torch.cat([p[:, :3], torch.tanh(p[:, 3:])], dim=1) for p in points]
        # TODO output with sweep_ind
        sweep_ind = None
        if self.sweeps_num > 1:
            sweep_ind = [p[:, -1] for p in points]
            points = [p[:, :-1].contiguous() for p in points]

        seg_pred = []
        x, pts_coors, points = self.extract_feat(points, img_metas)
        feats = x[0]
        valid_pts_mask = x[1]
        points = points[valid_pts_mask]
        pts_coors = pts_coors[valid_pts_mask]

        # select points with sweep_ind==0
        if sweep_ind is not None:
            sweep_ind = torch.cat(sweep_ind, dim=0)
            sweep_ind = sweep_ind[valid_pts_mask]
            sweep_ind_mask = sweep_ind == 0
            feats = feats[sweep_ind_mask]
            points = points[sweep_ind_mask]
            pts_coors = pts_coors[sweep_ind_mask]

        if self.need_full_seg:
            seg_logits, seg_logits_full, vote_preds = \
                self.segmentation_head.forward_test(feats, img_metas, self.test_cfg, True)
        else:
            seg_logits, vote_preds, out = self.segmentation_head.forward_test(feats, img_metas, self.test_cfg)
            seg_logits_full = None

        offsets = self.segmentation_head.decode_vote_targets(vote_preds)

        output_dict = dict(
            seg_points=points,  # ori points 
            seg_logits=seg_logits,  # 
            seg_vote_preds=vote_preds,
            offsets=offsets,
            seg_feats=feats,
            batch_idx=pts_coors[:, 0],
            seg_logits_full=seg_logits_full
        )

        return output_dict
