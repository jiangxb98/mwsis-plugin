import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from mmdet.core import reduce_mean
from mmdet.models import HEADS
from mmcv.cnn import normal_init
from mmcv.runner import auto_fp16, force_fp32 

from mmseg.models.builder import build_loss
from mmdet.models.builder import build_loss as build_det_loss
from mmdet3d.models.decode_heads.decode_head import Base3DDecodeHead

from .utils import build_mlp, build_norm_act, get_in_2d_box_inds
from .fsd_ops import scatter_v2

@HEADS.register_module()
class VoteSegHead(Base3DDecodeHead):

    def __init__(self,
                 in_channel,
                 num_classes,
                 hidden_dims=[],
                 dropout_ratio=0.5,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='naiveSyncBN1d'),
                 act_cfg=dict(type='ReLU'),
                 loss_decode=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    class_weight=None,
                    loss_weight=1.0),
                 loss_segment=None,
                 loss_lova=None,
                 loss_vote=dict(
                     type='L1Loss',
                 ),
                 loss_aux=None,
                 ignore_index=255,
                 logit_scale=1,
                 checkpointing=False,
                 init_bias=None,
                 init_cfg=None,
                 need_full_seg=False,
                 num_classes_full=None,
                 ignore_illegal_label=False,
                 segment_range=None,
                 need_inst_seg=False,
                 loss_inst=None,
                 need_vote=True,):
        end_channel = hidden_dims[-1] if len(hidden_dims) > 0 else in_channel
        super(VoteSegHead, self).__init__(
                 end_channel,
                 num_classes,
                 dropout_ratio,
                 conv_cfg,
                 norm_cfg,
                 act_cfg,
                 loss_decode,
                 ignore_index,
                 init_cfg
        )

        self.pre_seg_conv = None
        if len(hidden_dims) > 0:
            self.pre_seg_conv = build_mlp(in_channel, hidden_dims, norm_cfg, act=act_cfg['type'])

        self.use_sigmoid = loss_decode.get('use_sigmoid', False)
        self.bg_label = self.num_classes
        if not self.use_sigmoid:
            self.num_classes += 1


        self.logit_scale = logit_scale
        self.need_full_seg = need_full_seg
        self.need_inst_seg = need_inst_seg
        self.need_vote = need_vote
        self.conv_seg = nn.Linear(end_channel, self.num_classes)
        self.ignore_illegal_label = ignore_illegal_label
        self.segment_range = segment_range
        if self.need_full_seg:
            self.num_classes_full = num_classes_full
            self.conv_seg_full = nn.Linear(end_channel, self.num_classes_full)
        if self.need_inst_seg:
            self.inst_seg = nn.Linear(end_channel, 1)
        if self.need_vote:
            self.voting = nn.Linear(end_channel, self.num_classes * 3)
        self.fp16_enabled = False
        self.checkpointing = checkpointing
        self.init_bias = init_bias
        
        if loss_inst is not None:
            self.loss_inst = build_loss(loss_inst)
        if loss_aux is not None:
            self.loss_aux = build_loss(loss_aux)
        else:
            self.loss_aux = None
        if loss_decode['type'] == 'FocalLoss':
            self.loss_decode = build_det_loss(loss_decode) # mmdet has a better focal loss supporting single class
        
        self.loss_vote = build_det_loss(loss_vote) if loss_vote is not None else None
        self.loss_segment = build_det_loss(loss_segment) if loss_segment is not None else None
        self.loss_lova = build_det_loss(loss_lova) if loss_lova is not None else None

    def init_weights(self):
        """Initialize weights."""
        super().init_weights()
        if self.init_bias is not None:
            self.conv_seg.bias.data.fill_(self.init_bias)
            if self.need_full_seg:
                self.conv_seg_full.bias.data.fill_(self.init_bias)
            if self.need_inst_seg:
                self.inst_seg.bias.data.fill_(self.init_bias)
            print(f'Segmentation Head bias is initialized to {self.init_bias}')
        else:
            normal_init(self.conv_seg, mean=0, std=0.01)
            if self.need_full_seg:
                normal_init(self.conv_seg_full, mean=0, std=0.01)
            if self.need_inst_seg:
                normal_init(self.inst_seg, mean=0, std=0.01)

    def cls_seg_full(self, feat):
        """Classify each points."""
        if self.dropout is not None:
            raise NotImplementedError
            feat = self.dropout(feat)
        output = self.conv_seg_full(feat)
        return output

    def forward_test(self, inputs, img_metas, test_cfg, return_full_logits=False):
        return self.forward(inputs, return_full_logits)

    @auto_fp16(apply_to=('voxel_feat',))
    def forward(self, voxel_feat, return_full_logits=False):
        """Forward pass.
        
        """
        output = voxel_feat
        if self.pre_seg_conv is not None:
            if self.checkpointing:
                output = checkpoint(self.pre_seg_conv, voxel_feat)
            else:
                output = self.pre_seg_conv(voxel_feat)  # (N,67)-->(N,128)
        if self.need_full_seg:
            logits_full = self.cls_seg_full(output)  # (N,128)-->(N,sem_class=22) Segment 

        logits = self.cls_seg(output)  # (N,128)-->(N,3) Detect 
        vote_preds = self.voting(output)  # (N,128)-->(N,3*3)，每个类别预测一个偏置，所以是3*3

        if return_full_logits:
            return logits, logits_full, vote_preds
        else:
            return logits, vote_preds, output

    def vote_targets(self, ):
        pass

    @force_fp32(apply_to=('seg_logit', 'vote_preds'))
    def losses(self, points, seg_logit, vote_preds, seg_label, vote_targets, vote_mask,
            seg_logit_full=None, seg_label_full=None, use_multiview=False, history_labels=None,
            batch_idx=None):
        """Compute semantic segmentation loss.

        Args:
            seg_logit (torch.Tensor): Predicted per-point segmentation logits \
                of shape [B, num_classes, N].
            seg_label (torch.Tensor): Ground-truth segmentation label of \
                shape [B, N].
        """
        seg_logit = seg_logit * self.logit_scale
        if seg_logit_full is not None:  # seg_logit_full present segment
            seg_logit_full = seg_logit_full * self.logit_scale
        loss = dict()

        if False:
            # 使用run和历史投票来修正3D Box
            bg_mask = points[:, 17] == 0
            vote_targets[bg_mask] = 0
            vote_mask[bg_mask] = False
            seg_label[bg_mask] = self.num_classes

        # optim targets
        use_history_labels = False
        if history_labels is not None:
            # n_pts, n_his, n_cls
            if isinstance(history_labels, list):
                history_targets = []
                for i in range(len(history_labels)):
                    history_targets.append(history_labels[i][(points[batch_idx==i][:, 5]//10).long()])
                history_targets = torch.cat(history_targets)
            if not (history_targets[:, -1, :]==-1).all():
                gt_prob = 0.8
                his_prob = 0.2
                # N, 5
                max_value, max_indices = history_targets.max(2)
                # bg_mask = max_value < 0.5
                bg_mask = max_value < 0.4
                max_indices[bg_mask] = self.num_classes
                ignore_mask = (0.4 <= max_value) & (max_value <= 0.6)
                max_indices[ignore_mask] = -1
                
                # vote = 5
                compare_label = max_indices[:, 0]
                mask = torch.all(max_indices==compare_label[:, None], dim=1)
                history_targets = torch.ones_like(seg_label) * -1
                history_targets[mask & (compare_label!=-1)] = compare_label[mask & (compare_label!=-1)]
                label_mask = history_targets != -1
                # prob = torch.tensor([gt_prob, his_prob], device=points.device).repeat(label_mask.sum(), 1)
                # sample_dist = torch.distributions.Categorical(prob).sample()
                # combine_labels = torch.stack((seg_label[label_mask], history_targets[label_mask])).T
                # refine_labels = combine_labels[torch.arange(label_mask.sum()), sample_dist]
                refine_labels = history_targets[label_mask]
                seg_label[label_mask] = refine_labels
                use_history_labels = True
                # if input all top points, need
                points[:, 17][points[:, 16]==0] = -1
                points[:, 17][label_mask & (points[:, 17]==-1)] = 2 # not ignore
                
                # vote >= 4
                # cls_counts = torch.zeros((seg_label.shape[0], self.num_classes + 1), device=points.device)
                # for i in range(self.num_classes + 1):
                #     cls_counts[:, i] = (max_indices==i).sum(1)
                # max_value2, max_indices2 = cls_counts.max(1)
                # max_indices2[max_value2 < 4] = -1
                # label_mask = max_indices2 !=- 1
                # prob = torch.tensor([gt_prob, his_prob], device=points.device).repeat(label_mask.sum(), 1)
                # sample_dist = torch.distributions.Categorical(prob).sample()
                # combine_labels = torch.stack((seg_label[label_mask], max_indices2[label_mask])).T
                # refine_labels = combine_labels[torch.arange(label_mask.sum()), sample_dist]
                # seg_label[label_mask] = max_indices2[label_mask]
                # refine_labels = combine_labels[torch.arange(label_mask.sum()), sample_dist]
                # # refine_labels = history_targets[label_mask]
                # seg_label[label_mask] = refine_labels
                # use_history_labels = True

                no_valid_mask = seg_label == 3
                vote_targets[no_valid_mask] = 0
                vote_mask[no_valid_mask] = False

        if use_multiview and not use_history_labels:
            mask_cls = (points[:, 16]==1) & (points[:, 17]!=-1)
        elif use_history_labels:
            # mask_cls = (points[:, 16]==1) & (points[:, 17]!=-1)
            mask_cls = (points[:, 17]!=-1)
        elif not use_multiview:
            # use for box 3d
            mask_cls = (points[:, 16]==1) & (points[:,17]!=-1)
        
        loss['loss_sem_seg'] = torch.tensor(0., device=points.device)
        loss['loss_sem_seg'] = self.loss_decode(seg_logit[mask_cls], seg_label[mask_cls].long())  #(N,3),(N,)

        if self.loss_aux is not None:
            loss['loss_aux'] = self.loss_aux(seg_logit, seg_label)

        vote_preds = vote_preds.reshape(-1, self.num_classes, 3)  # (N,9)-->(N,3,3)
        if not self.use_sigmoid:
            assert seg_label.max().item() == self.num_classes - 1
        else:
            assert seg_label.max().item() == self.num_classes
        valid_vote_preds = vote_preds[vote_mask] # [n_valid, num_cls, 3]
        valid_vote_preds = valid_vote_preds.reshape(-1, 3)
        num_valid = vote_mask.sum()

        valid_label = seg_label[vote_mask]

        if num_valid > 0:
            assert valid_label.max().item() < self.num_classes
            assert valid_label.min().item() >= 0
            
            indices = torch.arange(num_valid, device=valid_label.device) * self.num_classes + valid_label
            valid_vote_preds = valid_vote_preds[indices.long(), :] # [n_valid, 3]

            valid_vote_targets = vote_targets[vote_mask]
            # avg_factor = num_valid * self.num_classes
            # num_pos = max(reduce_mean(num_valid.double()), 1.0)
            loss['loss_vote'] = self.loss_vote(valid_vote_preds, valid_vote_targets)
        else:
            loss['loss_vote'] = vote_preds.sum() * 0

        train_cfg = self.train_cfg
        if train_cfg.get('score_thresh', None) is not None:
            score_thresh = train_cfg['score_thresh']  # [0.3,0.25,0.25]
            if self.use_sigmoid:
                scores = seg_logit.sigmoid()
                for i in range(len(score_thresh)):
                    thr = score_thresh[i]
                    name = train_cfg['class_names'][i]
                    this_scores = scores[:, i]
                    pred_true = this_scores > thr
                    real_true = seg_label == i
                    tp = (pred_true & real_true).sum().float()
                    loss[f'recall_{name}'] = tp / (real_true.sum().float() + 1e-5)
            else:
                score = seg_logit.softmax(1)
                group_lens = train_cfg['group_lens']
                group_score = self.gather_group(score[:, :-1], group_lens)
                num_fg = score.new_zeros(1)
                for gi in range(len(group_lens)):
                    pred_true = group_score[:, gi] > score_thresh[gi]
                    num_fg += pred_true.sum().float()
                    for i in range(group_lens[gi]):
                        name = train_cfg['group_names'][gi][i]
                        real_true = seg_label == train_cfg['class_names'].index(name)
                        tp = (pred_true & real_true).sum().float()
                        loss[f'recall_{name}'] = tp / (real_true.sum().float() + 1e-5)
                loss[f'num_fg'] = num_fg

        return loss

    def forward_train(self, points, inputs, img_metas, pts_semantic_mask,
                        vote_targets, vote_mask, return_preds=False,
                        pts_semantic_mask_full=None, use_multiview=False,
                        history_labels=None, batch_idx=None):
        # 
        if pts_semantic_mask_full is None:
            seg_logits, vote_preds, out = self.forward(inputs)
            seg_logits_full = None
        else:
            seg_logits, seg_logits_full, vote_preds = self.forward(inputs, return_full_logits=True)  # (N,3),(N,22),(N,3x3)
        losses = self.losses(points, seg_logits, vote_preds, pts_semantic_mask,
                             vote_targets, vote_mask, seg_logits_full, pts_semantic_mask_full,
                             use_multiview, history_labels, batch_idx)
        if return_preds:
            return losses, dict(seg_logits=seg_logits, vote_preds=vote_preds), out
        else:
            return losses

    def seg_forward_train(self, points, feats, img_metas, pts_semantic_mask, pts_instance_mask, batch_idx, history_labels=None):
        losses = dict()
        output = self.pre_seg_conv(feats)  # (N, 67) --> (N, 128)
        # logits_inst = self.inst_seg(output)# (N, 128) --> (N,1)
        logits = self.cls_seg(output)      # (N, 128) --> (N,2)

        # optim targets
        use_history_labels = False
        if history_labels is not None:
            seg_label = pts_semantic_mask.clone().detach()
            seg_label[seg_label==-1] = self.num_classes
            # n_pts, n_his, n_cls
            if isinstance(history_labels, list):
                history_targets = []
                for i in range(len(history_labels)):
                    history_targets.append(history_labels[i][(points[batch_idx==i][:,5]//10).long()])
                history_targets = torch.cat(history_targets)
            if not (history_targets[:, -1, :]==-1).all():
                # N, 5
                max_value, max_indices = history_targets.max(2)
                bg_mask = max_value < 0.5
                max_indices[bg_mask] = self.num_classes
                
                # vote = 5
                compare_label = max_indices[:, 0]
                compare_label = compare_label.type(seg_label.dtype)
                mask = torch.all(max_indices==compare_label[:, None], dim=1)
                history_targets = torch.ones_like(seg_label) * -1
                history_targets[mask & (compare_label!=-1)] = compare_label[mask & (compare_label!=-1)]
                label_mask = history_targets != -1
                refine_labels = history_targets[label_mask]
                seg_label[label_mask] = refine_labels
                use_history_labels = True
                no_valid_mask = seg_label == 3
                # vote_targets[no_valid_mask] = 0
                # vote_mask[no_valid_mask] = False

        if self.need_vote:
            vote_preds = self.voting(output)
            vote_mask = pts_instance_mask != 0
            if use_history_labels:
                vote_mask[no_valid_mask] = False
            valid_vote_preds = vote_preds[vote_mask] # [n_valid, num_cls, 3]
            valid_vote_preds = valid_vote_preds.reshape(-1, 3)
            num_valid = vote_mask.sum()
            valid_label = pts_semantic_mask[vote_mask]
            if num_valid > 0:
                assert valid_label.max().item() < self.num_classes
                assert valid_label.min().item() >= 0
                indices = torch.arange(num_valid, device=valid_label.device) * self.num_classes + valid_label
                valid_vote_preds = valid_vote_preds[indices.long(), :] #[n_valid, 3]
                # get vote targets
                # 1. get instance center
                inst_sets, invs, counts = torch.unique(torch.stack((pts_instance_mask, batch_idx), dim=1), return_inverse=True, return_counts=True, dim=0)
                center_xyz = torch.zeros((inst_sets.shape[0], 3), device=points.device)
                center_xyz.scatter_add_(0, invs.repeat(3, 1).T, points[:, 0:3])
                # or torch_scatter.scatter(points[:, 0:3], invs, dim=0, reduce='sum')
                center_xyz = center_xyz / counts.repeat(3,1).T
                delta = center_xyz[invs] - points[:, 0:3]
                vote_targets = self.encode_vote_targets(delta)
                valid_vote_targets = vote_targets[vote_mask]
                losses['loss_vote'] = self.loss_vote(valid_vote_preds, valid_vote_targets)
            else:
                losses['loss_vote'] = vote_preds.sum() * 0

        mask_cls = (points[:,16]==0) #& (points[:,17]!=-1)
        pts_semantic_mask[pts_semantic_mask==-1] = self.num_classes

        if use_history_labels:
            pts_semantic_mask[label_mask] = refine_labels
            mask_cls = points[:,16]==1

        if mask_cls.sum() == 0:
            losses['loss_sem_seg'] = logits[mask_cls].sum() * 0.
            # losses['loss_inst_seg'] = logits_inst[mask_cls].sum() * 0.
        else:
            losses['loss_sem_seg'] = self.loss_decode(logits[mask_cls], pts_semantic_mask[mask_cls].long())
            # losses['loss_inst_seg'] = self.loss_inst(logits_inst[mask_cls], pts_instance_mask[mask_cls][:, None].long())
        return losses, logits

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

    def get_targets(self, points_list, gt_bboxes_list, gt_labels_list, gt_box_type=1, img_metas=None):
        bsz = len(points_list)  # batch size
        label_list = []
        vote_target_list = []
        vote_mask_list = []  # what is vote_mask? only in bboxes points need to vote,in test the vote_mask is pred foreground and background
        # multi_view_flag = len(points_list) != len(gt_bboxes_list)
        batch_img_nums = int(len(gt_bboxes_list)/len(points_list))

        for i in range(bsz):
            # batch_id = int(i//batch_img_nums)
            points = points_list[i][:, :3]
            bboxes = gt_bboxes_list[i]
            bbox_labels = gt_labels_list[i]

            # if self.num_classes < 3: # I don't know why there are some -1 labels when train car-only model.
            valid_gt_mask = bbox_labels >= 0
            bboxes = bboxes[valid_gt_mask]
            bbox_labels = bbox_labels[valid_gt_mask]
            
            if len(bbox_labels) == 0 and len(points_list) == len(gt_bboxes_list):
                this_label = torch.ones(len(points), device=points.device, dtype=torch.long) * self.bg_label
                this_vote_target = torch.zeros_like(points)
                vote_mask = torch.zeros_like(this_label).bool()
                if len(points_list) != len(gt_bboxes_list):
                    raise ValueError("have empty gt labels")
            else:
                if gt_box_type == 1:
                    extra_width = self.train_cfg.get('extra_width', None) 
                    if extra_width is not None:
                        bboxes = bboxes.enlarged_box_hw(extra_width)
                    inbox_inds = bboxes.points_in_boxes(points).long()  #(N,),-1表示没有在box内部
                    this_label = self.get_point_labels(inbox_inds, bbox_labels)  # (N,)
                    this_vote_target, vote_mask = self.get_vote_target(inbox_inds, points, bboxes)
                # if 2d box
                elif gt_box_type == 2:
                    extra_width = self.train_cfg.get('extra_width', None)
                    if extra_width is not None:
                        bboxes = self.enlarged_2d_box_hw(bboxes, extra_width)  # 待完成
                    if len(points_list) == len(gt_bboxes_list):  # 如果为false
                        inbox_inds = get_in_2d_box_inds(points_list[i], bboxes, img_metas[i])  # (N,1) 即每个点对应的盒子id
                        this_label = self.get_point_labels(inbox_inds, bbox_labels)  # (N,)  #每个点依据在哪个盒子内分配一个标签
                        this_vote_target, vote_mask = self.get_vote_target_2d(inbox_inds, points, bboxes)  # 由于2d box没有几何中心，所以这里的3d几何中心用2dbox内点的平均值
                    else:
                        inbox_inds = get_in_2d_box_inds(points_list[i], gt_bboxes_list[i*batch_img_nums:(i+1)*batch_img_nums], img_metas[i])  # (N,1) 即每个点对应的盒子id
                        this_label = self.get_point_labels(inbox_inds, gt_labels_list[i*batch_img_nums:(i+1)*batch_img_nums])  # (N,)  #每个点依据在哪个盒子内分配一个标签
                        this_vote_target, vote_mask = self.get_vote_target_2d(inbox_inds, points, gt_bboxes_list[i*batch_img_nums:(i+1)*batch_img_nums])  # 由于2d box没有几何中心，所以这里的3d几何中心用2dbox内点的平均值
            label_list.append(this_label)
            vote_target_list.append(this_vote_target)
            vote_mask_list.append(vote_mask)

        labels = torch.cat(label_list, dim=0)
        vote_targets = torch.cat(vote_target_list, dim=0)
        vote_mask = torch.cat(vote_mask_list, dim=0)

        return labels.long(), vote_targets, vote_mask
    
    def get_point_labels(self, inbox_inds, bbox_labels):
        if isinstance(bbox_labels, list):
            bbox_labels = torch.cat(bbox_labels)
            bg_mask = inbox_inds < 0
            class_labels = bbox_labels[inbox_inds]
            class_labels[bg_mask] = self.bg_label
        else:
            bg_mask = inbox_inds < 0
            # label = -1 * torch.ones(len(inbox_inds), dtype=torch.long, device=inbox_inds.device)
            class_labels = bbox_labels[inbox_inds]
            class_labels[bg_mask] = self.bg_label

        return class_labels

    def get_vote_target(self, inbox_inds, points, bboxes):

        bg_mask = inbox_inds < 0
        if self.train_cfg.get('centroid_offset', False):
            import pdb;pdb.set_trace()
            centroid, _, inv = scatter_v2(points, inbox_inds, mode='avg', return_inv=True)
            center_per_point = centroid[inv]
        else:
            center_per_point = bboxes.gravity_center[inbox_inds]  #
        delta = center_per_point.to(points.device) - points
        delta[bg_mask] = 0
        target = self.encode_vote_targets(delta)
        vote_mask = ~bg_mask
        return target, vote_mask
    
    def encode_vote_targets(self, delta):
        return torch.sign(delta) * (delta.abs() ** 0.5)# 就是开根号了，下面的decode_vote_targets又乘回来了,sign符号函数来保持正负性
    
    def decode_vote_targets(self, preds):
        return preds * preds.abs()

    # 2d box的投票
    def get_vote_target_2d(self, inbox_inds, points, bboxes):
        '''
        Input: 
            inbox_inds:
            points:
            bboxes:
        Return: 
            this_vote_target:
            vote_mask:
        '''

        bg_mask = inbox_inds < 0
        if self.train_cfg.get('centroid_offset', False):# 未修改
            import pdb; pdb.set_trace()
            centroid, _, inv = scatter_v2(points, inbox_inds, mode='avg', return_inv=True)
            center_per_point = centroid[inv]
        else:

            fake_box_center = self.get_2d_box_center(inbox_inds, points, bboxes)

            center_per_point = fake_box_center[inbox_inds]

        delta = center_per_point - points
        delta[bg_mask] = 0
        target = self.encode_vote_targets(delta)
        vote_mask = ~bg_mask
        return target, vote_mask

    def get_2d_box_center(self, inbox_inds, points, bboxes):
        # 这里是通过2d box得到了一簇点, 这一簇点求平均得到投票结果
        # 当然也可以设计使用2d的投票点进行投票，我这里没有写，因为需要修改网络结构
        if isinstance(bboxes, list):
            bboxes=torch.cat(bboxes)
        fake_center = torch.zeros((bboxes.shape[0], 3), device=points.device)

        for i, box in enumerate(bboxes):
            if (inbox_inds==i).sum() == 0:
                continue
            fake_cluster_points = points[inbox_inds==i]
            fake_cluster_points = fake_cluster_points[:,:3]
            fake_center[i] = torch.mean(fake_cluster_points, dim=0)

        return fake_center
