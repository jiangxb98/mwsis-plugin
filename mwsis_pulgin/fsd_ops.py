
import torch
import traceback
import torch_scatter
from torch.autograd import Function

import ingroup_indices
import dynamic_point_pool_v2_ext


class IngroupIndicesFunction(Function):

    @staticmethod
    def forward(ctx, group_inds):
        out_inds = torch.zeros_like(group_inds) - 1
        ingroup_indices.forward(group_inds, out_inds)
        ctx.mark_non_differentiable(out_inds)
        return out_inds

    @staticmethod
    def backward(ctx, g):
        return None

get_inner_win_inds = IngroupIndicesFunction.apply



class DynamicPointPoolFunction(Function):

    @staticmethod
    def forward(ctx, rois, pts, extra_wlh, max_inbox_point, max_all_pts=50000):
        """RoIAwarePool3d function forward.

        Args:
            rois (torch.Tensor): [N, 7], in LiDAR coordinate,
                (x, y, z) is the bottom center of rois
            pts (torch.Tensor): [npoints, 3]
            pts_feature (torch.Tensor): [npoints, C]
            out_size (int or tuple): n or [n1, n2, n3]
            max_pts_per_voxel (int): m
            mode (int): 0 (max pool) or 1 (average pool)

        Returns:
            pooled_features (torch.Tensor): [N, out_x, out_y, out_z, C]
        """

        # pts_inds, roi_inds, pts_norm_xyz, pts_offset = dynamic_point_pool_ext.forward(rois, pts)
        out_pts_idx = -1 * pts.new_ones(max_all_pts, dtype=torch.long)
        out_roi_idx = -1 * pts.new_ones(max_all_pts, dtype=torch.long)
        out_pts_feats = pts.new_zeros(max_all_pts, 13, dtype=torch.float)

        assert len(rois) > 0
        dynamic_point_pool_v2_ext.forward(rois, pts, extra_wlh, max_inbox_point, out_pts_idx, out_roi_idx, out_pts_feats)
        # Because of cuda block layout, the out_roi_idx is automatically sorted, but not strictly guaranteed.
        valid_mask = out_pts_idx >= 0

        if not valid_mask.any():
            # fake a non-empty input
            out_pts_idx = out_pts_idx[0:1]
            out_roi_idx = out_roi_idx[0:1]
            out_pts_feats = out_pts_feats[0:1, :]
        else:
            out_pts_idx = out_pts_idx[valid_mask]
            out_roi_idx = out_roi_idx[valid_mask]
            out_pts_feats = out_pts_feats[valid_mask]
            unique_roi_idx = torch.unique(out_roi_idx)

        ctx.mark_non_differentiable(out_pts_idx)
        ctx.mark_non_differentiable(out_roi_idx)
        ctx.mark_non_differentiable(out_pts_feats)

        return out_pts_idx, out_roi_idx, out_pts_feats 

    @staticmethod
    def backward(ctx, g1, g2, g3):

        return None, None, None, None, None

dynamic_point_pool = DynamicPointPoolFunction.apply




def scatter_v2(feat, coors, mode, return_inv=True, min_points=0, unq_inv=None, new_coors=None):
    assert feat.size(0) == coors.size(0)
    if mode == 'avg':
        mode = 'mean'

    if unq_inv is None and min_points > 0:
        new_coors, unq_inv, unq_cnt = torch.unique(coors, return_inverse=True, return_counts=True, dim=0)
    elif unq_inv is None:
        new_coors, unq_inv = torch.unique(coors, return_inverse=True, return_counts=False, dim=0)
    else:
        assert new_coors is not None, 'please pass new_coors for interface consistency, caller: {}'.format(traceback.extract_stack()[-2][2])

    if min_points > 0:
        cnt_per_point = unq_cnt[unq_inv]
        valid_mask = cnt_per_point >= min_points
        feat = feat[valid_mask]
        coors = coors[valid_mask]
        new_coors, unq_inv, unq_cnt = torch.unique(coors, return_inverse=True, return_counts=True, dim=0)

    if mode == 'max':
        new_feat, argmax = torch_scatter.scatter_max(feat, unq_inv, dim=0)
    elif mode in ('mean', 'sum'):
        new_feat = torch_scatter.scatter(feat, unq_inv, dim=0, reduce=mode)
    else:
        raise NotImplementedError

    if not return_inv:
        return new_feat, new_coors
    else:
        return new_feat, new_coors, unq_inv
