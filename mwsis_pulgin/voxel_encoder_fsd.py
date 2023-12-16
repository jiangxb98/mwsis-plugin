import torch
from torch import nn
from torch.nn import functional as F
from mmcv.cnn import build_norm_layer
from mmcv.runner import force_fp32, auto_fp16
from mmdet3d.ops import DynamicScatter
from mmdet3d.models.builder import VOXEL_ENCODERS
from mmdet3d.models import builder
from .fsd_ops import scatter_v2
from .utils import build_mlp, get_activation_layer


class DynamicVFELayer(nn.Module):
    """Replace the Voxel Feature Encoder layer in VFE layers.

    This layer has the same utility as VFELayer above

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm_cfg (dict): Config dict of normalization layers
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01)  # 0.01
                 ):
        super(DynamicVFELayer, self).__init__()
        self.fp16_enabled = False
        # self.units = int(out_channels / 2)
        self.norm = build_norm_layer(norm_cfg, out_channels)[1]
        self.linear = nn.Linear(in_channels, out_channels, bias=False)


    @auto_fp16(apply_to=('inputs'), out_fp32=True)
    def forward(self, inputs):
        """Forward function.

        Args:
            inputs (torch.Tensor): Voxels features of shape (M, C).
                M is the number of points, C is the number of channels of point features.

        Returns:
            torch.Tensor: point features in shape (M, C).
        """
        # [K, T, 7] tensordot [7, units] = [K, T, units]
        x = self.linear(inputs)
        x = self.norm(x)
        pointwise = F.relu(x)
        return pointwise


class DynamicVFELayerV2(nn.Module):
    """Replace the Voxel Feature Encoder layer in VFE layers.
    This layer has the same utility as VFELayer above
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm_cfg (dict): Config dict of normalization layers
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 act='relu',
                 dropout=0.0,
                 ):
        super(DynamicVFELayerV2, self).__init__()
        self.fp16_enabled = False
        # self.units = int(out_channels / 2)
        self.norm = build_norm_layer(norm_cfg, out_channels)[1]
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.act = get_activation_layer(act, out_channels)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

    @auto_fp16(apply_to=('inputs'), out_fp32=True)
    def forward(self, inputs):
        """Forward function.
        Args:
            inputs (torch.Tensor): Voxels features of shape (M, C).
                M is the number of points, C is the number of channels of point features.
        Returns:
            torch.Tensor: point features in shape (M, C).
        """
        # [K, T, 7] tensordot [7, units] = [K, T, units]
        if self.dropout is not None:
            inputs = self.dropout(inputs)
        x = self.linear(inputs)
        x = self.norm(x)
        pointwise = self.act(x)
        return pointwise



@VOXEL_ENCODERS.register_module(force=True)
class DynamicVFE(nn.Module):
    """Dynamic Voxel feature encoder used in DV-SECOND.

    It encodes features of voxels and their points. It could also fuse
    image feature into voxel features in a point-wise manner.
    The number of points inside the voxel varies.

    Args:
        in_channels (int): Input channels of VFE. Defaults to 4.
        feat_channels (list(int)): Channels of features in VFE.
        with_distance (bool): Whether to use the L2 distance of points to the
            origin point. Default False.
        with_cluster_center (bool): Whether to use the distance to cluster
            center of points inside a voxel. Default to False.
        with_voxel_center (bool): Whether to use the distance to center of
            voxel for each points inside a voxel. Default to False.
        voxel_size (tuple[float]): Size of a single voxel. Default to
            (0.2, 0.2, 4).
        point_cloud_range (tuple[float]): The range of points or voxels.
            Default to (0, -40, -3, 70.4, 40, 1).
        norm_cfg (dict): Config dict of normalization layers.
        mode (str): The mode when pooling features of points inside a voxel.
            Available options include 'max' and 'avg'. Default to 'max'.
        fusion_layer (dict | None): The config dict of fusion layer used in
            multi-modal detectors. Default to None.
        return_point_feats (bool): Whether to return the features of each
            points. Default to False.
    """

    def __init__(self,
                 in_channels=4,
                 feat_channels=[],
                 with_distance=False,
                 with_cluster_center=False,
                 with_voxel_center=False,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max',
                 fusion_layer=None,
                 return_point_feats=False,
                 ):
        super(DynamicVFE, self).__init__()
        self.in_pts_dim = in_channels
        # assert mode in ['avg', 'max']
        assert len(feat_channels) > 0
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 3
        if with_distance:
            in_channels += 3
        self.in_channels = in_channels
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self.return_point_feats = return_point_feats
        self.fp16_enabled = False

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.point_cloud_range = point_cloud_range
        self.scatter = DynamicScatter(voxel_size, point_cloud_range, True)

        feat_channels = [self.in_channels] + list(feat_channels)
        vfe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i > 0:
                in_filters *= 2

            vfe_layers.append(
                DynamicVFELayer(
                    in_filters,
                    out_filters,
                    norm_cfg))
        self.vfe_layers = nn.ModuleList(vfe_layers)
        self.num_vfe = len(vfe_layers)
        self.vfe_scatter = DynamicScatter(voxel_size, point_cloud_range,
                                          (mode != 'max'))
        self.cluster_scatter = DynamicScatter(
            voxel_size, point_cloud_range, average_points=True)
        self.fusion_layer = None
        if fusion_layer is not None:
            self.fusion_layer = builder.build_fusion_layer(fusion_layer)

    
    def map_voxel_center_to_point(self, pts_coors, voxel_mean, voxel_coors):
        """Map voxel features to its corresponding points.

        Args:
            pts_coors (torch.Tensor): Voxel coordinate of each point.
            voxel_mean (torch.Tensor): Voxel features to be mapped.
            voxel_coors (torch.Tensor): Coordinates of valid voxels

        Returns:
            torch.Tensor: Features or centers of each point.
        """
        # Step 1: scatter voxel into canvas
        # Calculate necessary things for canvas creation
        canvas_z = round(
            (self.point_cloud_range[5] - self.point_cloud_range[2]) / self.vz)
        canvas_y = round(
            (self.point_cloud_range[4] - self.point_cloud_range[1]) / self.vy)
        canvas_x = round(
            (self.point_cloud_range[3] - self.point_cloud_range[0]) / self.vx)
        # canvas_channel = voxel_mean.size(1)
        batch_size = pts_coors[-1, 0].int() + 1
        canvas_len = canvas_z * canvas_y * canvas_x * batch_size
        # Create the canvas for this sample
        canvas = voxel_mean.new_zeros(canvas_len, dtype=torch.long)
        # Only include non-empty pillars
        indices = (
            voxel_coors[:, 0] * canvas_z * canvas_y * canvas_x +
            voxel_coors[:, 1] * canvas_y * canvas_x +
            voxel_coors[:, 2] * canvas_x + voxel_coors[:, 3])
        # Scatter the blob back to the canvas
        canvas[indices.long()] = torch.arange(
            start=0, end=voxel_mean.size(0), device=voxel_mean.device)

        # Step 2: get voxel mean for each point
        voxel_index = (
            pts_coors[:, 0] * canvas_z * canvas_y * canvas_x +
            pts_coors[:, 1] * canvas_y * canvas_x +
            pts_coors[:, 2] * canvas_x + pts_coors[:, 3])
        voxel_inds = canvas[voxel_index.long()]
        center_per_point = voxel_mean[voxel_inds, ...]
        return center_per_point

    # if out_fp16=True, the large numbers of points 
    # lead to overflow error in following layers
    @force_fp32(out_fp16=False)
    def forward(self,
                features,
                coors,
                points=None,
                img_feats=None,
                img_metas=None):
        """Forward functions.

        Args:
            features (torch.Tensor): Features of voxels, shape is NxC.
            coors (torch.Tensor): Coordinates of voxels, shape is  Nx(1+NDim).
            points (list[torch.Tensor], optional): Raw points used to guide the
                multi-modality fusion. Defaults to None.
            img_feats (list[torch.Tensor], optional): Image fetures used for
                multi-modality fusion. Defaults to None.
            img_metas (dict, optional): [description]. Defaults to None.

        Returns:
            tuple: If `return_point_feats` is False, returns voxel features and
                its coordinates. If `return_point_feats` is True, returns
                feature of each points inside voxels.
        """
        features_ls = [features]
        origin_point_coors = features[:, :3]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            voxel_mean, mean_coors = self.cluster_scatter(features, coors)
            points_mean = self.map_voxel_center_to_point(
                coors, voxel_mean, mean_coors)
            # TODO: maybe also do cluster for reflectivity
            f_cluster = features[:, :3] - points_mean[:, :3]
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        if self._with_voxel_center:
            f_center = features.new_zeros(size=(features.size(0), 3))
            f_center[:, 0] = features[:, 0] - (
                coors[:, 3].type_as(features) * self.vx + self.x_offset)
            f_center[:, 1] = features[:, 1] - (
                coors[:, 2].type_as(features) * self.vy + self.y_offset)
            f_center[:, 2] = features[:, 2] - (
                coors[:, 1].type_as(features) * self.vz + self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :3], 2, 1, keepdim=True)
            features_ls.append(points_dist)


        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)

        low_level_point_feature = features
        for i, vfe in enumerate(self.vfe_layers):
            point_feats = vfe(features)

            if (i == len(self.vfe_layers) - 1 and self.fusion_layer is not None
                    and img_feats is not None):
                point_feats = self.fusion_layer(img_feats, points, point_feats,
                                                img_metas)
            voxel_feats, voxel_coors = self.vfe_scatter(point_feats, coors)
            if i != len(self.vfe_layers) - 1:
                # need to concat voxel feats if it is not the last vfe
                feat_per_point = self.map_voxel_center_to_point(
                    coors, voxel_feats, voxel_coors)
                features = torch.cat([point_feats, feat_per_point], dim=1)
        if self.return_point_feats:
            return point_feats
        return voxel_feats, voxel_coors



@VOXEL_ENCODERS.register_module()
class DynamicScatterVFE(DynamicVFE):
    """ Same with DynamicVFE but use torch_scatter to avoid construct canvas in map_voxel_center_to_point.
    The canvas is very memory-consuming when use tiny voxel size (5cm * 5cm * 5cm) in large 3D space.
    """

    def __init__(self,
                 in_channels=4,
                 feat_channels=[],
                 with_distance=False,
                 with_cluster_center=False,
                 with_voxel_center=False,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max',
                 fusion_layer=None,
                 return_point_feats=False,
                 return_inv=True,
                 rel_dist_scaler=1.0,
                 unique_once=False,
                 ):
        super(DynamicScatterVFE, self).__init__(
            in_channels,
            feat_channels,
            with_distance,
            with_cluster_center,
            with_voxel_center,
            voxel_size,
            point_cloud_range,
            norm_cfg,
            mode,
            fusion_layer,
            return_point_feats,
        )
        # overwrite
        self.scatter = None
        self.vfe_scatter = None
        self.cluster_scatter = None
        self.rel_dist_scaler = rel_dist_scaler
        self.mode = mode
        self.unique_once = unique_once

    def map_voxel_center_to_point(self, voxel_mean, voxel2point_inds):

        return voxel_mean[voxel2point_inds]

    # if out_fp16=True, the large numbers of points 
    # lead to overflow error in following layers
    @force_fp32(out_fp16=False)
    def forward(self,
                features,
                coors,
                points=None,
                img_feats=None,
                img_metas=None,
                return_inv=False):
            
        if self.unique_once:
            new_coors, unq_inv_once = torch.unique(coors, return_inverse=True, return_counts=False, dim=0)
        else:
            new_coors = unq_inv_once = None

        features_ls = [features[:, :self.in_pts_dim]]
        origin_point_coors = features[:, :3]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            voxel_mean, _, unq_inv = scatter_v2(features[:, :3], coors, mode='avg', new_coors=new_coors, unq_inv=unq_inv_once)
            points_mean = self.map_voxel_center_to_point(voxel_mean, unq_inv)
            # TODO: maybe also do cluster for reflectivity
            f_cluster = features[:, :3] - points_mean[:, :3]
            features_ls.append(f_cluster / self.rel_dist_scaler)

        # Find distance of x, y, and z from pillar center
        if self._with_voxel_center:
            f_center = features.new_zeros(size=(features.size(0), 3))
            f_center[:, 0] = features[:, 0] - (
                coors[:, 3].type_as(features) * self.vx + self.x_offset)
            f_center[:, 1] = features[:, 1] - (
                coors[:, 2].type_as(features) * self.vy + self.y_offset)
            f_center[:, 2] = features[:, 2] - (
                coors[:, 1].type_as(features) * self.vz + self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :3], 2, 1, keepdim=True)
            features_ls.append(points_dist)


        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)

        for i, vfe in enumerate(self.vfe_layers):
            point_feats = vfe(features)

            if (i == len(self.vfe_layers) - 1 and self.fusion_layer is not None
                    and img_feats is not None):
                point_feats = self.fusion_layer(img_feats, points, point_feats,
                                                img_metas)
            voxel_feats, voxel_coors, unq_inv = scatter_v2(point_feats, coors, mode=self.mode, new_coors=new_coors, unq_inv=unq_inv_once)
            if i != len(self.vfe_layers) - 1:
                # need to concat voxel feats if it is not the last vfe
                feat_per_point = self.map_voxel_center_to_point(voxel_feats, unq_inv)
                features = torch.cat([point_feats, feat_per_point], dim=1)
        if self.return_point_feats:
            return point_feats

        if return_inv:
            return voxel_feats, voxel_coors, unq_inv
        else:
            return voxel_feats, voxel_coors



class SIRLayer(DynamicVFE):

    def __init__(self,
                 in_channels=4,
                 feat_channels=[],
                 with_distance=False,
                 with_cluster_center=False,
                 with_rel_mlp=True,
                 rel_mlp_hidden_dims=[16,],
                 rel_mlp_in_channel=3,
                 with_voxel_center=False,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max',
                 fusion_layer=None,
                 return_point_feats=False,
                 return_inv=True,
                 rel_dist_scaler=1.0,
                 with_shortcut=True,
                 xyz_normalizer=[1.0, 1.0, 1.0],
                 act='relu',
                 dropout=0.0,
                 ):
        super().__init__(
            in_channels,
            feat_channels,
            with_distance,
            with_cluster_center,
            with_voxel_center,
            voxel_size,
            point_cloud_range,
            norm_cfg,
            mode,
            fusion_layer,
            return_point_feats,
        )
        # overwrite
        self.scatter = None
        self.vfe_scatter = None
        self.cluster_scatter = None
        self.rel_dist_scaler = rel_dist_scaler
        self.mode = mode
        self.with_shortcut = with_shortcut
        self._with_rel_mlp = with_rel_mlp
        self.xyz_normalizer = xyz_normalizer
        if with_rel_mlp:
            rel_mlp_hidden_dims.append(in_channels) # not self.in_channels
            self.rel_mlp = build_mlp(rel_mlp_in_channel, rel_mlp_hidden_dims, norm_cfg, act=act)

        if act != 'relu' or dropout > 0: # do not double in_filter
            feat_channels = [self.in_channels] + list(feat_channels)
            vfe_layers = []
            for i in range(len(feat_channels) - 1):
                in_filters = feat_channels[i]
                out_filters = feat_channels[i + 1]
                if i > 0:
                    in_filters *= 2

                vfe_layers.append(
                    DynamicVFELayerV2(
                        in_filters,
                        out_filters,
                        norm_cfg,
                        act=act,
                        dropout=dropout,
                    )
                )
            self.vfe_layers = nn.ModuleList(vfe_layers)
            self.num_vfe = len(vfe_layers)
        

    def map_voxel_center_to_point(self, voxel_mean, voxel2point_inds):

        return voxel_mean[voxel2point_inds]

    # if out_fp16=True, the large numbers of points 
    # lead to overflow error in following layers
    @force_fp32(out_fp16=False)
    def forward(self,
                features,
                coors,
                f_cluster=None,
                points=None,
                img_feats=None,
                img_metas=None,
                return_inv=False,
                return_both=False,
                unq_inv_once=None,
                new_coors_once=None,
        ):

        xyz_normalizer = torch.tensor(self.xyz_normalizer, device=features.device, dtype=features.dtype)
        features_ls = [torch.cat([features[:, :3] / xyz_normalizer[None, :], features[:, 3:]], dim=1)]
        # origin_point_coors = features[:, :3]
        if self.with_shortcut:
            shortcut = features[:, 3:]
        if f_cluster is None:
            # Find distance of x, y, and z from cluster center
            voxel_mean, mean_coors, unq_inv = scatter_v2(features[:, :3], coors, mode='avg', unq_inv=unq_inv_once, new_coors=new_coors_once)
            points_mean = self.map_voxel_center_to_point(
                voxel_mean, unq_inv)
            # TODO: maybe also do cluster for reflectivity
            f_cluster = (features[:, :3] - points_mean[:, :3]) / self.rel_dist_scaler
        else:
            f_cluster = f_cluster / self.rel_dist_scaler

        if self._with_cluster_center:
            features_ls.append(f_cluster / 10.0)

        if self._with_rel_mlp:
            features_ls[0] = features_ls[0] * self.rel_mlp(f_cluster)

        if self._with_distance:
            points_dist = torch.norm(features[:, :3], 2, 1, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)

        voxel_feats_list = []
        for i, vfe in enumerate(self.vfe_layers):
            point_feats = vfe(features)

            voxel_feats, voxel_coors, unq_inv = scatter_v2(point_feats, coors, mode=self.mode, unq_inv=unq_inv_once, new_coors=new_coors_once)
            voxel_feats_list.append(voxel_feats)
            if i != len(self.vfe_layers) - 1:
                # need to concat voxel feats if it is not the last vfe
                feat_per_point = self.map_voxel_center_to_point(voxel_feats, unq_inv)
                features = torch.cat([point_feats, feat_per_point], dim=1)

        voxel_feats = torch.cat(voxel_feats_list, dim=1)

        if return_both:
            if self.with_shortcut and point_feats.shape == shortcut.shape:
                point_feats = point_feats + shortcut
            return point_feats, voxel_feats, voxel_coors

        if self.return_point_feats:
            if self.with_shortcut and point_feats.shape == shortcut.shape:
                point_feats = point_feats + shortcut
            return point_feats, voxel_feats

        if return_inv:
            return voxel_feats, voxel_coors, unq_inv
        else:
            return voxel_feats, voxel_coors
