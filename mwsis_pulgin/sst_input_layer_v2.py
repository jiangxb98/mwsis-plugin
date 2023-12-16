# move the computation of position embeding and mask in middle_encoder_layer

from torch import nn
from mmcv.runner import auto_fp16

from mmdet3d.models.middle_encoders import SparseUNet
from mmdet3d.models.builder import MIDDLE_ENCODERS, BACKBONES


from mmdet3d.ops.spconv import IS_SPCONV2_AVAILABLE
if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor, SparseSequential
    print('\n **********import spconv.pytorch')
else:
    raise NotImplementedError("ERRRPR: MMCV SparseConvTensor is low efficiency")
    from mmcv.ops import SparseConvTensor, SparseSequential
    print('\n **********import mmcv.spconv')



@MIDDLE_ENCODERS.register_module()
class PseudoMiddleEncoderForSpconvFSD(nn.Module):

    def __init__(self,):
        super().__init__()

    @auto_fp16(apply_to=('voxel_feat', ))
    def forward(self, voxel_feats, voxel_coors, batch_size=None):
        '''
        Args:
            voxel_feats: shape=[N, C], N is the voxel num in the batch.
            coors: shape=[N, 4], [b, z, y, x]
        Returns:
            feat_3d_dict: contains region features (feat_3d) of each region batching level. Shape of feat_3d is [num_windows, num_max_tokens, C].
            flat2win_inds_list: two dict containing transformation information for non-shifted grouping and shifted grouping, respectively. The two dicts are used in function flat2window and window2flat.
            voxel_info: dict containing extra information of each voxel for usage in the backbone.
        '''

        voxel_info = {}
        voxel_info['voxel_feats'] = voxel_feats
        voxel_info['voxel_coors'] = voxel_coors

        return voxel_info


@BACKBONES.register_module()
class SimpleSparseUNet(SparseUNet):
    r""" A simpler SparseUNet, removing the densify part
    """

    def __init__(self,
                 in_channels,
                 sparse_shape,
                 order=('conv', 'norm', 'act'),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 base_channels=16,
                 output_channels=128,
                 ndim=3,
                 encoder_channels=((16, ), (32, 32, 32), (64, 64, 64), (64, 64,
                                                                        64)),
                 encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1,
                                                                 1)),
                 decoder_channels=((64, 64, 64), (64, 64, 32), (32, 32, 16),
                                   (16, 16, 16)),
                 decoder_paddings=((1, 0), (1, 0), (0, 0), (0, 1)),
                 keep_coors_dims=None,
                 act_type='relu',
                 init_cfg=None):
        super().__init__(
            in_channels=in_channels,
            sparse_shape=sparse_shape,
            order=order,
            norm_cfg=norm_cfg,
            base_channels=base_channels,
            output_channels=output_channels,
            encoder_channels=encoder_channels,
            encoder_paddings=encoder_paddings,
            decoder_channels=decoder_channels,
            decoder_paddings=decoder_paddings,
            # ndim=ndim,
            # act_type=act_type,
            init_cfg=init_cfg,
        )
        self.conv_out = None # override
        self.ndim = ndim
        self.keep_coors_dims = keep_coors_dims

    @auto_fp16(apply_to=('voxel_features', ))
    def forward(self, voxel_info):
        """Forward of SparseUNet.

        Args:
            voxel_features (torch.float32): Voxel features in shape [N, C].
            coors (torch.int32): Coordinates in shape [N, 4],
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size.

        Returns:
            dict[str, torch.Tensor]: Backbone features.
        """
        coors = voxel_info['voxel_coors']
        if self.ndim == 2:
            assert (coors[:, 1] == 0).all()
            coors = coors[:, [0, 2, 3]] # remove the z-axis indices
        if self.keep_coors_dims is not None:
            coors = coors[:, self.keep_coors_dims]
        voxel_features = voxel_info['voxel_feats']
        coors = coors.int()
        batch_size = coors[:, 0].max().item() + 1
        input_sp_tensor = SparseConvTensor(voxel_features, coors,
                                                  self.sparse_shape,
                                                  batch_size)
        x = self.conv_input(input_sp_tensor)

        encode_features = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encode_features.append(x)

        x = encode_features[-1]
        for i in range(self.stage_num, 0, -1):
            x = self.decoder_layer_forward(encode_features[i - 1], x,
                                           getattr(self, f'lateral_layer{i}'),
                                           getattr(self, f'merge_layer{i}'),
                                           getattr(self, f'upsample_layer{i}'))
            # decode_features.append(x)

        seg_features = x.features
        ret = {'voxel_feats':x.features}
        ret = [ret,] # keep consistent with SSTv2

        return ret
