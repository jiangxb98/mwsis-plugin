import math
import random
from ...utils import ground_segmentation, calculate_ground
import torch
import numpy as np
from scipy.sparse.csgraph import connected_components  # CCL
from mmcv.utils import build_from_cfg
from mmdet3d.core.bbox import box_np_ops, Coord3DMode, Box3DMode
from mmdet3d.datasets.builder import PIPELINES
from mmdet3d.datasets.builder import OBJECTSAMPLERS
from mmdet3d.datasets.pipelines import RandomFlip3D
from mmdet.core import BitmapMasks, PolygonMasks
from mmdet3d.core.points import get_points_type, BasePoints
import warnings
import mmcv
from PIL import Image
from skimage.util.shape import view_as_windows
from torchvision.transforms.transforms import ColorJitter
import time
import torch
from deploy3d.symfun.ops.ccl import VoxelSPCCL3D, voxel_spccl3d
import pickle
import os
from io import BytesIO
from pycocotools import mask as mask_utils
import copy

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

@PIPELINES.register_module(force=True)
class ObjectSample(object):
    def __init__(self, db_sampler, sample_2d=False, use_ground_plane=False):
        self.sampler_cfg = db_sampler
        self.sample_2d = sample_2d
        if 'type' not in db_sampler.keys():
            db_sampler['type'] = 'DataBaseSampler'
        self.db_sampler = build_from_cfg(db_sampler, OBJECTSAMPLERS)
        self.use_ground_plane = use_ground_plane

    @staticmethod
    def remove_points_in_boxes(points, boxes):
        masks = box_np_ops.points_in_rbbox(points.coord.numpy(), boxes)
        points = points[np.logical_not(masks.any(-1))]
        return points

    def __call__(self, input_dict):
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']
        if self.sample_2d:
            img = input_dict['img']
            gt_bboxes_2d = input_dict['gt_bboxes']

        if self.use_ground_plane and 'plane' in input_dict['ann_info']:
            ground_plane = input_dict['ann_info']['plane']
            input_dict['plane'] = ground_plane
        # change to float for blending operation
        points = input_dict['points']
        sampled_dict = self.db_sampler.sample_all(input_dict, self.sample_2d)
        if sampled_dict is not None:
            sampled_gt_bboxes_3d = sampled_dict['gt_bboxes_3d']
            sampled_points = sampled_dict['points']
            sampled_gt_labels = sampled_dict['gt_labels_3d']

            gt_labels_3d = np.concatenate([gt_labels_3d, sampled_gt_labels],
                                          axis=0)
            gt_bboxes_3d = gt_bboxes_3d.new_box(
                np.concatenate(
                    [gt_bboxes_3d.tensor.numpy(), sampled_gt_bboxes_3d]))

            points = self.remove_points_in_boxes(points, sampled_gt_bboxes_3d)
            # check the points dimension
            points = points.cat([sampled_points, points])

            if self.sample_2d:
                sampled_gt_bboxes_2d = sampled_dict['gt_bboxes_2d']
                gt_bboxes_2d = np.concatenate(
                    [gt_bboxes_2d, sampled_gt_bboxes_2d]).astype(np.float32)

                input_dict['gt_bboxes'] = gt_bboxes_2d
                input_dict['img'] = sampled_dict['img']

        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d.astype(np.int64)
        input_dict['points'] = points

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f' sample_2d={self.sample_2d},'
        repr_str += f' data_root={self.sampler_cfg.data_root},'
        repr_str += f' info_path={self.sampler_cfg.info_path},'
        repr_str += f' rate={self.sampler_cfg.rate},'
        repr_str += f' prepare={self.sampler_cfg.prepare},'
        repr_str += f' classes={self.sampler_cfg.classes},'
        repr_str += f' sample_groups={self.sampler_cfg.sample_groups}'
        return repr_str

@PIPELINES.register_module(force=True)
class MyFilterBoxWithMinimumPointsCount:
    def __init__(self, num_points=1):
        self.num_points = num_points

    def __call__(self, input_dict):
        points = input_dict['points'].convert_to(Coord3DMode.LIDAR)
        gt_boxes_lidar = input_dict['gt_bboxes_3d'].convert_to(Box3DMode.LIDAR)
        points_xyz = points.coord.numpy()
        gt_boxes_lidar = gt_boxes_lidar.tensor[:, :7].numpy()
        indices = box_np_ops.points_in_rbbox(points_xyz, gt_boxes_lidar)
        mask = (np.count_nonzero(indices, axis=0) >= self.num_points)
        input_dict['gt_bboxes_3d'] = input_dict['gt_bboxes_3d'][
            torch.from_numpy(mask)]
        input_dict['gt_labels_3d'] = input_dict['gt_labels_3d'][mask]
        return input_dict


@PIPELINES.register_module(force=True)
class PadMultiViewImage:
    """Pad the image & masks & segmentation map.

    Args:
        size is list[(tuple, optional)]: Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_to_square (bool): Whether to pad the image into a square.
            Currently only used for YOLOX. Default: False.
        pad_val (dict, optional): A dict for padding value, the default
            value is `dict(img=0, masks=0, seg=255)`.
    """

    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_to_square=False,
                 pad_val=dict(img=0, masks=0, seg=-1)):
        self.size = size
        self.size_divisor = size_divisor
        if isinstance(pad_val, float) or isinstance(pad_val, int):
            warnings.warn(
                'pad_val of float type is deprecated now, '
                f'please use pad_val=dict(img={pad_val}, '
                f'masks={pad_val}, seg=255) instead.', DeprecationWarning)
            pad_val = dict(img=pad_val, masks=pad_val, seg=255)
        assert isinstance(pad_val, dict)
        self.pad_val = pad_val
        self.pad_to_square = pad_to_square

        if pad_to_square:
            assert size is None and size_divisor is None, \
                'The size and size_divisor must be None ' \
                'when pad2square is True'
        else:
            assert size is not None or size_divisor is not None, \
                'only one of size and size_divisor should be valid'
            assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        pad_val = self.pad_val.get('img', 0)
        for key in results.get('img_fields', ['img']):
            for i in range(len(results[key])):
                if self.size[i] == results[key][i].shape[:2]:
                    continue
                if self.pad_to_square:
                    max_size = max(results[key][i].shape[:2])
                    self.size = (max_size, max_size)
                if self.size is not None:
                    # if only pad top. del: shape=, add parm: padding=(0,1280-886,0,0)
                    padded_img = mmcv.impad(
                        results[key][i], shape=self.size[i][:2], pad_val=pad_val)
                elif self.size_divisor is not None:
                    padded_img = mmcv.impad_to_multiple(
                        results[key][i], self.size_divisor, pad_val=pad_val)
                # Image.fromarray(np.uint8(results[key][i])).save("ori_img_{}.jpeg".format(i))
                # Image.fromarray(np.uint8(padded_img)).save("pad_img_{}.jpeg".format(i))
                results[key][i] = padded_img
        results['pad_shape'] = [padded_img.shape for padded_img in results[key]]
        # results['img_shape'] = results['pad_shape']
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor


    def _pad_masks(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        pad_val = self.pad_val.get('masks', 0)
        for key in results.get('mask_fields', []):
            for i in range(len(results['img'])):
                if self.size[i] == results[key][i].masks.shape[1:3]:
                    continue
                pad_shape = results['pad_shape'][i][:2]
                results[key][i] = results[key][i].pad(pad_shape, pad_val=pad_val)


    def _pad_seg(self, results):
        """Pad semantic segmentation map according to
        ``results['pad_shape']``."""
        pad_val = self.pad_val.get('seg', -1)
        for key in results.get('seg_fields', []):
            for i in range(len(results['img'])):
                if self.size[i] == results[key][i].shape:
                    continue
                results[key][i] = mmcv.impad(results[key][i], shape=results['pad_shape'][i][:2], pad_val=pad_val)

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        self._pad_masks(results)
        self._pad_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_to_square={self.pad_to_square}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str

@PIPELINES.register_module(force=True)
class ResizeMultiViewImage:
    """Resize images & bbox & mask.
    
    Args:
        results['img'] is a list or array, to resize fixed size 
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 keep_ratio=True,
                 bbox_clip_border=True,
                 backend='cv2',
                 interpolation='bilinear',):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        self.backend = backend
        self.multiscale_mode = multiscale_mode
        self.keep_ratio = keep_ratio
        # TODO: refactor the override option in Resize
        self.interpolation = interpolation
        self.bbox_clip_border = bbox_clip_border

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``, \
                where ``img_scale`` is the selected image scale and \
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        for key in results.get('img_fields', ['img']):
            multi_view_img = []
            multi_scale_factor = []
            for i in range(len(results['img'])):
                if self.keep_ratio:
                    img, scale_factor = mmcv.imrescale(
                        results['img'][i],
                        results['scale'][i],
                        return_scale=True,
                        interpolation=self.interpolation,
                        backend=self.backend)
                    # the w_scale and h_scale has minor difference
                    # a real fix should be done in the mmcv.imrescale in the future
                    new_h, new_w = img.shape[:2]
                    h, w = results[key][i].shape[:2]
                    w_scale = new_w / w
                    h_scale = new_h / h
                else:
                    img, w_scale, h_scale = mmcv.imresize(
                        results['img'][i],
                        results['scale'][i],
                        return_scale=True,
                        interpolation=self.interpolation,
                        backend=self.backend)
                multi_view_img.append(img)
                multi_scale_factor.append(np.array([w_scale, h_scale, w_scale, h_scale],
                        dtype=np.float32))
                
            results[key] = multi_view_img
            scale_factor = multi_scale_factor

            results['img_shape'] = [img.shape for img in multi_view_img]
            # in case that there is no padding
            results['pad_shape'] = [img.shape for img in multi_view_img]
            results['scale_factor'] = scale_factor
            results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        for key in results.get('bbox_fields', []):
            multi_gt_bboxes = []
            for i in range(len(results['img'])):
                # if None
                # if (results[key][i] == np.array([-1,-1,-1,-1])).all():
                if len(results[key][i]) == 0:
                    multi_gt_bboxes.append(np.array([]))
                # if  Not None
                else:
                    bboxes = results[key][i] * results['scale_factor'][i]
                    if self.bbox_clip_border:
                        img_shape = results['img_shape'][i]
                        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
                    multi_gt_bboxes.append(bboxes)

            results[key] = multi_gt_bboxes

    def _resize_masks(self, results):
        """Resize masks with ``results['scale']``"""
        for key in results.get('mask_fields', []):
            if results[key] is None:
                continue
            results['ori_gt_masks'] = copy.deepcopy(results[key])
            for i in range(len(results['img'])):
                # !!! both rescale and resize both are BitmapMasks(), not a np.array
                if self.keep_ratio:
                    # to contrast origin of transforms
                    results[key][i] = results[key][i].rescale(results['scale'][i], interpolation='bilinear')
                else:
                    new_shape=results['img_shape'][i][:2]
                    results[key][i] = results[key][i].resize(new_shape)

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            multi_seg = []
            for i in range(len(results['img'])):
                if self.keep_ratio:
                    gt_seg = mmcv.imrescale(
                        results[key][i],
                        results['scale'][i],
                        interpolation='nearest',
                        backend=self.backend)
                else:
                    gt_seg = mmcv.imresize(
                        results[key][i],
                        results['scale'][i],
                        interpolation='nearest',
                        backend=self.backend)
                # if results['scale'][i][::-1] == results[key][i].shape, the gt_seg=results[key][i]
                multi_seg.append(gt_seg)

            results[key] = multi_seg

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """
        results['scale'] = self.img_scale
        if self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
            results['scale'] = [scale for _ in range(len(results['img']))]
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'multiscale_mode={self.multiscale_mode}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str

# No Test
@PIPELINES.register_module(force=True)
class MultiViewCrop:
    def __init__(self,
                crop_size=(0.3, 0.7),
                crop_type='relative_range',  # relative absolute
                bbox_clip_border=True,
                recompute_bbox=False):
        self.crop_size = crop_size
        self.bbox_clip_border = bbox_clip_border
        self.crop_type = crop_type
        self.recompute_bbox = recompute_bbox
        # The key correspondence from bboxes to labels and masks.
        self.bbox2label = {
            'gt_bboxes': 'gt_labels',
        }
        self.bbox2mask = {
            'gt_bboxes': 'gt_masks',
        }
    
    def _crop_data(self, results, crop_size):
        assert crop_size[0][0] > 0 and crop_size[0][1] > 0
        img_shape_crop = []
        img_crop = []
        crop_coords = []
        offset_wh = []
        bbox_crop = []
        sem_crop = []
        valid_inds_ = []
        for key in results.get('img_fields', ['img']):
            imgs = results['img'] # mutli images = 5
            for i in range(len(results['img'])):
                margin_h = max(imgs[i].shape[0] - crop_size[i][0], 0)
                margin_w = max(imgs[i].shape[1] - crop_size[i][1], 0)
                offset_h = np.random.randint(0, margin_h + 1)
                offset_w = np.random.randint(0, margin_w + 1)
                crop_y1, crop_y2 = offset_h, offset_h + crop_size[i][0]
                crop_x1, crop_x2 = offset_w, offset_w + crop_size[i][1]
                # crop the image
                img = imgs[i][crop_y1:crop_y2, crop_x1:crop_x2, ...]
                img_crop.append(img)
                img_shape_crop.append(img.shape)
                crop_coords.append([crop_x1, crop_y1, crop_x2, crop_y2])
                offset_wh.append(np.array([offset_w, offset_h, offset_w, offset_h],
                                dtype=np.float32))
            results[key] = img_crop
        results['img_shape'] = img_shape_crop

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            # e.g. gt_bboxes
            for i in range(len(imgs)):
                img_shape = results['img_shape'][i]
                # if gt_bboxes is None, skip this image, continue
                if  len(results[key][i])==0 and key == 'gt_bboxes':
                    bbox_crop.append(np.array([]))
                    valid_inds_.append(np.array([]))
                    continue
                bboxes = results[key][i] - offset_wh[i]
                if self.bbox_clip_border:
                    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
                # If the crop does not contain any gt-bbox area, skip continue
                valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1])
                valid_inds_.append(valid_inds)
                if key == 'gt_bboxes' and not valid_inds.any():
                    bbox_crop.append(np.array([]))
                else:
                    bbox_crop.append(bboxes[valid_inds])
                # label fields. e.g. gt_labels and gt_labels_ignore
                label_key = self.bbox2label.get(key)
                if label_key in results:
                    if not valid_inds.any():
                        results[label_key][i] = np.array([])
                    else:
                        results[label_key][i] = results[label_key][i][valid_inds]
            results[key] = bbox_crop
        results['crop_coords'] = crop_coords
        # crop semantic seg
        for key in results.get('seg_fields', []):
            for i in range(len(imgs)):
                crop_x1, crop_y1, crop_x2, crop_y2 = crop_coords[i]
                sem_crop.append(results[key][i][crop_y1:crop_y2, crop_x1:crop_x2])
            results[key] = sem_crop

        # mask fields
        for key in results.get('mask_fields',[]):
            for i in range(len(imgs)):
                crop_x1, crop_y1, crop_x2, crop_y2 = crop_coords[i]
                valid_inds = valid_inds_[i]
                if len(valid_inds) == 0:
                    continue
                results[key][i] = results[key][i][valid_inds.nonzero()[0]].crop(np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))
                assert len(results[key][i]) == len(results['gt_bboxes'][i])
                if self.recompute_bbox:
                    results['gt_bboxes'][i] = results[key][i].get_bboxes()
        return results

    def _get_crop_size(self, image_shape):
        out_crop_size = []
        for i in range(len(image_shape)):
            h, w = image_shape[i][:2]
            if self.crop_type == "relative":
                crop_h, crop_w = self.crop_size
                out_crop_size.append((int(h * crop_h ), int(w * crop_w)))
            elif self.crop_type == "relative_range":
                crop_size = np.asarray(self.crop_size, dtype=np.float32)
                crop_h, crop_w = crop_size + np.random.rand(2) * (1-crop_size)
                out_crop_size.append((int(h * crop_h), int(w * crop_w)))
            else:
                out_crop_size.append((h, w))
        return out_crop_size

    def __call__(self, results):
        image_shape = results['img_shape']
        crop_size = self._get_crop_size(image_shape)
        results['crop_shape'] = crop_size
        results = self._crop_data(results, crop_size)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(crop_size={self.crop_size}, '
        repr_str += f'crop_type={self.crop_type}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str

@PIPELINES.register_module(force=True)
class NormalizeMultiViewImage:
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """
    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        for key in results.get('img_fields', ['img']):
            for i in range(len(results[key])):
                results[key][i] = mmcv.imnormalize(results[key][i], self.mean, self.std,
                                                self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module(force=True)
class FilterAndReLabel:
    '''filter and relabel seg label of 2d and 3d

    - filter_class_name: the class to use
    - with_mask: True or False
    - with_seg: True or False
    - with_mask_3d: True or False
    - with_seg_3d: True or False
    '''
    def __init__(self,
                filter_class_name=None,
                with_mask=True,
                with_seg=True,
                with_mask_3d=True,
                with_seg_3d=True,
                is_filter_small_object=True,
                *arg, **kwargs):
        self.filter_class_name = filter_class_name
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.with_mask_3d = with_mask_3d
        self.with_seg_3d = with_seg_3d
        seg_class = {'Car':[2,3,4,5,7,8,11], 'Pedestrian':[9], 'Cyclist':[10]}#'Car':[2,3,4,5,7,8,11] 1=ego car 6=bicycle 10=cyclist
        seg_3d_class = {'Car':[1,2,3,4,5,13], 'Pedestrian':[7], 'Cyclist':[6]}#'Car':[1,2,3,4,5,13] 6=bicyclist 12=bicycle
        self.seg_class = [val for key, val in seg_class.items() if key in filter_class_name]
        self.seg_3d_class = [val for key, val in seg_3d_class.items() if key in filter_class_name]
        self.is_filter_small_object = is_filter_small_object

    def _get_single_annos(self, results):
        if self.with_seg:
            semantic_seg = results['gt_semantic_seg'][0].squeeze()
            tmp = np.zeros(semantic_seg.shape)
            for j in range(len(self.seg_class)):
                for k in range(len(self.seg_class[j])):
                    tmp += np.where(semantic_seg==self.seg_class[j][k], j+1, 0)
            results['gt_semantic_seg'] = tmp - 1
        if self.with_mask:
            mask_gt_bboxes = []
            mask_gt_labels = []
            seg_ind = (tmp-1 >= 0)  # True or False array
            new_masks = results['gt_masks'][0].masks * seg_ind  # new_mask shape=(1,1280,1920)
            new_masks = new_masks.squeeze(0)
            h, w = new_masks.shape
            out_masks = []
            mask_sets, counts = np.unique(new_masks, return_counts=True)
            for j in range(len(mask_sets)):
                tmp_mask = np.ones((h, w), dtype=new_masks.dtype)
                if mask_sets[j] == 0:
                    continue
                if self.is_filter_small_object and counts[j] <= 200:
                    continue
                else:
                    tmp_mask = tmp_mask * (new_masks==mask_sets[j])
                    out_masks.append(tmp_mask)
                    # get mask bbox
                    y0 = np.where(tmp_mask.sum(1)!=0)[0][0]
                    y1 = np.where(tmp_mask.sum(1)!=0)[0][-1]
                    x0 = np.where(tmp_mask.sum(0)!=0)[0][0]
                    x1 = np.where(tmp_mask.sum(0)!=0)[0][-1]
                    mask_gt_bboxes.append(np.array([x0, y0, x1+1, y1+1]).astype(np.float32))
                    # get label
                    labels = int((tmp-1)[tmp_mask.astype(np.bool)][0])  # int((tmp_mask * tmp-1).max())
                    mask_gt_labels.append(np.array([labels]).astype(np.int))

            if len(out_masks)==0:
                results['gt_masks'] = BitmapMasks(np.zeros((0, h, w)), h, w)  # np.array([]).astype(np.int)  # 
                results['gt_bboxes'] = np.array([]).astype(np.float32)
                results['gt_labels'] = np.array([]).astype(np.int)
                return None
            else:
                out_masks = np.stack(out_masks)
                results['gt_masks'] = BitmapMasks(out_masks, h, w)
                results['gt_bboxes'] = np.stack(mask_gt_bboxes)
                results['gt_labels'] = (np.concatenate(mask_gt_labels)).astype(np.int32)
                assert len(results['gt_bboxes']) == len(results['gt_labels'])
        return results

    def __call__(self, results):
        '''return the new semantic label, and instance only filtered
        - label: Car=0, Pedestrian=1, Cyclist=2
        - seg -1 is uncorrelated, instance only filtered
        '''
        # results['ori_gt_masks'] = results['gt_masks'].copy()
        if 'img' in results.keys():
            if isinstance(results['img'], np.ndarray):
                results = self._get_single_annos(results)
                return results
            
            img_num = len(results['img'])
            for i in range(img_num):
                if False:
                    import cv2
                    img = results['img'][i]
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    gt_bboxes = results['gt_bboxes'][i]
                    for b, gt_bbox in enumerate(gt_bboxes):
                        cv2.rectangle(img, (int(gt_bbox[0]), int(gt_bbox[1])), (int(gt_bbox[2]), int(gt_bbox[3])), (0, 255, 0), 2)
                    cv2.imwrite('img{}.jpg'.format(i), img)
                    cv2.imwrite('mask{}.jpg'.format(i), results['gt_masks'][i].masks.squeeze(0).astype(np.uint8)*20)
                if self.with_seg:
                    semantic_seg = results['gt_semantic_seg'][i].squeeze()
                    tmp = np.zeros(semantic_seg.shape)
                    for j in range(len(self.seg_class)):
                        for k in range(len(self.seg_class[j])):
                            tmp += np.where(semantic_seg==self.seg_class[j][k], j+1, 0)
                    results['gt_semantic_seg'][i] = tmp - 1
                if self.with_mask:
                    mask_gt_bboxes = []
                    mask_gt_labels = []
                    seg_ind = (tmp-1 >= 0)  # True or False array
                    new_masks = results['gt_masks'][i].masks * seg_ind  # new_mask shape=(1,1280,1920)
                    new_masks = new_masks.squeeze(0)
                    h, w = new_masks.shape
                    out_masks = []
                    mask_sets, counts = np.unique(new_masks, return_counts=True)
                    for j in range(len(mask_sets)):
                        tmp_mask = np.ones((h, w), dtype=new_masks.dtype)
                        if mask_sets[j] == 0:
                            continue
                        if self.is_filter_small_object and counts[j] <= 200:
                            continue
                        else:
                            tmp_mask = tmp_mask*(new_masks==mask_sets[j])
                            out_masks.append(tmp_mask)
                            # get mask bbox
                            y0 = np.where(tmp_mask.sum(1)!=0)[0][0]
                            y1 = np.where(tmp_mask.sum(1)!=0)[0][-1]
                            x0 = np.where(tmp_mask.sum(0)!=0)[0][0]
                            x1 = np.where(tmp_mask.sum(0)!=0)[0][-1]
                            mask_gt_bboxes.append(np.array([x0, y0, x1+1, y1+1]).astype(np.float32))
                            # get label
                            labels = int((tmp-1)[tmp_mask.astype(np.bool)][0])  # int((tmp-1)[tmp_mask.astype(np.bool)][0])
                            mask_gt_labels.append(np.array([labels]).astype(np.int))

                    if len(out_masks)==0:
                        # results['ori_gt_masks'][i] = BitmapMasks(np.zeros((0, h, w)), h, w)  # np.array([]).astype(np.int)  # 
                        results['gt_masks'][i] = BitmapMasks(np.zeros((0, h, w)), h, w)  # np.array([]).astype(np.int)  # 
                        results['gt_bboxes'][i] = np.array([]).astype(np.float32)
                        results['gt_labels'][i] = np.array([]).astype(np.int)
                    else:
                        out_masks = np.stack(out_masks)
                        results['gt_masks'][i] = BitmapMasks(out_masks, h, w)
                        results['gt_bboxes'][i] = np.stack(mask_gt_bboxes)
                        results['gt_labels'][i] = np.concatenate(mask_gt_labels)
                        # results['ori_gt_masks'][i] = BitmapMasks(out_masks, h, w)  # np.array([]).astype(np.int)  # 
                        assert len(results['gt_bboxes'][i]) == len(results['gt_labels'][i])

        if self.with_seg_3d:
            if 'pts_semantic_mask' in results.keys():
                semantic_seg_3d = results['pts_semantic_mask'].squeeze()
                tmp_3d = np.zeros(semantic_seg_3d.shape)
                for i in range(len(self.seg_3d_class)):
                    for k in range(len(self.seg_3d_class[i])):
                        tmp_3d += np.where(semantic_seg_3d==self.seg_3d_class[i][k], i+1, 0)
                results['pts_semantic_mask'] = tmp_3d - 1
        if self.with_mask_3d:
            if 'pts_instance_mask' in results.keys():
                seg_ind_3d = (tmp_3d-1 >= 0)
                results['pts_instance_mask'] = results['pts_instance_mask'] * seg_ind_3d

        return results


@PIPELINES.register_module(force=True)
class SampleFrameImage:
    def __init__(self,
                sample = 'random',
                guide = 'gt_bboxes',
                training = True):
        self.sample = sample
        self.guide = guide
        self.training = training
    
    def _random_sample(self, results):
        ''''each frame random select a image which has 2d gt_bboxes
        '''
        results['sample_img_id'] = []
        sample_image_id = 0
        if self.training:
            if self.guide == 'gt_bboxes':
                for i in range(len(results['gt_labels'])):
                    gt_label = results['gt_labels'][i]
                    if (gt_label==-1).all():
                        continue
                    else:
                        results['sample_img_id'].append(i)
                if len(results['sample_img_id']) != 0:
                    sample_image_id = random.choice(results['sample_img_id'])
                sample_image_id = 0  # To Test，only sample 0 img
                results['sample_img_id'] = sample_image_id
        else:
            sample_image_id = random.choice(range(5))
            sample_image_id = 0  # To Test，only sample 0 img
            results['sample_img_id'] = sample_image_id

        results['img'] = results['img'][sample_image_id]
        results['filename'] = results['filename'][sample_image_id]
        results['img_shape'] = results['img_shape'][sample_image_id]
        results['ori_shape'] = results['ori_shape'][sample_image_id]
        results['pad_shape'] = results['pad_shape'][sample_image_id]
        results['scale'] = results['scale'][sample_image_id]
        results['scale_factor'] = results['scale_factor'][sample_image_id]
        results['lidar2img'] = results['lidar2img'][sample_image_id]
        results['pad_fixed_size'] = results['pad_fixed_size'][sample_image_id]
        if 'gt_labels' in results.keys():
            results['gt_labels'] = results['gt_labels'][sample_image_id]
            if (results['gt_labels']==-1).all():
                results['gt_labels'] = np.array([])
        if 'gt_bboxes' in results.keys():
            results['gt_bboxes'] = results['gt_bboxes'][sample_image_id]
            if results.get('flip', False) and 'flip' in results.keys():
                results['ori_gt_bboxes'] = results['ori_gt_bboxes'][sample_image_id]
            if (results['gt_bboxes']==-1).all():
                results['gt_bboxes'] = np.array([])
                if results.get('flip', False) and 'flip' in results.keys():
                    results['ori_gt_bboxes'] = np.array([])
        if 'gt_masks' in results.keys():
            results['gt_masks'] = results['gt_masks'][sample_image_id]
        if 'gt_semantic_seg' in results.keys():
            results['gt_semantic_seg'] = results['gt_semantic_seg'][sample_image_id]
        results.update(dict(img_sample='random'))


        return results
        
    def _resample(self, results):
        pass

    def all_img(self, results):
        return results

    def __call__(self, results):

        if self.sample == 'random':
            results = self._random_sample(results)
        elif self.sample == 'resample':
            results = self._resample(results)
        else:
            return results
        return results

@PIPELINES.register_module(force=True)
class RemoveGroundPoints:
    # 已丢弃
    def __init__(self, coord_type, remove=False):
        self.coord_type = coord_type
        self.remove = remove

    def __call__(self, results):
        # points = results['points'].tensor.numpy()
        # # RANSAC remove ground points
        # if self.remove:
        #     ground_points, segment_points = ground_segmentation(points)
        #     ground_points[:, 5] = 0  # 这个位置已经是第几次回波，所以不能表示地面
        #     segment_points[:, 5] = 1
        #     new_points = np.concatenate((ground_points, segment_points), axis=0)
        # else:
            
        #     new_points = points
        # points_class = get_points_type(self.coord_type)
        # results['points'] = points_class(new_points, points_dim=new_points.shape[-1])  # 实例化，LiDARPoints
        return results

@PIPELINES.register_module(force=True)
class FilterPointsByImage:
    """
    The project point cloud is obtained by the Image idx
    """
    def __init__(self, coord_type, kernel_size=3, threshold_depth=0.5,
                dist=(0.6,0.1,0.4), training=True, 
                relative_threshold=0.91, use_run_seg=True,
                only_img=False,
                use_pseudo_label=False):
        self.coord_type = coord_type
        self.kernel_size = kernel_size
        self.threshold_depth = threshold_depth
        self.dist = dist
        self.training = training
        self.relative_threshold = relative_threshold
        self.use_run_seg = use_run_seg
        self.only_img = only_img
        self.use_pseudo_label = use_pseudo_label

    def _filter_points(self, results):
        points = results['points'].tensor.numpy()  # (N,14)
        top_mask = points[:,6]==0
        points = points[top_mask] # 只要主雷达 
        if results.get('pts_semantic_mask', None) is not None:
            results['pts_semantic_mask'] = results['pts_semantic_mask'][top_mask]
        if results.get('pts_instance_mask', None) is not None:
            results['pts_instance_mask'] = results['pts_instance_mask'][top_mask]
        sample_img_id = results['sample_img_id']
        in_mask = (points[:,10]==sample_img_id) | (points[:,11]==sample_img_id)
        out_mask = ~in_mask

        # in_img_points = points[mask]
        # out_img_points = points[out_mask]
        # x,y,z,thanh(r),e, bg/fg mask, out/in img, points_inds,x,y,r,g,b 13
        new_points = np.zeros(((points.shape[0]), 14)).astype(np.float32)
        new_points[:,0:11] = points[:, 0:11]  # x,y,z,r,e,return_id,lidar_idx,row,column
        mask1 = points[:,10] == sample_img_id
        new_points[:,12:14][mask1] = np.stack((points[mask1][:,12], points[mask1][:,14]),1)
        mask2 = points[:,11] == sample_img_id
        new_points[:,12:14][mask2] = np.stack((points[mask2][:,13], points[mask2][:,15]),1)
        
        new_points[:,10][in_mask] = 1
        new_points[:,10][out_mask] = 0
        new_points[:,11] = 0
        
        # 根据图片大小将点云对应的2D坐标缩放，如果改变了points的维度，记得修改

        new_points[:,12:14] = new_points[:,12:14] * results['scale_factor'][0]

        # To Test，只加载落在图片上的点云
        if results.get('pts_semantic_mask', None) is not None:
            results['pts_semantic_mask'] = results['pts_semantic_mask'][new_points[:,10]==1]
        if results.get('pts_instance_mask', None) is not None:
            results['pts_instance_mask'] = results['pts_instance_mask'][new_points[:,10]==1]
        new_points = new_points[new_points[:,10]==1]

        points_class = get_points_type(self.coord_type)
        results['points'] = points_class(new_points, points_dim=new_points.shape[-1])  # 实例化，LiDARPoints

        # 过滤pseudolabel
        if self.use_pseudo_label and 'pseudo_labels' in results.keys():
            pseudo_labels = results['pseudo_labels']
            pseudo_labels = np.stack(
                        [pseudo_labels['run_id'],
                         pseudo_labels['run'],
                         pseudo_labels['ignore'],
                         pseudo_labels['run_ccl'],
                         pseudo_labels['collision']],axis=-1) # 1 is collision(bg points)
            in_img_mask = (points[:,10]!=-1) | (points[:,11]!=-1)
            points = points[in_img_mask]
            assert len(points) == len(pseudo_labels)
            in_tmp_img = (points[:,10]==sample_img_id) | (points[:,11]==sample_img_id)
            results['pseudo_labels'] = pseudo_labels[in_tmp_img]

        return results

    def find_connected_componets_single_batch(self, points, dist):

        this_points = points
        dist_mat = this_points[:, None, :2] - this_points[None, :, :2] # only care about xy
        dist_mat = (dist_mat ** 2).sum(2) ** 0.5
        adj_mat = dist_mat < dist
        adj_mat = adj_mat
        c_inds = connected_components(adj_mat, directed=False)[1]

        return c_inds

    def get_in_2d_box_points(self, results):
        points_ = results['points'].tensor.numpy()
        # 在图片的点
        fg_mask = (points_[:,10]==1)
        
        gt_bboxes = results['gt_bboxes']
        labels = results['gt_labels']
        roi_points = []
        
        out_gt_bboxes = []
        out_gt_labels = []
        gt_mask_all = np.zeros((points_.shape[0])).astype(np.bool)
        gt_mask_list = []
        for i, gt_bbox in enumerate(gt_bboxes):    
            gt_mask = (((points_[:, 12] >= gt_bbox[0]) & (points_[:, 12] < gt_bbox[2])) &
                        ((points_[:, 13] >= gt_bbox[1]) & (points_[:, 13] < gt_bbox[3]) &
                        fg_mask))
            gt_mask_all = gt_mask | gt_mask_all
            in_box_points = points_[gt_mask]
            # 若2d框里没有点，那么roi_points为空
            out_gt_bboxes.append(gt_bbox)
            out_gt_labels.append(labels[i])
            roi_points.append(in_box_points)  # 10
            gt_mask_list.append(gt_mask)  # 保存当前roi的mask
        points_[:,11][gt_mask_all] = 1  # 在2D box内的点大部分是前景点，后面进行筛选

        if self.use_run_seg:
            for i in range(len(gt_mask_list)):
                if len(roi_points[i]) == 0:
                    continue
                run_sets, inv_inds, counts = np.unique(roi_points[i][:,14], return_inverse=True, return_counts=True)
                for j in range(len(run_sets)):
                    out_box_points = points_[~gt_mask_all]
                    prop = (out_box_points[:,14]==run_sets[j]).sum()/(points_[:,14]==run_sets[j]).sum()
                    # 这个有问题，取1/2高度，有些点的坐标差值很大导致1/2高度并不能区分开地面
                    height = (roi_points[i][:,2].max()+roi_points[i][:,2].min())/2
                    # 对于非主雷达的点云我们就默认
                    if run_sets[j] == -1:
                        points_[:,11][(gt_mask_list[i])&(points_[:,14]==run_sets[j])] = -1
                        roi_points[i][:,11][roi_points[i][:,14]==run_sets[j]] = -1
                    # 若2D框外的run seg点大于总run seg的0.5，那么一定为背景点 0
                    elif prop >= 0.5:
                        # points_[:,11][(gt_mask_list[i])&(points_[:,14]==run_sets[j])&(points_[:,2]<height)] = 0
                        # roi_points[i][:,11][(roi_points[i][:,14]==run_sets[j])&(roi_points[i][:,2]<height)] = 0
                        # # 高于1/2的忽略掉
                        # points_[:,11][(gt_mask_list[i])&(points_[:,14]==run_sets[j])&(points_[:,2]>=height)] = 0
                        # roi_points[i][:,11][(roi_points[i][:,14]==run_sets[j])&(roi_points[i][:,2]>=height)] = 0
                        points_[:,11][(gt_mask_list[i])&(points_[:,14]==run_sets[j])] = 0
                        roi_points[i][:,11][(roi_points[i][:,14]==run_sets[j])] = 0
                    # 若[0.1,0.5]，则ignore掉 -1
                    elif 0.5 > prop >= 0.05:
                        points_[:,11][(gt_mask_list[i])&(points_[:,14]==run_sets[j])] = -1
                        roi_points[i][:,11][(roi_points[i][:,14]==run_sets[j])] = -1
                    else:
                        points_[:,11][(gt_mask_list[i])&(points_[:,14]==run_sets[j])] = 1
                        roi_points[i][:,11][(roi_points[i][:,14]==run_sets[j])] = 1
                roi_points[i] = roi_points[i][roi_points[i][:,11]>0]

        points_class = get_points_type(self.coord_type)
        results['points'] = points_class(points_, points_dim=points_.shape[-1])  # 实例化，LiDARPoints N,10

        assert len(roi_points) == len(out_gt_bboxes)
        results['gt_labels'] = np.array(out_gt_labels)
        # results['gt_bboxes'] = np.array(out_gt_bboxes)
        results['roi_points'] = roi_points
        return roi_points, results
    
    def points_assign_by_bboxes(self, results, roi_points):
        """2D box内的点云存在遮挡,依照距离远近来分配点云到对应的2D box
        Input: roi_points(list) [ndarray,ndarray,……] length is no empty 2D boxes numbers
        """
        all_roi_min_dist = np.ones((len(roi_points))) * 999999
        for i in range(len(roi_points)):
            if len(roi_points[i])==0:
                continue
            single_roi_points_dist = roi_points[i][:, 7]
            # 得到最近的距离，这里有问题！改成得到距离最近的3个点，如果不够3个点，就直接平均即可
            if len(single_roi_points_dist) > 3:
                all_roi_min_dist[i] = np.mean(np.partition(single_roi_points_dist, 3)[:3]).astype(np.float32)
            else:
                all_roi_min_dist[i] = np.mean(single_roi_points_dist)
        # 解决overlap问题
        bboxes = results['gt_bboxes']  # gt_box是resize过的
        labels = results['gt_labels']

        # 将box按照距离排序
        all_roi_min_dist, inds = np.sort(all_roi_min_dist), np.argsort(all_roi_min_dist)
        bboxes = bboxes[inds]
        labels = labels[inds]
        new_roi_points = []
        for i in range(len(roi_points)):
            new_roi_points.append(roi_points[inds[i]])
        roi_points = new_roi_points
        bboxes_nums = len(bboxes)
        # 1. 查找所有盒子的碰撞关系
        # 相交矩阵 N,N,5 1 is overlap 0 is none overlap
        all_overlap_mask = np.zeros((bboxes_nums, bboxes_nums, 5))  # 5=[overlap_flag, inter(x1,y1,x2,y2)]
        for i in range(bboxes_nums):
            if i == bboxes_nums-1:  #z最后一个box不用计算碰撞
                continue
            for j in range(i+1, bboxes_nums):
                flag, inter_box = self.if_overlap(bboxes[i], bboxes[j])
                if flag:
                    all_overlap_mask[i][j][0] = 1
                    all_overlap_mask[i][j][1:5] = inter_box
                else:
                    all_overlap_mask[i][j][0] = 0
        # 2. 判断相交的box内的点属于哪个box
        # all_ignore_indx = [[] for _ in range(bboxes_nums)] # 记录碰撞关系

        for i in range(bboxes_nums):
            overlap_mask = all_overlap_mask[i][:, 0]       # 1表示有碰撞，0表示未碰撞
            overlap_indx = np.where(overlap_mask==1)[0]    # 记录与当前盒子的碰撞的盒子索引
            # 这个放前面是为了避免重复计算碰撞盒关系
            # for j in range(len(overlap_indx)):
            #     all_ignore_indx[overlap_indx[j]].append(i) # 将当前第i个盒子索引放入碰撞盒子overlap_indx[j]的ignor_indx列表里面
            # ignore_indx = all_ignore_indx[i]
            if len(roi_points[i]) == 0:
                continue
            front_box_points = roi_points[i][:, 0:3]
            c_inds = self.find_connected_componets_single_batch(front_box_points, self.dist[labels[i]])
            set_c_inds = list(set(c_inds))
            c_ind = np.argmax([np.sum(c_inds == i) for i in set_c_inds])
            c_mask = c_inds == set_c_inds[c_ind]
            front_box_points = front_box_points[c_mask]  # 获得最大聚类簇的点

            # 3.获取overlap的盒子区域内的点云，然后选择正确的盒子
            for j in range(len(overlap_indx)):
                # if overlap_indx[j] not in ignore_indx:
                # 3.1 获取相交区域
                # inter_box = all_overlap_mask[i][overlap_indx[j]][1:5]
                # 3.2 找到两个盒子中距离车最近的距离，然后将相交区域的点赋给最近的盒子，去掉远的盒子的相交区域的点
                tmp_dist, overlap_dist = all_roi_min_dist[i], all_roi_min_dist[overlap_indx[j]]
                # 如果当前的盒子在前面，那么保留当前盒子的点云不动，删除碰撞盒内的相交点云
                assert tmp_dist <= overlap_dist
                eq_mask = np.isin(roi_points[overlap_indx[j]][:,0:3], front_box_points[:,0:3])  # 维度大小和前一个值相同 np.isin(a1, a2)也就是a1
                sum_eq_mask = np.sum(eq_mask, axis=1) == 3  # 过滤掉不是三个坐标值都相等的点
                roi_points[overlap_indx[j]] = roi_points[overlap_indx[j]][~sum_eq_mask]

        # # 过滤掉空的roi_points
        # out_roi_points = []
        # out_bboxes = []
        # out_labels = []
        # for i in range(len(bboxes)):
        #     if len(roi_points[i]) != 0:
        #         out_bboxes.append(bboxes[i])
        #         out_labels.append(labels[i])
        #         out_roi_points.append(roi_points[i])
        # results['gt_labels'] = np.array(out_labels)
        # results['gt_bboxes'] = np.array(out_bboxes)
        results['roi_points'] = roi_points  # 这个保存的是处理掉overlap问题后每个box内的点云,合起来的点云数是不变的
        return roi_points, results

    def if_overlap(self, box1, box2):

        min_x1, min_y1, max_x1, max_y1 = box1[0], box1[1], box1[2], box1[3]
        min_x2, min_y2, max_x2, max_y2 = box2[0], box2[1], box2[2], box2[3]

        top_x, top_y = max(min_x1, min_x2), max(min_y1, min_y2)
        bot_x, bot_y = min(max_x1, max_x2), min(max_y1, max_y2)
        
        if bot_x >= top_x and bot_y >= top_y:
            return True, np.array([top_x, top_y, bot_x, bot_y])
        else:
            return False, np.array([0, 0, 0, 0])

    def cluster_filter(self, roi_points, results):
        # Car=0.6, Pedestrian=0.1, Cyclist=0.4,
        labels = results['gt_labels']
        for i in range(len(roi_points)):
            if len(roi_points[i])==0:
                continue
            # 聚类过滤
            c_inds = self.find_connected_componets_single_batch(roi_points[i][:, 0:3], self.dist[labels[i]])
            set_c_inds = list(set(c_inds))
            c_ind = np.argmax([np.sum(c_inds == i) for i in set_c_inds])
            c_mask = c_inds == set_c_inds[c_ind]
            roi_points[i] = roi_points[i][c_mask]
        results['roi_points'] = roi_points
        return results

    def points_assign_by_bboxes_area(self, results, roi_points):
        # 记录相对面积表，只需要过滤大于0.92的
        record_real_ares = np.zeros((len(roi_points), len(roi_points)))
        gt_bboxes = results['gt_bboxes']
        labels = results['gt_labels']
        num_boxes = len(gt_bboxes)
        assert num_boxes == len(roi_points)

        for i, gt_bbox in enumerate(gt_bboxes):
            if i == (num_boxes -1):
                break
            for j in range((i+1),num_boxes):
                r1, r2 = self.cal_relative_area_scale(gt_bbox, gt_bboxes[j])  # 相交面积占自身面积的比值
                record_real_ares[i][j] = r1
                record_real_ares[j][i] = r2
        coords = np.where(record_real_ares > self.relative_threshold)
        x = coords[0]
        y = coords[1]
        if len(x) == 0:
            return roi_points, results
        # 上面是得到一个NXN的矩阵，N表示boxes的数量，
        # x,y表示得到的相交面积与自身面积之比大于0.92的小目标索引
        # 后续目标就是去除大目标内的点（小目标内的点）
        # 去除之后，记得过滤一下确保大目标内是否还有点，正常来看是有点的
        for i in range(len(x)):
            # 对小目标内的点进行聚类
            small_box_points = roi_points[x[i]][:,0:3]
            c_inds = self.find_connected_componets_single_batch(small_box_points, self.dist[labels[x[i]]])
            set_c_inds = list(set(c_inds))
            c_ind = np.argmax([np.sum(c_inds == i) for i in set_c_inds])
            c_mask = c_inds == set_c_inds[c_ind]
            small_box_points = small_box_points[c_mask]  # 获得最大聚类簇的点
            # 去除最大簇
            remove_mask = (np.isin(roi_points[y[i]][:,0:3], small_box_points).sum(1)) == 3
            roi_points[y[i]] = roi_points[y[i]][~remove_mask]
        # 过滤防止出现空的
        out_labels = []
        out_boxes = []
        out_roi_points = []
        for i in range(len(roi_points)):
            if len(roi_points[i]) == 0:
                continue
            out_labels.append(labels[i])
            out_boxes.append(gt_bboxes[i])
            out_roi_points.append(roi_points[i])
        assert len(out_labels) == len(out_boxes) == len(out_roi_points)
        results['gt_labels'] = np.array(out_labels)
        results['gt_bboxes'] = np.array(out_boxes)
        results['roi_points'] = out_roi_points

        return out_roi_points, results

    def cal_relative_area_scale(self, box1, box2):
        relative_scale = 0
        relative_scale2 = 0
        min_x1, min_y1, max_x1, max_y1 = box1[0], box1[1], box1[2], box1[3]
        min_x2, min_y2, max_x2, max_y2 = box2[0], box2[1], box2[2], box2[3]

        top_x, top_y = max(min_x1, min_x2), max(min_y1, min_y2)
        bot_x, bot_y = min(max_x1, max_x2), min(max_y1, max_y2)
        
        if bot_x >= top_x and bot_y >= top_y:
            s1 = (box1[2]-box1[0])*(box1[3]-box1[1])
            s2 = (box2[2]-box2[0])*(box2[3]-box2[1])
            s_inter = (bot_x - top_x)*(bot_y-top_y)
            relative_scale = s_inter / s1
            relative_scale2 = s_inter / s2
            # 如果IOU过大，那么就不分配了
            if s_inter / (s1 + s2 - s_inter) > 0.7:
                relative_scale = 0
                relative_scale2 = 0
        return relative_scale, relative_scale2

    def grad_guide_filter_v2(self, results):
        points_ = results['points'].tensor.numpy()
        # bg_points = points_[points_[:, 5] == 0]
        points = points_[((points_[:, 5] == 1) & (points_[:, 6] ==1))]

        # 0. 获得points2img的矩阵表示HxWx(x,y,z)
        h, w, c = results['pad_shape']
        points2img = np.zeros((h, w, c)).astype(np.float32)
        points2img_idx = (np.ones((h, w)) * -1).astype(np.int32)  # img上的点 对照到 points的索引
        for i, point in enumerate(points):
            x_0 = point[8]
            y_0 = point[9]
            points2img[int(y_0), int(x_0)] = point[0:3]
            points2img_idx[int(y_0), int(x_0)] = i

        # 1. 获得深度图
        depth_img = np.sqrt(points2img[:,:,0]**2 + points2img[:,:, 1]**2)

        # 2.滑动窗口过滤噪声点
        pad_width = self.kernel_size // 2  # 给深度图四周补0
        pad_depth_img = np.pad(depth_img, pad_width = pad_width, mode = 'constant', constant_values = 0)
        # 划分窗口
        depth_img_windows = view_as_windows(pad_depth_img, (self.kernel_size, self.kernel_size), 1)
        depth_img_windows_999 = np.copy(depth_img_windows)
        depth_img_windows_mask = depth_img_windows == 0
        depth_img_windows_999[depth_img_windows_mask] = 999
        depth_img_mask = depth_img != 0
        # 计算梯度关系（相对距离关系）
        relative_dis = (depth_img - np.min(depth_img_windows_999, axis=(2,3))) / np.max(depth_img_windows, axis=(2,3))
        relative_dis[~depth_img_mask] = 999  # 999表示空的像素位置
        points_near_mask = relative_dis < self.threshold_depth  # 640,960

        # points_far_mask = (relative_dis >= self.threshold_depth) & (relative_dis != 999)
        # 通过img2points可以获得对应depth_img每个像素点对应的点云的索引
        near_points_indx = points2img_idx[points_near_mask].astype(np.int)
        # far_points_indx = img2points[points_far_mask].astype(np.int)

        # -1 is 无关
        points2img_idx = (points_near_mask * (points2img_idx+1)) - 1

        points = points[near_points_indx]

        # points_class = get_points_type(self.coord_type)
        results['fg_points'] = points # points_class(points, points_dim=points.shape[-1])  # 当前图片的前景点
        results['points2img_idx'] = points2img_idx  # 当前图片上的点
        results['points2img'] = points2img
        return results

    def grad_guide_filter(self, results):
        points = results['points'].tensor.numpy()
        sample_img_id = results['sample_img_id']
   
        # 1. 获得深度距离图
        h, w, _ = results['pad_shape']
        depth_img = np.zeros((h, w))
        img2points = np.ones((h, w)) * -1
        points_dist = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        for i, point in enumerate(points):
            if point[6] == sample_img_id:
                x_0 = point[8]
                y_0 = point[10]
                depth_img[int(y_0), int(x_0)] = points_dist[i]
                img2points[int(y_0), int(x_0)] = i

            if point[7] == sample_img_id:
                x_1 = point[9]
                y_1 = point[11]
                depth_img[int(y_1), int(x_1)] = points_dist[i]
                img2points[int(y_1), int(x_1)] = i
        # 2.滑动窗口过滤噪声点
        pad_width = self.kernel_size // 2  # 给深度图四周补0
        pad_depth_img = np.pad(depth_img, pad_width = pad_width, mode = 'constant', constant_values = 0)
        # 划分窗口
        depth_img_windows = view_as_windows(pad_depth_img, (self.kernel_size, self.kernel_size), 1)
        depth_img_windows_999 = np.copy(depth_img_windows)
        depth_img_windows_mask = depth_img_windows == 0
        depth_img_windows_999[depth_img_windows_mask] = 999
        depth_img_mask = depth_img != 0
        # 计算梯度关系（相对距离关系）
        relative_dis = (depth_img - np.min(depth_img_windows_999, axis=(2,3))) / np.max(depth_img_windows, axis=(2,3))
        relative_dis[~depth_img_mask] = 999  # 999表示空的像素位置
        points_near_mask = relative_dis < self.threshold_depth
        # points_far_mask = (relative_dis >= self.threshold_depth) & (relative_dis != 999)
        # 通过img2points可以获得对应depth_img每个像素点对应的点云的索引
        near_points_indx = img2points[points_near_mask].astype(np.int)
        # far_points_indx = img2points[points_far_mask].astype(np.int)
        points = points[near_points_indx]
        points_class = get_points_type(self.coord_type)
        results['points'] = points_class(points, points_dim=points.shape[-1])  # 实例化，LiDARPoints
        return results

    def points_box_flag(self, results):
        roi_points = results['roi_points']
        points = results['points'].tensor.numpy()
        box_inds_flag = np.ones((points.shape[0],1)).astype(np.float32)*-1
        mask_all = np.zeros(points.shape[0]).astype(np.bool)
        for i in range(len(roi_points)):
            if len(roi_points[i])==0:
                continue
            mask = (np.isin(points[:, 0:3], roi_points[i][:, 0:3]).sum(1))==3
            mask_all = mask_all | mask
            # 这里存在问题，roi_points这个标志位会少点，因为一个点可以投影到多个2d box内，
            # roi_points经过了ccl是比较准确的，但依旧存在一个点在多个2d box内，比如车前面的人
            box_inds_flag[mask] = i
            roi_points[i] = np.concatenate((roi_points[i], np.ones((roi_points[i].shape[0],1)).astype(np.float32)*i), axis=1)
        points[:,11][mask_all] = 1
        points[:,11][(~mask_all)&(points[:,11]!=-1)] = 0
        points = np.concatenate((points, box_inds_flag),axis=1)
        points_class = get_points_type(self.coord_type)
        results['points'] = points_class(points, points_dim=points.shape[-1])  # 实例化，LiDARPoints N,11
        results['roi_points'] = roi_points
        return results

    def get_points2img(self,results):
        h, w, c = results['pad_shape']
        points2img = np.zeros((h, w, c)).astype(np.float32)
        points2img_idx = (np.ones((h, w)) * -1).astype(np.int32)  # img上的点 对照到 points的索引
        
        if len(results['gt_bboxes'])!=0:
            roi_points = results['roi_points']
            points = np.concatenate(roi_points)
            # 获得points2img的矩阵表示HxWx(x,y,z)
            for i, point in enumerate(points):
                x_0 = point[12]
                y_0 = point[13]
                points2img[int(y_0), int(x_0)] = point[0:3]
                points2img_idx[int(y_0), int(x_0)] = i

        results['points2img_idx'] = points2img_idx
        results['points2img'] = points2img
        return results

    def range_seg(self, results):
        points = results['points'].tensor.numpy()
        # top_lidar and first return points mask
        points_mask = (points[:,6]==0)  # (points[:,5]==0) & 
        range_seg_inds = np.ones((points.shape[0])).astype(np.float32)*-1
        points_row = points[points_mask][:,8]
        row_set, inv_inds = np.unique(points_row, return_inverse=True)
        for r in range(len(row_set)):
            row_points = points[(points_mask & (points[:,8]==row_set[r]))]
            # sort
            column, col_inv_inds = np.sort(row_points[:,9]), np.argsort(row_points[:,9])
            range_dist = row_points[:,7][col_inv_inds] # np.sqrt((row_points[:,0]**2+row_points[:,1]**2))[col_inv_inds]  # 水平距离排序的结果
            start = 0
            right = 1
            inds = 0
            dist_threshold = 0.2
            watch_window = 6 # window size is watch_window - 1
            range_seg = []
            rang_seg_right_dist = []
            range_seg_inds_buffer = []
            max_dist = range_dist.max()+10
            for left in range(len(range_dist)):
                # 如果相邻两个column的点距离超过设定的值(0.1)，那么就分段
                if np.abs(range_dist[right] - range_dist[left]) > dist_threshold*(column[right]-column[left]) * max_dist/50:
                # if not self.if_connect(row_points[col_inv_inds][right], row_points[col_inv_inds][left], (column[right]-column[left]), max_dist):
                    range_seg_inds[(points_mask & (points[:,8]==row_set[r]) & 
                    (np.isin(points[:,11], row_points[col_inv_inds[start:right]][:,11])))] = inds
                    
                    range_seg.append(row_points[col_inv_inds[start:right]])
                    range_seg_inds_buffer.append(inds)
                    rang_seg_right_dist.append(range_dist[left])
                    # 看前面分段的最右边的几个深度
                    for i, right_dist in enumerate(rang_seg_right_dist[-watch_window:-1][::-1]):
                        if np.abs(range_seg[-1][0][7] - right_dist) < dist_threshold*(range_seg[-1][0][9]-range_seg[-watch_window:-1][::-1][i][-1][9]):
                        # if self.if_connect(range_seg[-1][0], range_seg[-watch_window:-1][::-1][i][-1], (range_seg[-1][0][9]-range_seg[-watch_window:-1][::-1][i][-1][9]), max_dist):
                            range_seg_inds_buffer[-1] = range_seg_inds_buffer[-watch_window:-1][::-1][i]
                            range_seg_inds[(points_mask & (points[:,8]==row_set[r]) & 
                            (np.isin(points[:,11], row_points[col_inv_inds[start:right]][:,11])))] = range_seg_inds_buffer[-1]
                            inds -= 1  # 表示驻留，与下面的inds+1做抵消
                            break

                    inds += 1
                    start = right
                right += 1
                if right==len(range_dist):
                    range_seg_inds[(points_mask & (points[:,8]==row_set[r]) & 
                    (np.isin(points[:,11], row_points[col_inv_inds[start:right]][:,11])))] = inds

                    range_seg.append(row_points[col_inv_inds[start:right]])
                    range_seg_inds_buffer.append(inds)
                    rang_seg_right_dist.append(range_dist[start])
                    # 看之前分段的最右边的几个深度
                    for i, right_dist in enumerate(rang_seg_right_dist[-watch_window:-1][::-1]):
                        if np.abs(range_seg[-1][0][7] - right_dist) < dist_threshold*(range_seg[-1][0][9]-range_seg[-watch_window:-1][::-1][i][-1][9]):
                        # if self.if_connect(range_seg[-1][0], range_seg[-watch_window:-1][::-1][i][-1], dist_threshold*(range_seg[-1][0][9]-range_seg[-watch_window:-1][::-1][i][-1][9]), max_dist):
                            range_seg_inds_buffer[-1] = range_seg_inds_buffer[-watch_window:-1][::-1][i]
                            range_seg_inds[(points_mask & (points[:,8]==row_set[r]) & 
                            (np.isin(points[:,11], row_points[col_inv_inds[start:right]][:,11])))] = range_seg_inds_buffer[-1]
                            inds -= 1  # 表示驻留，与下面的inds+1做抵消
                            break
                    break
        points = np.concatenate((points, range_seg_inds.reshape(points.shape[0],1)), axis=1)
        points_class = get_points_type(self.coord_type)
        results['points'] = points_class(points, points_dim=points.shape[-1])
        import cv2

        img = results['img'].astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        r=32
        # for r in range(len(row_set)):
        p =  points[(points_mask & (points[:,8]==row_set[r]))]
        for i in range(int(p[:,14].max())+1):
            rgb = np.array([np.random.randint(255) for _ in range(3)])
            pp = p[p[:,14] == i]
            for j in pp:
                x = int(j[12])
                y = int(j[13])
                cv2.circle(img, (x, y), 1, rgb.tolist(), 1)
        cv2.imwrite('range_seg_img.jpeg', img)
        return results

    def range_seg_ccl(self, results):
        points = results['points'].tensor.numpy()
        range_seg_inds = np.ones((points.shape[0])).astype(np.float32)*-1
        if not self.use_run_seg:
            points = np.concatenate((points, range_seg_inds.reshape(points.shape[0],1)), axis=1)
            points_class = get_points_type(self.coord_type)
            results['points'] = points_class(points, points_dim=points.shape[-1])
            return results
        # top_lidar points mask two return all use
        top_points_mask = (points[:,6]==0)
        top_points  = points[top_points_mask]
        points_row = points[top_points_mask][:,8]
        row_set, inv_inds = np.unique(points_row, return_inverse=True)
        row_set = row_set.astype(np.int)
        clusters_id_all = []
        column_nums = 2650
        num_clusters = 0
        for r in range(64):
            if r not in row_set:
                clusters_id_all.append(np.ones((2, column_nums))*-1)
                continue
            
            # 第r行的点云深度排序(2,2650)
            range_dist = np.ones((2, column_nums))*-1
            # 第一次回波
            fir_return_points = top_points[(top_points[:,5]==0) & (top_points[:,8]==r)]
            for p, pts in enumerate(fir_return_points):
                range_dist[0][int(pts[9])] = pts[7]
                # range_dist[0][int(pts[9])] = np.sqrt((pts[0]**2+pts[1]**2))
            # 第二次回波
            sec_return_points = top_points[(top_points[:,5]==1) & (top_points[:,8]==r)]
            for p, pts in enumerate(sec_return_points):
                range_dist[1][int(pts[9])] = pts[7]
                # range_dist[1][int(pts[9])] = np.sqrt((pts[0]**2+pts[1]**2))
            
            max_dist = np.max(range_dist)

            # 初始化equalTable, num_cluster, clusters_id
            kernel_size = 10 / max_dist * 50
            kernel_size = np.clip(kernel_size, 12, 50)  # 50米看4个点
            equal_tabel = np.array([i for i in range(2*column_nums)]).reshape((2,column_nums))
            # num_clusters = 0
            clusters_id = np.ones((2, column_nums))*-1
            # 1. 建立equaltree
            for i in range(column_nums):
                if range_dist[0][i] == -1 and range_dist[1][i] == -1:
                    continue
                # 判断第一次回波
                if range_dist[0][i] != -1:
                    for j in range(int(kernel_size/2)):
                        if i >= j and range_dist[0][i-j]!=-1 and (j!=0):
                            dist_flag = self.if_connect(fir_return_points[(fir_return_points[:,8]==r)&(fir_return_points[:,9]==i)][0],
                            fir_return_points[(fir_return_points[:,8]==r)&(fir_return_points[:,9]==(i-j))][0],
                            j+1, max_dist)
                            if dist_flag:
                                equal_tabel[0][i] = equal_tabel[0][i-j]
                                break
                        # 第一次回波对应位置不与对应位置的第二次回波对比，只有第二次的才与对应第一次的对比
                        if i >= j and range_dist[1][i-j]!=-1 and (j!=0):
                            dist_flag = self.if_connect(fir_return_points[(fir_return_points[:,8]==r)&(fir_return_points[:,9]==i)][0],
                            sec_return_points[(sec_return_points[:,8]==r)&(sec_return_points[:,9]==(i-j))][0],
                            j+1, max_dist)
                            if dist_flag:
                                equal_tabel[0][i] = equal_tabel[1][i-j]
                                break
                # 判断第二次回波
                if range_dist[1][i] != -1:
                    for j in range(int(kernel_size/2)):                 
                        if i >= j and range_dist[0][i-j]!=-1:
                            dist_flag = self.if_connect(sec_return_points[(sec_return_points[:,8]==r)&(sec_return_points[:,9]==i)][0],
                                                        fir_return_points[(fir_return_points[:,8]==r)&(fir_return_points[:,9]==(i-j))][0],
                                                        j+1, max_dist)
                            if dist_flag:
                                equal_tabel[1][i] = equal_tabel[0][i-j]
                                break
                        if i >= j and range_dist[1][i-j]!=-1 and (j!=0):
                            dist_flag = self.if_connect(sec_return_points[(sec_return_points[:,8]==r)&(sec_return_points[:,9]==i)][0],
                                                        sec_return_points[(sec_return_points[:,8]==r)&(sec_return_points[:,9]==(i-j))][0],
                                                        j+1, max_dist)
                            if dist_flag:
                                equal_tabel[1][i] = equal_tabel[1][i-j]
                                break
            # 2. 统一label
            for i in range(column_nums):
                if range_dist[0][i] == -1 and range_dist[1][i] == -1:
                    continue
                if range_dist[0][i] != -1:
                    if equal_tabel[0][i] == i:
                        clusters_id[0][i] = num_clusters
                        num_clusters += 1
                if range_dist[1][i] != -1:
                    if equal_tabel[1][i] == i + column_nums:
                        clusters_id[1][i] = num_clusters
                        num_clusters += 1
            # 3. 重新label
            for i in range(column_nums):
                if range_dist[0][i] == -1 and range_dist[1][i] == -1:
                    continue
                if range_dist[0][i] != -1:
                    label = i
                    while label != equal_tabel[label//column_nums][label-(label//column_nums)*column_nums]:
                        batch_id = label//column_nums
                        label = equal_tabel[batch_id][label-(batch_id)*column_nums]
                    batch_id = label//column_nums
                    clusters_id[0][i] = int(clusters_id[batch_id][label-(batch_id)*column_nums])
                if range_dist[1][i] != -1:
                    label = column_nums + i
                    while label != equal_tabel[label//column_nums][label-(label//column_nums)*column_nums]:
                        batch_id = label//column_nums
                        label = equal_tabel[batch_id][label-(batch_id)*column_nums]
                    batch_id = label//column_nums
                    clusters_id[1][i] = int(clusters_id[batch_id][label-(batch_id)*column_nums])

            clusters_id_all.append(clusters_id)
        clusters_id_all = np.stack(clusters_id_all, 0).transpose(1,0,2)  # (64,2,2650)-->(2,64,2650)
        c, h, w = top_points[:,5].astype(np.int), top_points[:,8].astype(np.int), top_points[:,9].astype(np.int)
        range_seg_inds[top_points_mask] = clusters_id_all[(c,h,w)]

        points = np.concatenate((points, range_seg_inds.reshape(points.shape[0],1)), axis=1)
        points_class = get_points_type(self.coord_type)
        results['points'] = points_class(points, points_dim=points.shape[-1])
        return results

    def if_connect(self, pts1, pts2, times=1, max_dist=50):
        # 深度距离判断
        times = np.clip(times, 1, 1.5)
        if np.abs(pts1[7] - pts2[7]) < 0.24 * 1. * max_dist / 50:
            return True
        # 欧几里得距离判断 不加的原因是防止轮胎和地面连一块
        # elif np.sqrt((((pts1[0:3]-pts2[0:3])**2).sum())) < 0.05 * times:
        #     return True
        # elif np.abs(pts1[7] - pts2[7]) >= 0.5 and np.sqrt((((pts1[0:3]-pts2[0:3])**2).sum())) < 0.3:
        #     return True
        else:
            return False

    def get_in_2d_box_points_v2(self, results):
        points_ = results['points'].tensor.numpy()  # (N,19)
        # in image points
        in_mask = (points_[:,10]==1)

        sample_img_id = results['sample_img_id']
        gt_bboxes = results['gt_bboxes']
        labels = results['gt_labels']

        assert len(gt_bboxes)==len(labels)
        gt_mask_all = np.zeros((points_.shape[0])).astype(np.bool)
        # 如果不用叠帧就不要扩大2D框，因为这样会引入背景点！
        # 叠帧反而会抑制（&）掉扩大2D框引入的背景点
        dis = 0.
        dis_h = 0.
        for j, gt_bbox in enumerate(gt_bboxes):
            w, h = gt_bbox[2]-gt_bbox[0], gt_bbox[3]-gt_bbox[1]
            x1, y1 = gt_bbox[0]-dis*w, gt_bbox[1]-dis_h*h
            x2, y2 =  gt_bbox[2]+dis*w, gt_bbox[3]+dis*h
            gt_mask1 = (((points_[:, 12] >= x1) & (points_[:, 12] < x2)) &
                        ((points_[:, 13] >= y1) & (points_[:, 13] < y2)  &
                          in_mask))
            gt_mask = gt_mask1
            gt_mask_all = gt_mask_all | gt_mask
        points_[:,11][gt_mask_all] = 1  # 在2D box内的点大部分是前景点，后面进行筛选

        box_flag = np.zeros((points_.shape[0], 1)).astype(np.float32) # box_flag[:,1]是个标志位，表示第一个是否被填充
        out_box_points = points_[~gt_mask_all]
        points_index = np.array(range(0, len(points_))).astype(np.int)
        labels = results['gt_labels']
        # img
        dis = 0
        dis_h = 0
        for j, gt_bbox in enumerate(gt_bboxes):
            w,h = gt_bbox[2]-gt_bbox[0], gt_bbox[3]-gt_bbox[1]
            x1, y1 = gt_bbox[0]-dis*w, gt_bbox[1]-dis_h*h
            x2, y2 =  gt_bbox[2]+dis*w, gt_bbox[3]+dis*h
            gt_mask = (((points_[:, 12] >= x1) & (points_[:, 12] < x2)) &
                        ((points_[:, 13] >= y1) & (points_[:, 13] < y2)  &
                          in_mask))
            if gt_mask.sum() == 0:
                continue
            in_box_points = points_[gt_mask]
            # 进行run过滤计算
            if self.use_run_seg:
                run_sets = np.unique(in_box_points[:,14])
                for s in range(len(run_sets)):
                    prop = (out_box_points[:,14]==run_sets[s]).sum()/(points_[:,14]==run_sets[s]).sum()
                    if run_sets[s] == -1:
                        points_[:,11][(gt_mask)&(points_[:,14]==run_sets[s])] = -1
                    elif prop >= 0.5:
                        points_[:,11][(gt_mask)&(points_[:,14]==run_sets[s])] = 0
                    elif prop >= 0.05 and prop < 0.5:
                        points_[:,11][(gt_mask)&(points_[:,14]==run_sets[s])] = -1
                    else:
                        points_[:,11][(gt_mask)&(points_[:,14]==run_sets[s])] = 1
            # update gt_mask
            in_box_mask = gt_mask & (points_[:,11]>0)
            # cluster
            # Car=0.6, Pedestrian=0.1, Cyclist=0.4
            if in_box_mask.sum()==0:
                continue
            c_inds = self.find_connected_componets_single_batch(points_[in_box_mask][:,0:3], self.dist[labels[j]])
            set_c_inds = list(set(c_inds))
            c_ind = np.argmax([np.sum(c_inds == i) for i in set_c_inds])
            c_mask = c_inds == set_c_inds[c_ind]
            max_in_box_index = points_index[in_box_mask][c_mask]
            box_flag[max_in_box_index] = j+1
        
        box_flag = box_flag - 1
        bg_mask_ = (box_flag[:,0]==-1) & (points_[:,11]!=-1)
        points_[:, 11][bg_mask_] = 0
        points_ = np.concatenate((points_, box_flag), axis=1)
        points_class = get_points_type(self.coord_type)
        results['points'] = points_class(points_, points_dim=points_.shape[-1])  # 实例化，LiDARPoints N,10

        return results

    def assign_pseudo_label(self, results):
        if 'pseudo_labels' in results.keys() and self.use_pseudo_label:
            pseudo_labels = results['pseudo_labels']
            # [('run_id','i4'),('run','i2'), ('ignore','i2'), ('run_ccl','i2'), ('collision','i2')]
            points = results['points'].tensor.numpy()  # (N,19)
            # fg mask
            fg_mask = pseudo_labels[:,1].astype(np.bool)
            ignore_mask = pseudo_labels[:,2].astype(np.bool)
            bg_mask = (~fg_mask)&(~ignore_mask)
            # points[:,11][ignore_mask] = -1
            # points[:,11][bg_mask] = 0
            # points[:,11][fg_mask] = 1
            # points[:,11] = points[:,11]*(~(pseudo_labels[:,4].astype(np.bool))).astype('i2')

            # 为什么不用run_ccl这个标志位？因为这是ccl过滤后的，将前景点分类成背景点，如果与collision取背景点可能太hard，所以没有考虑这个，后面可以跑一个对照实验
            # 对照待定：使用run_ccl与collision来确定前景点、背景点、ignore点

            # 背景点一定是collision&run_bg_points
            points[:,11] = -1
            all_bg_mask = (pseudo_labels[:,4].astype(np.bool)) & bg_mask
            points[:,11][all_bg_mask] = 0
            # 前景点一定是都是前景点
            all_fg_mask = (~(pseudo_labels[:,4].astype(np.bool))) & fg_mask
            points[:,11][all_fg_mask] = 1
            # 剩下的点为ignore点

            # run id
            points = np.concatenate((points, pseudo_labels[:,0][:,None]), axis=1)

            dis = 0.
            dis_h = 0.
            sample_img_id = results['sample_img_id']
            # gt_bboxes = results['gt_bboxes']
            gt_bboxes = results['gt_bboxes']
            labels = results['gt_labels']
            assert len(gt_bboxes)==len(labels)

            box_flag = np.zeros((points.shape[0],1)).astype(np.float32) # box_flag[:,1]是个标志位，表示第一个是否被填充
            points_index = np.array(range(0,len(points))).astype(np.int)

            for j, gt_bbox in enumerate(gt_bboxes):
                w,h = gt_bbox[2]-gt_bbox[0], gt_bbox[3]-gt_bbox[1]
                x1, y1 = gt_bbox[0]-dis*w, gt_bbox[1]-dis_h*h
                x2, y2 = gt_bbox[2]+dis*w, gt_bbox[3]+dis*h
                gt_mask = (((points[:, 12] >= x1) & (points[:, 12] < x2)) &
                            ((points[:, 13] >= y1) & (points[:, 13] < y2) &
                            fg_mask))
                in_box_mask = gt_mask & (points[:,11]>0)
                if in_box_mask.sum()==0:
                    continue
                # Car=0.6, Pedestrian=0.1
                c_inds = self.find_connected_componets_single_batch(points[in_box_mask][:,0:3], self.dist[labels[j]])
                set_c_inds = list(set(c_inds))
                c_ind = np.argmax([np.sum(c_inds == i) for i in set_c_inds])
                c_mask = c_inds == set_c_inds[c_ind]
                max_in_box_index = points_index[in_box_mask][c_mask]
                box_flag[max_in_box_index] = j+1

            box_flag = box_flag - 1
            bg_mask_ = (box_flag[:,0] == -1) & (points[:,11] != -1)
            points[:, 11][bg_mask_] = 0
            points = np.concatenate((points, box_flag), axis=1)
            points_class = get_points_type(self.coord_type)
            results['points'] = points_class(points, points_dim=points.shape[-1])

        return results

    def load_ccl_labels(self, results):
        points = results['points'].tensor.numpy()
        ccl_id = results['pseudo_labels'][:, 0].reshape(-1, 1)
        assert len(ccl_id) == len(points)
        points = np.concatenate((points, ccl_id.reshape(points.shape[0], 1)), axis=1)
        points_class = get_points_type(self.coord_type)
        results['points'] = points_class(points, points_dim=points.shape[-1])
        return results

    def __call__(self, results):
        if self.only_img:
            results = self._filter_points(results)
        else:
            results = self._filter_points(results)
            if self.training:
                # if self.use_pseudo_label and 'pseudo_labels' in results.keys():
                #     results = self.assign_pseudo_label(results)
                # else:
                if self.use_pseudo_label and 'pseudo_labels' in results.keys():
                    results = self.load_ccl_labels(results)
                else:
                    results = self.range_seg_ccl(results)  # 全点云分段，不在图片上的也可以分段
                roi_points, results = self.get_in_2d_box_points(results)  # (gt_box的数量, 3) 不在2dbox内的点的points的fg_points标志位置-1
                results = self.cluster_filter(roi_points, results)
                results = self.points_box_flag(results)  # 依据roi_points 给对应的点添加对应2d box的索引
                # results = self.get_in_2d_box_points_v2(results)
                results['pseudo_labels'] = np.array([])
                # print((results['points'].tensor.numpy()[:,-2] == results['points'].tensor.numpy()[:,-1]).sum(), len(results['points'].tensor.numpy()))
        return results

@PIPELINES.register_module(force=True)
class AssignPseudoLabel:

    def __init__(self, coord_type, kernel_size=3, threshold_depth=0.5,
                dist=(0.6,0.1,0.4), training=True, 
                relative_threshold=0.91, use_run_seg=True,
                only_img=False):
        self.coord_type = coord_type
        self.kernel_size = kernel_size
        self.threshold_depth = threshold_depth
        self.dist = dist
        self.training = training
        self.relative_threshold = relative_threshold
        self.use_run_seg = use_run_seg
        self.only_img = only_img

    def filter_points(self, results):
        pass

    def __call__(self, results):
        return results

@PIPELINES.register_module(force=True)
class GetOrientation:
    """
    得到2D框内对象的朝向角
    """
    def __init__(self, gt_box_type=1, sample_roi_points=100, th_dx=4, dist=0.6, use_geomtry_loss=False):
        self.gt_box_type = gt_box_type
        self.sample_roi_points = sample_roi_points
        self.th_dx = th_dx  # 阈值
        self.dist = dist
        self.use_geomtry_loss = use_geomtry_loss

    def find_connected_componets_single_batch(self, points, dist):

        this_points = points
        dist_mat = this_points[:, None, :2] - this_points[None, :, :2] # only care about xy
        dist_mat = (dist_mat ** 2).sum(2) ** 0.5
        adj_mat = dist_mat < dist
        adj_mat = adj_mat
        c_inds = connected_components(adj_mat, directed=False)[1]

        return c_inds

    def get_in_2d_box_points(self, results):
        points = results['fg_points']
        gt_bboxes = results['gt_bboxes']
        labels = results['gt_labels']
        sample_img_id = results['sample_img_id']
        roi_batch_points = []
        
        out_gt_bboxes = []
        out_gt_labels = []

        for i, gt_bbox in enumerate(gt_bboxes):
            # 2d box是resize过得, 需要还原
            gt_bbox = gt_bbox
            # if 0 cam 8,10（列，行）
            gt_mask = (((points[:, 8] >= gt_bbox[0]) & (points[:, 8] < gt_bbox[2])) &
                        ((points[:, 9] >= gt_bbox[1]) & (points[:, 9] < gt_bbox[3])))
            in_box_points = points[gt_mask]
            # 如果2d框里没有点，那么就过滤掉对应的2d box和label
            if len(in_box_points) == 0:
                continue
            out_gt_bboxes.append(gt_bbox)
            out_gt_labels.append(labels[i])
            roi_batch_points.append(in_box_points)  # 12
        assert len(roi_batch_points) == len(out_gt_bboxes)
        results['gt_labels'] = np.array(out_gt_labels)
        results['gt_bboxes'] = np.array(out_gt_bboxes)
        return roi_batch_points, results
    
    def points_assign_by_bboxes(self, results, roi_points):
        """2D box内的点云存在遮挡,依照距离远近来分配点云到对应的2D box
        Input: roi_points(list) [ndarray,ndarray,……] length is no empty 2D boxes numbers
        """
        all_roi_min_dist = np.ones((len(roi_points))) * 999
        for i in range(len(roi_points)):
            single_roi_points = roi_points[i]
            single_roi_points_dist = np.sqrt(single_roi_points[:, 0]**2 + single_roi_points[:, 1]**2)
            all_roi_min_dist[i] = np.min(single_roi_points_dist).astype(np.float32)
        # 解决overlap问题
        bboxes = results['gt_bboxes']  # gt_box是resize过的
        bboxes_nums = len(bboxes)
        # 1. 查找所有盒子的碰撞关系
        # 相交矩阵
        all_overlap_mask = np.zeros((bboxes_nums, bboxes_nums, 5))  # 5=[overlap_flag, inter(x1,y1,x2,y2)]
        for i in range(bboxes_nums):
            for j in range(bboxes_nums):
                flag, inter_box = self.if_overlap(bboxes[i], bboxes[j])
                if flag:
                    all_overlap_mask[i][j][0] = 1
                    all_overlap_mask[i][j][1:5] = inter_box
                else:
                    all_overlap_mask[i][j][0] = 0
        # 2. 判断相交的box内的点属于哪个box
        all_ignore_indx = [[] for _ in range(bboxes_nums)]  # 记录碰撞关系
        remove_points = []  # 记录删除的点
        for i in range(bboxes_nums):
            overlap_mask = all_overlap_mask[i][:, 0]  # 1表示有碰撞，0表示未碰撞
            overlap_indx = np.where(overlap_mask==1)[0]
            # 这个放前面是为了避免计算自身的碰撞盒关系
            for j in range(len(overlap_indx)):
                all_ignore_indx[overlap_indx[j]].append(i)
            ignore_indx = all_ignore_indx[i]
            # 3.获取overlap的盒子区域内的点云，然后选择正确的盒子
            for j in range(len(overlap_indx)):
                if overlap_indx[j] not in ignore_indx:
                    # 3.1 获取相交区域
                    inter_box = all_overlap_mask[i][overlap_indx[j]][1:5]
                    # 3.2 找到两个盒子中距离车最近的距离，然后将相交区域的点赋给最近的盒子，去掉远的盒子的相交区域的点
                    tmp_dist, overlap_dist = all_roi_min_dist[i], all_roi_min_dist[overlap_indx[j]]
                    # 如果当前的盒子在前面，那么保留当前盒子的点云不动，删除碰撞盒内的相交点云
                    if tmp_dist < overlap_dist:
                        eq_mask = np.isin(roi_points[overlap_indx[j]][:,0:3], roi_points[i][:,0:3])  # 维度大小和前一个值相同 np.isin(a1, a2)也就是a1
                        sum_eq_mask = np.sum(eq_mask,axis=1) == 3  # 过滤掉不是三个坐标值都相等的点
                        remove_points.append(roi_points[overlap_indx[j]][sum_eq_mask])
                        roi_points[overlap_indx[j]] = roi_points[overlap_indx[j]][~sum_eq_mask]
                    # 如果是当前盒子在后面，删除当前盒子的点
                    else:
                        eq_mask = np.isin(roi_points[i][:,0:3], roi_points[overlap_indx[j]][:,0:3])
                        sum_eq_mask = np.sum(eq_mask, axis=1) == 3
                        remove_points.append(roi_points[i][sum_eq_mask])
                        roi_points[i] = roi_points[i][~sum_eq_mask]
        if len(remove_points) != 0:
            remove_points = np.concatenate(remove_points, axis=0)
        # 过滤掉空的roi_points
        out_roi_points = []
        out_bboxes = []
        out_labels = []
        for i in range(len(bboxes)):
            if len(roi_points[i]) != 0:
                out_bboxes.append(bboxes[i])
                out_labels.append(results['gt_labels'][i])
                out_roi_points.append(roi_points[i])
        results['gt_labels'] = np.array(out_labels)
        results['gt_bboxes'] = np.array(out_bboxes)
        results['ori_roi_points'] = out_roi_points  # 这个保存的是处理掉overlap问题后每个box内的点云
        # all_roi_points = np.concatenate(out_roi_points, axis=0)
        return out_roi_points, results

    def if_overlap(self, box1, box2):

        min_x1, min_y1, max_x1, max_y1 = box1[0], box1[1], box1[2], box1[3]
        min_x2, min_y2, max_x2, max_y2 = box2[0], box2[1], box2[2], box2[3]

        top_x, top_y = max(min_x1, min_x2), max(min_y1, min_y2)
        bot_x, bot_y = min(max_x1, max_x2), min(max_y1, max_y2)
        
        if bot_x >= top_x and bot_y >= top_y:
            return True, np.array([top_x, top_y, bot_x, bot_y])
        else:
            return False, np.array([0, 0, 0, 0])

    def get_orientation(self, roi_batch_points, results):
        
        RoI_points = roi_batch_points  # (N,12)
        gt_bboxes = results['gt_bboxes']
        assert len(RoI_points) == len(gt_bboxes) and len(gt_bboxes) == len(results['gt_labels'])

        batch_RoI_points = np.zeros((gt_bboxes.shape[0], self.sample_roi_points, 3), dtype=np.float32)
        batch_lidar_y_center = np.zeros((gt_bboxes.shape[0],), dtype=np.float32)  
        batch_lidar_orient = np.zeros((gt_bboxes.shape[0],), dtype=np.float32)
        batch_lidar_density = np.zeros((gt_bboxes.shape[0], self.sample_roi_points), dtype=np.float32)
        
        for i in range(len(RoI_points)):
            # 聚类过滤
            c_inds = self.find_connected_componets_single_batch(RoI_points[i][:, 0:3], self.dist)
            set_c_inds = list(set(c_inds))
            c_ind = np.argmax([np.sum(c_inds == i) for i in set_c_inds])
            c_mask = c_inds == set_c_inds[c_ind]
            RoI_points[i] = RoI_points[i][c_mask]
            
            z_coor = RoI_points[i][:, 2]  # height val
            batch_lidar_y_center[i] = np.mean(z_coor)
            z_thesh = (np.max(z_coor) + np.min(z_coor)) / 2
            z_ind = RoI_points[i][:, 2] < z_thesh  # 这里没看懂，为什么靠下？

            z_ind_points = RoI_points[i][z_ind]
            if z_ind_points.shape[0] < 10:
                z_ind_points = RoI_points[i]

            rand_ind = np.random.randint(0, z_ind_points.shape[0], 100)
            depth_points_sample = z_ind_points[rand_ind]
            batch_RoI_points[i] = depth_points_sample[:, 0:3]
            depth_points_np_xy = depth_points_sample[:, [0, 1]]  # 获得当前2d框内的点云的xy坐标

            '''orient'''
            orient_set = [(i[0] - j[0]) / - (i[1] - j[1]) for j in depth_points_np_xy  # 这里的y轴反向了一下，方便对照weakm3d得到的斜率k值
                          for i in depth_points_np_xy]  # 斜率，存在nan值，分母为0
            orient_sort = np.array(sorted(np.array(orient_set).reshape(-1)))
            orient_sort = np.arctan(orient_sort[~np.isnan(orient_sort)])  # 过滤掉nan值，然后得到角度 [-pi/2,pi/2]
            orient_sort_round = np.around(orient_sort, decimals=1)  # 对输入浮点数执行5舍6入，5做特殊处理 decimals保留1位小数
            set_orenit = list(set(orient_sort_round))  # 去重，得到直方图的bin  [-1.6, 1.6]
            try:
                ind = np.argmax([np.sum(orient_sort_round == i) for i in set_orenit])  # 得到直方图最高的点
                orient = set_orenit[ind]
                if orient < 0:  # 角度是钝角时，＜0，需要加上pi,变换到正方向
                    orient += np.pi
                
                # weakm3d提到车的行驶方向通常是45度到135度，但如果超过了阈值距离dx，那么就会再回到旧的距离，下面有写判断
                if orient > np.pi / 2 + np.pi * 3 / 8:
                    orient -= np.pi / 2
                if orient < np.pi / 8:
                    orient += np.pi / 2

                if np.max(RoI_points[i][:, 1]) - np.min(RoI_points[i][:, 1]) > self.th_dx and \
                        (orient >= np.pi / 8 and orient <= np.pi / 2 + np.pi * 3 / 8):
                    if orient < np.pi / 2:
                        orient += np.pi / 2
                    else:
                        orient -= np.pi / 2
                    # 这一步出来的是kitti下的yaw角范围是[0, pi]
                # 转到wamoy坐标系下，yaw [-pi/2, pi/2]
                orient = orient - np.pi/2
            except:
                orient = 0  # 如果np.argmax得不到值，就默认为沿x轴方向
            batch_lidar_orient[i] = orient

            '''density'''
            p_dis = np.array([(i[0] - depth_points_sample[:, 0]) ** 2 + (i[1] - depth_points_sample[:, 1]) ** 2
                                 for i in depth_points_sample])
            batch_lidar_density[i] = np.sum(p_dis < 0.04, axis=1)

        results['gt_yaw'] = batch_lidar_orient.astype(np.float32)
        results['roi_points'] = RoI_points  # roi_batch_points CCL聚类过滤得到的RoI_points
        if self.use_geomtry_loss:
            results['lidar_density'] = batch_lidar_density.astype(np.float32)
            results['batch_roi_points'] = batch_RoI_points.astype(np.float32)
            # results['y_center'] = batch_lidar_y_center.astype(np.float32)
        return results

    def get_orientation_v2(self, results):
        
        RoI_points = results['roi_points']  # (N,13)
        gt_bboxes = results['gt_bboxes']
        assert len(RoI_points) == len(gt_bboxes) and len(gt_bboxes) == len(results['gt_labels'])

        batch_RoI_points = np.zeros((gt_bboxes.shape[0], self.sample_roi_points, 3), dtype=np.float32)
        batch_lidar_y_center = np.zeros((gt_bboxes.shape[0],), dtype=np.float32)  
        batch_lidar_orient = np.zeros((gt_bboxes.shape[0],), dtype=np.float32)
        batch_lidar_density = np.zeros((gt_bboxes.shape[0], self.sample_roi_points), dtype=np.float32)
        
        for i in range(len(RoI_points)):
           
            z_coor = RoI_points[i][:, 2]  # height val
            batch_lidar_y_center[i] = np.mean(z_coor)
            z_thesh = (np.max(z_coor) + np.min(z_coor)) / 2
            z_ind = RoI_points[i][:, 2] < z_thesh  # 这里没看懂，为什么靠下？

            z_ind_points = RoI_points[i][z_ind]
            if z_ind_points.shape[0] < 10:
                z_ind_points = RoI_points[i]

            rand_ind = np.random.randint(0, z_ind_points.shape[0], 100)
            depth_points_sample = z_ind_points[rand_ind]
            batch_RoI_points[i] = depth_points_sample[:, 0:3]
            depth_points_np_xy = depth_points_sample[:, [0, 1]]  # 获得当前2d框内的点云的xy坐标

            '''orient'''
            orient_set = [(i[0] - j[0]) / - (i[1] - j[1]) for j in depth_points_np_xy  # 这里的y轴反向了一下，方便对照weakm3d得到的斜率k值
                          for i in depth_points_np_xy]  # 斜率，存在nan值，分母为0
            orient_sort = np.array(sorted(np.array(orient_set).reshape(-1)))
            orient_sort = np.arctan(orient_sort[~np.isnan(orient_sort)])  # 过滤掉nan值，然后得到角度 [-pi/2,pi/2]
            orient_sort_round = np.around(orient_sort, decimals=1)  # 对输入浮点数执行5舍6入，5做特殊处理 decimals保留1位小数
            set_orenit = list(set(orient_sort_round))  # 去重，得到直方图的bin  [-1.6, 1.6]
            try:
                ind = np.argmax([np.sum(orient_sort_round == i) for i in set_orenit])  # 得到直方图最高的点
                orient = set_orenit[ind]
                if orient < 0:  # 角度是钝角时，＜0，需要加上pi,变换到正方向
                    orient += np.pi
                
                # weakm3d提到车的行驶方向通常是45度到135度，但如果超过了阈值距离dx，那么就会再回到旧的距离，下面有写判断
                if orient > np.pi / 2 + np.pi * 3 / 8:
                    orient -= np.pi / 2
                if orient < np.pi / 8:
                    orient += np.pi / 2

                if np.max(RoI_points[i][:, 1]) - np.min(RoI_points[i][:, 1]) > self.th_dx and \
                        (orient >= np.pi / 8 and orient <= np.pi / 2 + np.pi * 3 / 8):
                    if orient < np.pi / 2:
                        orient += np.pi / 2
                    else:
                        orient -= np.pi / 2
                    # 这一步出来的是kitti下的yaw角范围是[0, pi]
                # 转到wamoy坐标系下，yaw [-pi/2, pi/2]
                orient = orient - np.pi/2
            except:
                orient = 0  # 如果np.argmax得不到值，就默认为沿x轴方向
            batch_lidar_orient[i] = orient

            '''density'''
            p_dis = np.array([(i[0] - depth_points_sample[:, 0]) ** 2 + (i[1] - depth_points_sample[:, 1]) ** 2
                                 for i in depth_points_sample])
            batch_lidar_density[i] = np.sum(p_dis < 0.04, axis=1)

        results['gt_yaw'] = batch_lidar_orient.astype(np.float32)
        # results['roi_points'] = RoI_points  # roi_batch_points CCL聚类过滤得到的RoI_points
        if self.use_geomtry_loss:
            results['lidar_density'] = batch_lidar_density.astype(np.float32)
            results['batch_roi_points'] = batch_RoI_points.astype(np.float32)
            # results['y_center'] = batch_lidar_y_center.astype(np.float32)
        return results

    def __call__(self, results):
        if self.gt_box_type == 2:
            # v1版本
            # roi_points, results = self.get_in_2d_box_points(results)  # (gt_box的数量, 3)
            # roi_points, results = self.points_assign_by_bboxes(results, roi_points)
            # results = self.get_orientation(roi_points, results)
            # v2版本
            results = self.get_orientation_v2(results)
        return results

@PIPELINES.register_module(force=True)
class FilterPointByMultiImage:
    """
    The project point cloud is obtained by the Image idx
    """
    def __init__(self,
                 coord_type,
                 kernel_size=3,
                 threshold_depth=0.5,
                 dist=(0.6, 0.1, 0.4),
                 training=True, 
                 relative_threshold=0.91,
                 use_run_seg=False,
                 only_img=False,
                 use_pseudo_label=False,
                 point_cloud_range=[-80, -80, -2, 80, 80, 4],
                 use_collision=False,  # discard
                 use_augment = False,
                 only_img_points=True):# 3D points project onto images
        self.coord_type = coord_type
        self.kernel_size = kernel_size
        self.threshold_depth = threshold_depth
        self.dist = dist
        self.training = training
        self.relative_threshold = relative_threshold
        self.use_run_seg = use_run_seg
        self.only_img = only_img
        self.use_pseudo_label = use_pseudo_label
        self.use_collision = use_collision
        self.point_cloud_range = [-80, -80, -2, 80, 80, 4]
        self.voxel_size = [[0.15, 0.15, 6], [0.05, 0.05, 0.05], [0.1, 0.1, 6]]
        self.dist_size = [[0.6, 0.6, 0.6], [0.1, 0.1, 0.1], [0.4, 0.4, 0]]
        self.kernel_size_ = [[1, 9, 9], [1, 5, 5], [1, 9, 9]]  # [z, y, x]
        self.min_points = 1
        self.use_augment = use_augment
        self.only_img_points = only_img_points

    def filter_points(self, results):
        points = results['points'].tensor.numpy()  # (N,16)
        top_mask = points[:, 6]==0
        points = points[top_mask]
        if results.get('pts_semantic_mask', None) is not None:
            results['pts_semantic_mask'] = results['pts_semantic_mask'][top_mask]
        if results.get('pts_instance_mask', None) is not None:
            results['pts_instance_mask'] = results['pts_instance_mask'][top_mask]
        # points x,y,z,r,e,return_id,lidar_idx,range_dist,lidar_row,lidar_colunm,camid,camid,col1,col2,row1,row2
        in_mask = (points[:,10] != -1) | (points[:,11] != -1)
        # x,y,z,thanh(r),e, bg/fg mask, out/in img, points_inds,x,y,r,g,b 13
        new_points = np.zeros(((points.shape[0]), 18)).astype(np.float32)
        new_points[:, 0:16] = points[:, 0:16]  # x,y,z,r,e,return_id,lidar_idx,row,column
        new_points[:, 16][in_mask] = 1
        new_points[:, 16][~in_mask] = 0
        new_points[:, 17] = 0

        if self.use_augment:
            # resize project coords
            # 1.crop
            # 2.resize
            # 3.flip
            # 4.pad ignore
            # if crop_coords
            crop_coords = results.get('crop_coords', None)
            img_shape = results.get('img_shape', None)
            points_index = np.arange(0, new_points.shape[0]).astype(np.int32)
            flip = results.get('flip', None)
            for i in range(len(results['img'])):
                in_crop_index1 = points_index[new_points[:, 10] == i]
                in_crop_index2 = points_index[new_points[:, 11] == i]
                if 'crop_coords' in results.keys():
                    img_mask = new_points[:, 10] == i
                    in_crop_mask = (new_points[:, 12][img_mask] >= crop_coords[i][0]) & (new_points[:, 12][img_mask] <= crop_coords[i][2]) & \
                        (new_points[:, 14][img_mask] >= crop_coords[i][1]) & (new_points[:, 14][img_mask] <= crop_coords[i][3])
                    in_crop_index1 = points_index[img_mask][in_crop_mask]
                    out_crop_index1 = points_index[img_mask][~in_crop_mask]
                    new_points[:, 12][in_crop_index1] -= crop_coords[i][0]
                    new_points[:, 14][in_crop_index1] -= crop_coords[i][1]
                    new_points[:, 10][out_crop_index1] = -1
                    img_mask = new_points[:, 11] == i
                    in_crop_mask = (new_points[:, 13][img_mask] >= crop_coords[i][0]) & (new_points[:, 13][img_mask] <= crop_coords[i][2]) & \
                        (new_points[:, 15][img_mask] >= crop_coords[i][1]) & (new_points[:, 15][img_mask] <= crop_coords[i][3])
                    in_crop_index2 = points_index[img_mask][in_crop_mask]
                    out_crop_index2 = points_index[img_mask][~in_crop_mask]
                    new_points[:, 13][in_crop_index2] -= crop_coords[i][0]
                    new_points[:, 15][in_crop_index2] -= crop_coords[i][1]
                    new_points[:, 11][out_crop_index2] = -1
                if 'scale_factor' in results.keys():
                    new_points[:, 12][in_crop_index1] *= results['scale_factor'][i][0]
                    new_points[:, 14][in_crop_index1] *= results['scale_factor'][i][0]
                    new_points[:, 13][in_crop_index2] *= results['scale_factor'][i][0]
                    new_points[:, 15][in_crop_index2] *= results['scale_factor'][i][0]
                if 'flip' in results.keys():
                    _, w, _ = img_shape[i]  # not pad shape
                    try:
                        if flip[i] and results['flip_direction'][i]=='horizontal':
                            new_points[:, 12][in_crop_index1] = w - new_points[:, 12][in_crop_index1]
                            new_points[:, 15][in_crop_index2] = w - new_points[:, 15][in_crop_index2]
                    except:
                        pass
            # if crop filter the out img points
            if 'crop_coords' in results.keys():
                in_mask_new = (new_points[:,10]!=-1) | (new_points[:,11]!=-1)
                new_points[:, 16][in_mask_new] = 1
                new_points[:, 16][~in_mask_new] = 0
        else:
            scale = results.get('scale_factor', None)
            if scale is not None:
                new_points[:, 12:16] = new_points[:, 12:16] * scale[0][0]

        if self.only_img_points:
            if results.get('pts_semantic_mask', None) is not None:
                results['pts_semantic_mask'] = results['pts_semantic_mask'][new_points[:, 16]==1]
            if results.get('pts_instance_mask', None) is not None:
                results['pts_instance_mask'] = results['pts_instance_mask'][new_points[:, 16]==1]
            new_points = new_points[new_points[:, 16]==1]

        points_class = get_points_type(self.coord_type)
        results['points'] = points_class(new_points, points_dim=new_points.shape[-1])  # 实例化，LiDARPoints

        if self.use_pseudo_label and 'pseudo_labels' in results.keys():
            # if self.use_augment:
            #     results['pseudo_labels'] = results['pseudo_labels'][in_mask_new]
            assert len(new_points) == len(results['pseudo_labels'])
        return results

    def find_connected_componets_single_batch(self, points, dist):

        this_points = points
        dist_mat = this_points[:, None, :2] - this_points[None, :, :2] # only care about xy
        dist_mat = (dist_mat ** 2).sum(2) ** 0.5
        adj_mat = dist_mat < dist
        adj_mat = adj_mat
        c_inds = connected_components(adj_mat, directed=False)[1]

        return c_inds

    def get_in_2d_box_points(self, results):
        points = results['points'].tensor.numpy()  # (N, 19)
        
        # use pseudo label directly
        if 'pseudo_labels' in results.keys():
            if results['pseudo_labels'].shape[1] > 9:
                mask = results['pseudo_labels'][:, 9]
                points[:, 17] = mask
                # not need
                if self.use_collision and 'pseudo_labels' in results.keys():
                    # discarded
                    collision = results['pseudo_labels'][:, 4][:, None].astype(np.float32)
                    points = np.concatenate((points, collision), axis=1)
                else:
                    points = np.concatenate((points, np.zeros((points.shape[0], 1)).astype(np.float32)), axis=1)
                points_class = get_points_type(self.coord_type)
                results['points'] = points_class(points, points_dim=points.shape[-1])  # 实例化，LiDARPoints N,10
                return results

        # filter points using the ring segment, ccl algorithm is implemented in model forward propagation
        gt_bboxes = results['gt_bboxes']
        labels = results['gt_labels']
        assert len(gt_bboxes)==len(labels)

        gt_mask_all = np.zeros((points.shape[0])).astype(np.bool)
        gt_mask_list = [[] for _ in range(len(labels))]
        for i in range(len(gt_bboxes)):
            if len(gt_bboxes[i])==0:
                continue
            for j, gt_bbox in enumerate(gt_bboxes[i]):
                gt_mask1 = (((points[:, 12] >= gt_bbox[0]) & (points[:, 12] < gt_bbox[2])) &
                            ((points[:, 14] >= gt_bbox[1]) & (points[:, 14] < gt_bbox[3])  &
                             (points[:,10]==i)))
                gt_mask2 = (((points[:, 13] >= gt_bbox[0]) & (points[:, 13] < gt_bbox[2])) &
                            ((points[:, 15] >= gt_bbox[1]) & (points[:, 13] < gt_bbox[3])  &
                             (points[:,11]==i)))
                gt_mask = gt_mask1 | gt_mask2
                gt_mask_all = gt_mask_all | gt_mask
                gt_mask_list[i].append(gt_mask)
        points[:, 17][gt_mask_all] = 1

        out_box_points = points[~gt_mask_all]
        # box_flag = np.zeros((points.shape[0], 3)).astype(np.float32) # box_flag[:,2]是个标志位，表示第一个是否被填充
        # points_index = np.array(range(0, len(points))).astype(np.int)
        labels = results['gt_labels']
        # t = 0
        
        # ring segment-based refine lables
        out_run_sets = np.unique(out_box_points[:, 18])
        for i in range(len(gt_bboxes)):
            if len(gt_bboxes[i])==0:
                continue
            for j, gt_bbox in enumerate(gt_bboxes[i]):
                gt_mask = gt_mask_list[i][j]
                if gt_mask.sum() == 0:
                    continue
                in_box_points = points[gt_mask]
                if self.use_run_seg:
                    # t3 = time.time()
                    run_sets = np.unique(in_box_points[:, 18])
                    for run_id in run_sets:
                        if run_id in out_run_sets:
                            run_mask = points[:, 18]==run_id
                            out_box_mask = out_box_points[:, 18]==run_id
                            prop = out_box_mask.sum() / run_mask.sum()
                            # if labels[i][j] == 1:
                            #     if run_id == -1:
                            #         points[:, 17][(gt_mask) & (run_mask)] = -1
                            #     elif prop >= 0.9:
                            #         points[:, 17][(gt_mask) & (run_mask)] = 0
                            #     # elif prop >= 0.5 and prop < 0.9:
                            #     #     points[:, 17][(gt_mask) & (run_mask)] = -1
                            #     else:
                            #         points[:, 17][(gt_mask) & (run_mask)] = 1
                            # else:
                            if run_id == -1:
                                points[:, 17][(gt_mask) & (run_mask)] = -1
                            elif prop >= 0.5:
                                points[:, 17][(gt_mask) & (run_mask)] = 0
                            elif prop >= 0.05 and prop < 0.5:
                                points[:, 17][(gt_mask) & (run_mask)] = -1
                            else:
                                points[:, 17][(gt_mask) & (run_mask)] = 1
                    # t4 = time.time()
                    # t = t + round(t4 - t3, 3)
                # # CCL
                # in_box_mask = gt_mask & (points[:, 17] > 0)
                # if in_box_mask.sum()==0:
                #     continue
                # cls_id = labels[i][j]
                # c_inds = self.find_connected_componets_single_batch(points[in_box_mask][:, 0:3], self.dist[cls_id])  # use scipy...connected_components (not in GPU)
                # set_c_inds = list(set(c_inds))
                # c_ind = np.argmax([np.sum(c_inds == i) for i in set_c_inds])
                # c_mask = c_inds == set_c_inds[c_ind]
                # c_mask_ = box_flag[:, 2][in_box_mask] == 0
                # max_in_box_index = points_index[in_box_mask][c_mask & c_mask_]
                # box_flag[:, 2][max_in_box_index] = 1
                # box_flag[:, 0][max_in_box_index] = i*1000+j+1
                # max_in_box_index_2 = points_index[in_box_mask][c_mask & (~c_mask_)]
                # box_flag[:, 1][max_in_box_index_2] = i*1000+j+1

        # t6  = time.time()
        # print("\n 去除points时间 {}".format(t))
        # box_flag[:, 0:2] = box_flag[:, 0:2] - 1
        # fg_mask_ = (box_flag[:, 0] != -1) | (box_flag[:,1] != -1)
        # bg_mask_ = ~(fg_mask_ | (points[:, 17]==-1))
        # points[:, 17][bg_mask_] = 0
        # points = np.concatenate((points, box_flag[:,0:2]),axis=1)
        
        # points = self.torch_ccl(points, results) # 可视化需要

        if self.use_collision and 'pseudo_labels' in results.keys():
            # discarded
            collision = results['pseudo_labels'][:, 4][:, None].astype(np.float32)
            points = np.concatenate((points, collision), axis=1)
        else:
            points = np.concatenate((points, np.zeros((points.shape[0], 1)).astype(np.float32)), axis=1)
        points_class = get_points_type(self.coord_type)
        results['points'] = points_class(points, points_dim=points.shape[-1])  # 实例化，LiDARPoints N,10
        return results

    def depth_clustering_segment(self, results):
        points = results['points'].tensor.numpy()
        range_seg_inds = np.ones((points.shape[0])).astype(np.float32)*-1
        if not self.use_run_seg:
            points = np.concatenate((points, range_seg_inds.reshape(points.shape[0], 1)), axis=1)
            points_class = get_points_type(self.coord_type)
            results['points'] = points_class(points, points_dim=points.shape[-1])
            return results
        # top_lidar points mask two return all use
        top_points_mask = (points[:, 6]==0)
        top_points  = points[top_points_mask]
        points_row = points[top_points_mask][:, 8]

        # why? if use history lables, the return_idx is changed, need to regain.
        if np.any(top_points[:, 5] > 1):
            top_points[:, 5] = top_points[:, 5] % 10
        
        row_set, inv_inds = np.unique(points_row, return_inverse=True)
        row_set = row_set.astype(np.int)
        clusters_id_all = []
        column_nums = 2650
        num_clusters = 0
        for r in range(64):
            if r not in row_set:
                clusters_id_all.append(np.ones((2, column_nums))*-1)
                continue

            # 第r行的点云深度排序(2, 2650)
            range_dist = np.ones((2, column_nums))*-1
            # 第一次回波
            fir_return_points = top_points[(top_points[:, 5]==0) & (top_points[:, 8]==r)]
            for p, pts in enumerate(fir_return_points):
                range_dist[0][int(pts[9])] = pts[7]
                # range_dist[0][int(pts[9])] = np.sqrt((pts[0]**2+pts[1]**2))
            # 第二次回波
            sec_return_points = top_points[(top_points[:, 5]==1) & (top_points[:, 8]==r)]
            for p, pts in enumerate(sec_return_points):
                range_dist[1][int(pts[9])] = pts[7]
                # range_dist[1][int(pts[9])] = np.sqrt((pts[0]**2+pts[1]**2))
            
            max_dist = np.max(range_dist)

            # 初始化equalTable, num_cluster, clusters_id
            kernel_size = 10 / max_dist * 50
            kernel_size = np.clip(kernel_size, 12, 50)  # 50米看4个点
            equal_tabel = np.array([i for i in range(2*column_nums)]).reshape((2,column_nums))
            
            clusters_id = np.ones((2, column_nums))*-1
            # 1. 建立equaltree
            for i in range(column_nums):
                if range_dist[0][i] == -1 and range_dist[1][i] == -1:
                    continue
                # 判断第一次回波
                if range_dist[0][i] != -1:
                    for j in range(int(kernel_size/2)):
                        if i >= j and range_dist[0][i-j]!=-1 and (j!=0):
                            dist_flag = self.if_connect(fir_return_points[(fir_return_points[:, 8]==r)&(fir_return_points[:, 9]==i)][0],
                            fir_return_points[(fir_return_points[:, 8]==r)&(fir_return_points[:, 9]==(i-j))][0],
                            j+1, max_dist)
                            if dist_flag:
                                equal_tabel[0][i] = equal_tabel[0][i-j]
                                break
                        # 第一次回波对应位置不与对应位置的第二次回波对比，只有第二次的才与对应第一次的对比
                        if i >= j and range_dist[1][i-j]!=-1 and (j!=0):
                            dist_flag = self.if_connect(fir_return_points[(fir_return_points[:, 8]==r)&(fir_return_points[:, 9]==i)][0],
                            sec_return_points[(sec_return_points[:, 8]==r)&(sec_return_points[:, 9]==(i-j))][0],
                            j+1, max_dist)
                            if dist_flag:
                                equal_tabel[0][i] = equal_tabel[1][i-j]
                                break
                # 判断第二次回波
                if range_dist[1][i] != -1:
                    for j in range(int(kernel_size/2)):                 
                        if i >= j and range_dist[0][i-j]!=-1:
                            dist_flag = self.if_connect(sec_return_points[(sec_return_points[:, 8]==r)&(sec_return_points[:, 9]==i)][0],
                                                        fir_return_points[(fir_return_points[:, 8]==r)&(fir_return_points[:, 9]==(i-j))][0],
                                                        j+1, max_dist)
                            if dist_flag:
                                equal_tabel[1][i] = equal_tabel[0][i-j]
                                break
                        if i >= j and range_dist[1][i-j]!=-1 and (j!=0):
                            dist_flag = self.if_connect(sec_return_points[(sec_return_points[:, 8]==r)&(sec_return_points[:, 9]==i)][0],
                                                        sec_return_points[(sec_return_points[:, 8]==r)&(sec_return_points[:, 9]==(i-j))][0],
                                                        j+1, max_dist)
                            if dist_flag:
                                equal_tabel[1][i] = equal_tabel[1][i-j]
                                break
            # 2. 统一label
            for i in range(column_nums):
                if range_dist[0][i] == -1 and range_dist[1][i] == -1:
                    continue
                if range_dist[0][i] != -1:
                    if equal_tabel[0][i] == i:
                        clusters_id[0][i] = num_clusters
                        num_clusters += 1
                if range_dist[1][i] != -1:
                    if equal_tabel[1][i] == i + column_nums:
                        clusters_id[1][i] = num_clusters
                        num_clusters += 1
            # 3. 重新label
            for i in range(column_nums):
                if range_dist[0][i] == -1 and range_dist[1][i] == -1:
                    continue
                if range_dist[0][i] != -1:
                    label = i
                    while label != equal_tabel[label//column_nums][label-(label//column_nums)*column_nums]:
                        batch_id = label//column_nums
                        label = equal_tabel[batch_id][label-(batch_id)*column_nums]
                    batch_id = label//column_nums
                    clusters_id[0][i] = int(clusters_id[batch_id][label-(batch_id)*column_nums])
                if range_dist[1][i] != -1:
                    label = column_nums + i
                    while label != equal_tabel[label//column_nums][label-(label//column_nums)*column_nums]:
                        batch_id = label//column_nums
                        label = equal_tabel[batch_id][label-(batch_id)*column_nums]
                    batch_id = label//column_nums
                    clusters_id[1][i] = int(clusters_id[batch_id][label-(batch_id)*column_nums])

            clusters_id_all.append(clusters_id)
        clusters_id_all = np.stack(clusters_id_all, 0).transpose(1, 0, 2)  # (64, 2, 2650) --> (2, 64, 2650)
        c, h, w = top_points[:, 5].astype(np.int), top_points[:, 8].astype(np.int), top_points[:, 9].astype(np.int)
        range_seg_inds[top_points_mask] = clusters_id_all[(c, h, w)]

        points = np.concatenate((points, range_seg_inds.reshape(points.shape[0], 1)), axis=1)
        points_class = get_points_type(self.coord_type)
        results['points'] = points_class(points, points_dim=points.shape[-1])
        return results

    def torch_ccl(self, points, results):
        self.voxel_size = [[0.15, 0.15, 6], [0.05, 0.05, 6], [0.1, 0.1, 6]]
        self.dist_size = [[0.6, 0.6, 0], [0.1, 0.1, 0], [0.4, 0.4, 0]]
        self.kernel_size_ = [[1, 9, 9], [1, 5, 5], [1, 9, 9]]  # [z, y, x]
        gt_bboxes = results['gt_bboxes']
        gt_labels = results['gt_labels']
        device = torch.cuda.current_device()
        points = torch.tensor(points, device=device)
        points_index = torch.arange(0, len(points), device=device, dtype=torch.int32)
        box_flag = torch.zeros((points.shape[0], 3), device=device, dtype=torch.int32)
        for j in range(5):
            if len(gt_bboxes[j])==0:
                continue
            for b, gt_bbox in enumerate(gt_bboxes[j]):
                gt_mask1 = (((points[:, 12] >= gt_bbox[0]) & (points[:, 12] < gt_bbox[2])) &
                            ((points[:, 14] >= gt_bbox[1]) & (points[:, 14] < gt_bbox[3])  &
                            (points[:,10]==j)))
                gt_mask2 = (((points[:, 13] >= gt_bbox[0]) & (points[:, 13] < gt_bbox[2])) &
                            ((points[:, 15] >= gt_bbox[1]) & (points[:, 13] < gt_bbox[3])  &
                            (points[:,11]==j)))
                gt_mask = gt_mask1 | gt_mask2
                in_box_mask = gt_mask & (points[:, 17] > 0)
                if in_box_mask.sum() == 0:
                    continue
                cls_id = gt_labels[j][b]
                class_id = torch.zeros((in_box_mask.sum().long()), device=device).long()
                xyz = points[in_box_mask][:, 0:3].contiguous()
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
                # num_act_in = int(20000)
                # c_inds = self.find_connected_componets_single_batch(xyz.cpu().numpy(), self.dist[cls_id])
                # scipy_sets, invs, counts = np.unique(c_inds, return_inverse=True, return_counts=True)
                # nums__ = (invs == scipy_sets[counts.argmax()]).sum()
                # set_c_inds = list(set(c_inds))
                # cluster_inds = torch.tensor(c_inds, device=device)
                xyz = points_padding(xyz, num_act_in, 0).contiguous()
                class_id = points_padding(class_id, num_act_in, -1)
                batch_id = points_padding(batch_id, num_act_in, -1)
                cluster_inds, valid_ind, num_valid, num_clusters = voxel_spccl3d(xyz, 
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
                cluster_inds = cluster_inds[0 : nums]
                cluster_sets, cluster_invs, cluster_counts = torch.unique(cluster_inds, return_inverse=True, return_counts=True)
                c_max_inds = cluster_counts.argmax()
                c_mask = cluster_invs==cluster_sets[c_max_inds]
                # print('\n scipy: 聚类簇数量{},最大簇点数{}'.format(len(scipy_sets), nums__), 
                    #   '\n spccl: 聚类簇数量{},最大簇点数{}'.format(len(cluster_sets), c_mask.sum()))
                # c_mask = torch.ones(cluster_inds.shape, dtype=torch.bool, device=device)
                c_mask_ = box_flag[:, 2][in_box_mask] == 0
                max_in_box_index = points_index[in_box_mask][c_mask & c_mask_].long()
                box_flag[:, 2][max_in_box_index] = 1
                box_flag[:, 0][max_in_box_index] = j*1000+b+1
                max_in_box_index_2 = points_index[in_box_mask][c_mask & (~c_mask_)].long()
                box_flag[:, 1][max_in_box_index_2] = j*1000+b+1
        box_flag[:, 0:2] = box_flag[:, 0:2] - 1
        fg_mask = (box_flag[:, 0] != -1) | (box_flag[:, 1] != -1)
        bg_mask = ~(fg_mask | (points[:, 17]==-1))
        points[:, 17][bg_mask] = 0
        points = torch.cat((points, box_flag[:, 0:2]), dim=1)
        return points.cpu().numpy()

    def if_connect(self, pts1, pts2, times=1, max_dist=50):

        times = np.clip(times, 1, 1.5)
        if np.abs(pts1[7] - pts2[7]) < 0.24 * 1. * max_dist / 50:
            return True
        else:
            return False
            
    def load_ccl_labels(self, results):
        points = results['points'].tensor.numpy()
        run_id = results['pseudo_labels'][:, 0].reshape(-1, 1).astype(np.float64)
        assert len(run_id) == len(points)
        points = np.concatenate((points, run_id), axis=1)
        points_class = get_points_type(self.coord_type)
        results['points'] = points_class(points, points_dim=points.shape[-1])
        return results
    
    def __call__(self, results):
        if self.only_img:
            results = self.filter_points(results)
        else:
            results = self.filter_points(results)
            if self.training:
                if self.use_pseudo_label and 'pseudo_labels' in results.keys():
                    results = self.load_ccl_labels(results)   # get run id
                else:
                    results = self.depth_clustering_segment(results)     # get run id
                    
                results = self.get_in_2d_box_points(results)
        return results

@PIPELINES.register_module(force=True)
class ImageAugment:
    def __init__(self,
                 brightness_range=10,        # 亮度
                 contrast_range=(0.5, 1.5),  # 对比度
                 saturation_range=(0.5, 1.5),# 饱和度
                 hue_range=0.1               # 色调
                 ):        
        self.brightness_range=brightness_range
        self.contrast_range=contrast_range
        self.saturation_range=saturation_range
        self.hue_range=hue_range
    
    def __call__(self, results):
        aug_op = ColorJitter(self.brightness_range,
                             self.contrast_range,
                             self.saturation_range,
                             self.hue_range)
        if 'img' in results.keys():
            for i in range(len(results['img'])):
                if False:
                    import cv2
                    img = cv2.cvtColor(results['img'][i], cv2.COLOR_RGB2BGR)
                    cv2.imwrite('img_{}.jpg'.format(i), img)
                # img rgb
                results['img'][i] = np.array(aug_op(Image.fromarray(results['img'][i])))
                if False:
                    import cv2
                    img = cv2.cvtColor(results['img'][i], cv2.COLOR_RGB2BGR)
                    cv2.imwrite('aug_img_{}.jpg'.format(i), img)
        return results

@PIPELINES.register_module(force=True)
class ResizeMultiViewImageAug:
    """Resize images & bbox & mask.
    
    Args:
        results['img'] is a list or array, to resize fixed size

    Example:
        dict(
            type='ResizeMultiViewImageAug',
            # Target size (w, h)
            img_scale=(1920, 1280),
            ratio_range=(0.5, 0.6),
            multiscale_mode='range',
            keep_ratio=True),
        dict(type='PadMultiViewImageAug', size_divisor=32),
    """

    def __init__(self,
                 img_scale=None,
                 ratio_range=None,
                 multiscale_mode='range',
                 keep_ratio=True,
                 bbox_clip_border=True,
                 backend='cv2',
                 interpolation='bilinear',):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = img_scale
        if ratio_range is not None:
            self.ratio_range = ratio_range
        else:
            self.ratio_range = None
        self.backend = backend
        self.multiscale_mode = multiscale_mode
        self.keep_ratio = keep_ratio
        # TODO: refactor the override option in Resize
        self.interpolation = interpolation
        self.bbox_clip_border = bbox_clip_border

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        for key in results.get('img_fields', ['img']):
            multi_view_img = []
            multi_scale_factor = []
            for i in range(len(results['img'])):
                if self.keep_ratio:
                    img, scale_factor = mmcv.imrescale(
                        results['img'][i],
                        results['scale'][i],
                        return_scale=True,
                        interpolation=self.interpolation,
                        backend=self.backend)
                    # the w_scale and h_scale has minor difference
                    # a real fix should be done in the mmcv.imrescale in the future
                    new_h, new_w = img.shape[:2]
                    h, w = results[key][i].shape[:2]
                    w_scale = new_w / w
                    h_scale = new_h / h

                multi_view_img.append(img)
                multi_scale_factor.append(np.array([w_scale, h_scale, w_scale, h_scale],
                        dtype=np.float32))
                
            results[key] = multi_view_img
            scale_factor = multi_scale_factor

            results['img_shape'] = [img.shape for img in multi_view_img]
            # in case that there is no padding
            results['pad_shape'] = [img.shape for img in multi_view_img]
            results['scale_factor'] = scale_factor
            results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        for key in results.get('bbox_fields', []):
            multi_gt_bboxes = []
            for i in range(len(results['img'])):
                # if None
                # if (results[key][i] == np.array([-1,-1,-1,-1])).all():
                if len(results[key][i]) == 0:
                    multi_gt_bboxes.append(np.array([]))
                # if  Not None
                else:
                    bboxes = results[key][i] * results['scale_factor'][i]
                    if self.bbox_clip_border:
                        img_shape = results['img_shape'][i]
                        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
                    multi_gt_bboxes.append(bboxes)

            results[key] = multi_gt_bboxes

    def _resize_masks(self, results):
        """Resize masks with ``results['scale']``"""
        for key in results.get('mask_fields', []):
            if results[key] is None:
                continue
            for i in range(len(results['img'])):
                # !!! both rescale and resize both are BitmapMasks(), not a np.array
                if self.keep_ratio:
                    # to contrast origin of transforms
                    results[key][i] = results[key][i].rescale(results['scale'][i])
                else:
                    new_shape=results['img_shape'][i][:2]
                    results[key][i] = results[key][i].resize(new_shape)

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            multi_seg = []
            for i in range(len(results['img'])):
                if self.keep_ratio:
                    gt_seg = mmcv.imrescale(
                        results[key][i],
                        results['scale'][i],
                        interpolation='nearest',
                        backend=self.backend)
                else:
                    gt_seg = mmcv.imresize(
                        results[key][i],
                        results['scale'][i],
                        interpolation='nearest',
                        backend=self.backend)
                # if results['scale'][i][::-1] == results[key][i].shape, the gt_seg=results[key][i]
                multi_seg.append(gt_seg)

            results[key] = multi_seg

    def random_sample_ratio(self, img_scale, ratio_range):
        assert isinstance(img_scale, tuple) and len(img_scale)==2
        min_ratio, max_ratio = ratio_range[0], ratio_range[1]
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = (int(img_scale[0] * ratio), int(img_scale[1] * ratio))
        return scale

    def __call__(self, results):
        if self.ratio_range is not None and self.img_scale is not None:
            self.new_scale = [self.random_sample_ratio(self.img_scale, self.ratio_range) for _ in range(len(results['img']))]
        results['scale'] = self.new_scale
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'multiscale_mode={self.multiscale_mode}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str

@PIPELINES.register_module(force=True)
class PadMultiViewImageAug:
    """Pad the image & masks & segmentation map.

    Args:
        size is list[(tuple, optional)]: Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_to_square (bool): Whether to pad the image into a square.
            Currently only used for YOLOX. Default: False.
        pad_val (dict, optional): A dict for padding value, the default
            value is `dict(img=0, masks=0, seg=255)`.
    """

    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_to_square=False,
                 pad_val=dict(img=0, masks=0, seg=-1)):
        self.size = size
        self.size_divisor = size_divisor
        if isinstance(pad_val, float) or isinstance(pad_val, int):
            warnings.warn(
                'pad_val of float type is deprecated now, '
                f'please use pad_val=dict(img={pad_val}, '
                f'masks={pad_val}, seg=255) instead.', DeprecationWarning)
            pad_val = dict(img=pad_val, masks=pad_val, seg=255)
        assert isinstance(pad_val, dict)
        self.pad_val = pad_val
        self.pad_to_square = pad_to_square

        if pad_to_square:
            assert size is None and size_divisor is None, \
                'The size and size_divisor must be None ' \
                'when pad2square is True'
        else:
            assert size is not None or size_divisor is not None, \
                'only one of size and size_divisor should be valid'
            assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        pad_val = self.pad_val.get('img', 0)
        if self.size is None and self.size_divisor is not None:
            img_shape = np.array(results['img_shape'])
            pad_h = int(np.ceil(img_shape[:,0].max() / self.size_divisor)) * self.size_divisor
            pad_w = int(np.ceil(img_shape[:,1].max() / self.size_divisor)) * self.size_divisor
            new_size = [(pad_h, pad_w) for _ in range(len(results['img']))]
        for i in range(len(results['img'])):
            # if only pad top. del: shape=, add parm: padding=(0,1280-886,0,0)
            padded_img = mmcv.impad(
                results['img'][i], shape=new_size[i][:2], pad_val=pad_val)
            results['img'][i] = padded_img
        results['pad_shape'] = [padded_img.shape for padded_img in results['img']]
        results['pad_fixed_size'] = new_size
        results['pad_size_divisor'] = self.size_divisor


    def _pad_masks(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        pad_val = self.pad_val.get('masks', 0)
        for key in results.get('mask_fields', []):
            for i in range(len(results['img'])):
                pad_shape = results['pad_shape'][i][:2]
                results[key][i] = results[key][i].pad(pad_shape, pad_val=pad_val)


    def _pad_seg(self, results):
        """Pad semantic segmentation map according to
        ``results['pad_shape']``."""
        pad_val = self.pad_val.get('seg', -1)
        for key in results.get('seg_fields', []):
            for i in range(len(results['img'])):
                results[key][i] = mmcv.impad(results[key][i], shape=results['pad_shape'][i][:2], pad_val=pad_val)

    def __call__(self, results):
        self._pad_img(results)
        self._pad_masks(results)
        self._pad_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_to_square={self.pad_to_square}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str

@PIPELINES.register_module(force=True)
class RandomFlipMultiImage:
    """Flip the image & bbox & mask.
    Args:
        flip_ratio (float | list[float], optional): The flipping probability.
            Default: None.
        direction(str | list[str], optional): The flipping direction. Options
            are 'horizontal', 'vertical', 'diagonal'. Default: 'horizontal'.
            If input is a list, the length must equal ``flip_ratio``. Each
            element in ``flip_ratio`` indicates the flip probability of
            corresponding direction.
    """
    def __init__(self, flip_ratio=None, direction='horizontal'):
        if isinstance(flip_ratio, list):
            assert mmcv.is_list_of(flip_ratio, float)
            assert 0 <= sum(flip_ratio) <= 1
        elif isinstance(flip_ratio, float):
            assert 0 <= flip_ratio <= 1
        elif flip_ratio is None:
            pass
        else:
            raise ValueError('flip_ratios must be None, float, '
                             'or list of float')
        self.flip_ratio = flip_ratio

        valid_directions = ['horizontal', 'vertical', 'diagonal']
        if isinstance(direction, str):
            assert direction in valid_directions
        elif isinstance(direction, list):
            assert mmcv.is_list_of(direction, str)
            assert set(direction).issubset(set(valid_directions))
        else:
            raise ValueError('direction must be either str or list of str')
        self.direction = direction

        if isinstance(flip_ratio, list):
            assert len(self.flip_ratio) == len(self.direction)

    def bbox_flip(self, bboxes, img_shape, direction):
        """Flip bboxes horizontally.

        Args:
            bboxes (numpy.ndarray): Bounding boxes, shape (..., 4*k)
            img_shape (tuple[int]): Image shape (height, width)
            direction (str): Flip direction. Options are 'horizontal',
                'vertical'.

        Returns:
            numpy.ndarray: Flipped bounding boxes.
        """

        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.copy()
        if direction == 'horizontal':
            w = img_shape[1]
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
        elif direction == 'vertical':
            h = img_shape[0]
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        elif direction == 'diagonal':
            w = img_shape[1]
            h = img_shape[0]
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        else:
            raise ValueError(f"Invalid flipping direction '{direction}'")
        return flipped

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added \
                into result dict.
        """

        if 'flip' not in results:
            if isinstance(self.direction, list):
                # None means non-flip
                direction_list = self.direction + [None]
            else:
                # None means non-flip
                direction_list = [self.direction, None]

            if isinstance(self.flip_ratio, list):
                non_flip_ratio = 1 - sum(self.flip_ratio)
                flip_ratio_list = self.flip_ratio + [non_flip_ratio]
            else:
                non_flip_ratio = 1 - self.flip_ratio
                # exclude non-flip
                single_ratio = self.flip_ratio / (len(direction_list) - 1)
                flip_ratio_list = [single_ratio] * (len(direction_list) -
                                                    1) + [non_flip_ratio]

            cur_dir = [np.random.choice(direction_list, p=flip_ratio_list) for _ in range(len(results['img']))]

            results['flip'] = [cur_dir[i] is not None for i in range(len(results['img']))]
        if 'flip_direction' not in results:
            results['flip_direction'] = cur_dir
        results['ori_gt_bboxes'] = results['gt_bboxes']
        for i in range(len(results['img'])):
            if results['flip'][i]:
                # flip image
                for key in results.get('img_fields', ['img']):
                    results[key][i] = mmcv.imflip(
                        results[key][i], direction=results['flip_direction'][i])
                # flip bboxes
                for key in results.get('bbox_fields', []):
                    if len(results[key][i]) != 0:
                        results[key][i] = self.bbox_flip(results[key][i],
                                                    results['img_shape'][i],
                                                    results['flip_direction'][i])
                # flip masks
                for key in results.get('mask_fields', []):
                    results[key][i] = results[key][i].flip(results['flip_direction'][i])

                # flip segs
                for key in results.get('seg_fields', []):
                    results[key][i] = mmcv.imflip(
                        results[key][i], direction=results['flip_direction'][i])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(flip_ratio={self.flip_ratio})'

@PIPELINES.register_module(force=True)
class RandomFlip3D(RandomFlipMultiImage):
    """Flip the points & bbox.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        sync_2d (bool, optional): Whether to apply flip according to the 2D
            images. If True, it will apply the same flip as that to 2D images.
            If False, it will decide whether to flip randomly and independently
            to that of 2D images. Defaults to True.
        flip_ratio_bev_horizontal (float, optional): The flipping probability
            in horizontal direction. Defaults to 0.0.
        flip_ratio_bev_vertical (float, optional): The flipping probability
            in vertical direction. Defaults to 0.0.
    """

    def __init__(self,
                 sync_2d=True,
                 flip_ratio_bev_horizontal=0.0,
                 flip_ratio_bev_vertical=0.0,
                 **kwargs):
        super(RandomFlip3D, self).__init__(
            flip_ratio=flip_ratio_bev_horizontal, **kwargs)
        self.sync_2d = sync_2d
        self.flip_ratio_bev_vertical = flip_ratio_bev_vertical
        if flip_ratio_bev_horizontal is not None:
            assert isinstance(
                flip_ratio_bev_horizontal,
                (int, float)) and 0 <= flip_ratio_bev_horizontal <= 1
        if flip_ratio_bev_vertical is not None:
            assert isinstance(
                flip_ratio_bev_vertical,
                (int, float)) and 0 <= flip_ratio_bev_vertical <= 1

    def random_flip_data_3d(self, input_dict, direction='horizontal'):
        """Flip 3D data randomly.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            direction (str, optional): Flip direction.
                Default: 'horizontal'.

        Returns:
            dict: Flipped results, 'points', 'bbox3d_fields' keys are
                updated in the result dict.
        """
        assert direction in ['horizontal', 'vertical']
        # for semantic segmentation task, only points will be flipped.
        if 'bbox3d_fields' not in input_dict:
            input_dict['points'].flip(direction)
            return
        if len(input_dict['bbox3d_fields']) == 0:  # test mode
            input_dict['bbox3d_fields'].append('empty_box3d')
            input_dict['empty_box3d'] = input_dict['box_type_3d'](
                np.array([], dtype=np.float32))
        assert len(input_dict['bbox3d_fields']) == 1
        for key in input_dict['bbox3d_fields']:
            if 'points' in input_dict:
                input_dict['points'] = input_dict[key].flip(
                    direction, points=input_dict['points'])
            else:
                input_dict[key].flip(direction)
        if 'centers2d' in input_dict:
            assert self.sync_2d is True and direction == 'horizontal', \
                'Only support sync_2d=True and horizontal flip with images'
            w = input_dict['ori_shape'][1]
            input_dict['centers2d'][..., 0] = \
                w - input_dict['centers2d'][..., 0]
            # need to modify the horizontal position of camera center
            # along u-axis in the image (flip like centers2d)
            # ['cam2img'][0][2] = c_u
            # see more details and examples at
            # https://github.com/open-mmlab/mmdetection3d/pull/744
            input_dict['cam2img'][0][2] = w - input_dict['cam2img'][0][2]

    def __call__(self, input_dict):
        """Call function to flip points, values in the ``bbox3d_fields`` and
        also flip 2D image and its annotations.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction',
                'pcd_horizontal_flip' and 'pcd_vertical_flip' keys are added
                into result dict.
        """
        # Not flip 2D image and its annotations
        # super(RandomFlip3D, self).__call__(input_dict)

        if self.sync_2d:
            input_dict['pcd_horizontal_flip'] = input_dict['flip']
            input_dict['pcd_vertical_flip'] = False
        else:
            if 'pcd_horizontal_flip' not in input_dict:
                flip_horizontal = True if np.random.rand(
                ) < self.flip_ratio else False
                input_dict['pcd_horizontal_flip'] = flip_horizontal
            if 'pcd_vertical_flip' not in input_dict:
                flip_vertical = True if np.random.rand(
                ) < self.flip_ratio_bev_vertical else False
                input_dict['pcd_vertical_flip'] = flip_vertical

        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        if input_dict['pcd_horizontal_flip']:
            self.random_flip_data_3d(input_dict, 'horizontal')
            input_dict['transformation_3d_flow'].extend(['HF'])
        if input_dict['pcd_vertical_flip']:
            self.random_flip_data_3d(input_dict, 'vertical')
            input_dict['transformation_3d_flow'].extend(['VF'])
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(sync_2d={self.sync_2d},'
        repr_str += f' flip_ratio_bev_vertical={self.flip_ratio_bev_vertical})'
        return repr_str

@PIPELINES.register_module(force=True)
class SamFilterPointsByImage:
    """
    The project point cloud is obtained by the Image idx
    """
    def __init__(self, coord_type, kernel_size=3, threshold_depth=0.5,
                dist=(0.6,0.1,0.4), training=True, 
                relative_threshold=0.91, use_run_seg=True,
                only_img=False, use_pseudo_label=False,
                sample_img_id=None):
        self.coord_type = coord_type
        self.kernel_size = kernel_size
        self.threshold_depth = threshold_depth
        self.dist = dist
        self.training = training
        self.relative_threshold = relative_threshold
        self.use_run_seg = use_run_seg
        self.only_img = only_img
        self.use_pseudo_label = use_pseudo_label
        self.sample_img_id = sample_img_id

    def _filter_points(self, results):
        points = results['points'].tensor.numpy()  # (N,16)
        top_mask = points[:,6]==0
        points = points[top_mask] # top lidar
        if results.get('pts_semantic_mask', None) is not None:
            results['pts_semantic_mask'] = results['pts_semantic_mask'][top_mask]
        if results.get('pts_instance_mask', None) is not None:
            results['pts_instance_mask'] = results['pts_instance_mask'][top_mask]
        # points x,y,z,r,e,return_id,lidar_idx,range_dist,lidar_row,lidar_colunm,camid,camid,col1,col2,row1,row2
        in_mask = (points[:,10]!=-1) | (points[:,11]!=-1)

        # x,y,z,thanh(r),e, bg/fg mask, out/in img, points_inds,x,y,r,g,b 13
        new_points = np.zeros(((points.shape[0]), 18)).astype(np.float32)
        new_points[:,0:16] = points[:, 0:16]  # x,y,z,r,e,return_id,lidar_idx,row,column

        new_points[:,16][in_mask] = 1
        new_points[:,16][~in_mask] = 0
        new_points[:,17] = 0

        # resize project coords
        new_points[:,12:16] = new_points[:,12:16] * results['scale_factor'][0]

        # To Test，only load in img points
        if results.get('pts_semantic_mask', None) is not None:
            results['pts_semantic_mask'] = results['pts_semantic_mask'][new_points[:,16]==1]
        if results.get('pts_instance_mask', None) is not None:
            results['pts_instance_mask'] = results['pts_instance_mask'][new_points[:,16]==1]
        new_points = new_points[new_points[:,16]==1]

        points_class = get_points_type(self.coord_type)
        results['points'] = points_class(new_points, points_dim=new_points.shape[-1])  # 实例化，LiDARPoints

        # filter pseudolabel
        if self.use_pseudo_label and 'pseudo_labels' in results.keys():
            # pseudo_labels = results['pseudo_labels']
            # pseudo_labels = np.stack(
            #             [pseudo_labels['run_id'],
            #              pseudo_labels['run'],
            #              pseudo_labels['ignore'],
            #              pseudo_labels['run_ccl'],
            #              pseudo_labels['collision']],axis=-1) # 1 is collision(bg points)
            # assert len(new_points) == len(pseudo_labels)
            if self.sample_img_id is not None:
                # filter single img points
                points = points[in_mask]
                in_tmp_img = (points[:,10]==int(self.sample_img_id)) | (points[:,11]==int(self.sample_img_id))
                pseudo_labels = pseudo_labels[in_tmp_img]
            results['pseudo_labels'] = pseudo_labels

        return results

    def find_connected_componets_single_batch(self, points, dist):

        this_points = points
        dist_mat = this_points[:, None, :2] - this_points[None, :, :2] # only care about xy
        dist_mat = (dist_mat ** 2).sum(2) ** 0.5
        adj_mat = dist_mat < dist
        adj_mat = adj_mat
        c_inds = connected_components(adj_mat, directed=False)[1]

        return c_inds

    def get_in_2d_box_points(self, results):
        points_ = results['points'].tensor.numpy()  # (N,19)
        # in image points
        in_mask = (points_[:,16]==1)
        
        gt_bboxes = results['gt_bboxes']
        labels = results['gt_labels']

        assert len(gt_bboxes)==len(labels)
        gt_mask_all = np.zeros((points_.shape[0])).astype(np.bool)
        # gt_mask_list = [[] for _ in range(len(labels))]

        for i in range(len(gt_bboxes)):
            if len(labels[i])==0:
                continue
            for j, gt_bbox in enumerate(gt_bboxes[i]):
                # 过滤掉空的
                gt_mask1 = (((points_[:, 12] >= gt_bbox[0]) & (points_[:, 12] < gt_bbox[2])) &
                            ((points_[:, 14] >= gt_bbox[1]) & (points_[:, 14] < gt_bbox[3])  &
                              in_mask & (points_[:,10]==i)))
                gt_mask2 = (((points_[:, 13] >= gt_bbox[0]) & (points_[:, 13] < gt_bbox[2])) &
                            ((points_[:, 15] >= gt_bbox[1]) & (points_[:, 13] < gt_bbox[3])  &
                              in_mask & (points_[:,11]==i)))
                gt_mask = gt_mask1 | gt_mask2
                gt_mask_all = gt_mask_all | gt_mask
        points_[:,17][gt_mask_all] = 1  # 在2D box内的点大部分是前景点，后面进行筛选

        box_flag = np.zeros((points_.shape[0],3)).astype(np.float32) # box_flag[:,2]是个标志位，表示第一个是否被填充
        # box_flag[:,2] = 0
        out_box_points = points_[~gt_mask_all]
        points_index = np.array(range(0,len(points_))).astype(np.int)
        labels = results['gt_labels']
        # img
        for i in range(len(gt_bboxes)):
            if len(labels[i])==0:
                continue
            for j, gt_bbox in enumerate(gt_bboxes[i]):
                gt_mask1 = (((points_[:, 12] >= gt_bbox[0]) & (points_[:, 12] < gt_bbox[2])) &
                            ((points_[:, 14] >= gt_bbox[1]) & (points_[:, 14] < gt_bbox[3])  &
                              in_mask & (points_[:,10]==i)))
                gt_mask2 = (((points_[:, 13] >= gt_bbox[0]) & (points_[:, 13] < gt_bbox[2])) &
                            ((points_[:, 15] >= gt_bbox[1]) & (points_[:, 13] < gt_bbox[3])  &
                              in_mask & (points_[:,11]==i)))
                gt_mask = gt_mask1 | gt_mask2
                in_box_points = points_[gt_mask]
                # 进行run过滤计算
                if self.use_run_seg:
                    in_box_points
                    run_sets, inv_inds, counts = np.unique(in_box_points[:,18], return_inverse=True, return_counts=True)
                    for s in range(len(run_sets)):
                        prop = (out_box_points[:,18]==run_sets[s]).sum()/(points_[:,18]==run_sets[s]).sum()
                        if run_sets[s] == -1:
                            points_[:,17][(gt_mask)&(points_[:,18]==run_sets[s])] = -1
                        elif prop >= 0.5:
                            points_[:,17][(gt_mask)&(points_[:,18]==run_sets[s])] = 0
                        elif prop >= 0.05 and prop < 0.5:
                            points_[:,17][(gt_mask)&(points_[:,18]==run_sets[s])] = -1
                        else:
                            points_[:,17][(gt_mask)&(points_[:,18]==run_sets[s])] = 1
                # 更新gt_mask
                in_box_mask = gt_mask & (points_[:,17]>0)
                # 聚类
                # Car=0.6, Pedestrian=0.1, Cyclist=0.4
                # labels[i][j]
                if in_box_mask.sum()==0:
                    continue
                c_inds = self.find_connected_componets_single_batch(points_[in_box_mask][:,0:3], self.dist[labels[i][j]])
                set_c_inds = list(set(c_inds))
                c_ind = np.argmax([np.sum(c_inds == i) for i in set_c_inds])
                c_mask = c_inds == set_c_inds[c_ind]
                # 对box flag 位赋值
                # 一个点可能投射到两个box内
                # c_mask_1表示能够往第一个格子放box索引的位置mask
                c_mask_1 = box_flag[:,2][in_box_mask] == 0
                max_in_box_index = points_index[in_box_mask][c_mask&c_mask_1]
                box_flag[:,2][max_in_box_index] = 1
                box_flag[:,0][max_in_box_index] = i*1000+j+1
                # 第一个位置有box id的地方不能赋值，往第二个里放
                c_mask_2 = box_flag[:,2][in_box_mask] == 0
                max_in_box_index_2 = points_index[in_box_mask][c_mask&c_mask_2]
                box_flag[:,1][max_in_box_index_2] = i*1000+j+1

        box_flag[:,0:2] = box_flag[:,0:2] - 1
        fg_mask_ = (box_flag[:, 0] != -1) | (box_flag[:,1] != -1)
        bg_mask_ = ~(fg_mask_ | (points_[:, 17]==-1))
        points_[:, 17][bg_mask_] = 0
        points_ = np.concatenate((points_, box_flag[:,0:2]),axis=1)
        points_class = get_points_type(self.coord_type)
        results['points'] = points_class(points_, points_dim=points_.shape[-1])  # 实例化，LiDARPoints N,10

        return results

    def range_seg_ccl(self, results):
        points = results['points'].tensor.numpy()
        range_seg_inds = np.ones((points.shape[0])).astype(np.float32)*-1
        if not self.use_run_seg:
            points = np.concatenate((points, range_seg_inds.reshape(points.shape[0],1)), axis=1)
            points_class = get_points_type(self.coord_type)
            results['points'] = points_class(points, points_dim=points.shape[-1])
            return results
        # top_lidar points mask two return all use
        top_points_mask = (points[:,6]==0)
        top_points  = points[top_points_mask]
        points_row = points[top_points_mask][:,8]
        row_set, inv_inds = np.unique(points_row, return_inverse=True)
        row_set = row_set.astype(np.int)
        clusters_id_all = []
        column_nums = 2650
        num_clusters = 0
        for r in range(64):
            if r not in row_set:
                clusters_id_all.append(np.ones((2, column_nums))*-1)
                continue

            # 第r行的点云深度排序(2,2650)
            range_dist = np.ones((2, column_nums))*-1
            # 第一次回波
            fir_return_points = top_points[(top_points[:,5]==0) & (top_points[:,8]==r)]
            for p, pts in enumerate(fir_return_points):
                range_dist[0][int(pts[9])] = pts[7]
                # range_dist[0][int(pts[9])] = np.sqrt((pts[0]**2+pts[1]**2))
            # 第二次回波
            sec_return_points = top_points[(top_points[:,5]==1) & (top_points[:,8]==r)]
            for p, pts in enumerate(sec_return_points):
                range_dist[1][int(pts[9])] = pts[7]
                # range_dist[1][int(pts[9])] = np.sqrt((pts[0]**2+pts[1]**2))
            
            max_dist = np.max(range_dist)

            # 初始化equalTable, num_cluster, clusters_id
            kernel_size = 10 / max_dist * 50
            kernel_size = np.clip(kernel_size, 12, 50)  # 50米看4个点
            equal_tabel = np.array([i for i in range(2*column_nums)]).reshape((2,column_nums))
            # num_clusters = 0
            clusters_id = np.ones((2, column_nums))*-1
            # 1. 建立equaltree
            for i in range(column_nums):
                if range_dist[0][i] == -1 and range_dist[1][i] == -1:
                    continue
                # 判断第一次回波
                if range_dist[0][i] != -1:
                    for j in range(int(kernel_size/2)):
                        if i >= j and range_dist[0][i-j]!=-1 and (j!=0):
                            dist_flag = self.if_connect(fir_return_points[(fir_return_points[:,8]==r)&(fir_return_points[:,9]==i)][0],
                            fir_return_points[(fir_return_points[:,8]==r)&(fir_return_points[:,9]==(i-j))][0],
                            j+1, max_dist)
                            if dist_flag:
                                equal_tabel[0][i] = equal_tabel[0][i-j]
                                break
                        # 第一次回波对应位置不与对应位置的第二次回波对比，只有第二次的才与对应第一次的对比
                        if i >= j and range_dist[1][i-j]!=-1 and (j!=0):
                            dist_flag = self.if_connect(fir_return_points[(fir_return_points[:,8]==r)&(fir_return_points[:,9]==i)][0],
                            sec_return_points[(sec_return_points[:,8]==r)&(sec_return_points[:,9]==(i-j))][0],
                            j+1, max_dist)
                            if dist_flag:
                                equal_tabel[0][i] = equal_tabel[1][i-j]
                                break
                # 判断第二次回波
                if range_dist[1][i] != -1:
                    for j in range(int(kernel_size/2)):                 
                        if i >= j and range_dist[0][i-j]!=-1:
                            dist_flag = self.if_connect(sec_return_points[(sec_return_points[:,8]==r)&(sec_return_points[:,9]==i)][0],
                                                        fir_return_points[(fir_return_points[:,8]==r)&(fir_return_points[:,9]==(i-j))][0],
                                                        j+1, max_dist)
                            if dist_flag:
                                equal_tabel[1][i] = equal_tabel[0][i-j]
                                break
                        if i >= j and range_dist[1][i-j]!=-1 and (j!=0):
                            dist_flag = self.if_connect(sec_return_points[(sec_return_points[:,8]==r)&(sec_return_points[:,9]==i)][0],
                                                        sec_return_points[(sec_return_points[:,8]==r)&(sec_return_points[:,9]==(i-j))][0],
                                                        j+1, max_dist)
                            if dist_flag:
                                equal_tabel[1][i] = equal_tabel[1][i-j]
                                break
            # 2. 统一label
            for i in range(column_nums):
                if range_dist[0][i] == -1 and range_dist[1][i] == -1:
                    continue
                if range_dist[0][i] != -1:
                    if equal_tabel[0][i] == i:
                        clusters_id[0][i] = num_clusters
                        num_clusters += 1
                if range_dist[1][i] != -1:
                    if equal_tabel[1][i] == i + column_nums:
                        clusters_id[1][i] = num_clusters
                        num_clusters += 1
            # 3. 重新label
            for i in range(column_nums):
                if range_dist[0][i] == -1 and range_dist[1][i] == -1:
                    continue
                if range_dist[0][i] != -1:
                    label = i
                    while label != equal_tabel[label//column_nums][label-(label//column_nums)*column_nums]:
                        batch_id = label//column_nums
                        label = equal_tabel[batch_id][label-(batch_id)*column_nums]
                    batch_id = label//column_nums
                    clusters_id[0][i] = int(clusters_id[batch_id][label-(batch_id)*column_nums])
                if range_dist[1][i] != -1:
                    label = column_nums + i
                    while label != equal_tabel[label//column_nums][label-(label//column_nums)*column_nums]:
                        batch_id = label//column_nums
                        label = equal_tabel[batch_id][label-(batch_id)*column_nums]
                    batch_id = label//column_nums
                    clusters_id[1][i] = int(clusters_id[batch_id][label-(batch_id)*column_nums])

            clusters_id_all.append(clusters_id)
        clusters_id_all = np.stack(clusters_id_all, 0).transpose(1,0,2)  # (64,2,2650)-->(2,64,2650)
        c, h, w = top_points[:,5].astype(np.int), top_points[:,8].astype(np.int), top_points[:,9].astype(np.int)
        range_seg_inds[top_points_mask] = clusters_id_all[(c,h,w)]

        points = np.concatenate((points, range_seg_inds.reshape(points.shape[0],1)), axis=1)
        points_class = get_points_type(self.coord_type)
        results['points'] = points_class(points, points_dim=points.shape[-1])
        return results

    def if_connect(self, pts1, pts2, times=1, max_dist=50):
        times = np.clip(times, 1, 1.5)
        if np.abs(pts1[7] - pts2[7]) < 0.24 * 1. * max_dist / 50:
            return True
        else:
            return False

    def __call__(self, results):
        if self.only_img:
            results = self._filter_points(results)
        else:
            results = self._filter_points(results)
            if self.training:
                results = self.range_seg_ccl(results)
                results = self.get_in_2d_box_points(results)   # (gt_box的数量, 3) 不在2dbox内的点的points的fg_points标志位置-1
        if not self.use_pseudo_label:
            results['pseudo_labels'] = np.array([])
        return results

@PIPELINES.register_module(force=True)
class PointsRangeFilter(object):
    """Filter points by the range.

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, input_dict):
        """Call function to filter points by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'points', 'pts_instance_mask'
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = input_dict['points']
        points_mask = points.in_range_3d(self.pcd_range)
        clean_points = points[points_mask]
        input_dict['points'] = clean_points
        points_mask = points_mask.numpy()

        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)

        if pts_instance_mask is not None:
            input_dict['pts_instance_mask'] = pts_instance_mask[points_mask]

        if pts_semantic_mask is not None:
            input_dict['pts_semantic_mask'] = pts_semantic_mask[points_mask]

        if 'pseudo_labels' in input_dict.keys():
            if len(input_dict['pseudo_labels']) == len(points):
                input_dict['pseudo_labels'] = input_dict['pseudo_labels'][points_mask]
            
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(point_cloud_range={self.pcd_range.tolist()})'
        return repr_str

@PIPELINES.register_module(force=True)
class FilterPoints:
    """
    (Opotional) Filter out points that are not projected onto the images.
    """
    def __init__(self, 
                coord_type='LIDAR',
                num_classes=3,
                only_img_points=True):  # only load points map to 2D images
        self.coord_type = coord_type
        self.num_classes = num_classes
        self.only_img_points = only_img_points

    def filter_points(self, results):
        # filter points which is obtain by top lidar
        # may not remove other points get better results?
        points = results['points'].tensor.numpy()  # (N,16)
        top_mask = points[:, 6] == 0
        points = points[top_mask] # top lidar
        if results.get('pts_semantic_mask', None) is not None:
            results['pts_semantic_mask'] = results['pts_semantic_mask'][top_mask]
        if results.get('pts_instance_mask', None) is not None:
            results['pts_instance_mask'] = results['pts_instance_mask'][top_mask]

        # points x,y,z,r,e,return_id,lidar_idx,range_dist,lidar_row,lidar_colunm,camid,camid,col1,col2,row1,row2
        if self.only_img_points:
            in_mask = (points[:,10] !=- 1) | (points[:,11] != -1)
            points = points[in_mask]
            if results.get('pts_semantic_mask', None) is not None:
                results['pts_semantic_mask'] = results['pts_semantic_mask'][in_mask]
            if results.get('pts_instance_mask', None) is not None:
                results['pts_instance_mask'] = results['pts_instance_mask'][in_mask]

        points_class = get_points_type(self.coord_type)
        results['points'] = points_class(points, points_dim=points.shape[-1])  # 实例化，LiDARPoints

        # filter pseudo-labels
        if 'pseudo_labels' in results.keys():
            pseudo_labels = results['pseudo_labels']
            pseudo_labels = np.stack(
                        [pseudo_labels['run_id'],  # ring-segment id
                         pseudo_labels['run'],     # use ring-segment id as flag to filter outside 2D box points, get flag 1 is in box; 0 is bg; -1 is ignore (blurry)
                         pseudo_labels['ignore'],
                         pseudo_labels['run_ccl'],
                         pseudo_labels['collision']], axis=-1).astype(np.float32) # 1 is collision(bg points)
            # pseudo instance label
            if 'ccl_labels' in results.keys():
                pseudo_labels = np.concatenate((pseudo_labels, results['ccl_labels']), axis=1)

            if self.only_img_points:
                assert len(points) == len(pseudo_labels)
                results['pseudo_labels'] = pseudo_labels
            else:
                # input all points
                points = results['points'].tensor.numpy()
                # out_img_mask = (points[:, 10] == -1) & (points[:, 11] == -1)
                in_img_mask = (points[:, 10] != -1) | (points[:, 11] != -1)
                new_pseudo_labels = np.ones((points.shape[0], pseudo_labels.shape[1])).astype(np.float32) * -1
                new_pseudo_labels[in_img_mask] = pseudo_labels
                results['pseudo_labels'] = new_pseudo_labels

        return results
    
    def __call__(self, results):
        results = self.filter_points(results)
        return results

@PIPELINES.register_module(force=True)
class LoadHistoryLabel:
    """
    (Opotional) Load history points.
    (Opotional) Load sam labels.
    """
    def __init__(self, 
                coord_type='LIDAR',
                use_history_labels=False,
                num_classes=3,
                history_nums=4,
                use_sam_labels=False,
                only_img_points=True):  # only load points map to 2D images
        self.coord_type = coord_type
        self.use_history_labels = use_history_labels
        self.num_classes = num_classes
        self.history_nums = history_nums
        self.use_sam_labels = use_sam_labels
        self.only_img_points = only_img_points

    def load_history_labels(self, results):
        points = results['points'].tensor.numpy()
        # point's index * 10 + return_id
        points[:, 5] = points[:, 5] + np.arange(len(points)) * 10
        sample_idx = results['sample_idx']

        if not os.path.exists('./work_dirs/results/history_labels'):
            os.makedirs('./work_dirs/results/history_labels')
        path = './work_dirs/results/history_labels/{}.{}'.format(sample_idx, 'npy')

        if os.path.exists(path):
            while True:
                try:
                    labels = np.load(path)
                except ValueError:
                    labels = np.ones((len(points), self.history_nums, self.num_classes)).astype(np.float32) * -1 # self.num_classes
                    np.save(path, labels)
                    print('\n history labels ValueError!!!!!')
                    break
                except:
                    time.sleep(0.1)
                    print('\n sleep!!!!!')
                    continue
                break
        else:
            labels = np.ones((len(points), self.history_nums, self.num_classes)).astype(np.float32) * -1 # self.num_classes
            np.save(path, labels)
            # time.sleep(0.1)
        if labels.shape[0] != points.shape[0]:
            print("\n re-generate history labels")
            labels = np.ones((len(points), self.history_nums, self.num_classes)).astype(np.float32) * -1 # self.num_classes
            np.save(path, labels)
            time.sleep(0.1)
        
        results['history_labels'] = labels
        return results

    # need to finish sam
    def load_sam_labels(self, results):
        points = results['points'].tensor.numpy()
        # point's index * 10 + return_id
        if not self.use_history_labels:
            points[:, 5] = points[:, 5] + np.arange(len(points)) * 10
        sample_idx = results['sample_idx']

        if not os.path.exists('./work_dirs/sam_masks'):
            os.makedirs('./work_dirs/results/history_labels')

        if os.path.exists('/home/jiangguangfeng/桌面/codebase/'):
            path = '/home/jiangguangfeng/桌面/codebase/sam_masks/{}.{}'.format(sample_idx, 'npy')
        else:
            path = './work_dirs/sam_masks/{}.{}'.format(sample_idx, 'npy')

        if os.path.exists(path):
            while True:
                try:
                    labels = np.load(path)
                    if len(labels.shape) != 1:
                        labels = np.ones(len(points)).astype(np.float32) * -1 # self.num_classes
                        np.save(path, labels)
                except ValueError:
                    labels = np.ones(len(points)).astype(np.float32) * -1 # self.num_classes
                    # np.save(path, labels)
                    print('\n sam labels ValueError!!!!!')
                    break
                except:
                    time.sleep(0.1)
                    print('\n sleep!!!!!')
                    continue
                break
        else:
            labels = np.ones(len(points)).astype(np.float32) * -1 # self.num_classes
            np.save(path, labels)

        if self.only_img_points:
            assert labels.shape[0] == points.shape[0]
        else:
            if labels.shape[0] != points.shape[0]:
                # input all points
                in_img_mask = (points[:, 10] != -1) | (points[:, 11] != -1)
                new_sam_masks = np.ones((points.shape(0), labels.shape(1))).astype(np.float32) * -1
                new_sam_masks[in_img_mask] = labels
                results['sam_masks'] = new_sam_masks

        results['sam_masks'] = labels
        return results
    
    def __call__(self, results):
        if self.use_history_labels:
            results = self.load_history_labels(results)
        if self.use_sam_labels:
            results = self.load_sam_labels(results)
        return results

# @PIPELINES.register_module()
class CycAugment:

    def __init__(self, file_client_args, lwsis_client_args, sample_nums=1000):
        self.file_client_args = file_client_args.copy()
        self.lwsis_client_args = lwsis_client_args.copy()
        self.lwsis_client = None
        self.file_client = None
        self.cyc_image_id_list = self.get_image_id()
        self.sample_nums = sample_nums
   
    def get_image_id(self):
        if os.path.exists('/root/3D/work_dirs/dataset_infos/cyc_image_id.pkl'):
            f = open('/root/3D/work_dirs/dataset_infos/cyc_image_id.pkl', "rb+")
        elif os.path.exists('/home/jiangguangfeng/桌面/codebase/cyc_image_id.pkl'):
            f = open('/home/jiangguangfeng/桌面/codebase/cyc_image_id.pkl', "rb+")
        else:
            raise ValueError('please sure you have cyc dataset')
        cyc_ret = pickle.load(f)     
        return cyc_ret
    
    def load_data(self, results, sample_image_id):
        # load image
        image_path_prefix = 'training/image'
        image_path = "{}/{}/{}".format(image_path_prefix, '%07d'%1, str(sample_image_id%10))
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        pts_bytes = self.file_client.get(image_path)
        img = mmcv.imfrombytes(pts_bytes, flag=self.color_type, channel_order='bgr')
        # load lables
        labels_path_prefix = 'training/img'
        if self.lwsis_client is None:
            self.lwsis_client = mmcv.FileClient(**self.lwsis_client_args)
        path = "{}/{}".format(labels_path_prefix, str(sample_image_id))
        if self.lwsis_client.exists(path):
            label_bytes = self.lwsis_client.get(path)
            labels = np.load(BytesIO(label_bytes), allow_pickle=True)
            img_bbox = []
            img_label = []
            img_mask = []
            for j in range(len(labels)):
                tmp_bbox = np.array(labels[j][0]['bbox']) + np.array([0, 0, labels[j][0]['bbox'][0], labels[j][0]['bbox'][1]])
                img_bbox.append(tmp_bbox)
                img_label.append(labels[j][0]['category_id'] - 1)  # 1, 2, 3
                img_mask.append(mask_utils.decode(labels[j][0]['segmentation']))
            if len(img_bbox) != 0:
                img_bbox = np.array(img_bbox)
                img_label = np.array(img_label)
                img_mask = np.array(img_mask)
                n, h, w = img_mask.shape
                img_mask = BitmapMasks(img_mask, h, w)
            else:
                img_bbox = np.array([]).astype(np.float32)
                img_label = np.array([]).astype(np.float32)
                img_mask = np.array(img_mask)
                h, w, c = results['img'][sample_image_id%10].shape
                img_mask = BitmapMasks(np.array([]), h, w)
        else:
            raise ValueError("path: {} is not exists".format(path))
        # load pts
        
        return img, img_bbox, img_label, img_mask

    def __call__(self, results):
        # if cyc
        if np.random.choice(self.sample_nums) == 1:
            # sample image_id
            sample_image_id = random.choice(self.cyc_image_id_list)
            sample_frame = sample_image_id // 10
            sample_frame_img_indx = sample_image_id % 10
            # load cyc image
            img, img_bbox, img_label, img_mask = self.load_data(results, sample_image_id)
            
        return results

@PIPELINES.register_module(force=True)
class PointShuffle(object):
    """Shuffle input points."""

    def __call__(self, input_dict):
        """Call function to shuffle points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'points', 'pts_instance_mask'
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        idx = input_dict['points'].shuffle()
        idx = idx.numpy()

        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)

        if pts_instance_mask is not None:
            input_dict['pts_instance_mask'] = pts_instance_mask[idx]

        if pts_semantic_mask is not None:
            input_dict['pts_semantic_mask'] = pts_semantic_mask[idx]
        
        if 'pseudo_labels' in input_dict.keys():
            input_dict['pseudo_labels'] = input_dict['pseudo_labels'][idx]
            
        return input_dict

    def __repr__(self):
        return self.__class__.__name__

import cv2
@PIPELINES.register_module(force=True)
class YOLOXHSVRandomAug:
    """Apply HSV augmentation to image sequentially. It is referenced from
    https://github.com/Megvii-
    BaseDetection/YOLOX/blob/main/yolox/data/data_augment.py#L21.

    Args:
        hue_delta (int): delta of hue. Default: 5.
        saturation_delta (int): delta of saturation. Default: 30.
        value_delta (int): delat of value. Default: 30.
    """

    def __init__(self, hue_delta=5, saturation_delta=30, value_delta=30):
        self.hue_delta = hue_delta
        self.saturation_delta = saturation_delta
        self.value_delta = value_delta

    def __call__(self, results):
        img = results['img']
        for i in range(5):
            hsv_gains = np.random.uniform(-1, 1, 3) * [
                self.hue_delta, self.saturation_delta, self.value_delta
            ]
            # random selection of h, s, v
            hsv_gains *= np.random.randint(0, 2, 3)
            # prevent overflow
            hsv_gains = hsv_gains.astype(np.int16)
            img_hsv = cv2.cvtColor(img[i], cv2.COLOR_BGR2HSV).astype(np.int16)

            img_hsv[..., 0] = (img_hsv[..., 0] + hsv_gains[0]) % 180
            img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_gains[1], 0, 255)
            img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_gains[2], 0, 255)
            cv2.cvtColor(img_hsv.astype(img[i].dtype), cv2.COLOR_HSV2BGR, dst=img[i])

            results['img'][i] = img[i]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(hue_delta={self.hue_delta}, '
        repr_str += f'saturation_delta={self.saturation_delta}, '
        repr_str += f'value_delta={self.value_delta})'
        return repr_str