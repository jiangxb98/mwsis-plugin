import mmcv
import numpy as np
from mmdet3d.core.points import get_points_type, BasePoints
from mmdet3d.datasets.pipelines import LoadAnnotations3D
from mmdet.datasets.pipelines import LoadAnnotations
from io import BytesIO
from mmdet3d.datasets.builder import PIPELINES
import torch
from mmdet.core import BitmapMasks, PolygonMasks
import cv2
from pycocotools import mask as mask_utils

@PIPELINES.register_module(force=True)
class LoadPoints(object):
    def __init__(self,
                 coord_type,
                 remove_close=False,
                 file_client_args=dict(backend='disk')):
        self.coord_type = coord_type
        self.remove_close = remove_close
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _remove_close(self, points, radius=1.0):
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        not_close = np.linalg.norm(
            points_numpy[:, :2], ord=2, axis=-1) >= radius
        return points[not_close]

    def _load_points(self, results, token):
        # if self.file_client is None:
        #     self.file_client = mmcv.FileClient(**self.file_client_args)
        # pts_bytes = self.file_client.get(token)
        # points = results['pts_info']['pts_loader'](results, pts_bytes)
        if token.endswith('.npy'):
            points = np.load(token)
        else:
            points = np.fromfile(token, dtype=np.float32, count=-1).reshape([-1, 16])

        # have a bug need to fix
        # swap the dim
        points[:, 3] = np.tanh(points[:, 3])
        points[:, 5:8] = points[:, [6, 7, 5]]
        return points

    def __call__(self, results):
        token = results['pts_info']['path']
        points = self._load_points(results, token)
        if self.remove_close:
            points = self._remove_close(points)
        points_class = get_points_type(self.coord_type)
        points = points_class(points, points_dim=points.shape[-1])  # Instance，LiDARPoints
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'coord_type={self.coord_type}, '
        repr_str += f'remove_close={self.remove_close}, '
        repr_str += f'file_client_args={self.file_client_args})'
        return repr_str

@PIPELINES.register_module(force=True)
class MyLoadPoints(object):
    def __init__(self,
                 coord_type,
                 remove_close=False,
                 file_client_args=dict(backend='disk')):
        self.coord_type = coord_type
        self.remove_close = remove_close
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _remove_close(self, points, radius=1.0):
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        not_close = np.linalg.norm(
            points_numpy[:, :2], ord=2, axis=-1) >= radius
        return points[not_close]

    def _load_points(self, results, token):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        pts_bytes = self.file_client.get(token)
        points = results['pts_info']['pts_loader'](results, pts_bytes)
        return points

    def __call__(self, results):
        token = results['pts_info']['path']
        # load points from oss
        points = self._load_points(results, token)
        if self.remove_close:
            points = self._remove_close(points)
        points_class = get_points_type(self.coord_type)
        points = points_class(points, points_dim=points.shape[-1])  # Instance，LiDARPoints
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'coord_type={self.coord_type}, '
        repr_str += f'remove_close={self.remove_close}, '
        repr_str += f'file_client_args={self.file_client_args})'
        return repr_str


@PIPELINES.register_module(force=True)
class MyLoadSweeps(MyLoadPoints):
    def __init__(self,
                 sweeps_num,
                 coord_type,
                 file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False,
                 remove_close=False,
                 random_choose=True,
                 test_mode=False):
        super(MyLoadSweeps, self).__init__(
            coord_type, remove_close, file_client_args)
        self.sweeps_num = sweeps_num
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        self.random_choose = random_choose

    def __call__(self, results):
        points = results['points']
        sweep_ts = points.tensor.new_zeros((len(points), 1))
        sweep_points_list = [points]
        sweep_ts_list = [sweep_ts]
        pts_info = results['pts_info']
        ts = pts_info['timestamp']
        dts = pts_info['timestamp_step']
        if self.pad_empty_sweeps and len(pts_info['sweeps']) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    points_remove_close = self._remove_close(points)
                    sweep_ts_remove_close = points.tensor.new_zeros(
                        (len(points_remove_close), 1))
                    sweep_points_list.append(points_remove_close)
                    sweep_ts_list.append(sweep_ts_remove_close)
                else:
                    sweep_points_list.append(points)
                    sweep_ts_list.append(sweep_ts)
        else:
            if len(pts_info['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(pts_info['sweeps']))
            elif self.test_mode or (not self.random_choose):
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(
                    len(pts_info['sweeps']), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = pts_info['sweeps'][idx]
                points_sweep = self._load_points(results, sweep['path'])
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep['timestamp']
                rel_pose = sweep['rel_pose']
                points_sweep[:, :3] = points_sweep[:, :3] @ rel_pose[:3, :3].T
                points_sweep[:, :3] += rel_pose[:3, 3][None, :]
                points_sweep_ts = points.tensor.new_full(
                    (len(points_sweep), 1), (ts - sweep_ts) * dts)
                sweep_ts_list.append(points_sweep_ts)
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        sweep_ts = torch.cat(sweep_ts_list, dim=0)
        new_points = torch.cat((points.tensor, sweep_ts), dim=-1)
        points = type(points)(new_points, points_dim=new_points.shape[-1],
                              attribute_dims=points.attribute_dims)
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'

@PIPELINES.register_module(force=True)
class LoadAnnos3D(LoadAnnotations3D):
    def __init__(self,
                 *args,
                 **kwargs):
        super(LoadAnnos3D, self).__init__(*args, **kwargs)

    def _load_masks_3d(self, results):
        # instance mask and semantic have the same path
        pts_instance_mask_path = results['ann_info']['pts_instance_mask_path']
        pts_instance_mask_loader = results['ann_info']['pts_instance_mask_loader']

        if not pts_instance_mask_path.endswith('.npy'):
            pts_instance_mask_path = pts_instance_mask_path + '.npy'
        
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        mask_bytes = self.file_client.get(pts_instance_mask_path)
        pts_instance_mask = pts_instance_mask_loader(results, mask_bytes, 'instance_id')
        results['pts_instance_mask'] = pts_instance_mask
        results['pts_mask_fields'].append('pts_instance_mask')
        return results

    def _load_semantic_seg_3d(self, results):
        pts_semantic_mask_path = results['ann_info']['pts_semantic_mask_path']
        pts_semantic_mask_loader = results['ann_info']['pts_semantic_mask_loader']

        if not pts_semantic_mask_path.endswith('.npy'):
            pts_semantic_mask_path = pts_semantic_mask_path + '.npy'

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        mask_bytes = self.file_client.get(pts_semantic_mask_path)
        pts_semantic_mask = pts_semantic_mask_loader(results, mask_bytes, 'semseg_cls')
        results['pts_semantic_mask'] = pts_semantic_mask
        results['pts_seg_fields'].append('pts_semantic_mask')
        return results

@PIPELINES.register_module(force=True)
class MyLoadAnnos3D(LoadAnnotations3D):
    def __init__(self,
                 *args,
                 **kwargs):
        super(MyLoadAnnos3D, self).__init__(*args, **kwargs)

    def _load_masks_3d(self, results):
        # instance mask and semantic have the same path
        pts_instance_mask_path = results['ann_info']['pts_instance_mask_path']
        pts_instance_mask_loader = results['ann_info']['pts_instance_mask_loader']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        mask_bytes = self.file_client.get(pts_instance_mask_path)
        pts_instance_mask = pts_instance_mask_loader(results, mask_bytes, 'instance_id')
        results['pts_instance_mask'] = pts_instance_mask
        results['pts_mask_fields'].append('pts_instance_mask')
        return results

    def _load_semantic_seg_3d(self, results):
        pts_semantic_mask_path = results['ann_info']['pts_semantic_mask_path']
        pts_semantic_mask_loader = results['ann_info']['pts_semantic_mask_loader']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        mask_bytes = self.file_client.get(pts_semantic_mask_path)
        pts_semantic_mask = pts_semantic_mask_loader(results, mask_bytes, 'semseg_cls')
        results['pts_semantic_mask'] = pts_semantic_mask
        results['pts_seg_fields'].append('pts_semantic_mask')
        return results


@PIPELINES.register_module(force=True)
class LoadImages(object):
    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 pad_shape=None,
                 ):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_shape = pad_shape

    def _load_img(self, results, token):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        pts_bytes = self.file_client.get(token)
        # not need loader, use mmcv.imfrombytes
        # img = results['img_info']['img_loader'](results, pts_bytes)
        img = mmcv.imfrombytes(pts_bytes, flag=self.color_type, channel_order='bgr')
        return img

    def __call__(self, results):
        results['filename'] = []
        results['img'] = []
        results['img_shape'] = []
        results['ori_shape'] = []
        results['img_fields'] = ['img']
        results['lidar2img'] = []
        results['pad_shape'] = self.pad_shape
        for i in range(len(results['img_info']['img_path_info'])):
            filename = results['img_info']['img_path_info'][i]['filename']
            img = self._load_img(results, filename)
            results['img'].append(img)
            results['filename'].append(filename)
            results['img_shape'].append(img.shape)
            results['ori_shape'].append(img.shape)
            if 'lidar2img' in results.keys() and len(results['img_info']['img_path_info'][i]['lidar2img']) != 0:
                results['lidar2img'].append(results['img_info']['img_path_info'][i]['lidar2img'])
        return results
    
    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@PIPELINES.register_module(force=True)
class LoadAnnos(LoadAnnotations):
    def __init__(self, *arg, **kwargs):
        super(LoadAnnos, self).__init__(*arg, **kwargs)
    
    def _load_bboxes(self, results):
        ann_info = results['ann_info']
        # some imgaes only front camera has 2d gt bbox
        results['gt_bboxes'] = ann_info['gt_bboxes'].copy()
        if len(results['gt_bboxes']) == 0:
            return None
        # if no gt_bboxes then is np.array([-1,-1,-1,-1])
        # for i in range(len(results['gt_bboxes'])):
            # if results['gt_bboxes'][i].shape[0] == 0:
                # results['gt_bboxes'][i] = np.array([-1,-1,-1,-1])
                # pass
        results['bbox_fields'].append('gt_bboxes')

        return results
    
    def _load_labels(self, results):
        results['gt_labels'] = results['ann_info']['gt_labels'].copy()
        # if no gt_labels then is [-1]
        # for i in range(len(results['gt_labels'])):
            # if results['gt_labels'][i].shape[0] == 0:
                # results['gt_labels'][i] = np.array([-1])
                # pass
        return results

    def _load_semantic_seg(self, results):
        pan_semantic_mask_path = results['ann_info']['pan_semantic_mask_path']
        pan_semantic_mask_loader = results['ann_info']['pan_semantic_mask_loader']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        results['gt_semantic_seg'] = []

        for pan_seg in pan_semantic_mask_path:
            mask_bytes = self.file_client.get(pan_seg)
            pan_semantic_mask = pan_semantic_mask_loader(results, mask_bytes, 'panseg_cls').squeeze()
            pan_semantic_mask.dtype = "int16"
            results['gt_semantic_seg'].append(pan_semantic_mask)

        results['seg_fields'].append('gt_semantic_seg')
        results['gt_semantic_seg'] = results['gt_semantic_seg']
        return results

    def _load_masks(self, results):
        # instance mask and semantic have the same path
        pan_instance_mask_path = results['ann_info']['pan_instance_mask_path']
        pan_instance_mask_loader = results['ann_info']['pan_instance_mask_loader']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        results['gt_masks'] = []

        for pan_seg in pan_instance_mask_path:
            mask_bytes = self.file_client.get(pan_seg)
            pan_instance_mask = pan_instance_mask_loader(results, mask_bytes, 'panseg_instance_id')
            pan_instance_mask.dtype = "int16"
            # HxWx1 -->  1xHxW
            pan_instance_mask = pan_instance_mask.transpose((2,0,1))
            h, w = pan_instance_mask.shape[1], pan_instance_mask.shape[2]
            if self.poly2mask:
                gt_masks = BitmapMasks(pan_instance_mask, h, w)
            else:
                gt_masks = PolygonMasks(self.process_polygons(pan_instance_mask))

            results['gt_masks'].append(gt_masks)

        results['mask_fields'].append('gt_masks')
        return results

@PIPELINES.register_module()
class LoadPseudoLabel:
    '''
    load pseudo labels by the SPG, and then save to results['pseudo_label'].
    the loaded pseudo labels is preprocessed with code and soted local file.
    '''
    def __init__(self,
                 coord_type,
                 file_client_args,
                 pseudo_label_path):
        self.coord_type = coord_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pseudo_label_path = pseudo_label_path

    def load_pseudo_label(self, results):
        if  self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        frame_id = results['frame_id']
        path = f'{self.pseudo_label_path}/'+ f'{str(frame_id).zfill(7)}.npy'
        pseudo_labels_bytes = self.file_client.get(path)
        pseudo_labels = np.load(pseudo_labels_bytes)
        # ring segment id = points[:, 0], mask flag = points[:, 1], spg results = points[:, [2, 3]]
        results['pseudo_labels'] = pseudo_labels
        return results

    def __call__(self, results):
        results = self.load_pseudo_label(results)
        return results

import os.path as osp
@PIPELINES.register_module()
class LoadImageFromFile:
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 channel_order='bgr',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_img(self, results, token):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        pts_bytes = self.file_client.get(token)
        # not need loader, use mmcv.imfrombytes
        # img = results['img_info']['img_loader'](results, pts_bytes)
        img = mmcv.imfrombytes(pts_bytes, flag=self.color_type, channel_order=self.channel_order)
        # import tensorflow as tf
        # tf.image.decode_jpeg(pts_bytes) # both two function have the same size (1280,1920,3) but have tiny distinction
        return img

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        filename = results['img_info']['img_path_info']['filename']
        img = self._load_img(results, filename)
        results['filename'] = filename
        results['ori_filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"channel_order='{self.channel_order}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@PIPELINES.register_module(force=True)
class LoadAnnosLwsis:
    def __init__(self, file_client_args, prefix='training'):
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.prefix = prefix
        assert prefix in ['training', 'validation']
    
    def _load_labels(self, results):
        img_path_prefix = '{}/{}'.format(self.prefix, 'img') #'training/img'
        sample_idx = results['sample_idx']
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        results['bbox_fields'].append('gt_bboxes')
        results['mask_fields'].append('gt_masks')
        # results['seg_fields'].append('gt_semantic_seg')
        gt_bboxes = []
        gt_labels = []
        # gt_semantic_seg = []
        gt_masks = []
        num_img = 5 if len(results['img'])==5 else 1
        for i in range(num_img):
            if num_img == 5:
                path = "{}/{}".format(img_path_prefix, str(sample_idx*10+i))
            else:
                path = "{}/{}".format(img_path_prefix, str(sample_idx))
            if self.file_client.exists(path):
                label_bytes = self.file_client.get(path)
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
                    gt_bboxes.append(img_bbox.astype(np.float32))
                    gt_labels.append(img_label.astype(np.float32))
                    gt_masks.append(img_mask)
                else:
                    gt_bboxes.append(np.array([]).astype(np.float32))
                    gt_labels.append(np.array([]).astype(np.float32))
                    if num_img == 5:
                        h, w, c = results['img'][i].shape
                    else:
                        h, w, c = results['img'].shape
                    gt_masks.append(BitmapMasks(np.array([]), h, w))
            else:
                gt_bboxes.append(np.array([]).astype(np.float32))
                gt_labels.append(np.array([]).astype(np.float32))
                if num_img == 5:
                    h, w, c = results['img'][i].shape
                else:
                    h, w, c = results['img'].shape
                gt_masks.append(BitmapMasks(np.array([]), h, w))
        if num_img == 5:
            results['gt_bboxes'] = gt_bboxes
            results['gt_labels'] = gt_labels
            results['gt_masks'] = gt_masks
        if num_img == 1:
            results['gt_bboxes'] = gt_bboxes[0]
            results['gt_labels'] = gt_labels[0]
            results['gt_masks'] = gt_masks[0]
        if len(results['gt_bboxes']) == 0:
            return None
        # if num_img == 5 and len(results['gt_bboxes']) < 3:
        #     return None
        return results

    def __call__(self, results):
        results = self._load_labels(results)
        return results

@PIPELINES.register_module()
class LoadValRingSegLabel:
    # load pseudo lables
    # some points project onto two images, so have two instance id.
    # no assign[:, 0:2] means directlt project to get the instance id,
    # assign[:, 2:4] means some methods to get instance id
    # [instance id, instance id, instance id, instance id, ring segment, mask flag]
    def __init__(self,
                 coord_type,
                 file_client_args,
                 ccl_mode=None,
                 load_ccl=False,
                 online_dcs=True):
        self.coord_type = coord_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.mode = ccl_mode
        self.load_ccl = load_ccl
        self.online_dcs = online_dcs
    
    def load_pseudo_label(self, reuslts):
        # TopLidar & InImages & PointsRangeFilter
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        if self.mode is not None and self.load_ccl:
            if self.mode == 'mwsis_ccl':
                path = reuslts['pts_info']['path']
                path = path.split('/')
                path = '{}/{}/{}'.format(path[0], 'mwsis_ccl', path[2])
            else:
                raise ValueError('path is error, the correct path is {}'.format(path = '{}/{}/{}'.format(path[0], 'mwsis_ccl', path[2])))
            ccl_bytes = self.file_client.get(path)
            ccl_label = np.load(BytesIO(ccl_bytes))
            # no assign[:,0:2], assign[:,2:4]
            ccl_label = np.stack(
                        [ccl_label['norm_box_1'],
                         ccl_label['norm_box_2'],
                         ccl_label['assi_box_1'],
                         ccl_label['assi_box_2'],
                         ccl_label['run_id'],  # ring segment
                         ccl_label['mask_flag']], axis=1).astype(np.float32)
            reuslts['ccl_labels'] = ccl_label

        return reuslts
    
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
        
    def dcs(self, results):
        points = results['points'].tensor.numpy()
        range_seg_inds = np.ones((points.shape[0])).astype(np.float32)*-1
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

        # points = np.concatenate((points, range_seg_inds.reshape(points.shape[0], 1)), axis=1)
        # points_class = get_points_type(self.coord_type)
        # results['points'] = points_class(points, points_dim=points.shape[-1])
        pseudo_labels = np.zeros((len(range_seg_inds), 5))
        pseudo_labels[:, 4] = range_seg_inds
        results['ccl_labels'] = pseudo_labels
        
        return results


    def __call__(self, results):
        if self.load_ccl:
            if self.online_dcs:
                results = self.dcs(results)
            else:
                results = self.load_pseudo_label(results)
        return results