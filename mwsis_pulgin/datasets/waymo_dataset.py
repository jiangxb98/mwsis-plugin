import tempfile
import time
from os import path as osp
import csv
import zlib
import mmcv
import tqdm
import numpy as np
from collections import OrderedDict, defaultdict
from terminaltables import AsciiTable
from mmcv.utils import print_log
from mmdet3d.datasets.builder import DATASETS
from mmdet3d.core.bbox import get_box_type, LiDARInstance3DBoxes
from mmdet3d.datasets.custom_3d import Custom3DDataset
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.core.points import get_points_type
from io import BytesIO
from multiprocessing import Pool
import torch
import json
from pycocotools import mask
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from datetime import datetime
from waymo_open_dataset import dataset_pb2 as open_dataset
try:
    from waymo_open_dataset.protos import segmentation_metrics_pb2
    from waymo_open_dataset.protos import segmentation_submission_pb2
except ImportError:
    raise ImportError(
        'Please run "pip install waymo-open-dataset-tf-2-6-0==1.4.9" '
        'to install the official devkit first.')
import random

TOP_LIDAR_ROW_NUM = 64
TOP_LIDAR_COL_NUM = 2650

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf-8');
        return json.JSONEncoder.default(self, obj)

class COCOeval_(COCOeval):
    def summarize(self, catId=None, logger=None):
        class_name = ['Car', 'Ped', 'Cyc']
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.4f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if isinstance(catId, int):
                    s = s[:, :, catId, aind, mind]
                elif isinstance(catId, list):
                    s = s[:, :, int(catId[0]):int(catId[1])+1, aind, mind]
                else:
                    s = s[:, :, :, aind, mind]
                # s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                # s = s[:,:,aind,mind]
                if isinstance(catId, int):
                    s = s[:,catId, aind, mind]
                elif isinstance(catId, list):
                    s = s[:, int(catId[0]):int(catId[1])+1, aind, mind]
                else:
                    s = s[:, :, aind, mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            # print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            if isinstance(catId, int):
                class_str = class_name[catId]
            else:
                class_str = 'Mul'
            print_log(class_str + iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s), logger=logger)
            return mean_s
        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[7] = _summarize(0, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[8] = _summarize(0, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats
        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats
        if isinstance(catId, int):
            class_name = ['Car','Ped','Cyc']
            print_log('Eval {} Results:'.format(class_name[catId]),logger=logger)
        if isinstance(catId, list):
            print_log('Eval MAP Results:', logger=logger)            
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()
        return self.stats


# validation 39987 panseg 1881*5=9405 semseg 5976
# train 158081 panseg 12296*5=61480 semseg 23691
@DATASETS.register_module(name='MyWaymoDataset', force=True)
class MyWaymoDataset(Custom3DDataset):
    CLASSES = ('Car', 'Pedestrian', 'Cyclist')
    def __init__(self,
                 info_path,
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 load_interval=1,
                 pcd_limit_range=[-85, -85, -5, 85, 85, 5],
                 datainfo_client_args=None,
                 load_semseg=False,
                 semseg_classes=None,
                 semseg_info_path=None,
                 load_img=False,
                 load_panseg=False,
                 panseg_classes=None,
                 panseg_info_path=None,
                 load_img_index=[0,1,2,3,4],
                 finetune=False):

        super(Custom3DDataset, self).__init__()
        self.info_path = info_path
        self.datainfo_client_args = datainfo_client_args
        self.test_mode = test_mode
        self.modality = modality
        self.filter_empty_gt = filter_empty_gt
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        self.load_semseg = load_semseg
        self.load_panseg = load_panseg  # 2d segmentation
        self.load_img = load_img
        self.load_img_index = load_img_index  #collect images to use
        if len(classes) == 2:
            self.filter_classes = ['DontCare', 'Sign','Cyclist']
        elif len(classes) == 3:
            self.filter_classes = ['DontCare', 'Sign']
        self.CLASSES = self.get_classes(classes)
        self.infos_reader = None
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        if self.load_semseg:
            self.SEMSEG_CLASSES = self.get_classes(semseg_classes)
            self.segCat2id = {name: i for i, name in enumerate(self.SEMSEG_CLASSES)}
            self.semseg_info_path = semseg_info_path
        if self.load_panseg:
            self.panseg_CLASSES = self.get_classes(panseg_classes)
            self.pansegCat2id = {name: i for i, name in enumerate(self.panseg_CLASSES)}
            self.panseg_info_path = panseg_info_path

        self.pipeline_types = [p['type'] for p in pipeline]  # pipline name list
        self._skip_type_keys = None
        self.finetune = finetune

        # load annotations
        # 这个是所有在training/infos的数据(每帧点云)索引，如果用语义信息，就建议用语义的索引覆盖掉
        self.data_infos = self.load_annotations(self.info_path)  # 158081
        # 这个是load所有的语义标签在OSS上的路径
        if self.load_semseg:  # 23691
            self.semseg_frame_infos = self.read_semseg_infos(self.semseg_info_path, filter_sem='semseg_info')
            # intersection
            self.data_infos = self.semseg_frame_infos # self.semseg_frame_infos # 
            # self.data_infos = list(set(self.data_infos).intersection(self.semseg_frame_infos)) #

            if not self.test_mode and self.finetune:
                # self.data_infos = random.sample(self.data_infos, 237)
                self.data_infos = self.data_infos[::10]
        # train len(self.panseg_frame_infos)=12295
        if self.load_panseg:
            self.panseg_frame_infos = self.read_semseg_infos(self.panseg_info_path, filter_sem='panseg_info')
            # intersection
            self.data_infos = self.panseg_frame_infos # self.panseg_frame_infos # 
            # self.data_infos = list(set(self.data_infos).intersection(self.panseg_frame_infos)) #
            if not self.test_mode and self.finetune:
                self.data_infos = self.data_infos[::10]
        
        # cyclist_list = [1025,1027,1028,1031,1038,1052,1056,1003,1007,1008,1015,1016,1063,1072,1073,1074,1078,1079,1086,1092,
        #                 1095,1104,1109,1119,1120,1124,1125,1126,1127,1129,1135,1139,1141,1147,1152,1157,1161,1166,1173,1176,1177,
        #                 1182,1183,1185,1187,1191,1192]
        # new_data_infos = []
        # for i in range(len(self.data_infos)):
        #     if self.data_infos[i]//1000 in cyclist_list:
        #         new_data_infos.append(self.data_infos[i])
        # self.data_infos = new_data_infos
        # self.data_infos = [1092048,1092058,1092068,1092098]
        # self.data_infos = self.data_infos[0:4]
        # random.shuffle(self.data_infos)
        # self.data_infos = [1030148,1108025,1150054,1155074,1178065]

        # process pipeline
        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        # set group flag for the samplers
        if not self.test_mode:
            self._set_group_flag()

        assert self.modality is not None
        self.pcd_limit_range = pcd_limit_range
        self.data_infos = self.data_infos[::load_interval]
        # if hasattr(self, 'flag'):
        #     self.flag = self.flag[::load_interval]

    def load_annotations(self, info_path):
        _infos_reader = mmcv.FileClient(**self.datainfo_client_args,
                                        scope='main_process')
        data_infos = sorted(_infos_reader.client.query_index(info_path))
        return data_infos

    # get sementic label indx in mongodb database
    def read_semseg_infos(self, semseg_info_path, filter_sem=None):
        _infos_reader = mmcv.FileClient(**self.datainfo_client_args,
                                        scope='main_process')
        data_infos = sorted(_infos_reader.client.query_index(semseg_info_path, filter_sem=filter_sem, test_model=self.test_mode))
        return data_infos

    def get_data_info(self, index):
        if self.infos_reader is None:
            self.infos_reader = mmcv.FileClient(**self.datainfo_client_args)
        info = self.infos_reader.get((self.info_path, self.data_infos[index]))  # 进入mongodb.py改写的get方法 get a frame info
        sample_idx = info['sample_idx']

        if self.load_panseg and (not self.test_mode):
            if sample_idx not in self.panseg_frame_infos:
                return None
        if self.load_semseg and (not self.test_mode):
            if sample_idx not in self.semseg_frame_infos:
                return None

        # get lidar points from oss
        info['pts_info']['pts_loader'] = self.pts_loader  # 只是定义了函数，没有调用，用到才调用
        # get image info : image path, transform matrix
        img_path_info = []
        for img in info['images']:
            lidar2img = np.array(img['cam_intrinsic']) @ np.array(
                img['tf_lidar_to_cam'])
            img_path_info.append(dict(filename=img['path'], lidar2img=lidar2img))

        info['ego_pose'] = np.array(info['ego_pose'])
        for sweep in info['pts_info']['sweeps']:
            sweep['rel_pose'] = np.array(sweep['rel_pose'])

        input_dict = dict(
            sample_idx=sample_idx,
            pts_info=info['pts_info'],
            img_info=dict(img_loader=self.img_loader,
                img_path_info=img_path_info),)

        # 这里放入了2D的label
        #import pdb;pdb.set_trace()
        if 'annos' in info:
            annos = self.get_ann_info(info)
            input_dict['ann_info'] = annos
            # save semseg and panseg info{path, loader}
            if self.load_semseg:
                input_dict['ann_info']['pts_semantic_mask_loader'] = self.semseg_loader
                input_dict['ann_info']['pts_semantic_mask_path'] = info['semseg_info']['path']
                input_dict['ann_info']['pts_instance_mask_loader'] = self.semseg_loader
                input_dict['ann_info']['pts_instance_mask_path'] = info['semseg_info']['path']
            if self.load_panseg:
                panseg_path_info = []
                # each frame have 5 panseg path
                for panseg in info['panseg_info']:
                    panseg_path_info.append(panseg['path'])
                input_dict['ann_info']['pan_semantic_mask_loader'] = self.panseg_loader
                input_dict['ann_info']['pan_semantic_mask_path'] = panseg_path_info
                input_dict['ann_info']['pan_instance_mask_loader'] = self.panseg_loader
                input_dict['ann_info']['pan_instance_mask_path'] = panseg_path_info

        return input_dict

    @staticmethod
    def pts_loader(results, pts_bytes):
        points = np.load(BytesIO(pts_bytes))
        return np.stack([points['x'].astype('f4'),
                         points['y'].astype('f4'),
                         points['z'].astype('f4'),
                         np.tanh(points['intensity'].astype('f4')), # 3
                        #  points['intensity'].astype('f4'), input lidar intensity
                         points['elongation'].astype('f4'),  # 4
                         points['return_idx'].astype('i2'),  # 5
                         points['lidar_idx'].astype('i2'),   # 6
                         points['range_dist'].astype('f4'),  # range distance 7
                         points['lidar_row'].astype('i2'),   # range image 行 64 8
                         points['lidar_column'].astype('i2'),# range image 列 2650 9
                         points['cam_idx_0'].astype('i2'),   # 10
                         points['cam_idx_1'].astype('i2'),   # 11
                         points['cam_column_0'].astype('i2'),
                         points['cam_column_1'].astype('i2'),
                         points['cam_row_0'].astype('i2'),
                         points['cam_row_1'].astype('i2'),   # 15
                         ], axis=-1)

    @staticmethod
    def img_loader(results, pts_bytes):
        return None

    @staticmethod
    def semseg_loader(results, pts_bytes, semseg_name):
        semseg_labels = np.load(BytesIO(pts_bytes))
        return semseg_labels[semseg_name]  # return semantic or instance id

    @staticmethod
    def panseg_loader(results, pts_bytes, panseg_name):
        panseg_labels = np.load(BytesIO(pts_bytes))
        return panseg_labels[panseg_name]  # return semantic or instance id

    def class_name_to_label_index(self, gt_names):
        gt_labels = []
        for cat in gt_names:
            if cat in self.CLASSES:
                gt_labels.append(self.CLASSES.index(cat))
            else:
                gt_labels.append(-1)
        return np.array(gt_labels).astype(np.int64)

    def get_ann_info(self, info):
        annos = info['annos']
        difficulty = annos['difficulty']
        # we need other objects to avoid collision when sample
        gt_names = [np.array(n) for n in annos['name']]  # 2d
        gt_bboxes = [np.array(b, dtype=np.float32) for b in annos['bbox']]  # 2d
        selected = [self.drop_arrays_by_name(n, self.filter_classes) for n in  # filter obejcts which we not need
                    gt_names]
        gt_names = [n[s] for n, s in zip(gt_names, selected)]  # select the need objects
        gt_bboxes = [b[s] for b, s in zip(gt_bboxes, selected)]

        gt_names_3d = np.array(annos['name_3d'])
        gt_bboxes_3d = np.array(annos['bbox_3d']).astype('float32').reshape(-1,
                                                                            7)
        selected_3d = self.drop_arrays_by_name(gt_names_3d,
                                               self.filter_classes)
        gt_names_3d = gt_names_3d[selected_3d]
        gt_bboxes_3d = gt_bboxes_3d[selected_3d]
        # exchange width&length
        #gt_bboxes_3d = gt_bboxes_3d[:, [0, 1, 2, 4, 3, 5, 6]]
        gt_bboxes_3d = LiDARInstance3DBoxes(gt_bboxes_3d)  # 定义3Dbox实例，gt3d的相对位置关系是(0.5,0.5,0)

        # NOTE yaw: LiDAR -> CAM
        # gt_bboxes_3d.tensor[:, 6] = - gt_bboxes_3d.tensor[:, 6] - np.pi / 2
        # gt_bboxes_3d.tensor[:, 6][gt_bboxes_3d.tensor[:, 6] < -np.pi] += 2 * np.pi
        # gt_bboxes_3d.tensor[:, 6][gt_bboxes_3d.tensor[:, 6] > np.pi] -= 2 * np.pi


        gt_labels = [self.class_name_to_label_index(n) for n in gt_names]  # encode label
        gt_labels_3d = self.class_name_to_label_index(gt_names_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names_3d=gt_names_3d,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
            gt_names=gt_names,
            plane=None,
            difficulty=difficulty)
        return anns_results

    @staticmethod
    def drop_arrays_by_name(gt_names, drop_classes):
        inds = [i for i, x in enumerate(gt_names) if x not in drop_classes]
        inds = np.array(inds, dtype=np.int64)
        return inds

    @staticmethod
    def keep_arrays_by_name(gt_names, use_classes):
        inds = [i for i, x in enumerate(gt_names) if x in use_classes]
        inds = np.array(inds, dtype=np.int64)
        return inds

    # 由Custom3DDataset.__getitem__(self, idx)方法来进入这个函数
    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)  # Initialization before data preparation
        # into pipline. eg: LoadPoints, LoadAnnos3D, 调用每个class的__call__(self,example)
        example = input_dict
        for transform, transform_type in zip(self.pipeline.transforms, self.pipeline_types):
            if self._skip_type_keys is not None and transform_type in self._skip_type_keys:
                continue
            if example is None:
                return None
            example = transform(example)
        # example = self.pipeline(input_dict)
        if 'gt_labels_3d' in example.keys():
            if self.filter_empty_gt and \
                    (example is None or
                        ~(example['gt_labels_3d']._data != -1).any()):
                return None
        if 'gt_labels' in example.keys():
            if isinstance(example['gt_labels']._data, list):
                tmp_gt_labels = example['gt_labels']._data
                nums = 0
                for i in range(len(tmp_gt_labels)):
                    nums += len(tmp_gt_labels[i])
                if nums == 0:
                    return None
            if isinstance(example['gt_bboxes']._data, list):
                tmp_gt_labels = example['gt_bboxes']._data
                nums = 0
                for i in range(len(tmp_gt_labels)):
                    nums += len(tmp_gt_labels[i])
                if nums == 0:
                    return None
            else:
                if len(example['gt_labels']._data) == 0:
                    return None
                if len(example['gt_bboxes']._data) == 0:
                    return None
        if 'points' in example.keys():
            if len(example['points']._data) == 0:
                return None
        return example

    def update_skip_type_keys(self, skip_type_keys):
        self._skip_type_keys = skip_type_keys

    def format_results(self, outputs, pklfile_prefix=None):
        if 'pts_bbox' in outputs[0]:
            outputs = [out['pts_bbox'] for out in outputs]
        result_serialized = self.bbox2result_waymo(outputs)

        waymo_results_final_path = f'{pklfile_prefix}.bin'

        with open(waymo_results_final_path, 'wb') as f:
            f.write(result_serialized)

        return waymo_results_final_path

    def evaluate(self, results, **kwargs):

        if kwargs.get('eval_2d', False):
            out = self.evaluate_mask_2d(results, **kwargs)

        if kwargs.get('eval_3d', False):
            out2 =  self.evaluate_segment_3d(results, **kwargs)

        if kwargs.get('eval_3dmask', False):
            out3 = self.evaluate_mask_3d(results, **kwargs)

        return dict()

    def evaluate_mask_2d(self,
                    results,
                    logger=None,
                    pklfile_prefix=None,
                    seg_format_worker=0,
                    **kwargs,
                    ):
        self.seg_format_worker = seg_format_worker
        print("\n seg_format_worker:{}\n".format(self.seg_format_worker))
        if pklfile_prefix is None:
            eval_tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(eval_tmp_dir.name, 'results')
        else:
            eval_tmp_dir = None

        if 'img_preds' in results[0].keys():
            img_preds = [out['img_preds'] for out in results if out.get('img_preds') is not None]
            img_metas = []
            img_mask_preds = []
            img_bboxes_preds = []
            for i, img_infos in enumerate(img_preds):  # imgs
                for j, img_info in enumerate(img_infos):  # 5 imgs
                    if img_info.get('img_bbox') is not None:
                        img_bboxes_preds.append(img_info['img_bbox'])
                    if img_info.get('img_mask') is not None:
                        img_mask_preds.append(img_info['img_mask'])
                    if img_info.get('img_bbox') is not None or img_info.get('img_mask') is not None:
                        img_metas.append(img_info['img_metas'])
            if len(img_metas)==0:
                return dict()
        else:
            img_metas = [out['img_metas'] for out in results if out.get('img_metas') is not None]
            if len(img_metas)==0:
                return dict()
            img_mask_preds = [out['img_mask'] for out in results if out.get('img_mask') is not None]
            img_bboxes_preds = [out['img_bbox'] for out in results if out.get('img_bbox') is not None]
            assert len(img_metas) == len(img_mask_preds)
        results_mask_2d = self.format_2d_mask(img_mask_preds, img_bboxes_preds, img_metas)

        # results_mask_2d_path = '/root/3D/work_dirs/results/mask_2d_preds.json'
        # results_mask_2d_path = '/home/jiangguangfeng/桌面/codebase/mask_2d_preds.json'
        results_mask_2d_path = f'{pklfile_prefix}_mask_2d_preds.json'
        if 'img_preds' in results[0].keys():
            if osp.exists('/root/3D/work_dirs/dataset_infos/waymo_lwsis_val1.0_my.json'):
                # validation_path = '/root/3D/work_dirs/dataset_infos/validation_instance_infos_lwsis.json'
                validation_path = '/root/3D/work_dirs/dataset_infos/waymo_lwsis_val1.0_my.json'
                results_mask_2d_path = '/root/3D/work_dirs/results/mask_2d_preds{}.json'.format(datetime.now().strftime("%m-%d=%H:%M"))
            elif osp.exists('/home/jiangguangfeng/桌面/codebase/waymo_lwsis_val1.0_my.json'):
                validation_path = '/home/jiangguangfeng/桌面/codebase/waymo_lwsis_val1.0_my.json'
                results_mask_2d_path = '/home/jiangguangfeng/桌面/codebase/mask_2d_preds.json'
        else:
            if osp.exists('/root/3D/work_dirs/dataset_infos/validation_instance_infos.json'):
                validation_path = '/root/3D/work_dirs/dataset_infos/validation_instance_infos_lwsis.json'
                results_mask_2d_path = '/root/3D/work_dirs/results/mask_2d_preds.json'
            elif osp.exists('/home/jiangguangfeng/桌面/codebase/validation_instance_infos.json'):
                validation_path = '/home/jiangguangfeng/桌面/codebase/validation_instance_infos_final.json'
                results_mask_2d_path = '/home/jiangguangfeng/桌面/codebase/mask_2d_preds.json'
        
        print_log("save results path {}".format(results_mask_2d_path), logger=logger) 
        with open(results_mask_2d_path,'w') as f:
            json.dump(results_mask_2d, f, indent=4, ensure_ascii=False, cls=MyEncoder)

        annType = ['segm','bbox','keypoints']
        annType = annType[0]  # specify type here
        cocoGt = COCO(validation_path)
        cocoDt = cocoGt.loadRes(results_mask_2d_path)
        cocoEval = COCOeval_(cocoGt, cocoDt, annType)
        # cocoEval.params.imgIds  = imgIds
        cocoEval.evaluate()
        cocoEval.accumulate()
        map = cocoEval.summarize(logger=logger)
        # result = cocoEval.summarize(catId=[0,1], logger=logger)
        result0 = cocoEval.summarize(catId=0, logger=logger)
        result1 = cocoEval.summarize(catId=1, logger=logger)
        result2 = cocoEval.summarize(catId=2, logger=logger)
        # m_result = (result1+result2)/2
        # result2 = cocoEval.summarize(catId=2)
        results_dict = dict()
        return results_dict

    def evaluate_segment_3d(self, results, metric='seg', ignore_class=['unlab'],
                    logger=None, show=False, out_dir=None, pipeline=None, **kwargs):
        # pts_semantic_confusion_matrix is in ./utils.py
        num_classes = len(self.CLASSES) + 1
        confusion_matrixs = []
        seg_targets = [res['pts_semantic_mask'] for res in results]
        seg_preds = [res['segment3d'] for res in results]
        for i in range(len(seg_targets)):
            pts_cond = seg_targets[i] * num_classes + seg_preds[i]
            pts_cond = pts_cond.type(torch.long)
            pts_cond_count = pts_cond.bincount(minlength=num_classes * num_classes)
            # confusion_matrix.sum(1) is targets nums
            confusion_matrix = pts_cond_count[:num_classes * num_classes].reshape(num_classes, num_classes).numpy()
            confusion_matrixs.append(confusion_matrix)
        histogram = np.sum(np.stack(confusion_matrixs, axis=0), axis=0)
        # del confusion_matrixs
        # del confusion_matrix
        # del seg_targets
        # del seg_preds
        eval_mask = []
        eval_classes = []
        seg_classes = self.CLASSES.copy()
        seg_classes.append('BackGround')
        assert len(seg_classes) == len(self.CLASSES) + 1
        for c in seg_classes:
            eval_mask.append(c not in ignore_class)
            if c not in ignore_class:
                eval_classes.append(c)
        histogram = histogram[eval_mask][:, eval_mask]
        inter = np.diag(histogram)
        union = np.sum(histogram, axis=0) + np.sum(histogram, axis=1) - inter
        iou = inter / np.clip(union, 1, None)
        eval_result = OrderedDict()
        eval_classes = eval_classes[:-1]
        iou = iou[:-1]
        for c, score in zip(eval_classes, iou):
            eval_result[c] = score
        eval_result['all(mean)'] = iou.mean()
        eval_result['accuracy'] = inter[:-1].sum() / histogram.sum(1)[:-1].sum()

        table_data = [
            ['Class', 'mIoU']]
        for c, score in eval_result.items():
            table_data.append(
                [c, f'{100 * score:.3f}'])
        table = AsciiTable(table_data)
        # table.inner_footing_row_border = True
        print_log('\n' + table.table, logger=logger)
        print_log(histogram, logger=logger)
        return eval_result

    def evaluate_mask_3d(self,
                    results,
                    logger=None,
                    pklfile_prefix=None,
                    seg_format_worker=0,
                    **kwargs
                    ):
        self.seg_format_worker = seg_format_worker
        if pklfile_prefix is None:
            eval_tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(eval_tmp_dir.name, 'results')
        else:
            eval_tmp_dir = None

        img_metas = [out['img_metas'] for out in results if out.get('img_metas') is not None]
        # 0 is bg points
        pts_mask_preds = [out['mask_3d_rle'] for out in results if out.get('mask_3d_rle') is not None]
        seg_scores = [out['mask_3d_scores'] for out in results if out.get('mask_3d_scores') is not None]

        if len(img_metas)==0:
            return dict()

        results_mask_3d = self.format_3d_mask(pts_mask_preds, seg_scores, img_metas)

        # results_mask_3d_path = '/home/jiangguangfeng/桌面/codebase/mask3d_preds.json'
        # results_mask_3d_path = '/root/3D/work_dirs/dataset_infos/mask3d_preds.json'
        # default json path
        results_mask_3d_path = f'{pklfile_prefix}_mask_3d_preds.json'
        # with open(results_mask_3d_path,'w') as f:
        #     json.dump(results_mask_3d, f, indent=4, ensure_ascii=False, cls=MyEncoder)

        if osp.exists('/root/3D/work_dirs/dataset_infos/validation_3dmask_infos.json'):
            validation_path = '/root/3D/work_dirs/dataset_infos/validation_3dmask_infos_full.json'
            results_mask_3d_path = '/root/3D/work_dirs/results/mask_3d_preds{}.json'.format(datetime.now().strftime("%m-%d=%H:%M"))
        elif osp.exists('/home/jiangguangfeng/桌面/codebase/validation_3dmask_infos.json'):
            validation_path = '/home/jiangguangfeng/桌面/codebase/validation_3dmask_infos.json'
            results_mask_3d_path = '/home/jiangguangfeng/桌面/codebase/mask3d_preds.json'
        with open(results_mask_3d_path,'w') as f:
            json.dump(results_mask_3d, f, indent=4, ensure_ascii=False, cls=MyEncoder)
        # del results_mask_3d
        # del pts_mask_preds
        # del seg_scores
        # del img_metas
        print_log("save results path {}".format(results_mask_3d_path), logger=logger) 
        annType = ['segm','bbox','keypoints']
        annType = annType[0]  # specify type here
        cocoGt = COCO(validation_path)
        cocoDt = cocoGt.loadRes(results_mask_3d_path)
        cocoEval = COCOeval_(cocoGt, cocoDt, annType)
        # cocoEval.params.imgIds  = imgIds
        cocoEval.evaluate()
        cocoEval.accumulate()
        map = cocoEval.summarize(logger=logger)
        # result = cocoEval.summarize(catId=[0,1])
        result0 = cocoEval.summarize(catId=0, logger=logger)
        result1 = cocoEval.summarize(catId=1, logger=logger)
        result2 = cocoEval.summarize(catId=2, logger=logger)
        # m_result = (result1+result2)/2
        # result2 = cocoEval.summarize(catId=2)
        results_dict = dict()
        return results_dict

    def format_3d_mask(self, pts_mask_preds, seg_socres, img_metas):
        pred_list = []
        mask_args = [[i, pts_mask_preds[i], seg_socres[i], img_metas[i]] for i in range(len(pts_mask_preds))]
        if self.seg_format_worker == 0:
            results = []
            for i in mmcv.track_iter_progress(range(len(pts_mask_preds))):
                results.append(self.format_3d_mask_one(mask_args[i]))
        else:
            results = mmcv.track_parallel_progress(self.format_3d_mask_one, mask_args, self.seg_format_worker)
        for result in mmcv.track_iter_progress(results):
            pred_list.extend(result)
        return pred_list

    def format_3d_mask_one(self, mask_args):
        # img_mask_preds is list[[car],[ped],[cyc]]
        indx, pts_mask_preds, seg_socres, img_metas = mask_args
        assert len(pts_mask_preds) == len(seg_socres)
        pts_id = int(img_metas['sample_idx'])
        out_preds = []
        for i in range(len(pts_mask_preds)):
            for j, instance_mask in enumerate(pts_mask_preds[i]):
                rle = instance_mask
                # instance_mask = np.asfortranarray(instance_mask)
                # rle = mask.encode(instance_mask)
                # area = float(mask.area(rle))
                # bbox = mask.toBbox(rle)
                out_preds.append({
                    'image_id': int(pts_id),
                    'category_id': int(i),
                    "segmentation": rle,
                    'score': float(seg_socres[i][j])
                })
        return out_preds

    def format_2d_mask(self, img_mask_preds, img_bboxes_preds, img_metas):
        pred_list = []
        if len(img_bboxes_preds) == 0:
            img_bboxes_preds = [[[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]] for _ in range(len(img_mask_preds))]
        mask_args = [[img_mask_preds[i], img_bboxes_preds[i], img_metas[i]] for i in range(len(img_mask_preds))]
        if self.seg_format_worker == 0:
            results = []
            for i in mmcv.track_iter_progress(range(len(img_mask_preds))):
                results.append(self.format_2d_mask_one(mask_args[i]))
        else:
            results = mmcv.track_parallel_progress(self.format_2d_mask_one, mask_args, self.seg_format_worker)
        for result in mmcv.track_iter_progress(results):
            pred_list.extend(result)
        return pred_list

    def format_2d_mask_one(self, mask_args):
        # img_mask_preds is list[[car],[ped],[cyc]]
        img_mask_preds, img_bboxes_preds, img_metas = mask_args
        filename = img_metas['filename']
        split_path = filename.split('/')[-2:]
        for p in range(len(split_path)):
            split_path[p] = int(split_path[p])
        image_id = split_path[0]*10+split_path[1]
        out_preds = []
        for i in range(len(img_mask_preds)):
            for j, instance_mask in enumerate(img_mask_preds[i]):
                rle = instance_mask
                # instance_mask = np.asfortranarray(instance_mask)
                # rle = mask.encode(instance_mask)
                # area = float(mask.area(rle))
                # bbox = mask.toBbox(rle)
                out_preds.append({
                    'image_id': int(image_id),
                    'category_id': int(i),
                    "segmentation": rle,
                    'score': float(img_bboxes_preds[i][j][4])# float(img_bboxes_preds[i][j][4])
                })
        return out_preds

    def bbox2result_waymo(self, net_outputs):
        from waymo_open_dataset import label_pb2
        from waymo_open_dataset.protos import metrics_pb2
        class2proto = {
            'Car': label_pb2.Label.TYPE_VEHICLE,
            'Pedestrian': label_pb2.Label.TYPE_PEDESTRIAN,
            'Sign': label_pb2.Label.TYPE_SIGN,
            'Cyclist': label_pb2.Label.TYPE_CYCLIST,
        }
        label2proto = {}
        for c in class2proto:
            if c in self.cat2id:
                label2proto[self.cat2id[c]] = class2proto[c]

        assert len(net_outputs) == len(self.data_infos), \
            'invalid list length of network outputs'

        all_objects_serialized = []

        _infos_reader = mmcv.FileClient(**self.datainfo_client_args,
                                        scope='main_process')
        context_info = _infos_reader.client.query(
            self.info_path, projection=['context', 'timestamp'])
        context_info = {info['_id']: info for info in context_info}

        for idx, pred_dicts in enumerate(
                mmcv.track_iter_progress(net_outputs)):
            info = context_info[self.data_infos[idx]]
            objects_proto = metrics_pb2.Objects()
            for box_3d, score_3d, label_3d in zip(
                    pred_dicts['boxes_3d'].tensor.tolist(),
                    pred_dicts['scores_3d'].tolist(),
                    pred_dicts['labels_3d'].tolist()):
                o = objects_proto.objects.add()
                x, y, z, length, width, height, yaw = box_3d
                #x, y, z, width, length, height, yaw = box_3d   # NOTE deprecated
                o.object.box.center_x = x
                o.object.box.center_y = y
                o.object.box.center_z = z + height / 2
                o.object.box.length = length
                o.object.box.width = width
                o.object.box.height = height
                o.object.box.heading = yaw
                #o.object.box.heading = -yaw - np.pi / 2   # yam: CAM back to LiDAR  NOTE deprecated
                o.object.type = label2proto[label_3d]
                o.score = score_3d
                o.context_name = info['context']
                o.frame_timestamp_micros = info['timestamp']

            all_objects_serialized.append(objects_proto.SerializeToString())
        all_objects_serialized = b''.join(all_objects_serialized)
        return all_objects_serialized

    def format_one_frame_seg(self, args):
        idx, net_out, context_info, pc_range = args
        info = context_info[self.data_infos_seg_frames[idx]]
        range_image_pred = np.zeros(
            (TOP_LIDAR_ROW_NUM, TOP_LIDAR_COL_NUM, 2), dtype=np.int32)
        range_image_pred_ri2 = np.zeros(
            (TOP_LIDAR_ROW_NUM, TOP_LIDAR_COL_NUM, 2), dtype=np.int32)

        # fill back seg labels to range image format
        points_bytes = self.file_client.get(info['pts_info']['path'])
        points = np.load(BytesIO(points_bytes))
        points_indexing_row = points['lidar_row']
        points_indexing_column = points['lidar_column']
        points_return_idx = points['return_idx']
        points_range_mask = np.ones_like(points_indexing_row, dtype=np.bool)
        if pc_range is not None:
            xyz = np.stack([points['x'].astype('f4'),
                            points['y'].astype('f4'),
                            points['z'].astype('f4')], axis=-1)
            points_class = get_points_type('LIDAR')
            xyz = points_class(xyz, points_dim=xyz.shape[-1])
            points_range_mask = xyz.in_range_3d(pc_range).numpy()
        points_lidar_mask = points['lidar_idx'] == 0    # 0 for open_dataset.LaserName.TOP

        points_mask = points_range_mask & points_lidar_mask
        points_indexing_row = points_indexing_row[points_mask]
        points_indexing_column = points_indexing_column[points_mask]
        points_return_idx = points_return_idx[points_mask]

        points_lidar_in_range_mask = points_lidar_mask[points_range_mask]
        net_out = net_out[points_lidar_in_range_mask]

        assert points_indexing_row.shape[0] == net_out.shape[0]
        assert points_indexing_row.shape[0] == points_indexing_column.shape[0]
        range_image_pred[points_indexing_row[points_return_idx==0],
                points_indexing_column[points_return_idx==0], 1] = net_out[:int((points_return_idx==0).sum())]
        range_image_pred_ri2[points_indexing_row[points_return_idx==1],
                points_indexing_column[points_return_idx==1], 1] = net_out[int((points_return_idx==0).sum()):]

        segmentation_frame = segmentation_metrics_pb2.SegmentationFrame()
        segmentation_frame.context_name = info['context']
        segmentation_frame.frame_timestamp_micros = info['timestamp']
        laser_semseg = open_dataset.Laser()
        laser_semseg.name = open_dataset.LaserName.TOP
        laser_semseg.ri_return1.segmentation_label_compressed = self.compress_array(
            range_image_pred, is_int32=True)
        laser_semseg.ri_return2.segmentation_label_compressed = self.compress_array(
            range_image_pred_ri2, is_int32=True)
        segmentation_frame.segmentation_labels.append(laser_semseg)

        return segmentation_frame

    def format_segmap(self, net_outputs, pc_range=None):

        semseg_frames = self.semseg_frame_info
        _infos_reader = mmcv.FileClient(**self.datainfo_client_args,
                                        scope='main_process')
        context_info = _infos_reader.client.query(
            self.info_path, projection=['context', 'timestamp', 'pts_info'])
        context_info = {info['_id']: info for info in context_info if info['_id'] in semseg_frames}

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        # for idx, net_out in enumerate(mmcv.track_iter_progress(net_outputs)):
        seg_args = [[i, net_outputs[i], context_info, pc_range] for i in range(len(net_outputs))]
        if self.seg_format_worker == 0:
            segmentation_res = []
            for idx in mmcv.track_iter_progress(range(len(net_outputs))):
                segmentation_res.append(self.format_one_frame_seg(seg_args[idx]))
        else:
            # segmentation_res = mmcv.track_parallel_progress(self.format_one_frame_seg,
            #                                                 seg_args,
            #                                                 self.seg_format_worker)
            with Pool(self.seg_format_worker) as p:
                segmentation_res = list(tqdm.tqdm(p.imap(self.format_one_frame_seg, seg_args), total=len(seg_args)))
        
        segmentation_frame_list = segmentation_metrics_pb2.SegmentationFrameList()
        for segmentation_frame in segmentation_res:
            segmentation_frame_list.frames.append(segmentation_frame)

        return segmentation_frame_list.SerializeToString()
       
    def compress_array(self, array: np.ndarray, is_int32: bool = False):
        """Compress a numpy array to ZLIP compressed serialized MatrixFloat/Int32.

        Args:
            array: A numpy array.
            is_int32: If true, use MatrixInt32, otherwise use MatrixFloat.

        Returns:
            The compressed bytes.
        """
        if is_int32:
            m = open_dataset.MatrixInt32()
        else:
            m = open_dataset.MatrixFloat()
        m.shape.dims.extend(list(array.shape))
        m.data.extend(array.reshape([-1]).tolist())
        return zlib.compress(m.SerializeToString())

    def evaluate_det(self,
                    results,
                    metric=None,
                    logger=None,
                    pklfile_prefix=None,
                    submission_prefix=None,
                    show=False,
                    out_dir=None,
                    pipeline=None,
                    *args,
                    **kwargs):
        waymo_root = 'data/waymo/waymo_format'
        if pklfile_prefix is None:
            eval_tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(eval_tmp_dir.name, 'results')
        else:
            eval_tmp_dir = None
        self.format_results(results, pklfile_prefix)
        import subprocess
        ret_bytes = subprocess.check_output(
            f'/root/3D/waymo_utils/'
            'compute_detection_metrics_main '
            f'{pklfile_prefix}.bin '
            f'/disk/deepdata/dataset/waymo_v1.2/waymo_format/gt.bin',
            shell=True)
        ret_texts = ret_bytes.decode('utf-8')
        print_log(ret_texts, logger=logger)
        # parse the text to get ap_dict
        ap_dict = {
            'Vehicle/L1 mAP': 0,
            'Vehicle/L1 mAPH': 0,
            'Vehicle/L2 mAP': 0,
            'Vehicle/L2 mAPH': 0,
            'Pedestrian/L1 mAP': 0,
            'Pedestrian/L1 mAPH': 0,
            'Pedestrian/L2 mAP': 0,
            'Pedestrian/L2 mAPH': 0,
            'Sign/L1 mAP': 0,
            'Sign/L1 mAPH': 0,
            'Sign/L2 mAP': 0,
            'Sign/L2 mAPH': 0,
            'Cyclist/L1 mAP': 0,
            'Cyclist/L1 mAPH': 0,
            'Cyclist/L2 mAP': 0,
            'Cyclist/L2 mAPH': 0,
            'Overall/L1 mAP': 0,
            'Overall/L1 mAPH': 0,
            'Overall/L2 mAP': 0,
            'Overall/L2 mAPH': 0
        }
        mAP_splits = ret_texts.split('mAP ')
        mAPH_splits = ret_texts.split('mAPH ')
        for idx, key in enumerate(ap_dict.keys()):
            split_idx = int(idx / 2) + 1
            if idx % 2 == 0:  # mAP
                ap_dict[key] = float(mAP_splits[split_idx].split(']')[0])
            else:  # mAPH
                ap_dict[key] = float(mAPH_splits[split_idx].split(']')[0])
        ap_dict['Overall/L1 mAP'] = \
            (ap_dict['Vehicle/L1 mAP'] + ap_dict['Pedestrian/L1 mAP'] +
             ap_dict['Cyclist/L1 mAP']) / 3
        ap_dict['Overall/L1 mAPH'] = \
            (ap_dict['Vehicle/L1 mAPH'] + ap_dict['Pedestrian/L1 mAPH'] +
             ap_dict['Cyclist/L1 mAPH']) / 3
        ap_dict['Overall/L2 mAP'] = \
            (ap_dict['Vehicle/L2 mAP'] + ap_dict['Pedestrian/L2 mAP'] +
             ap_dict['Cyclist/L2 mAP']) / 3
        ap_dict['Overall/L2 mAPH'] = \
            (ap_dict['Vehicle/L2 mAPH'] + ap_dict['Pedestrian/L2 mAPH'] +
             ap_dict['Cyclist/L2 mAPH']) / 3
        if eval_tmp_dir is not None:
            eval_tmp_dir.cleanup()
        if show or out_dir:
            raise NotImplementedError
        return ap_dict
