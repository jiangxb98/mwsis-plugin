import mmcv
import numpy as np
from mmdet3d.core.bbox import box_np_ops
from mmdet3d.datasets.pipelines import data_augment_utils, Compose
from mmdet3d.datasets.builder import OBJECTSAMPLERS
from mmdet3d.datasets.pipelines.dbsampler import BatchSampler
import os.path as osp


@OBJECTSAMPLERS.register_module(force=True)
class DataBaseSampler(object):
    def __init__(self,
                 info_path,
                 rate,
                 filter,
                 sample_groups,
                 classes=None,
                 datainfo_client_args=None,
                 points_loader=dict(
                     type='LoadPointsFromFile',
                     coord_type='LIDAR',
                     load_dim=4,
                     use_dim=[0, 1, 2, 3])):
        super().__init__()
        self.info_path = dict()
        self.rate = rate
        self.filter = filter
        self.classes = classes
        self.cat2label = {name: i for i, name in enumerate(classes)}
        self.label2cat = {i: name for i, name in enumerate(classes)}
        self.points_loader = Compose(points_loader)
        self.datainfo_client_args = datainfo_client_args
        self.db_infos_reader = None

        # filter database infos
        from mmdet3d.utils import get_root_logger
        logger = get_root_logger(name='mmdet')
        db_infos = dict()
        _db_infos_reader = mmcv.FileClient(**self.datainfo_client_args,
                                           scope='main_process')

        self.sample_classes = []
        self.sample_max_nums = []
        for name, num in sample_groups.items():
            self.sample_classes.append(name)
            self.sample_max_nums.append(int(num))
            self.info_path[name] = osp.join(info_path, name)
            db_infos[name] = _db_infos_reader.client.query_index(
                self.info_path[name], filter=filter.get(name, {}))
            logger.info(f'load {len(db_infos[name])} {name} database infos '
                        f'with filter: {filter.get(name, {})}')

        self.db_infos = db_infos

        self.sampler_dict = {}
        for k, v in self.db_infos.items():
            self.sampler_dict[k] = BatchSampler(v, k, shuffle=True)

    def sample_all(self, input_dict, sample_2d=False):
        gt_bboxes = input_dict['gt_bboxes_3d'].tensor.numpy()
        gt_labels = input_dict['gt_labels_3d']
        img = None
        gt_bboxes_2d = None
        if sample_2d:
            img = input_dict['img']
            gt_bboxes_2d = input_dict['gt_bboxes']
        ground_plane = input_dict.get('plane', None)

        sampled_num_dict = {}
        sample_num_per_class = []
        for class_name, max_sample_num in zip(self.sample_classes,
                                              self.sample_max_nums):
            class_label = self.cat2label[class_name]
            sampled_num = int(max_sample_num -
                              np.sum([n == class_label for n in gt_labels]))
            sampled_num = np.round(self.rate * sampled_num).astype(np.int64)
            sampled_num_dict[class_name] = sampled_num
            sample_num_per_class.append(sampled_num)

        sampled = []
        sampled_gt_bboxes = []
        avoid_coll_boxes = gt_bboxes

        for class_name, sampled_num in zip(self.sample_classes,
                                           sample_num_per_class):
            if sampled_num > 0:
                sampled_cls = self.sample_class_v2(class_name, sampled_num,
                                                   avoid_coll_boxes)

                sampled += sampled_cls
                if len(sampled_cls) > 0:
                    sampled_gt_box = self.get_sample_boxes(sampled_cls,
                                                           gt_bboxes.dtype)
                    sampled_gt_bboxes += [sampled_gt_box]
                    avoid_coll_boxes = np.concatenate(
                        [avoid_coll_boxes, sampled_gt_box], axis=0)

        ret = None
        if len(sampled) > 0:
            sampled_gt_bboxes = np.concatenate(sampled_gt_bboxes, axis=0)
            s_points_list = []
            count = 0
            for info in sampled:
                info['pts_info']['pts_loader'] = input_dict['pts_info'][
                    'pts_loader']
                info['ego_pose'] = np.array(info['ego_pose'])
                for sweep in info['pts_info']['sweeps']:
                    sweep['rel_pose'] = np.array(sweep['rel_pose'])
                s_points = self.points_loader(info)['points']

                count += 1

                s_points_list.append(s_points)

            gt_labels = np.array([self.cat2label[s['name']] for s in sampled],
                                 dtype=np.long)

            if ground_plane is not None:
                xyz = sampled_gt_bboxes[:, :3]
                dz = (ground_plane[:3][None, :] *
                      xyz).sum(-1) + ground_plane[3]
                sampled_gt_bboxes[:, 2] -= dz
                for i, s_points in enumerate(s_points_list):
                    s_points.tensor[:, 2].sub_(dz[i])

            ret = {
                'gt_labels_3d':
                    gt_labels,
                'gt_bboxes_3d':
                    sampled_gt_bboxes,
                'points':
                    s_points_list[0].cat(s_points_list),
                'group_ids':
                    np.arange(gt_bboxes.shape[0],
                              gt_bboxes.shape[0] + len(sampled))
            }

        return ret

    def sample_class_v2(self, name, num, gt_bboxes):
        if self.db_infos_reader is None:
            self.db_infos_reader = mmcv.FileClient(**self.datainfo_client_args)
        sampled = self.sampler_dict[name].sample(num)
        sampled = [self.db_infos_reader.get((self.info_path[name], i))
                   for i in sampled]
        num_gt = gt_bboxes.shape[0]
        num_sampled = len(sampled)
        gt_bboxes_bv = box_np_ops.center_to_corner_box2d(
            gt_bboxes[:, 0:2], gt_bboxes[:, 3:5], gt_bboxes[:, 6])

        sp_boxes = self.get_sample_boxes(sampled, dtype=gt_bboxes.dtype)
        boxes = np.concatenate([gt_bboxes, sp_boxes], axis=0).copy()

        sp_boxes_new = boxes[gt_bboxes.shape[0]:]
        sp_boxes_bv = box_np_ops.center_to_corner_box2d(
            sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, 6])

        total_bv = np.concatenate([gt_bboxes_bv, sp_boxes_bv], axis=0)
        coll_mat = data_augment_utils.box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                valid_samples.append(sampled[i - num_gt])
        return valid_samples

    @staticmethod
    def get_sample_boxes(sampled_infos, dtype='float32'):
        sp_boxes = []
        for sp in sampled_infos:
            sp_box = list(sp['box3d_lidar']) + list(sp.get('velocity', []))
            sp_boxes.append(sp_box)
        return np.array(sp_boxes, dtype=dtype)
