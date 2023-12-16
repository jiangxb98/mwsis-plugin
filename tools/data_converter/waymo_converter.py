# Copyright (c) OpenMMLab. All rights reserved.
r"""Adapted from `Waymo to KITTI converter
    <https://github.com/caizhongang/waymo_kitti_converter>`_.
"""

try:
    from waymo_open_dataset import dataset_pb2
    from waymo_open_dataset.protos import segmentation_metrics_pb2
    from waymo_open_dataset.protos import segmentation_submission_pb2
except ImportError:
    raise ImportError(
        'Please run "pip install waymo-open-dataset-tf-2-1-0==1.2.0" '
        'to install the official devkit first.')

from glob import glob
from os.path import join
import os.path as osp

import mmcv
import numpy as np
import tensorflow as tf
from waymo_open_dataset.utils import range_image_utils, transform_utils, camera_segmentation_utils
from waymo_open_dataset.utils.frame_utils import \
    parse_range_image_and_camera_projection
from mmdet3d.core.bbox.box_np_ops import points_in_rbbox

def cat_gtdb(dict_a, dict_b):
    for k in dict_b:
        if k not in dict_a:
            dict_a[k] = []
        dict_a[k].extend(dict_b[k])
    return dict_a

def numpy2list(data):
    if isinstance(data, (list, tuple)):
        data = list(data)
        for idx, data_idx in enumerate(data):
            data[idx] = numpy2list(data_idx)
    elif isinstance(data, dict):
        for key in data:
            data[key] = numpy2list(data[key])
    elif isinstance(data, np.ndarray):
        data = data.tolist()
    return data

class Waymo2KITTI(object):
    """Waymo to KITTI converter.

    This class serves as the converter to change the waymo raw data to KITTI
    format.

    Args:
        load_dir (str): Directory to load waymo raw data.
        save_dir (str): Directory to save data in KITTI format.
        prefix (str): Prefix of filename. In general, 0 for training, 1 for
            validation and 2 for testing.
        workers (int, optional): Number of workers for the parallel process.
        test_mode (bool, optional): Whether in the test_mode. Default: False.
    """

    def __init__(self,
                 load_dir,
                 save_dir,
                 prefix,
                 workers=64,
                 test_mode=False,
                 sweeps=5,
                 load_img_panseg=True,
                 load_lidar_panseg=True):
        self.filter_empty_3dboxes = True
        self.filter_no_label_zone_points = True

        self.selected_waymo_classes = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']

        # Only data collected in specific locations will be converted
        # If set None, this filter is disabled
        # Available options: location_sf (main dataset)
        self.selected_waymo_locations = None
        self.save_track_id = False

        # turn on eager execution for older tensorflow versions
        if int(tf.__version__.split('.')[0]) < 2:
            tf.enable_eager_execution()

        self.lidar_list = [
            '_FRONT', '_FRONT_RIGHT', '_FRONT_LEFT', '_SIDE_RIGHT',
            '_SIDE_LEFT'
        ]
        self.type_list = [
            'UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST'
        ]
        self.waymo_to_kitti_class_map = {
            'UNKNOWN': 'DontCare',
            'PEDESTRIAN': 'Pedestrian',
            'VEHICLE': 'Car',
            'CYCLIST': 'Cyclist',
            'SIGN': 'Sign'  # not in kitti
        }

        self.load_dir = load_dir
        self.save_dir = save_dir
        self.prefix = prefix
        self.workers = int(workers)
        self.test_mode = test_mode

        # add for jiangxb
        self.sweeps = sweeps
        self.load_img_panseg = load_img_panseg
        self.load_lidar_panseg = load_lidar_panseg
        self.lidar_panseg_label_save_dir = f'{self.save_dir}/lidar_panseg_label'
        self.img_panseg_label_save_dir = f'{self.save_dir}/img_panseg_label_'
        self.frame_info_save_dir = f'{self.save_dir}/frame_infos'
        self.gtdb_save_dir = f'{self.save_dir}/gtdb_infos'
        self.img_panseg_index = []
        self.pts_panseg_index = []

        self.tfrecord_pathnames = sorted(
            glob(join(self.load_dir, '*.tfrecord')))

        self.label_save_dir = f'{self.save_dir}/label_'
        self.label_all_save_dir = f'{self.save_dir}/label_all'
        self.image_save_dir = f'{self.save_dir}/image_'
        self.calib_save_dir = f'{self.save_dir}/calib'
        self.point_cloud_save_dir = f'{self.save_dir}/velodyne'
        self.pose_save_dir = f'{self.save_dir}/pose'
        self.timestamp_save_dir = f'{self.save_dir}/timestamp'

        self.create_folder()

    def sample_index(self, file_idx, frame_idx):
        return int(self.prefix) * 1000000 + file_idx * 1000 + frame_idx

    def convert(self):
        """Convert action."""
        print('\nStart converting ...')

        if self.workers == 0:
            resutls = []
            for idx in mmcv.track_iter_progress(range(len(self))):
                # if idx == 2:
                #     break
                resutls.append(self.convert_one(idx))
        else:
            resutls = mmcv.track_parallel_progress(self.convert_one, range(len(self)),
                                                   self.workers)
        
        # save img and pts panseg index
        import pickle
        if 'validation' in self.load_dir:
            prefix = 'val'
        elif 'test' in self.load_dir:
            prefix = 'test'
        else:
            prefix = 'train'
        if self.load_img_panseg or self.load_lidar_panseg:
            img_panseg_index_save_path = f'{self.save_dir}/img_panseg_index_'+prefix + '.pkl'
            pts_panseg_index_save_path = f'{self.save_dir}/pts_panseg_index_'+prefix + '.pkl'
            with open(img_panseg_index_save_path, 'wb') as f:
                pickle.dump(self.img_panseg_index, f)
            with open(pts_panseg_index_save_path, 'wb') as f:
                pickle.dump(self.pts_panseg_index, f)

        print('\nOrganizing...')
        gtdb_infos = {}
        frames_infos = []
        for clip_index, clip_gtdb in mmcv.track_iter_progress(resutls):
            frames_infos.extend(clip_index)
            gtdb_infos = cat_gtdb(gtdb_infos, clip_gtdb)

        parts = self.save_dir.split('/')[:-1]
        all_infos_save_path = '/' + osp.join(*parts) + '/my_waymo_' + prefix + '.pkl'
        with open(all_infos_save_path, 'wb') as f:
            pickle.dump(frames_infos, f)

        all_gt_database_save_path = '/' + osp.join(*parts) + '/my_waymo_gt_database_' + prefix + '.pkl'
        with open(all_gt_database_save_path, 'wb') as f:
            pickle.dump(gtdb_infos, f)

        print('\nFinished ...' + prefix)

    def convert_one(self, file_idx):
        """Convert action for single file.

        Args:
            file_idx (int): Index of the file to be converted.
        """
        pathname = self.tfrecord_pathnames[file_idx]
        dataset = tf.data.TFRecordDataset(pathname, compression_type='')

        frame_infos = []
        gtdb_tracking = []
        gtdb_infos = dict()

        for frame_idx, data in enumerate(dataset):
            # if frame_idx == 1:
            #     break
            sample_idx = self.sample_index(file_idx, frame_idx)
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            frame_info = dict(_id=sample_idx,
                              sample_idx=sample_idx,
                              frame_idx=frame_idx,
                              record=osp.split(pathname)[-1],
                              context=frame.context.name,
                              timestamp=frame.timestamp_micros)
            if (self.selected_waymo_locations is not None
                    and frame.context.stats.location
                    not in self.selected_waymo_locations):
                continue

            self.save_image(frame, file_idx, frame_idx, frame_info, sample_idx)
            self.save_calib(frame, file_idx, frame_idx, frame_info)
            points = self.save_lidar(frame, file_idx, frame_idx, frame_info, frame_infos, sample_idx)
            self.save_pose(frame, file_idx, frame_idx)
            self.save_timestamp(frame, file_idx, frame_idx)

            if not self.test_mode:
                self.save_label(frame, file_idx, frame_idx)
                gtdb_info = self.save_label_my(frame, file_idx, frame_idx, points, frame_info, gtdb_tracking)
                cat_gtdb(gtdb_infos, gtdb_info)
            
            frame_infos.append(frame_info)

            frame_info_path = f'{self.frame_info_save_dir}/' + \
                f'{self.prefix}{str(file_idx).zfill(3)}' + \
                f'{str(frame_idx).zfill(3)}.npy'
            
            np.save(frame_info_path, frame_info)  # np.load or with open(filepath, 'rb') as f: value_buf = f.read()

        frame_infos = [numpy2list(frame_info) for frame_info
                       in frame_infos]
        gtdb_infos = {k: [numpy2list(vi) for vi in v]
                      for k, v in gtdb_infos.items()}

        return frame_infos, gtdb_infos

    def __len__(self):
        """Length of the filename list."""
        return len(self.tfrecord_pathnames)

    def save_image(self, frame, file_idx, frame_idx, frame_info, sample_idx):
        """Parse and save the images in png format.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        frame_info['images'] = [dict() for _ in range(5)]
        if self.load_img_panseg and frame.images[0].camera_segmentation_label.panoptic_label:
            frame_info['panseg_info'] = [dict() for _ in range(5)]
            self.img_panseg_index.append(sample_idx)

        for img in frame.images:
            # save image
            img_path = f'{self.image_save_dir}{str(img.name - 1)}/' + \
                f'{self.prefix}{str(file_idx).zfill(3)}' + \
                f'{str(frame_idx).zfill(3)}.png'
            # ./data/waymo/kitti_format/training/image_0/0000000.png
            frame_info['images'][img.name - 1]['path'] = img_path  # start with 0
            img_ = mmcv.imfrombytes(img.image)
            mmcv.imwrite(img_, img_path)

            # save panseg label
            if self.load_img_panseg and frame.images[0].camera_segmentation_label.panoptic_label:
                # decode a single panoptic label.
                panoptic_label = camera_segmentation_utils.decode_single_panoptic_label_from_proto(
                    img.camera_segmentation_label
                )
                # separate the panoptic label into semantic and instance labels.
                # 28 classes
                semantic_label, instance_label = self.decode_semantic_and_instance_labels_from_panoptic_label(
                    panoptic_label,
                    img.camera_segmentation_label.panoptic_label_divisor
                )
                panseg_path = f'{self.img_panseg_label_save_dir}{str(img.name - 1)}/' + \
                    f'{self.prefix}{str(file_idx).zfill(3)}' + \
                    f'{str(frame_idx).zfill(3)}.npz'
                np.savez_compressed(panseg_path, panseg_cls=semantic_label, panseg_instance_id=instance_label)
                frame_info['panseg_info'][img.name - 1]['path'] = panseg_path

    def save_calib(self, frame, file_idx, frame_idx, frame_info):
        """Parse and save the calibration data.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        # waymo front camera to kitti reference camera
        T_front_cam_to_ref = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0],
                                       [1.0, 0.0, 0.0]])
        camera_calibs = []
        R0_rect = [f'{i:e}' for i in np.eye(3).flatten()]
        Tr_velo_to_cams = []
        calib_context = ''

        for camera in frame.context.camera_calibrations:
            # extrinsic parameters
            T_cam_to_vehicle = np.array(camera.extrinsic.transform).reshape(
                4, 4)
            T_vehicle_to_cam = np.linalg.inv(T_cam_to_vehicle)
            Tr_velo_to_cam = \
                self.cart_to_homo(T_front_cam_to_ref) @ T_vehicle_to_cam
            frame_info['images'][camera.name - 1]['tf_lidar_to_cam'] = Tr_velo_to_cam    # 从waymo的vehicle的坐标系到kitti相机坐标系下的变换矩阵
            if camera.name == 1:  # FRONT = 1, see dataset.proto for details
                self.T_velo_to_front_cam = Tr_velo_to_cam.copy()
            Tr_velo_to_cam = Tr_velo_to_cam[:3, :].reshape((12, ))
            Tr_velo_to_cams.append([f'{i:e}' for i in Tr_velo_to_cam])

            # intrinsic parameters
            camera_calib = np.zeros((3, 4))
            camera_calib[0, 0] = camera.intrinsic[0]
            camera_calib[1, 1] = camera.intrinsic[1]
            camera_calib[0, 2] = camera.intrinsic[2]
            camera_calib[1, 2] = camera.intrinsic[3]
            camera_calib[2, 2] = 1
            frame_info['images'][camera.name - 1]['cam_intrinsic'] = self.cart_to_homo(camera_calib)
            camera_calib = list(camera_calib.reshape(12))
            camera_calib = [f'{i:e}' for i in camera_calib]
            camera_calibs.append(camera_calib)

        # all camera ids are saved as id-1 in the result because
        # camera 0 is unknown in the proto
        for i in range(5):
            calib_context += 'P' + str(i) + ': ' + \
                ' '.join(camera_calibs[i]) + '\n'
        calib_context += 'R0_rect' + ': ' + ' '.join(R0_rect) + '\n'
        for i in range(5):
            calib_context += 'Tr_velo_to_cam_' + str(i) + ': ' + \
                ' '.join(Tr_velo_to_cams[i]) + '\n'

        with open(
                f'{self.calib_save_dir}/{self.prefix}' +
                f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.txt',
                'w+') as fp_calib:
            fp_calib.write(calib_context)
            fp_calib.close()

    def save_lidar(self, frame, file_idx, frame_idx, frame_info, frame_infos, sample_idx):
        """Parse and save the lidar data in psd format.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        range_images, camera_projections, segmentation_labels, range_image_top_pose = \
            parse_range_image_and_camera_projection(frame)

        # First return
        points_0, cp_points_0, range_0, intensity_0, elongation_0, mask_indices_0 = \
            self.convert_range_image_to_point_cloud(
                frame,
                range_images,
                camera_projections,
                range_image_top_pose,
                ri_index=0
            )
        sensor_index_0 = np.concatenate([np.full_like(s, sid) for sid, s in enumerate(range_0)])
        points_0 = np.concatenate(points_0, axis=0)
        cp_points_0 = np.concatenate(cp_points_0, axis=0)
        range_0 = np.concatenate(range_0, axis=0)
        intensity_0 = np.concatenate(intensity_0, axis=0)
        elongation_0 = np.concatenate(elongation_0, axis=0)
        mask_indices_0 = np.concatenate(mask_indices_0, axis=0)
        ri_index_0 = np.full_like(range_0, 0)

        # Second return
        points_1, cp_points_1, range_1, intensity_1, elongation_1, mask_indices_1 = \
            self.convert_range_image_to_point_cloud(
                frame,
                range_images,
                camera_projections,
                range_image_top_pose,
                ri_index=1
            )
        sensor_index_1 = np.concatenate([np.full_like(s, sid, dtype=np.uint16) for sid, s in enumerate(range_1)])
        points_1 = np.concatenate(points_1, axis=0)
        cp_points_1 = np.concatenate(cp_points_1, axis=0)
        range_1 = np.concatenate(range_1, axis=0)
        intensity_1 = np.concatenate(intensity_1, axis=0)
        elongation_1 = np.concatenate(elongation_1, axis=0)
        mask_indices_1 = np.concatenate(mask_indices_1, axis=0)
        ri_index_1 = np.full_like(range_1, 1)

        sensor_index = np.concatenate([sensor_index_0, sensor_index_1], axis=0)
        points = np.concatenate([points_0, points_1], axis=0)
        cp_points = np.concatenate([cp_points_0, cp_points_1], axis=0)
        range_dist = np.concatenate([range_0, range_1], axis=0)
        intensity = np.concatenate([intensity_0, intensity_1], axis=0)
        elongation = np.concatenate([elongation_0, elongation_1], axis=0)
        mask_indices = np.concatenate([mask_indices_0, mask_indices_1], axis=0)
        ri_index = np.concatenate([ri_index_0, ri_index_1], axis=0)

        point_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                       ('intensity', 'f4'), ('elongation', 'f4'),
                       ('range_dist', 'f4'), ('return_idx', 'i2'),
                       ('lidar_idx', 'i2'), ('lidar_row', 'i2'),
                       ('lidar_column', 'i2'), ('cam_idx_0', 'i2'),
                       ('cam_row_0', 'i2'), ('cam_column_0', 'i2'),
                       ('cam_idx_1', 'i2'), ('cam_row_1', 'i2'),
                       ('cam_column_1', 'i2')]

        point_cloud = np.empty(len(points), dtype=point_dtype)
        point_cloud['x'] = points[:, 0]  # (N,)
        point_cloud['y'] = points[:, 1]  # (N,)
        point_cloud['z'] = points[:, 2]  # (N,)
        point_cloud['intensity'] = intensity    # (N,)
        point_cloud['elongation'] = elongation  # (N,)
        point_cloud['range_dist'] = range_dist  # (N,)
        point_cloud['return_idx'] = ri_index    # (N,) 记录是第几次回波
        point_cloud['lidar_idx'] = sensor_index  # 5个雷达的索引值
        point_cloud['lidar_row'] = mask_indices[:, 0]   # lidar对应range image的行
        point_cloud['lidar_column'] = mask_indices[:, 1]
        point_cloud['cam_idx_0'] = cp_points[:, 0] - 1  # -1 denotes no image
        point_cloud['cam_idx_1'] = cp_points[:, 3] - 1
        point_cloud['cam_column_0'] = cp_points[:, 1]
        point_cloud['cam_column_1'] = cp_points[:, 4]
        point_cloud['cam_row_0'] = cp_points[:, 2]
        point_cloud['cam_row_1'] = cp_points[:, 5]

        pc_path = f'{self.point_cloud_save_dir}/{self.prefix}' + \
            f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.bin'
        
        ego_pose = np.array(frame.pose.transform).reshape(4, 4)  # is in self.save_pose()
        frame_info['ego_pose'] = ego_pose

        point_cloud.tofile(pc_path)  # npz,npy,bin speed test ?

        sweeps = []
        for f in frame_infos[-self.sweeps:][::-1]:
            sweep = f['pts_info']
            sweeps.append(dict(
                path=sweep['path'],
                rel_pose=np.linalg.inv(ego_pose) @ f['ego_pose'],
                timestamp=sweep['timestamp']))

        frame_info['pts_info'] = dict(
            path=pc_path,
            range_image_shape=[],
            timestamp=frame.timestamp_micros,
            timestamp_step=1e-6,
            sweeps=sweeps)
        for c in sorted(frame.context.laser_calibrations,
                        key=lambda c: c.name):
            frame_info['pts_info']['range_image_shape'].append(
                list(range_images[c.name][0].shape.dims))

        # load and save lidar panseg labels
        if self.load_lidar_panseg and frame.lasers[0].ri_return1.segmentation_label_compressed:
            self.pts_panseg_index.append(sample_idx)
            semseg_path = f'{self.lidar_panseg_label_save_dir}/{self.prefix}' + \
                f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}'
            semseg_type = [('semseg_cls', 'i2'), ('instance_id', 'i2')]
            semseg_labels = np.empty(len(points), dtype=semseg_type)

            point_labels = self.convert_range_image_to_point_cloud_semseg(
                frame, range_images, segmentation_labels)
            point_labels_ri2 = self.convert_range_image_to_point_cloud_semseg(
                frame, range_images, segmentation_labels, ri_index=1)

            point_labels_all = np.concatenate(point_labels, axis=0)
            point_labels_all_ri2 = np.concatenate(point_labels_ri2, axis=0)
            point_labels_all = np.concatenate(
                [point_labels_all, point_labels_all_ri2], axis=0)

            semseg_labels['semseg_cls'] = point_labels_all[:, 1]
            semseg_labels['instance_id'] = point_labels_all[:, 0]
            np.save(semseg_path, semseg_labels)
            frame_info['semseg_info'] = dict(path=semseg_path)

        return point_cloud

    def save_label_my(self, frame, file_idx, frame_idx, points, frame_info, gtdb_tracking):

        sample_idx = self.sample_index(file_idx, frame_idx)

        annos = dict()
        annos['bbox'] = [[] for _ in range(5)]
        annos['name'] = [[] for _ in range(5)]
        annos['bbox_3d'] = []
        annos['name_3d'] = []
        annos['difficulty'] = []
        annos['track_id'] = []
        annos['track_difficulty'] = []
        
        # get camera labels
        for labels in frame.camera_labels:
            for obj in labels.labels:
                # 获取对象类别，obj.type是int类型，{0:"DontCare",1:"Car",2,3,4}
                my_type = self.type_list[obj.type]
                bbox = [
                    obj.box.center_x - obj.box.length / 2,
                    obj.box.center_y - obj.box.width / 2,
                    obj.box.center_x + obj.box.length / 2,
                    obj.box.center_y + obj.box.width / 2
                ]  # 存放的是2D box的左上和右下的 两个角点
                annos['bbox'][labels.name - 1].append(bbox)     # 放入对应相机 bbox内，
                annos['name'][labels.name - 1].append(my_type)  # 放入对应相机 类别内，
        
        # get lidar labels
        for obj in frame.laser_labels:
            my_type = self.type_list[obj.type]
            height = obj.box.height
            width = obj.box.width
            length = obj.box.length

            x = obj.box.center_x
            y = obj.box.center_y
            z = obj.box.center_z - height / 2  # 这里的中心点坐标记录的是3D box的底面的

            yaw = obj.box.heading  # 偏航角，逆时针偏离x轴
            track_id = obj.id
            det_difficulty = obj.detection_difficulty_level
            track_difficulty = obj.tracking_difficulty_level

            annos['bbox_3d'].append([x, y, z, length, width, height, yaw])
            annos['name_3d'].append(my_type)
            annos['difficulty'].append(det_difficulty)
            annos['track_id'].append(track_id)
            annos['track_difficulty'].append(track_difficulty)

        xyz = np.stack([points['x'], points['y'], points['z']], axis=-1)  # (N, 3), 表示所有点的坐标

        if len(annos['bbox_3d']) == 0:
            annos['bbox_3d'] = np.zeros((0, 7), dtype=np.float64)
        else:
            annos['bbox_3d'] = np.array(annos['bbox_3d'], dtype=np.float64)
        # 获得在该帧下所有在box内的点的真值True, False
        inside_mask = points_in_rbbox(xyz, annos['bbox_3d'])
        annos['num_lidar_points_in_box'] = inside_mask.sum(axis=0).tolist()
        frame_info['annos'] = annos     # 每一帧的所有标签信息
        gtdb = dict()                   # 一帧frame下的所有obj字典
        tracking = dict()
        # 遍历每个对象的inside_mask
        for obj_idx, obj_mask in enumerate(inside_mask.T):
            obj_bbox_3d = annos['bbox_3d'][obj_idx]
            obj_name = annos["name_3d"][obj_idx]
            obj_track_id = annos['track_id'][obj_idx]
            obj_points = points[obj_mask]
            gtdb_pc_path = f'{self.gtdb_save_dir}/{sample_idx}_{obj_name}_{obj_idx}'
            np.save(gtdb_pc_path, obj_points)  # obj_points.nbytes/1024/1024 MB

            obj_sweeps = []
            for t in gtdb_tracking[-self.sweeps:][::-1]:
                obj_sweep = t.get(obj_track_id, None)
                if obj_sweep is not None:
                    obj_sweeps.append(dict(
                        path=obj_sweep['pts_info']['path'],
                        rel_pose=np.linalg.inv(
                            frame_info['ego_pose']) @ obj_sweep['ego_pose'],
                        box3d_lidar=obj_sweep['box3d_lidar'],
                        timestamp=obj_sweep['pts_info']['timestamp']))

            obj_gtdb = dict(
                name=obj_name,
                pts_info=dict(
                    path=gtdb_pc_path,
                    timestamp=frame_info['pts_info']['timestamp'],
                    timestamp_step=frame_info['pts_info']['timestamp_step'],
                    sweeps=obj_sweeps),
                image_idx=sample_idx,  # 表示一个tfrecord内的第几帧数据
                gt_idx=obj_idx,
                box3d_lidar=obj_bbox_3d,
                num_points_in_gt=annos['num_lidar_points_in_box'][obj_idx],
                difficulty=annos['difficulty'][obj_idx],
                ego_pose=frame_info['ego_pose'],
                track_id=obj_track_id)
            if obj_name not in gtdb:
                gtdb[obj_name] = []
            gtdb[obj_name].append(obj_gtdb)
            tracking[obj_track_id] = obj_gtdb
        gtdb_tracking.append(tracking)
        return gtdb

    def save_label(self, frame, file_idx, frame_idx):
        """Parse and save the label data in txt format.
        The relation between waymo and kitti coordinates is noteworthy:
        1. x, y, z correspond to l, w, h (waymo) -> l, h, w (kitti)
        2. x-y-z: front-left-up (waymo) -> right-down-front(kitti)
        3. bbox origin at volumetric center (waymo) -> bottom center (kitti)
        4. rotation: +x around y-axis (kitti) -> +x around z-axis (waymo)

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        fp_label_all = open(
            f'{self.label_all_save_dir}/{self.prefix}' +
            f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.txt', 'w+')
        id_to_bbox = dict()
        id_to_name = dict()
        for labels in frame.projected_lidar_labels:
            name = labels.name
            for label in labels.labels:
                # TODO: need a workaround as bbox may not belong to front cam
                bbox = [
                    label.box.center_x - label.box.length / 2,
                    label.box.center_y - label.box.width / 2,
                    label.box.center_x + label.box.length / 2,
                    label.box.center_y + label.box.width / 2
                ]
                id_to_bbox[label.id] = bbox
                id_to_name[label.id] = name - 1

        for obj in frame.laser_labels:
            bounding_box = None
            name = None
            id = obj.id
            for lidar in self.lidar_list:
                if id + lidar in id_to_bbox:
                    bounding_box = id_to_bbox.get(id + lidar)
                    name = str(id_to_name.get(id + lidar))
                    break

            if bounding_box is None or name is None:
                name = '0'
                bounding_box = (0, 0, 0, 0)

            my_type = self.type_list[obj.type]

            if my_type not in self.selected_waymo_classes:
                continue

            if self.filter_empty_3dboxes and obj.num_lidar_points_in_box < 1:
                continue

            my_type = self.waymo_to_kitti_class_map[my_type]

            height = obj.box.height
            width = obj.box.width
            length = obj.box.length

            x = obj.box.center_x
            y = obj.box.center_y
            z = obj.box.center_z - height / 2

            # project bounding box to the virtual reference frame
            pt_ref = self.T_velo_to_front_cam @ \
                np.array([x, y, z, 1]).reshape((4, 1))
            x, y, z, _ = pt_ref.flatten().tolist()

            rotation_y = -obj.box.heading - np.pi / 2
            track_id = obj.id

            # not available
            truncated = 0
            occluded = 0
            alpha = -10

            line = my_type + \
                ' {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(
                    round(truncated, 2), occluded, round(alpha, 2),
                    round(bounding_box[0], 2), round(bounding_box[1], 2),
                    round(bounding_box[2], 2), round(bounding_box[3], 2),
                    round(height, 2), round(width, 2), round(length, 2),
                    round(x, 2), round(y, 2), round(z, 2),
                    round(rotation_y, 2))

            if self.save_track_id:
                line_all = line[:-1] + ' ' + name + ' ' + track_id + '\n'
            else:
                line_all = line[:-1] + ' ' + name + '\n'

            fp_label = open(
                f'{self.label_save_dir}{name}/{self.prefix}' +
                f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.txt', 'a')
            fp_label.write(line)
            fp_label.close()

            fp_label_all.write(line_all)

        fp_label_all.close()

    def save_pose(self, frame, file_idx, frame_idx):
        """Parse and save the pose data.

        Note that SDC's own pose is not included in the regular training
        of KITTI dataset. KITTI raw dataset contains ego motion files
        but are not often used. Pose is important for algorithms that
        take advantage of the temporal information.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        pose = np.array(frame.pose.transform).reshape(4, 4)
        np.savetxt(
            join(f'{self.pose_save_dir}/{self.prefix}' +
                 f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.txt'),
            pose)

    def save_timestamp(self, frame, file_idx, frame_idx):
        """Save the timestamp data in a separate file instead of the
        pointcloud.

        Note that SDC's own pose is not included in the regular training
        of KITTI dataset. KITTI raw dataset contains ego motion files
        but are not often used. Pose is important for algorithms that
        take advantage of the temporal information.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        with open(
                join(f'{self.timestamp_save_dir}/{self.prefix}' +
                     f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.txt'),
                'w') as f:
            f.write(str(frame.timestamp_micros))

    def create_folder(self):
        """Create folder for data preprocessing."""
        if not self.test_mode:
            dir_list1 = [
                self.label_all_save_dir, self.calib_save_dir,
                self.point_cloud_save_dir, self.pose_save_dir,
                self.timestamp_save_dir, self.lidar_panseg_label_save_dir,
                self.frame_info_save_dir, self.gtdb_save_dir
            ]
            dir_list2 = [self.label_save_dir, self.image_save_dir,
                         self.img_panseg_label_save_dir]
        else:
            dir_list1 = [
                self.calib_save_dir, self.point_cloud_save_dir,
                self.pose_save_dir, self.timestamp_save_dir,
                self.frame_info_save_dir
            ]
            dir_list2 = [self.image_save_dir]
        for d in dir_list1:
            mmcv.mkdir_or_exist(d)
        for d in dir_list2:
            for i in range(5):
                mmcv.mkdir_or_exist(f'{d}{str(i)}')

    def convert_range_image_to_point_cloud(self,
                                           frame,
                                           range_images,
                                           camera_projections,
                                           range_image_top_pose,
                                           ri_index=0):
        """Convert range images to point cloud.

        Args:
            frame (:obj:`Frame`): Open dataset frame.
            range_images (dict): Mapping from laser_name to list of two
                range images corresponding with two returns.
            camera_projections (dict): Mapping from laser_name to list of two
                camera projections corresponding with two returns.
            range_image_top_pose (:obj:`Transform`): Range image pixel pose for
                top lidar.
            ri_index (int, optional): 0 for the first return,
                1 for the second return. Default: 0.

        Returns:
            tuple[list[np.ndarray]]: (List of points with shape [N, 3],
                camera projections of points with shape [N, 6], intensity
                with shape [N, 1], elongation with shape [N, 1], points'
                position in the depth map (element offset if points come from
                the main lidar otherwise -1) with shape[N, 1]). All the
                lists have the length of lidar numbers (5).
        """
        calibrations = sorted(
            frame.context.laser_calibrations, key=lambda c: c.name)
        points = []
        cp_points = []
        range_dist = []
        intensity = []
        elongation = []
        mask_indices = []

        frame_pose = tf.convert_to_tensor(
            value=np.reshape(np.array(frame.pose.transform), [4, 4]))
        # [H, W, 6]
        range_image_top_pose_tensor = tf.reshape(
            tf.convert_to_tensor(value=range_image_top_pose.data),
            range_image_top_pose.shape.dims)
        # [H, W, 3, 3]
        range_image_top_pose_tensor_rotation = \
            transform_utils.get_rotation_matrix(
                range_image_top_pose_tensor[..., 0],
                range_image_top_pose_tensor[..., 1],
                range_image_top_pose_tensor[..., 2])
        range_image_top_pose_tensor_translation = \
            range_image_top_pose_tensor[..., 3:]
        range_image_top_pose_tensor = transform_utils.get_transform(
            range_image_top_pose_tensor_rotation,
            range_image_top_pose_tensor_translation)
        for c in calibrations:
            range_image = range_images[c.name][ri_index]
            if len(c.beam_inclinations) == 0:
                beam_inclinations = range_image_utils.compute_inclination(
                    tf.constant(
                        [c.beam_inclination_min, c.beam_inclination_max]),
                    height=range_image.shape.dims[0])
            else:
                beam_inclinations = tf.constant(c.beam_inclinations)

            beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
            extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(value=range_image.data),
                range_image.shape.dims)
            pixel_pose_local = None
            frame_pose_local = None
            if c.name == dataset_pb2.LaserName.TOP:
                pixel_pose_local = range_image_top_pose_tensor
                pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
                frame_pose_local = tf.expand_dims(frame_pose, axis=0)
            range_image_mask = range_image_tensor[..., 0] > 0

            if self.filter_no_label_zone_points:
                nlz_mask = range_image_tensor[..., 3] != 1.0  # 1.0: in NLZ
                range_image_mask = range_image_mask & nlz_mask

            range_image_cartesian = \
                range_image_utils.extract_point_cloud_from_range_image(
                    tf.expand_dims(range_image_tensor[..., 0], axis=0),
                    tf.expand_dims(extrinsic, axis=0),
                    tf.expand_dims(tf.convert_to_tensor(
                        value=beam_inclinations), axis=0),
                    pixel_pose=pixel_pose_local,
                    frame_pose=frame_pose_local)

            mask_index = tf.where(range_image_mask)

            range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
            points_tensor = tf.gather_nd(range_image_cartesian, mask_index)

            cp = camera_projections[c.name][ri_index]
            cp_tensor = tf.reshape(
                tf.convert_to_tensor(value=cp.data), cp.shape.dims)
            cp_points_tensor = tf.gather_nd(cp_tensor, mask_index)
            points.append(points_tensor.numpy())
            cp_points.append(cp_points_tensor.numpy())

            range_tensor = tf.gather_nd(range_image_tensor[..., 0], 
                                            mask_index)
            range_dist.append(range_tensor.numpy())

            intensity_tensor = tf.gather_nd(range_image_tensor[..., 1],
                                            mask_index)
            intensity.append(intensity_tensor.numpy())

            elongation_tensor = tf.gather_nd(range_image_tensor[..., 2],
                                             mask_index)
            elongation.append(elongation_tensor.numpy())

            # 这一步的作用？注释掉，不需要，我需要的是对应在range image的位置
            # if c.name == 1:
            #     mask_index = (ri_index * range_image_mask.shape[0] +
            #                   mask_index[:, 0]
            #                   ) * range_image_mask.shape[1] + mask_index[:, 1]
            #     mask_index = mask_index.numpy().astype(elongation[-1].dtype)
            # else:
            #     mask_index = np.full_like(elongation[-1], -1)

            mask_indices.append(mask_index)

        return points, cp_points, range_dist, intensity, elongation, mask_indices

    def convert_range_image_to_point_cloud_semseg(self,
                                                  frame,
                                                  range_images,
                                                  segmentation_labels,
                                                  ri_index=0):
        """Convert segmentation labels from range images to point clouds.
        Args:
            frame: open dataset frame
            range_images: A dict of {laser_name, [range_image_first_return,
            range_image_second_return]}.
            segmentation_labels: A dict of {laser_name, [range_image_first_return,
            range_image_second_return]}.
            ri_index: 0 for the first return, 1 for the second return.
        Returns:
            point_labels: {[N, 2]} list of 3d lidar points's segmentation labels. 0 for
            points that are not labeled.
        """
        calibrations = sorted(
            frame.context.laser_calibrations, key=lambda c: c.name)
        point_labels = []
        for c in calibrations:
            range_image = range_images[c.name][ri_index]
            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(range_image.data), range_image.shape.dims)
            range_image_mask = range_image_tensor[..., 0] > 0

            if self.filter_no_label_zone_points:
                nlz_mask = range_image_tensor[..., 3] != 1.0  # 1.0: in NLZ
                range_image_mask = range_image_mask & nlz_mask

            if c.name in segmentation_labels:
                sl = segmentation_labels[c.name][ri_index]
                sl_tensor = tf.reshape(
                    tf.convert_to_tensor(sl.data), sl.shape.dims)
                sl_points_tensor = tf.gather_nd(
                    sl_tensor, tf.where(range_image_mask))
            else:
                num_valid_point = tf.math.reduce_sum(
                    tf.cast(range_image_mask, tf.int32))
                sl_points_tensor = tf.zeros(
                    [num_valid_point, 2], dtype=tf.int32)

            point_labels.append(sl_points_tensor.numpy())
        return point_labels

    def cart_to_homo(self, mat):
        """Convert transformation matrix in Cartesian coordinates to
        homogeneous format.

        Args:
            mat (np.ndarray): Transformation matrix in Cartesian.
                The input matrix shape is 3x3 or 3x4.

        Returns:
            np.ndarray: Transformation matrix in homogeneous format.
                The matrix shape is 4x4.
        """
        ret = np.eye(4)
        if mat.shape == (3, 3):
            ret[:3, :3] = mat
        elif mat.shape == (3, 4):
            ret[:3, :] = mat
        else:
            raise ValueError(mat.shape)
        return ret

    def decode_semantic_and_instance_labels_from_panoptic_label(self,
        panoptic_label,
        panoptic_label_divisor):
        """Converts a panoptic label into semantic and instance segmentation labels.
        Args:
            panoptic_label: A 2D array where each pixel is encoded as: semantic_label *
            panoptic_label_divisor + instance_label.
            panoptic_label_divisor: an int used to encode the panoptic labels.
        Returns:
            A tuple containing the semantic and instance labels, respectively.
        """
        if panoptic_label_divisor <= 0:
            raise ValueError("panoptic_label_divisor must be > 0.")
        return np.divmod(panoptic_label, panoptic_label_divisor)