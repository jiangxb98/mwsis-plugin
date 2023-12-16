# coding: utf-8


try:
    from waymo_open_dataset import dataset_pb2
    from waymo_open_dataset.protos import segmentation_metrics_pb2
    from waymo_open_dataset.protos import segmentation_submission_pb2
except ImportError:
    raise ImportError(
        'Please run "pip install waymo-open-dataset-tf-2-6-0==1.4.9" '
        'to install the official devkit first.')

import io
import time
import mmcv
import numpy as np
import os.path as osp
import tensorflow as tf

from glob import glob
from minio import Minio
from pymongo import MongoClient

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


class WaymoPrep(object):
    def __init__(self,
                 load_dir,
                 bucket,
                 database,
                 split,
                 minio_cfg,
                 mongo_cfg,
                 sweeps=5,
                 workers=8,
                 test_mode=False,
                 load_semseg=True):

        self.minio_cfg = minio_cfg
        self.mongo_cfg = mongo_cfg

        self.filter_empty_3dboxes = True
        self.filter_no_label_zone_points = True  # 非标记区域的位置

        self.selected_waymo_classes = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']

        self.selected_waymo_locations = None  #
        self.save_track_id = True

        if int(tf.__version__.split('.')[0]) < 2:
            tf.enable_eager_execution()

        self.lidar_list = [
            '_FRONT', '_FRONT_RIGHT', '_FRONT_LEFT', '_SIDE_RIGHT',
            '_SIDE_LEFT'
        ]
        self.type_list = ['DontCare', 'Car', 'Pedestrian', 'Sign', 'Cyclist']

        self.load_dir = load_dir
        self.bucket = bucket
        self.database = database
        self.split = split
        self.prefix = dict(training=0, validation=1, testing=2)[split]
        self.workers = int(workers)
        self.test_mode = test_mode
        self.load_semseg = load_semseg

        self.tfrecord_pathnames = sorted(
            glob(osp.join(self.load_dir, '*.tfrecord')))   # 获取所有的tfrecord文件名
        self.image_prefix = f'{self.split}/image'
        self.lidar_prefix = f'{self.split}/lidar'
        self.gtdb_prefix = f'{self.split}/gtdb'
        self.semseg_prefix = f'{self.split}/semseg'
        self.panseg_prefix = f'{self.split}/panseg'  # 记录全景语义和实例标签
        self.sweeps = sweeps
        self.infos_fname = f'{self.split}/infos'
        self.dbinfos_fname = f'{self.split}/dbinfos/''{}'

    def sample_index(self, file_idx, frame_idx):
        return self.prefix * 1000000 + file_idx * 1000 + frame_idx

    def convert(self):
        print('Start converting ...')
        # data_client = Minio(**self.minio_cfg)
        # found = data_client.bucket_exists(self.bucket)
        # if not found:
        #     data_client.make_bucket(self.bucket)
        if self.workers == 0:
            result = []
            for idx in mmcv.track_iter_progress(range(len(self))):
                # result放的是798个训练场景，一个tfrecord是一个场景，一个798个
                result.append(self.convert_one(idx))
        else:
            result = mmcv.track_parallel_progress(self.convert_one,
                                                  range(len(self)),
                                                  self.workers)
        print('\nOrganizing ...')
        # infos_client = MongoClient(**self.mongo_cfg)
        # infos_database = infos_client[self.database]

        gtdb_infos = {}
        frames_infos = []
        for clip_index, clip_gtdb in mmcv.track_iter_progress(result):
            frames_infos.extend(clip_index)
            gtdb_infos = cat_gtdb(gtdb_infos, clip_gtdb)

        print(f'\nDumping {len(frames_infos)} items'
              f'into mongodb collection {self.infos_fname} ...')
        # while True:
        #     try:
        #         if self.infos_fname in infos_database.list_collection_names():
        #             infos_database.drop_collection(self.infos_fname)
        #         infos_collection = infos_database.create_collection(
        #             self.infos_fname)
        #         infos_collection.insert_many(frames_infos)
        #         pass
        #     except Exception as e:
        #         print(e)
        #         time.sleep(0.1)
        #         continue
        #     break

        for cls in gtdb_infos:
            dbinfos_fname = self.dbinfos_fname.format(cls)
            print(f'\nDumping {len(gtdb_infos[cls])} items'
                  f' dbinfos into mongodb collection {dbinfos_fname} ...')
            # while True:
            #     try:
            #         if dbinfos_fname in infos_database.list_collection_names():
            #             infos_database.drop_collection(dbinfos_fname)
            #         dbinfos_collection = infos_database.create_collection(
            #             dbinfos_fname)
            #         dbinfos_collection.insert_many(gtdb_infos[cls])
            #         pass
            #     except Exception as e:
            #         print(e)
            #         time.sleep(0.1)
            #         continue
            #     break

        print('\nFinished ...')

    def convert_one(self, file_idx):
        # 加载一个tfrecord数据（里面199帧数据）
        pathname = self.tfrecord_pathnames[file_idx]
        dataset = tf.data.TFRecordDataset(pathname, compression_type='')

        frame_infos = []
        gtdb_tracking = []
        gtdb_infos = dict()
        # data_client = Minio(**self.minio_cfg)
        data_client = None  # 防止上传，一定注意workers=0,设置好！ mongodb也得注释掉，不然会清空
        # 迭代遍历dataset，获取每帧的信息
        for frame_idx, data in enumerate(dataset):
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

            # 将五张图片信息的地址存到frame_info，也有放到Minio
            self.save_image(data_client, frame, sample_idx, frame_info)
            self.save_calib(frame, sample_idx, frame_info)  # 存放相机内外参
            points = self.save_lidar(
                data_client, frame, sample_idx, frame_info, frame_infos)
            if not self.test_mode:
                gtdb_info = self.save_label(
                    data_client, frame, sample_idx, points, frame_info,
                    gtdb_tracking)
                # 将得到的一帧的gtdb_info放入一个总的gtdb_infos下，这个gtdb_infos是指一个tfrecord的(199帧)
                cat_gtdb(gtdb_infos, gtdb_info)
            frame_infos.append(frame_info)
        frame_infos = [numpy2list(frame_info) for frame_info
                       in frame_infos]
        gtdb_infos = {k: [numpy2list(vi) for vi in v]
                      for k, v in gtdb_infos.items()}
        return frame_infos, gtdb_infos

    def __len__(self):
        return len(self.tfrecord_pathnames)

    def save_image(self, client, frame, sample_idx, frame_info):
        # 5个空字典,存放五张图片的信息[{'path': 'training/image/0000000/0'}, {'path': 'training/image/0000000/1'}, {}, {}, {}]
        frame_info['images'] = [dict() for _ in range(5)]
  
        if self.load_semseg and frame.images[0].camera_segmentation_label.panoptic_label:
            frame_info['panseg_info'] = [dict() for _ in range(5)]  
        for img in frame.images:
            img_path = f'{self.image_prefix}/{sample_idx:07d}/{str(img.name - 1)}'
            panseg_path = f'{self.panseg_prefix}/{sample_idx:07d}/{str(img.name - 1)}'
            frame_info['images'][img.name - 1]['path'] = img_path  # img.name - 1,为了统一从0开始计数
            
            # 获取语义标签
            if self.load_semseg and frame.images[0].camera_segmentation_label.panoptic_label:
                # Decode a single panoptic label.
                panoptic_label = camera_segmentation_utils.decode_single_panoptic_label_from_proto(
                    img.camera_segmentation_label
                )
                # Separate the panoptic label into semantic and instance labels.
                # 28类
                semantic_label, instance_label = self.decode_semantic_and_instance_labels_from_panoptic_label(
                    panoptic_label,
                    img.camera_segmentation_label.panoptic_label_divisor
                )
                # while True:
                #     try:
                #         buf = io.BytesIO()
                #         np.savez_compressed(buf, panseg_cls=semantic_label, panseg_instance_id=instance_label)
                #         buf.seek(0, 2)
                #         buf_size = buf.tell()
                #         buf.seek(0, 0)
                #         client.put_object(self.bucket, panseg_path, buf, buf_size)
                #     except Exception as e:
                #         print(e)
                #         time.sleep(0.1)
                #         continue
                #     break
                frame_info['panseg_info'][img.name - 1]['path'] = panseg_path
            # while True:
            #     try:
            #         client.put_object(self.bucket, img_path,
            #                           io.BytesIO(img.image), len(img.image))
            #     except Exception as e:
            #         print(e)
            #         time.sleep(0.1)
            #         continue
            #     break

    def save_calib(self, frame, sample_idx, frame_info):
        # waymo front camera to kitti reference camera 从waymo相机坐标系转到kitti相机坐标系
        tf_front_cam_to_ref = np.array([[0.0, -1.0, 0.0, 0.0],
                                        [0.0, 0.0, -1.0, 0.0],
                                        [1.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0]])

        for camera in sorted(frame.context.camera_calibrations,
                             key=lambda c: c.name):
            # extrinsic parameters 相机外参，P_vehicle = T x P_camera 将相机坐标系的坐标变换到vehicle下
            tf_cam_to_vehicle = np.array(camera.extrinsic.transform).reshape(
                4, 4)
            tf_vehicle_to_cam = np.linalg.inv(tf_cam_to_vehicle)    # 逆矩阵
            # = tf_velo_to_cam，这里的lidar=vehicle
            tf_lidar_to_cam = tf_front_cam_to_ref @ tf_vehicle_to_cam
            frame_info['images'][camera.name -
                                 1]['tf_lidar_to_cam'] = tf_lidar_to_cam    # 从waymo的vehicle的坐标系到kitti相机坐标系下的变换矩阵

            # intrinsic parameters 在相机坐标系的点转到图像上 P_2d = K x P_cam
            cam_intrinsic = np.eye(4)
            cam_intrinsic[0, 0] = camera.intrinsic[0]   # fx
            cam_intrinsic[1, 1] = camera.intrinsic[1]   # fy
            cam_intrinsic[0, 2] = camera.intrinsic[2]   # u
            cam_intrinsic[1, 2] = camera.intrinsic[3]   # v
            frame_info['images'][camera.name -
                                 1]['cam_intrinsic'] = cam_intrinsic

    def save_lidar(self, client, frame, sample_idx, frame_info, frame_infos):
        range_images, camera_projections, segmentation_labels, range_image_top_pose = \
            parse_range_image_and_camera_projection(
                frame)  # segmentation_label这个版本是没有的[instance_id, semantic_class]

        # First return
        (points_0, cp_points_0, range_0, intensity_0, elongation_0,
         mask_indices_0) = self.convert_range_image_to_point_cloud(
            frame,
            range_images,  # 5个雷达(key)×[(64,2650,4), (64,2650,4)]列表里表示两次回波
            # 5个相机(key)×[(64,2650,6), (64,2650,6)] 6=(camera_name,x,y,camera_name,x,y),点云能投影到多个相机，只取前两个投影
            camera_projections,
            # 两次回波的位姿都一样[64, 2650, 6],range image pixel pose for top lidar.6表示在此时刻top雷达的pose[roll, pitch, yaw, x, y, z]
            range_image_top_pose,
            ri_index=0)  # 第几次回波的索引
        sensor_index_0 = np.concatenate(
            [np.full_like(s, sid) for sid, s in enumerate(range_0)])  # 五个雷达的点云放一块，然后标记处所属lidar的索引
        points_0 = np.concatenate(points_0, axis=0)
        cp_points_0 = np.concatenate(cp_points_0, axis=0)
        range_0 = np.concatenate(range_0, axis=0)
        intensity_0 = np.concatenate(intensity_0, axis=0)
        elongation_0 = np.concatenate(elongation_0, axis=0)
        # (171606, 2),range image 索引值，每个lidar点所以在的range image的位置
        mask_indices_0 = np.concatenate(mask_indices_0, axis=0)
        ri_index_0 = np.full_like(range_0, 0)  # 全覆盖为0，表示是第一次回波

        # Second return
        (points_1, cp_points_1, range_1, intensity_1, elongation_1,
         mask_indices_1) = self.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose,
            ri_index=1)  # 第二次回波的数据
        sensor_index_1 = np.concatenate(
            [np.full_like(s, sid, dtype=np.uint16) for sid, s in
             enumerate(range_1)])
        points_1 = np.concatenate(points_1, axis=0)
        cp_points_1 = np.concatenate(cp_points_1, axis=0)
        range_1 = np.concatenate(range_1, axis=0)
        intensity_1 = np.concatenate(intensity_1, axis=0)
        elongation_1 = np.concatenate(elongation_1, axis=0)
        mask_indices_1 = np.concatenate(mask_indices_1, axis=0)
        ri_index_1 = np.full_like(range_1, 1)   # 全覆盖为1，标识是第二次回波

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
        point_cloud['intensity'] = intensity  # (N,)
        point_cloud['elongation'] = elongation  # (N,)
        point_cloud['range_dist'] = range_dist  # (N,)
        point_cloud['return_idx'] = ri_index  # (N,) 记录是第几次回波
        point_cloud['lidar_idx'] = sensor_index  # 5个雷达的索引值
        point_cloud['lidar_row'] = mask_indices[:, 0]  # lidar对应range image的行
        point_cloud['lidar_column'] = mask_indices[:, 1]
        point_cloud['cam_idx_0'] = cp_points[:, 0] - 1  # 这里为啥-1, 和上面的img.name - 1保持一致，-1表示没有
        point_cloud['cam_idx_1'] = cp_points[:, 3] - 1
        point_cloud['cam_column_0'] = cp_points[:, 1]
        point_cloud['cam_column_1'] = cp_points[:, 4]
        point_cloud['cam_row_0'] = cp_points[:, 2]
        point_cloud['cam_row_1'] = cp_points[:, 5]

        pc_path = f'{self.lidar_prefix}/{sample_idx:07d}'
        # vehicle在global下的位姿，Pglobal = ego_pose x Pvehicle
        ego_pose = np.array(frame.pose.transform).reshape(4, 4)
        frame_info['ego_pose'] = ego_pose
        # 没看懂这里的sweeps是做什么的？记录他的上一帧
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

        # while True:
        #     try:
        #         buf = io.BytesIO()
        #         np.save(buf, point_cloud)
        #         buf.seek(0, 2)
        #         buf_size = buf.tell()
        #         buf.seek(0, 0)
        #         client.put_object(self.bucket, pc_path, buf, buf_size)
        #     except Exception as e:
        #         print(e)
        #         time.sleep(0.1)
        #         continue
        #     break

        # Point sematic & instance segmentation labels
        if self.load_semseg and frame.lasers[0].ri_return1.segmentation_label_compressed:
            semseg_path = f'{self.semseg_prefix}/{sample_idx:07d}'
            semseg_type = [('semseg_cls', 'i2'), ('instance_id', 'i2')]
            semseg_labels = np.empty(len(points), dtype=semseg_type)

            point_labels = self.convert_range_image_to_point_cloud_semseg(
                frame, range_images, segmentation_labels)
            point_labels_ri2 = self.convert_range_image_to_point_cloud_semseg(
                frame, range_images, segmentation_labels, ri_index=1)

            # point labels.
            point_labels_all = np.concatenate(point_labels, axis=0)
            point_labels_all_ri2 = np.concatenate(point_labels_ri2, axis=0)
            point_labels_all = np.concatenate(
                [point_labels_all, point_labels_all_ri2], axis=0)

            semseg_labels['semseg_cls'] = point_labels_all[:, 1]
            semseg_labels['instance_id'] = point_labels_all[:, 0]

            # while True:
            #     try:
            #         buf = io.BytesIO()
            #         np.save(buf, semseg_labels)
            #         buf.seek(0, 2)
            #         buf_size = buf.tell()
            #         buf.seek(0, 0)
            #         client.put_object(self.bucket, semseg_path, buf, buf_size)
            #     except Exception as e:
            #         print(e)
            #         time.sleep(0.1)
            #         continue
            #     break

            frame_info['semseg_info'] = dict(path=semseg_path)

        return point_cloud

    def save_label(self, client, frame, sample_idx, points, frame_info,
                   gtdb_tracking):
        annos = dict()
        annos['bbox'] = [[] for _ in range(5)]
        annos['name'] = [[] for _ in range(5)]
        annos['bbox_3d'] = []  # annos 存放的是一帧frame下的所有对象的3D boxes信息
        annos['name_3d'] = []
        annos['difficulty'] = []
        annos['track_id'] = []
        annos['track_difficulty'] = []
        # 获得camera的labels
        for labels in frame.camera_labels:  # for 循环五个相机
            for obj in labels.labels:  # for 循环每个相机内的对象
                # 获取对象类别，obj.type是int类型，{0:"DontCare",1:"Car",2,3,4}
                my_type = self.type_list[obj.type]
                bbox = [
                    obj.box.center_x - obj.box.length / 2,
                    obj.box.center_y - obj.box.width / 2,
                    obj.box.center_x + obj.box.length / 2,
                    obj.box.center_y + obj.box.width / 2
                ]  # 存放的是2D box的左上和右下的 两个角点
                annos['bbox'][labels.name - 1].append(bbox)  # 放入对应相机 bbox内，
                annos['name'][labels.name - 1].append(my_type)  # 放入对应相机 类别内，
        # 获得lidar的labels
        for obj in frame.laser_labels:
            my_type = self.type_list[obj.type]  # 同上，这里和2D的一样
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

        xyz = np.stack([points['x'], points['y'], points['z']],
                       axis=-1)  # (N, 3), 表示所有点的坐标
        if len(annos['bbox_3d']) == 0:
            annos['bbox_3d'] = np.zeros((0, 7), dtype=np.float64)
        else:
            annos['bbox_3d'] = np.array(annos['bbox_3d'], dtype=np.float64)
        # 获得在该帧下所有在box内的点的真值True,False
        inside_mask = points_in_rbbox(xyz, annos['bbox_3d'])
        annos['num_lidar_points_in_box'] = inside_mask.sum(
            axis=0).tolist()  # 统计每个obj内有多少点在里面，是个列表
        frame_info['annos'] = annos  # 每一帧的所有标签信息
        gtdb = dict()  # 一帧frame下的所有obj字典
        tracking = dict()
        # 遍历每个对象的inside_mask
        for obj_idx, obj_mask in enumerate(inside_mask.T):
            obj_bbox_3d = annos['bbox_3d'][obj_idx]
            obj_name = annos["name_3d"][obj_idx]
            obj_track_id = annos['track_id'][obj_idx]
            obj_points = points[obj_mask]
            gtdb_pc_path = f'{self.gtdb_prefix}/{sample_idx}_{obj_name}_{obj_idx}'

            # while True:
            #     try:
            #         buf = io.BytesIO()
            #         np.save(buf, obj_points)
            #         buf.seek(0, 2)
            #         buf_size = buf.tell()
            #         buf.seek(0, 0)
            #         client.put_object(self.bucket, gtdb_pc_path, buf, buf_size)
            #     except Exception as e:
            #         print(e)
            #         time.sleep(0.1)
            #         continue
            #     break

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

    def convert_range_image_to_point_cloud(self,
                                           frame,
                                           range_images,
                                           camera_projections,
                                           range_image_top_pose,
                                           ri_index=0):
        calibrations = sorted(
            frame.context.laser_calibrations, key=lambda c: c.name)
        points = []  # len(points)=5 points[0].shape=(N,3)
        # len(cp_points)=5 cp_points[0].shape=(N,6)，两个投影，一次占三个维度id,x,y,id,x,y
        cp_points = []
        range_dist = []  # len()=5, range_dist[0].shape(N,)
        intensity = []  # len()=5, intensity[0].shape(N,)
        elongation = []  # len()=5, elongation[0].shape(N,) 伸长率
        mask_indices = []  # len()=5, mask_indices[0].shape(N,2) 表示N点在range image的第几行第几列
        # 给出vehicle在global frame的位姿，Pglobal = T x Pvehicle
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
            range_image_top_pose_tensor_translation)    # TensorShape([64, 2650, 4, 4]) 每个像素下的top lidar的位姿
        for c in calibrations:  # 遍历5个雷达的坐标变换
            # range_image.shape.dims = [64, 2650, 4]
            range_image = range_images[c.name][ri_index]
            if len(c.beam_inclinations) == 0:  # 为什么不是第一个lidar，就会这样？需要计算这个beam_inclinations
                beam_inclinations = range_image_utils.compute_inclination(
                    tf.constant(
                        [c.beam_inclination_min, c.beam_inclination_max]),
                    height=range_image.shape.dims[0])
            else:
                # TensorShape([64]),64线激光，如果不是top lidar 则需要计算得到
                beam_inclinations = tf.constant(c.beam_inclinations)

            beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
            extrinsic = np.reshape(np.array(c.extrinsic.transform), [
                                   4, 4])  # 得到雷达的外参T，P_vehicle = T x P_lidar

            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(value=range_image.data),
                range_image.shape.dims)
            pixel_pose_local = None
            frame_pose_local = None
            if c.name == dataset_pb2.LaserName.TOP:  # 如果是top雷达则需要计算，如果不是就不需要这两个参数
                # TensorShape([64, 2650, 4, 4])
                pixel_pose_local = range_image_top_pose_tensor
                # TensorShape([1, 1, 64, 2650, 4, 4])
                pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
                frame_pose_local = tf.expand_dims(
                    frame_pose, axis=0)  # TensorShape([1, 4, 4]) 车的位姿
            # 真值mask，TensorShape([64, 2650]),深度大于0
            range_image_mask = range_image_tensor[..., 0] > 0

            if self.filter_no_label_zone_points:  # 过滤掉没有label的地方
                nlz_mask = range_image_tensor[..., 3] != 1.0  # 1.0: in NLZ
                range_image_mask = range_image_mask & nlz_mask
            # [B, H, W, 3] with {x, y, z} as inner dims in vehicle 坐标系 TensorShape([1, 64, 2650, 3])
            range_image_cartesian = \
                range_image_utils.extract_point_cloud_from_range_image(
                    tf.expand_dims(range_image_tensor[..., 0], axis=0),
                    tf.expand_dims(extrinsic, axis=0),
                    tf.expand_dims(tf.convert_to_tensor(
                        value=beam_inclinations), axis=0),
                    pixel_pose=pixel_pose_local,  # 仅top lidar有
                    frame_pose=frame_pose_local)  # 这个参数只有top lidar有

            mask_index = tf.where(range_image_mask)  # 生成索引mask来提取点

            range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
            # TensorShape([153830, 3]) 小于64*2650，因为过滤了一些点
            points_tensor = tf.gather_nd(range_image_cartesian, mask_index)

            cp = camera_projections[c.name][ri_index]  # 获取投影 [64, 2650, 6]
            cp_tensor = tf.reshape(
                tf.convert_to_tensor(value=cp.data), cp.shape.dims)  # TensorShape([64, 2650, 6])
            # 索引mask提取点，同上面的点云 TensorShape([153830, 6])
            cp_points_tensor = tf.gather_nd(cp_tensor, mask_index)
            points.append(points_tensor.numpy())
            cp_points.append(cp_points_tensor.numpy())

            range_tensor = tf.gather_nd(
                range_image_tensor[..., 0], mask_index)  # 提取深度信息，注意带mask
            range_dist.append(range_tensor.numpy())

            intensity_tensor = tf.gather_nd(range_image_tensor[..., 1],
                                            mask_index)
            intensity.append(intensity_tensor.numpy())

            elongation_tensor = tf.gather_nd(range_image_tensor[..., 2],
                                             mask_index)
            elongation.append(elongation_tensor.numpy())
            # 返回的mask索引值，这个索引值是在range image上的
            mask_indices.append(mask_index.numpy())

        return (points, cp_points, range_dist, intensity, elongation,
                mask_indices)

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

if __name__ == '__main__':
    in_dir = '/disk/deepdata/dataset/waymo_v1.4/'
    bucket = 'ai-waymo-v1.4'
    database = 'ai-waymo-v1_4'
    workers = 0  # 如何是0就是单进程
    splits = ['training', 'validation', 'testing']

    minio_cfg = dict(
        # endpoint='ossapi.cowarobot.cn:9000',
        # access_key='abcdef',
        # secret_key='12345678',
        # region='shjd-oss',
        # secure=False,
        )

    mongo_cfg = dict(
        # host="mongodb://root:root@172.16.110.100:27017/"
    )

    for i, split in enumerate(splits):
        test_mode = (split == 'testing')
        WaymoPrep(
            # indir='/disk/deepdata/dataset/waymo_v1.4/'
            load_dir=osp.join(in_dir, split),
            bucket=bucket,
            database=database,
            split=split,
            workers=workers,
            minio_cfg=minio_cfg,
            mongo_cfg=mongo_cfg,
            test_mode=test_mode,
            load_semseg=True).convert()
