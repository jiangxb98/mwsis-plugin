import io
import time
import minio
import pymongo
from io import BytesIO
import numpy as np
import pickle
from tqdm import tqdm
import torch
from deploy3d.libs.libdeploy3d_torch import HashMap, HashSet
import open3d as o3d
import matplotlib.pyplot as plt
from run_ccl import range_seg_ccl, get_in_2d_box_points, filter_ann
from mmdet3d.core.points import get_points_type, BasePoints
import numpy as np
from scipy.sparse.csgraph import connected_components  # CCL
import mmcv

class UploadLabel:
    def __init__(self,
                bucket,  # minio
                new_bucket,
                database,# mongodb
                minio_cfg,
                mongo_cfg,
                workers=4,
                collection='training/infos'
                ):
        self.bucket = bucket
        self.new_bucket = new_bucket
        self.database = database
        self.minio_cfg = minio_cfg
        self.mongo_cfg = mongo_cfg
        self.workers = workers
        self.collection = collection
        self.points_idx = np.array(self.query_index()).astype(np.int)
        self.records = np.unique((self.points_idx//1000).astype(np.int))[18:]

        # 量化参数
        self.voxel_size = [0.2, 0.2, 0.1]
        self.pts_range_min = [-200, -200, -50]
        self.pts_voxel_num = [5000000000, 500000, 1]

    def upload(self):

        new_client = minio.Minio(**self.minio_cfg)
        found = new_client.bucket_exists(self.new_bucket)
        if not found:
            new_client.make_bucket(self.new_bucket)

        if self.workers == 0:
            result = []
            for idx in mmcv.track_iter_progress(range(len(self.records))):
                result.append(self.upload_one(idx))
        else:
            result = mmcv.track_parallel_progress(self.upload_one,
                                                  range(len(self.records)),
                                                  self.workers)
        print('上传完成')
        print('\n上传地址:{}'.format(self.new_bucket))

    def upload_one(self, record_num):
        record_num = self.records[record_num]
        bg_points, fg_points = self.get_record_points(record_num)
        print("完成获取_{}_Record的背景点和前景点".format(record_num))
        print('GPU ID:{}'.format(int(record_num%4)))
        torch.cuda.set_device(int(record_num%4))
        # 碰撞检测
        results = self.collision_filter(bg_points, record_num)
        return results

    def collision_filter(self, bg_points, record_num):
        mongo_client = pymongo.MongoClient(**self.mongo_cfg)[self.database]
        minio_client = minio.Minio(**self.minio_cfg)

        ret_mask = (1000*record_num<=self.points_idx)&(self.points_idx<(record_num+1)*1000)
        sample_ret = self.points_idx[ret_mask].tolist()
        info_1 = mongo_client.get_collection(self.collection).find_one(sample_ret[0])
        ego_pose_1 = info_1['ego_pose']

        if not isinstance(bg_points, torch.Tensor):
            device_id = int(record_num%4)
            bg_points = torch.tensor(bg_points, device='cuda:{}'.format(device_id))
        quant_bg_points = (
                ((bg_points[:, :3] - bg_points.new_tensor(self.pts_range_min)) /
                bg_points.new_tensor(self.voxel_size)).long() *
                bg_points.new_tensor(self.pts_voxel_num, dtype=torch.long)).sum(-1)
        for ind in sample_ret:
            info = mongo_client.get_collection(self.collection).find_one(ind)
            tmp_ego_pose = info['ego_pose']
            pts_path = info['pts_info']['path']
            # load points
            pts_bytes = minio_client.get_object(self.bucket, pts_path).read()
            points_ = np.load(BytesIO(pts_bytes))
            points = np.stack([points_['x'].astype('f4'),
                                points_['y'].astype('f4'),
                                points_['z'].astype('f4'),
                                np.tanh(points_['intensity'].astype('f4')), # 3
                            #  points_['intensity'].astype('f4'), input lidar intensity
                                points_['elongation'].astype('f4'),  # 4
                                points_['return_idx'].astype('i2'),  # 5
                                points_['lidar_idx'].astype('i2'),   # 6
                                points_['range_dist'].astype('f4'),  # range distance 7
                                points_['lidar_row'].astype('i2'),   # range image 行 64 8
                                points_['lidar_column'].astype('i2'),# range image 列 2650 9
                                points_['cam_idx_0'].astype('i2'),   # 10
                                points_['cam_idx_1'].astype('i2'),   # 11
                                points_['cam_column_0'].astype('i2'),# 12
                                points_['cam_column_1'].astype('i2'),# 13
                                points_['cam_row_0'].astype('i2'),   # 14
                                points_['cam_row_1'].astype('i2'),   # 15
                                ], axis=-1)
            # 过滤超出范围的点
            points_class = get_points_type('LIDAR')
            points = points_class(points, points_dim=points.shape[-1])  # 实例化，LiDARPoints
            points_mask = points.in_range_3d([-80, -80, -2, 80, 80, 4]).numpy()
            # 过滤不在图片内的点
            points = points.tensor.numpy()
            top_lidar = points[:,6] == 0
            in_img_mask = (points[:,10]!=-1) | (points[:,11]!=-1)
            points = points[in_img_mask&top_lidar&points_mask]

            # run + ccl过滤
            gt_labels, gt_bboxes = filter_ann(info)
            points = np.concatenate((points, np.zeros((points.shape[0],2)).astype(points.dtype)),axis=1)
            points = range_seg_ccl(points)  # points[:,18] = run_id
            # range + ccl
            points = get_in_2d_box_points(points, gt_bboxes, gt_labels)
            # fg points
            fg_mask = (points[:,20]!=-1) | (points[:,19]!=-1)
            run_mask = points[:,17]==1
            ignore_mask = points[:,17]==-1
            run_id = points[:,18]

            # 坐标转换
            if ego_pose_1 == tmp_ego_pose:
                points = torch.tensor(points, device=quant_bg_points.device)
            else:
                rel_pose = np.linalg.inv(ego_pose_1) @ tmp_ego_pose
                points[:,:3] = points[:,:3] @ rel_pose[:3,:3].T
                points[:,:3] += rel_pose[:3,3][None, :]
                points = torch.tensor(points, device=quant_bg_points.device)

            table = HashSet(bg_points.shape[0] * 2)
            table.insert(quant_bg_points)

            # 存在量化误差，查询
            coll_list = []
            quant_stride = [-1,0,1]
            for i in range(3): # x
                for j in range(3): # y
                    for k in range(3): # z
                        tmp_points = torch.zeros((len(points),3), device=quant_bg_points.device)
                        tmp_points[:,i] = points[:,i] + quant_stride[i]*self.voxel_size[0]
                        tmp_points[:,j] = points[:,j] + quant_stride[j]*self.voxel_size[1]
                        tmp_points[:,k] = points[:,k] + quant_stride[k]*self.voxel_size[2]
                        # points_list.append(tmp_points)
                        quant_points = (
                                ((tmp_points[:, :3] - tmp_points.new_tensor(self.pts_range_min)) /
                                tmp_points.new_tensor(self.voxel_size)).long() *
                                tmp_points.new_tensor(self.pts_voxel_num, dtype=torch.long)).sum(-1)

                        # True is collision
                        is_coll = table.lookUp(quant_points)
                        coll_list.append(is_coll)

            coll_list
            coll_mask = torch.stack(coll_list).T
            coll_vote = coll_mask.sum(1)
            is_coll = coll_vote > 0

            assert len(is_coll) == len(fg_mask) == len(ignore_mask)
            mask_dtype = [('run_id','i4'),('run','i2'), ('ignore','i2'), ('run_ccl','i2'), ('collision','i2'),]
            mask = np.empty(len(points), dtype=mask_dtype)
            mask['run_id'] = run_id.astype('i4')
            mask['run'] = run_mask.astype('i2')        # points[:,17]==1
            mask['run_ccl'] = fg_mask.astype('i2')     # (points[:,20]!=-1) | (points[:,19]!=-1)
            mask['ignore'] = ignore_mask.astype('i2')  # points[:,17]==-1
            mask['collision'] = is_coll.cpu().numpy().astype('i2')

            while True:
                try:
                    buf = io.BytesIO()
                    np.save(buf, mask)
                    buf.seek(0, 2)
                    buf_size = buf.tell()
                    buf.seek(0, 0)
                    minio_client.put_object(self.new_bucket, pts_path, buf, buf_size)
                except Exception as e:
                    print(e)
                    time.sleep(0.1)
                    continue
                break       
        
        return record_num

    def get_record_points(self, record_num):
        mongo_client = pymongo.MongoClient(**self.mongo_cfg)[self.database]
        minio_client = minio.Minio(**self.minio_cfg)

        points_list = []
        in_box_points_list = []
        ret_mask = (1000*record_num<=self.points_idx)&(self.points_idx<(record_num+1)*1000)
        sample_ret = self.points_idx[ret_mask].tolist()
        info_1 = mongo_client.get_collection(self.collection).find_one(sample_ret[0])
        ego_pose_1 = info_1['ego_pose']

        for ind in sample_ret:
            info = mongo_client.get_collection(self.collection).find_one(ind)
            tmp_ego_pose = info['ego_pose']
            pts_path = info['pts_info']['path']
            # load points
            pts_bytes = minio_client.get_object(self.bucket, pts_path).read()
            points_ = np.load(BytesIO(pts_bytes))
            points = np.stack([points_['x'].astype('f4'),
                                points_['y'].astype('f4'),
                                points_['z'].astype('f4'),
                                np.tanh(points_['intensity'].astype('f4')), # 3
                            #  points_['intensity'].astype('f4'), input lidar intensity
                                points_['elongation'].astype('f4'),  # 4
                                points_['return_idx'].astype('i2'),  # 5
                                points_['lidar_idx'].astype('i2'),   # 6
                                points_['range_dist'].astype('f4'),  # range distance 7
                                points_['lidar_row'].astype('i2'),   # range image 行 64 8
                                points_['lidar_column'].astype('i2'),# range image 列 2650 9
                                points_['cam_idx_0'].astype('i2'),   # 10
                                points_['cam_idx_1'].astype('i2'),   # 11
                                points_['cam_column_0'].astype('i2'),# 12
                                points_['cam_column_1'].astype('i2'),# 13
                                points_['cam_row_0'].astype('i2'),   # 14
                                points_['cam_row_1'].astype('i2'),   # 15
                                ], axis=-1)
            # 过滤不在图片内的点
            top_lidar = points[:,5] == 0
            in_img_mask = (points[:,10]!=-1) | (points[:,11]!=-1)
            points = points[in_img_mask&top_lidar]

            # 坐标转换
            if (tmp_ego_pose == ego_pose_1):
                print('\nind:', ind)
            else:
                rel_pose = np.linalg.inv(ego_pose_1) @ tmp_ego_pose
                points[:,:3] = points[:,:3] @ rel_pose[:3,:3].T
                points[:,:3] += rel_pose[:3,3][None, :]
            # 这里的box包含car,ped,cyc,dontcare,sign
            bboxes = info['annos']['bbox']
            gt_mask_all = np.zeros((points.shape[0])).astype(np.bool)
            # 过滤掉2D Box内的点
            for i in range(len(bboxes)):
                if len(bboxes[i])==0:
                    continue
                dis = 0.05
                dis_h = 0.1
                for j, gt_bbox in enumerate(bboxes[i]):
                    w,h = gt_bbox[2]-gt_bbox[0], gt_bbox[3]-gt_bbox[1]
                    x1, y1 = gt_bbox[0]-dis*w, gt_bbox[1]-dis_h*h
                    x2, y2 =  gt_bbox[2]+dis*w, gt_bbox[3]+dis*h
                    gt_mask1 = (points[:, 12] >= x1) & (points[:, 12] < x2) & \
                                (points[:, 14] >= y1) & (points[:, 14] < y2)  & (points[:, 10]==i)
                    gt_mask2 = (((points[:, 13] >= x1) & (points[:, 13] < x2)) &
                                ((points[:, 15] >= y1) & (points[:, 15] < y2)  &
                                (points[:,11]==i)))
                    gt_mask = gt_mask1 | gt_mask2
                    gt_mask_all = gt_mask_all | gt_mask
            in_box_mask = gt_mask_all
            out_box_mask = ~gt_mask_all

            points_list.append(points[out_box_mask])
            in_box_points_list.append(points[in_box_mask])

        bg_points = np.concatenate(points_list)
        fg_points = np.concatenate(in_box_points_list)
        return bg_points, fg_points

    def query_index(self, filter=dict(), filter_sem=None):
        _client = pymongo.MongoClient(**self.mongo_cfg)[self.database]
        while True:
            try:
                if filter_sem is None:
                    ret = [o['_id'] for o in _client.get_collection(self.collection).find(filter, projection=[])]
                else:
                    ret = [o['_id'] for o in _client.get_collection(self.collection).find() if filter_sem in o.keys()]
            except Exception as e:
                print(e)
                time.sleep(0.1)
                continue
            return ret

if __name__=='__main__':
    torch.multiprocessing.set_start_method('spawn')
    bucket = 'ai-waymo-v1.4'
    new_bucket = 'ai-waymo-v1.4-pts-pseudo-labels'
    database = 'ai-waymo-v1_4'
    workers = 4
    collection = 'training/infos'
    minio_cfg=dict(endpoint='ossapi.cowarobot.cn:9000',
            access_key='abcdef',
            secret_key='12345678',
            region='shjd-oss',
            secure=False,)
    mongo_cfg=dict(host="mongodb://root:root@172.16.110.100:27017/")
    UploadLabel(
        bucket=bucket,
        new_bucket=new_bucket,
        database=database,
        minio_cfg=minio_cfg,
        mongo_cfg=mongo_cfg,
        workers=workers,
        collection=collection,
    ).upload()