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
from mmdet3d.core.points import get_points_type, BasePoints
import numpy as np
from scipy.sparse.csgraph import connected_components  # CCL
import mmcv
import os
from run_ccl import filter_ann
from deploy3d.symfun.ops.ccl import VoxelSPCCL3D, voxel_spccl3d
from pycocotools import mask as mask_utils
from mmdet.core import BitmapMasks, PolygonMasks
from scipy.sparse.csgraph import connected_components  # CCL
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class UploadLabel:
    def __init__(self,
                bucket,  # minio
                new_bucket,
                database,# mongodb
                minio_cfg,
                mongo_cfg,
                workers=4,
                collection='training/infos',
                prefix='training'
                ):
        self.bucket = bucket
        self.new_bucket = new_bucket
        self.database = database
        self.minio_cfg = minio_cfg
        self.mongo_cfg = mongo_cfg
        self.workers = workers
        self.collection = collection
        self.points_idx = self.query_index()
        self.prefix = prefix
        self.dist=(0.6,0.1,0.4)

    def upload(self):

        new_client = minio.Minio(**self.minio_cfg)
        found = new_client.bucket_exists(self.new_bucket)
        if not found:
            new_client.make_bucket(self.new_bucket)

        if self.workers == 0:
            result = []
            for idx in mmcv.track_iter_progress(range(len(self.points_idx))):
                result.append(self.upload_one(idx))
        else:
            result = mmcv.track_parallel_progress(self.upload_one,
                                                  range(len(self.points_idx)),
                                                  self.workers)
        print('上传完成')
        print('\n上传地址:{}'.format(self.new_bucket))

    def upload_one(self, ind):
        # load points
        # filter points, 超出边界， 不在图片内，top_lidar
        mongo_client = pymongo.MongoClient(**self.mongo_cfg)[self.database]
        minio_client = minio.Minio(**self.minio_cfg)
        info = mongo_client.get_collection(self.collection).find_one(self.points_idx[ind])
        pts_path = info['pts_info']['path']
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
        # 过滤不在图片内的点和非top lidar
        points = points.tensor.numpy()
        top_lidar = points[:,6] == 0
        in_img_mask = (points[:,10]!=-1) | (points[:,11]!=-1)
        points = points[in_img_mask&top_lidar&points_mask]

        # 读取2d框
        # lwsis
        sample_idx = info['sample_idx']
        # gt_bboxes, gt_labels = self.load_labels(sample_idx)
        # 正常 还需要映射一下
        gt_labels, gt_bboxes = filter_ann(info)


        # load run id
        pts_path = info['pts_info']['path']
        pts_bytes = minio_client.get_object(self.new_bucket, pts_path).read()
        pseudo_lables = np.load(BytesIO(pts_bytes))
        run_id = pseudo_lables['run_id']
        assert len(run_id) == len(points)
        points = np.concatenate((points, run_id[:,None]), axis=1)
        # filter points and ccl
        points = self.get_in_2d_box_points(points, gt_bboxes, gt_labels)

        points_box_idx = points[:, -4:].astype(np.float32)
        pts_path = pts_path.split('/')
        pts_path = '{}/{}/{}'.format(pts_path[0],'all_ccl',pts_path[2])
        while True:
            try:
                buf = io.BytesIO()
                np.save(buf, points_box_idx)
                buf.seek(0, 2)
                buf_size = buf.tell()
                buf.seek(0, 0)
                minio_client.put_object(self.new_bucket, pts_path, buf, buf_size)
            except Exception as e:
                print(e)
                time.sleep(0.1)
                continue
            break
        return ind

    def query_index(self, filter_sem=None, test_model=False):
        mongo_client = pymongo.MongoClient(**self.mongo_cfg)[self.database]
        while True:
            try:
                if filter_sem is None:
                    ret = [o['_id'] for o in mongo_client.get_collection(self.collection).find(dict(), projection=[])]
                elif filter_sem == 'panseg_info':
                    is_local = True
                    if test_model:
                        import pickle
                        if os.path.exists('/root/3D/work_dirs/dataset_infos/validation_infos_panseg.pkl'):
                            f = open('/root/3D/work_dirs/dataset_infos/validation_infos_panseg.pkl', "rb+")
                        elif os.path.exists('/home/jiangguangfeng/桌面/codebase/validation_infos_panseg.pkl'):
                            f = open('/home/jiangguangfeng/桌面/codebase/validation_infos_panseg.pkl', "rb+")
                    else:
                        import pickle
                        if os.path.exists('/root/3D/work_dirs/dataset_infos/training_infos_panseg.pkl'):
                            f = open('/root/3D/work_dirs/dataset_infos/training_infos_panseg.pkl', "rb+")
                        elif os.path.exists('/home/jiangguangfeng/桌面/codebase/training_infos_panseg.pkl'):
                            f = open('/home/jiangguangfeng/桌面/codebase/training_infos_panseg.pkl', "rb+")
                    if is_local:
                        ret = pickle.load(f)
                        f.close()
                    print("--------load {} dataset index------".format(filter_sem))
                else:# 'semseg_info'
                    is_local = True
                    # ret = [o['_id'] for o in self.get_collection(collection).find() if filter_sem in o.keys()]
                    if test_model:
                        import pickle
                        if os.path.exists('/root/3D/work_dirs/dataset_infos/validation_infos_semseg.pkl'):
                            f = open('/root/3D/work_dirs/dataset_infos/validation_infos_semseg.pkl', "rb+")
                        elif os.path.exists('/home/jiangguangfeng/桌面/codebase/validation_infos_semseg.pkl'):
                            f = open('/home/jiangguangfeng/桌面/codebase/validation_infos_semseg.pkl', "rb+")
                    else:
                        import pickle
                        if os.path.exists('/root/3D/work_dirs/dataset_infos/training_infos_semseg.pkl'):
                            f = open('/root/3D/work_dirs/dataset_infos/training_infos_semseg.pkl', "rb+")
                        elif os.path.exists('/home/jiangguangfeng/桌面/codebase/training_infos_semseg.pkl'):
                            f = open('/home/jiangguangfeng/桌面/codebase/training_infos_semseg.pkl', "rb+")
                    if is_local:
                        ret = pickle.load(f)
                        f.close()
                    print("--------load {} dataset index------".format(filter_sem))
            except Exception as e:
                print(e)
                time.sleep(0.1)
                continue
            return ret # sorted(random.sample(ret, 500)) ret[881:883] 4900 行人 车3262 [16090,24139] 周视图bug[751177,50138,693079,267021] [500::100]

    def get_in_2d_box_points(self, points, gt_bboxes, labels):
        n, c = points.shape
        if c < 19:
            points = np.concatenate((points, np.zeros((n, 19-c))), axis=1)
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

        out_run_sets = np.unique(out_box_points[:, 16])
        for i in range(len(gt_bboxes)):
            if len(gt_bboxes[i])==0:
                continue
            for j, gt_bbox in enumerate(gt_bboxes[i]):
                gt_mask = gt_mask_list[i][j]
                if gt_mask.sum() == 0:
                    continue
                in_box_points = points[gt_mask]
                if True:
                    # t3 = time.time()
                    run_sets = np.unique(in_box_points[:, 16])
                    for run_id in run_sets:
                        if run_id in out_run_sets:
                            run_mask = points[:, 16]==run_id
                            out_box_mask = out_box_points[:, 16]==run_id
                            prop = out_box_mask.sum() / run_mask.sum()
                            if labels[i][j] == 1:
                                if run_id == -1:
                                    points[:, 17][(gt_mask) & (run_mask)] = -1
                                elif prop >= 0.9:
                                    points[:, 17][(gt_mask) & (run_mask)] = 0
                                # elif prop >= 0.05 and prop < 0.9:
                                #     points[:, 17][(gt_mask) & (run_mask)] = -1
                                else:
                                    points[:, 17][(gt_mask) & (run_mask)] = 1
                            else:
                                if run_id == -1:
                                    points[:, 17][(gt_mask) & (run_mask)] = -1
                                elif prop >= 0.5:
                                    points[:, 17][(gt_mask) & (run_mask)] = 0
                                elif prop >= 0.05 and prop < 0.5:
                                    points[:, 17][(gt_mask) & (run_mask)] = -1
                                else:
                                    points[:, 17][(gt_mask) & (run_mask)] = 1

        points = self.torch_ccl(points, gt_bboxes, labels)

        return points

    def torch_ccl(self, points, gt_bboxes, gt_labels):
        self.voxel_size = [[0.15, 0.15, 6], [0.05, 0.05, 6], [0.1, 0.1, 6]]
        self.dist_size = [[0.6, 0.6, 0], [0.1, 0.1, 0], [0.4, 0.4, 0]]
        self.kernel_size_ = [[1, 9, 9], [1, 5, 5], [1, 9, 9]]  # [z, y, x]
        self.point_cloud_range = [-80, -80, -2, 80, 80, 4]
        device = torch.cuda.current_device()
        points = torch.tensor(points, device=device)
        points_index = torch.arange(0, len(points), device=device, dtype=torch.int32)
        box_flag = torch.zeros((points.shape[0], 3), device=device, dtype=torch.int32)
        box_flag2 = torch.zeros((points.shape[0], 3), device=device, dtype=torch.int32)
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
                cls_id = int(gt_labels[j][b])
                class_id = torch.zeros((in_box_mask.sum().long()), device=device).long()
                xyz = points[in_box_mask][:, 0:3].contiguous()
                batch_id = torch.zeros((xyz.shape[0]), device=device, dtype=torch.int32)
                nums = in_box_mask.sum()
                spatial_shape = self.gen_shape(self.point_cloud_range, self.voxel_size[cls_id])
                if nums < 100:
                    num_act_in = 100
                elif nums < 1000:
                    num_act_in = int((nums//100)*100+100)
                elif nums < 10000:
                    num_act_in = int((nums//1000)*1000+1000)
                else:
                    num_act_in = int((nums//10000)*10000+10000)

                # c_inds = self.find_connected_componets_single_batch(xyz.cpu().numpy(), self.dist[cls_id])
                # scipy_sets, invs, counts = np.unique(c_inds, return_inverse=True, return_counts=True)
                # nums__ = (invs == scipy_sets[counts.argmax()]).sum()

                xyz = self.points_padding(xyz, num_act_in, 0).contiguous()
                class_id = self.points_padding(class_id, num_act_in, -1)
                batch_id = self.points_padding(batch_id, num_act_in, -1)
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
                #         '\n spccl: 聚类簇数量{},最大簇点数{}'.format(len(cluster_sets), c_mask.sum()))

                c_mask_ = box_flag[:, 2][in_box_mask] == 0
                # 标准
                max_in_box_index = points_index[in_box_mask][c_mask & c_mask_].long()
                box_flag[:, 2][max_in_box_index] = 1
                box_flag[:, 0][max_in_box_index] = j*1000+b+1
                max_in_box_index_2 = points_index[in_box_mask][c_mask & (~c_mask_)].long()
                box_flag[:, 1][max_in_box_index_2] = j*1000+b+1 
                
                # 分配的
                if cls_id == 1:
                    max_in_box_index = points_index[in_box_mask][c_mask].long()
                    box_flag2[:, 2][max_in_box_index] = 1
                    box_flag2[:, 0][max_in_box_index] = j*1000+b+1
                else:
                    c_mask_ = box_flag2[:, 2][in_box_mask] == 0
                    max_in_box_index = points_index[in_box_mask][c_mask & c_mask_].long()
                    box_flag2[:, 2][max_in_box_index] = 1
                    box_flag2[:, 0][max_in_box_index] = j*1000+b+1
                    max_in_box_index_2 = points_index[in_box_mask][c_mask & (~c_mask_)].long()
                    box_flag2[:, 1][max_in_box_index_2] = j*1000+b+1

        box_flag[:, 0:2] = box_flag[:, 0:2] - 1
        fg_mask = (box_flag[:, 0] != -1) | (box_flag[:, 1] != -1)
        bg_mask = ~(fg_mask | (points[:, 17]==-1))
        points[:, 17][bg_mask] = 0

        box_flag2[:, 0:2] = box_flag2[:, 0:2] - 1
        
        points = torch.cat((points, box_flag[:, 0:2], box_flag2[:, 0:2]), dim=1)
        return points.cpu().numpy()

    @staticmethod    
    def gen_shape(pc_range, voxel_size):
        voxel_size = np.array(voxel_size).reshape(-1, 3)
        ncls = len(voxel_size)
        spatial_shape = []
        for i in range(ncls):
            spatial_shape.append([(pc_range[3]-pc_range[0])/voxel_size[i][0],
                                (pc_range[4]-pc_range[1])/voxel_size[i][1],
                                (pc_range[5]-pc_range[2])/voxel_size[i][2]])
        return np.array(spatial_shape).astype(np.int32).reshape(-1).tolist()

    @staticmethod 
    def points_padding(x, num_out, padding_nb):
        padding_shape = (num_out, ) + tuple(x.shape[1:])
        x_padding = x.new_ones(padding_shape) * (padding_nb)
        x_padding[:x.shape[0]] = x
        return x_padding

    def load_labels(self, sample_idx):
        img_path_prefix = '{}/{}'.format(self.prefix, 'img') #'training/img'
        self.file_client = minio.Minio(**self.minio_cfg)
        gt_bboxes = []
        gt_labels = []
        gt_masks = []
        num_img = 5
        for i in range(num_img):
            if num_img == 5:
                path = "{}/{}".format(img_path_prefix, str(sample_idx*10+i))
            else:
                path = "{}/{}".format(img_path_prefix, str(sample_idx))
            if self.exists(self.file_client, path):
                label_bytes = self.file_client.get_object(self.new_bucket, path).read()
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
                    # gt_masks.append(img_mask)
                else:
                    gt_bboxes.append(np.array([]).astype(np.float32))
                    gt_labels.append(np.array([]).astype(np.float32))
            else:
                gt_bboxes.append(np.array([]).astype(np.float32))
                gt_labels.append(np.array([]).astype(np.float32))
                # gt_masks.append(BitmapMasks(np.array([]), h, w))
        if len(gt_bboxes) == 0:
            return None

        return gt_bboxes, gt_labels


    def exists(self, file_client, path):
        try:
            file_client.stat_object(self.new_bucket, path)
            return True
        except:
            return False

    def find_connected_componets_single_batch(self, points, dist):

        this_points = points
        dist_mat = this_points[:, None, :2] - this_points[None, :, :2] # only care about xy
        dist_mat = (dist_mat ** 2).sum(2) ** 0.5
        adj_mat = dist_mat < dist
        adj_mat = adj_mat
        c_inds = connected_components(adj_mat, directed=False)[1]

        return c_inds
if __name__=='__main__':
    torch.multiprocessing.set_start_method('spawn')
    bucket = 'ai-waymo-v1.4'
    new_bucket = 'ai-waymo-v1.4-pts-pseudo-labels'
    database = 'ai-waymo-v1_4'
    workers = 0
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