import time
import minio
import pymongo
from io import BytesIO
import numpy as np
import pickle
from tqdm import tqdm

minio_cfg=dict(endpoint='ossapi.cowarobot.cn:9000',
                access_key='abcdef',
                secret_key='12345678',
                region='shjd-oss',
                secure=False,)
mongo_cfg=dict(host="mongodb://root:root@172.16.110.100:27017/")
bucket='ai-waymo-v1.4'
database='ai-waymo-v1_4'
collection='training/infos'

def query_index(mongo_cfg, database, collection='training/infos', filter=dict(), filter_sem=None):
    _client = pymongo.MongoClient(**mongo_cfg)[database]
    # client = minio.Minio(**self.minio_cfg)
    while True:
        try:
            if filter_sem is None:
                ret = [o['_id'] for o in _client.get_collection(collection).find(filter, projection=[])]
            else:
                ret = [o['_id'] for o in _client.get_collection(collection).find() if filter_sem in o.keys()]
        except Exception as e:
            print(e)
            time.sleep(0.1)
            continue
        return ret



mongo_client = pymongo.MongoClient(**mongo_cfg)[database]
minio_client = minio.Minio(**minio_cfg)

ret = query_index(mongo_cfg, database)
ret_ = np.array(ret)
# 选择第几个record
record = 2
ret_mask = (1000*record<=ret_)&(ret_<(record+1)*1000)
sample_ret = ret_[ret_mask].tolist()
points_list = []
in_box_points_list = []
info_1 = mongo_client.get_collection(collection).find_one(sample_ret[0])
ego_pose_1 = info_1['ego_pose']

for ind in tqdm(sample_ret):
    info = mongo_client.get_collection(collection).find_one(ind)
    tmp_ego_pose = info['ego_pose']
    pts_path = info['pts_info']['path']
    # load points
    pts_bytes = minio_client.get_object(bucket, pts_path).read()
    points_ = np.load(BytesIO(pts_bytes))
    points = np.stack([ points_['x'].astype('f4'),
                        points_['y'].astype('f4'),
                        points_['z'].astype('f4'),
                        np.tanh(points_['intensity'].astype('f4')), # 
                        points_['lidar_idx'].astype('i2'),   # 4
                        points_['cam_idx_0'].astype('i2'),   # 5
                        points_['cam_idx_1'].astype('i2'),   # 6
                        points_['cam_column_0'].astype('i2'),# 7
                        points_['cam_column_1'].astype('i2'),# 8
                        points_['cam_row_0'].astype('i2'),   # 9
                        points_['cam_row_1'].astype('i2'),   # 10
                        ], axis=-1)
    # 过滤不在图片内的点
    top_lidar = points[:,4] == 0
    in_img_mask = (points[:,5]!=-1) | (points[:,6]!=-1)
    points = points[in_img_mask&top_lidar]

    # 坐标转换
    if (tmp_ego_pose == ego_pose_1):
        print('ind:', ind)
    else:
        rel_pose = np.linalg.inv(ego_pose_1) @ tmp_ego_pose
        points[:,:3] = points[:,:3] @ rel_pose[:3,:3].T
        points[:,:3] += rel_pose[:3,3][None, :]
    bboxes = info['annos']['bbox']
    gt_mask_all = np.zeros((points.shape[0])).astype(np.bool)
    # 过滤掉2D Box内的点
    for i in range(len(bboxes)):
        if len(bboxes[i])==0:
            continue
        for j, gt_bbox in enumerate(bboxes[i]):
            gt_mask1 = (points[:, 7] >= gt_bbox[0]) & (points[:, 7] < gt_bbox[2]) & \
                        (points[:, 9] >= gt_bbox[1]) & (points[:, 9] < gt_bbox[3])  & (points[:, 5]==i)
            gt_mask2 = (((points[:, 8] >= gt_bbox[0]) & (points[:, 8] < gt_bbox[2])) &
                        ((points[:, 10] >= gt_bbox[1]) & (points[:, 10] < gt_bbox[3])  &
                        (points[:,6]==i)))
            gt_mask = gt_mask1 | gt_mask2
            gt_mask_all = gt_mask_all | gt_mask
    in_box_mask = gt_mask_all
    out_box_mask = ~gt_mask_all

    points_list.append(points[out_box_mask])
    in_box_points_list.append(points[in_box_mask])

points_list
in_box_points_list
out = np.concatenate(points_list)
f = open('points_x200.pkl', 'wb')
pickle.dump(out, f)
f.close()
out = np.concatenate(in_box_points_list)
f = open('points_fg_x200.pkl', 'wb')
pickle.dump(out, f)
f.close()