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

def draw_pts(points, color=None, ind=0):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    xyz = points[:, :3].astype(np.float64)
    clr = plt.get_cmap('gist_rainbow')(points[:, 4])[:, :3]
    if color is not None:
        clr[:, [0]], clr[:, [1]], clr[:, [2]] = color
    points = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(xyz))
    points.colors = o3d.utility.Vector3dVector(clr)
    # vis.clear_geometries()
    vis.add_geometry(points, ind==0)  # 必须带ind==0才可以保证每次更新你的放缩操作不改变

mongo_client = pymongo.MongoClient(**mongo_cfg)[database]
minio_client = minio.Minio(**minio_cfg)

ret = query_index(mongo_cfg, database)
ret_ = np.array(ret)
# 选择第几个record
record = 2
ret_mask = (1000*record<=ret_)&(ret_<(record+1)*1000)
sample_ret = ret_[ret_mask].tolist()

label_list = []
points_list = []
in_box_points_list = []

info_1 = mongo_client.get_collection(collection).find_one(sample_ret[0])
ego_pose_1 = info_1['ego_pose']

f = open('points_x200.pkl', 'rb+')
bg_points = pickle.load(f)
f.close()

bg_points = torch.tensor(bg_points, device='cuda')
voxel_size = [0.25, 0.25, 0.2]
pts_range_min = [-200, -200, -50]
pts_voxel_num = [25000000, 500000, 1]
quant_bg_points = (
        ((bg_points[:, :3] - bg_points.new_tensor(pts_range_min)) /
         bg_points.new_tensor(voxel_size)).long() *
        bg_points.new_tensor(pts_voxel_num, dtype=torch.long)).sum(-1)
# for i in tqdm(range(len(ret))):
ind = 0
def next_pcd(vis: o3d.visualization.Visualizer):
    global ind
    info = mongo_client.get_collection(collection).find_one(sample_ret[ind])
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
    rel_pose = np.linalg.inv(ego_pose_1) @ tmp_ego_pose
    points[:,:3] = points[:,:3] @ rel_pose[:3,:3].T
    points[:,:3] += rel_pose[:3,3][None, :]
    points = torch.tensor(points, device=quant_bg_points.device)
    quant_points = (
            ((points[:, :3] - points.new_tensor(pts_range_min)) /
            points.new_tensor(voxel_size)).long() *
            points.new_tensor(pts_voxel_num, dtype=torch.long)).sum(-1)

    torch.cuda.synchronize()
    s_time = time.time()

    table = HashSet(bg_points.shape[0] * 2)
    table.insert(quant_bg_points)
    # True is collision
    is_coll = table.lookUp(quant_points)
    coll_points = points[is_coll]

    vis.clear_geometries()
    draw_pts(coll_points, color=plt.get_cmap('tab10')(0)[:3], ind=ind)
    draw_pts(points[~is_coll], color=plt.get_cmap('tab10')(1)[:3], ind=ind)

    ind += 1


vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(width=1080, height=720)
op = vis.get_render_option()
# op.background_color = np.array([1., 1., 1.])
op.background_color = np.array([0., 0., 0.])
op.point_size = 1.0
vis.register_key_callback(ord(' '), next_pcd)
vis.run()
vis.destroy_window()