import io
import time
import minio
import pymongo
from io import BytesIO
import numpy as np
import pickle
from tqdm import tqdm
import torch
import open3d as o3d
import matplotlib.pyplot as plt
from run_ccl import range_seg_ccl, get_in_2d_box_points, filter_ann
from mmdet3d.core.points import get_points_type, BasePoints
import numpy as np
from scipy.sparse.csgraph import connected_components  # CCL
import mmcv
import os
from deploy3d.symfun.ops.ccl import VoxelSPCCL3D, voxel_spccl3d

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

class GeneratePseudoLabel:
    def __init__(self,
                frame_infos_path,
                pts_trainval_pth,
                index_filename,
                output_dir,
                workers=0,
                training=True,
                ):

        self.frame_infos_path = frame_infos_path
        self.pts_trainval_pth = pts_trainval_pth
        self.index_filename = index_filename
        self.workers = workers
        self.training = training
        self.output_dir = output_dir
        self.index = np.array(self.load_index()).astype(np.int)
        self.coord_type = 'LIDAR'
        self.point_cloud_range = [-80, -80, -2, 80, 80, 4]
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.waymo_to_kitti_class_map = {
            'UNKNOWN': 'DontCare',
            'PEDESTRIAN': 'Pedestrian',
            'VEHICLE': 'Car',
            'CYCLIST': 'Cyclist',
            'SIGN': 'Sign'  # not in kitti
        }

    def load_index(self):
        f = open(self.index_filename, "rb+")
        ret = pickle.load(f)
        f.close()
        return ret

    def load_local_infos(self, info_path, index):
        info_path = os.path.join(info_path, str(index).zfill(7)+'.npy')
        while True:
            try:
                data_infos = np.load(info_path, allow_pickle=True)
            except:
                time.sleep(0.1)
                continue
            return data_infos[()]  # Dict

    def waymo_to_kitti_class(self, annos):
        for i, img_name in enumerate(annos['name']):
            map_name = [self.waymo_to_kitti_class_map[i] for i in img_name]
            annos['name'][i] = map_name
        pts_map_name = [self.waymo_to_kitti_class_map[i] for i in annos['name_3d']]
        annos['name_3d'] = pts_map_name
        return annos

    def generate(self):

        if self.workers == 0:
            result = []
            for idx in mmcv.track_iter_progress(self.index):
                result.append(self.spg_one(idx))
        else:
            result = mmcv.track_parallel_progress(self.spg_one, self.index, self.workers)
        
        print('Finished')
        print('\n上传地址:{}'.format(self.output_dir))

    def spg_one(self, index):

        torch.cuda.set_device(int(index % 4))

        # 1. get frame info, e.g., gt
        info = self.load_local_infos(self.frame_infos_path, index)

        # 2. load points
        pts_path = info['pts_info']['path']
        points = np.fromfile(pts_path, dtype=np.float32, count=-1).reshape([-1, 16])
        points = np.concatenate((points, np.zeros((points.shape[0], 2)).astype(points.dtype)), axis=1)  # why? N,16 --> N,18
        # have a bug need to fix
        # there is a problem with the order of the point cloud array
        points[:, 3] = np.tanh(points[:, 3])
        points[:, 5:8] = points[:, [6, 7, 5]]  # points.shape = (N, 16)
        points_class = get_points_type(self.coord_type)
        points = points_class(points, points_dim=points.shape[-1])
        out_mask = points.in_range_3d(self.point_cloud_range).numpy()  # Remove points that are out of range
        points = points.tensor.numpy()
        # get the top LiDAR points Note: only top LiDAR has the panseg label
        in_img_mask = (points[:, 10] != -1) | (points[:, 11] != -1)
        points[:, 16][in_img_mask] = 1  # if in image
        top_mask = points[:, 6] == 0
        in_mask = top_mask & out_mask
        points = points[in_mask]
        
        # 3. DCS (depth clustering segment)
        points = range_seg_ccl(points)  # points[:, 18] = ring segment id
        if False:
            import cv2
            image_id = 0
            image_path = info['images'][image_id]['path']
            image = cv2.imread(image_path)
            # image = np.zeros((1280, 1920, 3), dtype=np.uint8)
            # front image
            front_mask = points[:, 10] == image_id
            f_points = points[front_mask]
            f_rs = f_points[:, 18]
            rs_sets = np.unique(f_rs)
            for i in range(len(rs_sets)):
                rgb = np.array([np.random.randint(255) for _ in range(3)])
                p = f_points[f_points[:, 18] == rs_sets[i]]
                for pp in p:
                    x = int(pp[12])
                    y = int(pp[14])
                    cv2.circle(image, (x,y), 2, rgb.tolist(), -1)
            cv2.imwrite('rs_image.png', image)
        
        # 4. get the pseudo labels by the ring segment
        annos = self.waymo_to_kitti_class(info['annos'])
        gt_labels, gt_bboxes = filter_ann(info)
        points = self.get_in_2d_box_points(points, gt_bboxes, gt_labels, index)
        # points = self.get_in_2d_box_points(points, gt_bboxes, gt_labels)

        # ring segment id = points[:, 18], mask flag = points[:, 17], spg results = points[:, [19, 20]]
        # mask_flag = points[:, 17]
        # ring_segment_id = points[:, 18]
        # spg_label = points[:, [19, 20]]

        # 5. save .npy
        output_path = f'{self.output_dir}/'+ f'{str(index).zfill(7)}.npy'
        np.save(output_path, points[:,[17, 18, 19, 20]])

        return None

    def get_in_2d_box_points(self, points, gt_bboxes, gt_labels, index=None):
        assert len(gt_bboxes)==len(gt_labels)
        gt_mask_all = np.zeros((points.shape[0])).astype(np.bool)
        gt_mask_list = [[] for _ in range(len(gt_labels))]

        for i in range(len(gt_bboxes)):
            if len(gt_bboxes[i])==0:
                continue
            for j, gt_bbox in enumerate(gt_bboxes[i]):
                gt_mask1 = (((points[:, 12] >= gt_bbox[0]) & (points[:, 12] < gt_bbox[2])) &
                            ((points[:, 14] >= gt_bbox[1]) & (points[:, 14] < gt_bbox[3])  &
                             (points[:, 10] == i)))
                gt_mask2 = (((points[:, 13] >= gt_bbox[0]) & (points[:, 13] < gt_bbox[2])) &
                            ((points[:, 15] >= gt_bbox[1]) & (points[:, 13] < gt_bbox[3])  &
                             (points[:, 11] == i)))
                gt_mask = gt_mask1 | gt_mask2
                gt_mask_all = gt_mask_all | gt_mask
                gt_mask_list[i].append(gt_mask)
        points[:, 17][gt_mask_all] = 1  # Note: 这里只给落在图片上的备注了1，那些没有落在图片上的就忽略，一定记得points[:, 16]==0
        out_box_points = points[~gt_mask_all]

        if False:
            import cv2
            image_id = 2
            # image_path = info['images'][image_id]['path']
            # image = cv2.imread(image_path)
            image = np.zeros((1280, 1920, 3), dtype=np.uint8)
            # front image
            front_mask = points[:, 10] == image_id
            f_points = points[front_mask]
            for pp in f_points:
                x = int(pp[12])
                y = int(pp[14])
                if pp[17] == 0:
                    rgb=[255,0,0]
                else:
                    rgb=[0,0,255]
                cv2.circle(image, (x,y), 2, rgb, -1)
            cv2.imwrite('rs_image.png', image)

        out_run_sets = np.unique(out_box_points[:, 18])
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
                    run_sets = np.unique(in_box_points[:, 18])
                    for run_id in run_sets:
                        if run_id in out_run_sets:
                            run_mask = points[:, 18]==run_id
                            out_box_mask = out_box_points[:, 18]==run_id
                            prop = out_box_mask.sum() / run_mask.sum()
                            if run_id == -1:
                                points[:, 17][(gt_mask) & (run_mask)] = -1
                            elif prop >= 0.5:
                                points[:, 17][(gt_mask) & (run_mask)] = 0
                            elif prop >= 0.05 and prop < 0.5:
                                points[:, 17][(gt_mask) & (run_mask)] = -1
                            else:
                                points[:, 17][(gt_mask) & (run_mask)] = 1
        if index is None:
            points = self.ccl(points, gt_bboxes, gt_labels)
        else:
            points = self.torch_ccl(points, gt_bboxes, gt_labels, index)

        return points

    def torch_ccl(self, points, gt_bboxes, gt_labels, index):
        self.voxel_size = [[0.15, 0.15, 6], [0.05, 0.05, 6], [0.1, 0.1, 6]]
        self.dist_size = [[0.6, 0.6, 0], [0.1, 0.1, 0], [0.4, 0.4, 0]]
        self.kernel_size_ = [[1, 9, 9], [1, 5, 5], [1, 9, 9]]  # [z, y, x]

        # to gpu
        device_id = int(index % 4)
        device = 'cuda:{}'.format(device_id)
        points = torch.tensor(points, device=device)
        points_index = torch.arange(0, len(points), device=device, dtype=torch.int32)
        box_flag = torch.zeros((points.shape[0], 3), device=device, dtype=torch.int32)

        for j in range(5):
            if len(gt_bboxes[j])==0:
                continue
            for b, gt_bbox in enumerate(gt_bboxes[j]):
                gt_mask1 = (((points[:, 12] >= gt_bbox[0]) & (points[:, 12] < gt_bbox[2])) &
                            ((points[:, 14] >= gt_bbox[1]) & (points[:, 14] < gt_bbox[3])  &
                            (points[:, 10] == j)))
                gt_mask2 = (((points[:, 13] >= gt_bbox[0]) & (points[:, 13] < gt_bbox[2])) &
                            ((points[:, 15] >= gt_bbox[1]) & (points[:, 13] < gt_bbox[3])  &
                            (points[:, 11] == j)))
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
                xyz = points_padding(xyz, num_act_in, 0).contiguous()
                class_id = points_padding(class_id, num_act_in, -1)
                batch_id = points_padding(batch_id, num_act_in, -1)
                cluster_inds, valid_ind, num_valid, num_clusters = voxel_spccl3d(xyz, 
                                                    batch_id.type(torch.int32),
                                                    class_id.type(torch.int32),
                                                    nums.type(torch.int32),
                                                    self.kernel_size_[cls_id],
                                                    self.point_cloud_range,
                                                    [1, 1, 1],
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

    def ccl(self, points, gt_bboxes, gt_labels):

        points_index = np.arange(0, len(points), dtype=np.int32)
        box_flag = np.zeros((points.shape[0], 3), dtype=np.int32)

        for j in range(5):
            if len(gt_bboxes[j])==0:
                continue
            for b, gt_bbox in enumerate(gt_bboxes[j]):
                gt_mask1 = (((points[:, 12] >= gt_bbox[0]) & (points[:, 12] < gt_bbox[2])) &
                            ((points[:, 14] >= gt_bbox[1]) & (points[:, 14] < gt_bbox[3])  &
                            (points[:, 10] == j)))
                gt_mask2 = (((points[:, 13] >= gt_bbox[0]) & (points[:, 13] < gt_bbox[2])) &
                            ((points[:, 15] >= gt_bbox[1]) & (points[:, 13] < gt_bbox[3])  &
                            (points[:, 11] == j)))
                gt_mask = gt_mask1 | gt_mask2
                in_box_mask = gt_mask & (points[:, 17] > 0)
                if in_box_mask.sum() == 0:
                    continue

                cls_id = gt_labels[j][b]
                xyz = points[in_box_mask][:, 0:3].contiguous()
                cluster_inds = self.find_connected_componets_single_batch(xyz.cpu().numpy(), self.dist[cls_id])
                cluster_sets, cluster_invs, cluster_counts = torch.unique(cluster_inds, return_inverse=True, return_counts=True)

                c_max_inds = cluster_counts.argmax()
                c_mask = cluster_invs==cluster_sets[c_max_inds]
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

if __name__=='__main__':
    torch.multiprocessing.set_start_method('spawn')
    workers = 10
    frame_infos_path = "data/waymo/kitti_format/training/frame_infos"
    pts_trainval_pth = "data/waymo/kitti_format/training/velodyne"
    # index_filename = "data/waymo/kitti_format/training/pts_panseg_index_train.pkl"
    index_filename = "data/waymo/kitti_format/training/pts_panseg_index_val.pkl"
    output_dir = "data/waymo/kitti_format/training/pseudo_label"

    GeneratePseudoLabel(
        frame_infos_path=frame_infos_path,
        pts_trainval_pth=pts_trainval_pth,
        index_filename=index_filename,
        workers=workers,
        output_dir=output_dir,
    ).generate()