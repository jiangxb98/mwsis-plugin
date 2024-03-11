import torch
from deploy3d.symfun.ops.ccl import VoxelSPCCL3D, voxel_spccl3d
import numpy as np

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

num_pts = 250000

pc_range = [-50.0, -50.0, -2.0, 50.0, 50.0, 4.0]
voxel_size = [0.5, 0.5, 0.5,  # unlabeled
                1.0, 1.0, 6.0,  # road_plane
                0.5, 0.5, 6.0,  # curb
                1.0, 1.0, 6.0,  # other_ground
                0.5, 0.5, 3.0,  # terrain
                0.2, 0.2, 1.0,  # vegetation
                0.1, 0.1, 1.0,  # pillars
                0.5, 0.5, 6.0,  # framework
                0.5, 0.5, 2.0,  # building
                0.2, 0.2, 0.5,  # fence
                0.2, 0.2, 1.0,  # traffic_sign
                0.5, 0.5, 1.0,  # other_structure
                0.5, 0.5, 0.5,  # noise
                0.5, 0.5, 6.0,  # road_users
                0.2, 0.2, 6.0]  # road_block

min_points = 1
kernel_size = [5, 5, 5]  # [z, y, x]
spatial_shape = gen_shape(pc_range, voxel_size)
ncls = 1

dist_size = [1.0, 1.0, 1.0,
                2.0, 2.0, 0,
                1.0, 1.0, 0,
                2.0, 2.0, 0,
                1.0, 1.0, 0,
                0.4, 0.4, 2.0,
                0.2, 0.2, 2.0,
                1.0, 1.0, 0,
                1.0, 1.0, 4.0,
                0.4, 0.4, 1.0,
                0.4, 0.4, 2.0,
                1, 1, 2,
                1, 1, 1,
                1, 1, 0,
                0.4, 0.4, 0]

device = "cuda:0"
type = torch.float32
x = torch.randn((num_pts, 1), device=device, dtype=type) * 30
y = torch.randn((num_pts, 1), device=device, dtype=type) * 30
z = torch.randn((num_pts, 1), device=device, dtype=type) * 2 + 1
other = torch.randn((num_pts, 1), device=device, dtype=type)  # 2
points = torch.cat([x, y, z, other], dim=-1)
class_id = torch.randint(
    0, ncls, (num_pts,), device=device, dtype=torch.int32)
batch_id = torch.randint(
    0, 2, (num_pts,), device=device, dtype=torch.int32)
numPts = class_id.new_ones(1) * points.shape[0]

num_act_in = 300000
points = points_padding(points, num_act_in, 0)
class_id = points_padding(class_id, num_act_in, -1)
batch_id = points_padding(batch_id, num_act_in, -1)

cluster_inds11, valid_ind11, num_valid11, num_clusters11 = voxel_spccl3d(points, batch_id, class_id, numPts,
                                                                    kernel_size, pc_range, voxel_size,
                                                                    dist_size, spatial_shape, ncls, min_points)
# valid_ind11 = valid_ind11[0: num_valid11]
# cluster_inds11[valid_ind11]
device