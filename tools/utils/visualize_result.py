import mmcv
import torch
import pickle
import argparse
import numpy as np
import open3d as o3d
import os.path as osp
import matplotlib.pyplot as plt
from pycocotools import mask as mask_utils
from mmcv import Config
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.apis import init_model
from mmdet3d.models import build_model
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmcv.image import tensor2imgs
import cv2

def plot_img(data, pred, flag, points=None):

    # get rgb 
    original_img = tensor2imgs(data['img'][0]._data.unsqueeze(0), mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],
                                to_rgb=False)[0]  # in RGB format. ori_image already rgb, no need to swap dim
    # to bgr
    original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    if flag == 0:
        boxes = np.concatenate(pred['img_bbox'])*0.5
        mask = boxes[:,4] > 0.1
        boxes = boxes[mask]
        for gt_bbox in boxes:
            cv2.rectangle(original_img, (int(gt_bbox[0]), int(gt_bbox[1])), (int(gt_bbox[2]), int(gt_bbox[3])), (0, 255, 0), 1)
        # if points is not None:
        #     for p, point in enumerate(points):
        #         x = point[12].astype(np.int)
        #         y = point[13].astype(np.int)
        #         if pred['segment3d'][p] == 3:
        #             cv2.circle(original_img,(x, y),1,(0,255,0),thickness=-1)
        cv2.imwrite('pred.jpeg', original_img)
        if mask.sum()==0:
            return None
        mask_2d = []
        for i in range(len(pred['img_mask'])):
            if len(pred['img_mask'][i])!=0:
                mask_2d.append(mask_utils.decode(pred['img_mask'][i]).transpose(2,0,1))
        if len(mask_2d) != 0:
            instance_mask = np.concatenate(mask_2d)[mask]
            original_mask = np.zeros((1280,1920,3))
            for i in range(len(instance_mask)):
                rgb = np.array([np.random.randint(255) for _ in range(3)])
                instance_mask_rgb = np.repeat(np.expand_dims(instance_mask[i],0),3,axis=0)
                instance_mask_rgb = instance_mask_rgb.transpose(1,2,0)
                original_mask = original_mask + instance_mask_rgb*rgb
            boxes = boxes * 2
            for gt_bbox in boxes:
                cv2.rectangle(original_mask, (int(gt_bbox[0]), int(gt_bbox[1])), (int(gt_bbox[2]), int(gt_bbox[3])), (0, 255, 0), 1)
            # if points is not None:
            #     for p, point in enumerate(points):
            #         x = (point[12]*2).astype(np.int)
            #         y = (point[13]*2).astype(np.int)
            #         if pred['segment3d'][p] == 3:
            #             cv2.circle(original_mask,(x, y),1,(255,255,255),thickness=-1,lineType=2)
            cv2.imwrite('pred_mask.jpeg', original_mask)
    elif flag == 1:
        boxes = data['gt_bboxes'][0]._data.numpy()
        for gt_bbox in boxes:
            cv2.rectangle(original_img, (int(gt_bbox[0]), int(gt_bbox[1])), (int(gt_bbox[2]), int(gt_bbox[3])), (0, 255, 0), 1)
        # if points is not None:
        #     for point in points:
        #         x = point[12].astype(np.int)
        #         y = point[13].astype(np.int)
        #         cv2.circle(original_img,(x, y),1,(0,255,0),thickness=-1)
        cv2.imwrite('label.jpeg', original_img)

def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--pth', help='train checkpoint path', default=None)
    parser.add_argument('--out', help='inference result', default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    dataset = build_dataset(cfg.data.test)

    model = None
    loader = None
    if args.pth:
        model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
        checkpoint = load_checkpoint(model, args.pth, map_location='cpu')
        # model = init_model(cfg, args.pth)
        model = revert_sync_batchnorm(model)
        model = MMDataParallel(model, device_ids=[0])
        model.eval()
        loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=0,
            dist=False,
            shuffle=False)

    out = None
    if args.out:
        out = mmcv.load(args.out)

    def renderbox(box3d, labels, color_amp=1.0):
        clr_map = plt.get_cmap('tab10').colors
        corners = box3d.corners
        if box3d.box_dim == 9:
            vels = box3d.tensor[:, -2:]
            vels_norm = vels.norm(dim=-1, p=2)
            vels_yaw = torch.atan2(vels[:, 1], vels[:, 0])
            vels_ctr = box3d.bottom_center

        cores = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
            (8, 4), (8, 5), (8, 6), (8, 7)
        ]
        ret = None
        vel_vectors = None
        for i, (corners_i, label_i) in enumerate(zip(corners, labels)):
            corners_i = corners_i.numpy().astype(np.float64)
            frontcenter = corners_i[[4, 5, 6, 7]].mean(axis=0, keepdims=True)
            heading = corners_i[4] - corners_i[0]
            frontcenter += 0.3 * heading / np.linalg.norm(heading)
            corners_i = np.concatenate((corners_i, frontcenter), axis=0)
            corners_i = o3d.utility.Vector3dVector(corners_i)
            corners_i = o3d.geometry.PointCloud(points=corners_i)

            if box3d.box_dim == 9:  # with velocity
                vel_norm = vels_norm[i].item()
                vel_yaw = vels_yaw[i].item()
                if vel_norm > 0:
                    vel_vector = o3d.geometry.TriangleMesh.create_arrow(
                        cylinder_radius=0.1, cone_radius=0.3,
                        cylinder_height=vel_norm, cone_height=0.5)
                    R = vel_vector.get_rotation_matrix_from_xyz(
                        (0, np.pi / 2, 0))
                    vel_vector.rotate(R, center=(0, 0, 0))
                    R = vel_vector.get_rotation_matrix_from_xyz(
                        (0, 0, vel_yaw))
                    vel_vector.rotate(R, center=(0, 0, 0))
                    vel_vector.translate(vels_ctr[i].numpy())
                    vel_vector.paint_uniform_color(
                        [color_amp * c for c in
                         clr_map[label_i % len(clr_map)]])
                    if vel_vectors is None:
                        vel_vectors = vel_vector
                    else:
                        vel_vectors += vel_vector

            box = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
                corners_i,
                corners_i,
                cores)
            box.paint_uniform_color(
                [color_amp * c for c in clr_map[label_i % len(clr_map)]])
            if ret is None:
                ret = box
            else:
                ret += box
        return ret, vel_vectors

    def rendergroundplane(plane, griddim=5, gridpts=21):
        a = np.linspace(-gridpts // 2, gridpts // 2, gridpts) * griddim
        b = np.linspace(0, gridpts - 1, gridpts) * griddim
        aa, bb = np.meshgrid(a, b)
        plane_x, plane_y, plane_z, plane_off = plane
        dir1 = np.array([0, plane_z, -plane_y])
        dir2 = np.array(
            [plane_y * plane_y + plane_z * plane_z, -plane_x * plane_y,
             -plane_x * plane_z])
        off_dir = -plane_off * np.array([plane_x, plane_y, plane_z])
        dir1 = dir1 / np.linalg.norm(dir1)
        dir2 = dir2 / np.linalg.norm(dir2)
        dirmat = np.stack((dir1, dir2), axis=0)
        pts = np.stack((aa, bb), axis=-1).reshape(-1, 2)
        pts = pts @ dirmat + off_dir
        pts = o3d.utility.Vector3dVector(pts)
        pts = o3d.geometry.PointCloud(points=pts)
        cores = [(p * gridpts + i, p * gridpts + j) for i, j in
                 zip(range(gridpts - 1), range(1, gridpts)) for p in
                 range(gridpts)]
        cores += [(p + i * gridpts, p + j * gridpts) for i, j in
                  zip(range(gridpts - 1), range(1, gridpts)) for p in
                  range(gridpts)]
        grid = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
            pts,
            pts,
            cores)
        grid.paint_uniform_color(((0.5), (0.5), (0.5)))
        return grid

    progress_bar = mmcv.ProgressBar(len(dataset))

    def vis_iter_pth(loader, model):
        for i, data in enumerate(loader):
            with torch.no_grad():
                det = model(return_loss=False, rescale=True, **data)
            yield i, loader.dataset[i], det[0]

    def vis_iter_out(dataset, out):
        for i, data in enumerate(dataset):
            det = out[i]
            yield i, data, det

    if out:
        vis_iter = vis_iter_out(dataset, out)
    elif model:
        vis_iter = vis_iter_pth(loader, model)

    def key_cbk(vis: o3d.visualization.Visualizer):
        try:
            idx, data, pred = next(vis_iter)
        except StopIteration:
            return True

        points = data['points'][0].data

        # if 'offsets' in pred:
        #     points[:, 0:3] = points[:, 0:3] + pred['offsets']

        if '3d_instance_id' in pred:
            instance_id = pred['3d_instance_id']

        pts_semantic_mask = None
        if 'pts_semantic_mask' in data:
            pts_semantic_mask = data['pts_semantic_mask'][0].data

        if 'pts_bbox' in pred:
            det = pred['pts_bbox']

        seg_label = None
        det_box3d = None
        det_names = None
        points_ = points.numpy().astype(np.float64)
        plot_img(data, pred, 0, points_)
        plot_img(data, data['gt_bboxes'][0]._data.numpy(), 1, points_)
        if 'boxes_3d' in pred:
            det_box3d = pred['boxes_3d']
            det_names = pred['labels_3d']
            det_scores = pred['scores_3d']
            v = det_scores.ge(0.4)
            det_box3d = det_box3d[v]
            det_names = det_names[v]
        if 'segment3d' in pred:
            seg_label = pred['segment3d']
            if points.shape[0] != seg_label.shape[0]:
                points = points[points[:, -1] == 0]
        
        # 观察range image 或 语义分割结果
        # seg_label = pred['segment3d'] * 0
        # seg_label[(points[:,5]==0)&(points[:,8]==32)] = 1
        # seg_label[(points[:,5]==1)&(points[:,8]==32)] = 2
        xyz = points[:, :3].numpy().astype(np.float64)

        # if seg_label is not None:
        #     clr = plt.get_cmap('tab20')((seg_label % 10) / 10)[:, :3]

        # if pts_semantic_mask is not None:
        #     clr = plt.get_cmap('tab20')((pts_semantic_mask % 10) / 10)[:, :3]

        # 可视化实例分割结果
        # clr = plt.get_cmap('tab20')((instance_id % 10) / 10)[:, :3]

        # 可视化run 或 可视化 CCL结果 15 11 14
        xyz = points[(points[:,6]==0)][:, :3].numpy().astype(np.float64)
        clr = plt.get_cmap('tab20')((points[points[:,6]==0][:,15] % 10) / 10)[:, :3]

        # if seg_label is None:
        #     clr = plt.get_cmap('gist_rainbow')(points[:, 3])[:, :3]

        # if 'seg_logits' in pred:
        #     # 可视化3D分割的loss
        #     seg_logits = pred['seg_logits'].sigmoid()
        #     gt_labels = data['gt_labels'][0].data
        #     mask = (points[:,6]==0) & (points[:,11]!=-1)
        #     targets = torch.zeros((mask.sum(),3))
        #     for i in range(3):
        #         cls_inds = torch.where(gt_labels == i)[0]
        #         for j in range(len(cls_inds)):
        #             targets[:,i][points[:,15][mask]==(cls_inds[j])] = 1.0
        #     seg_logits = seg_logits[mask]
        #     seg_logits[~(targets>0)] = 1 - seg_logits[~(targets>0)]
        #     loss = (-1 * seg_logits.log()).sum(dim=1)
        #     loss_rgb = loss/loss.max()
        #     loss_rgb = torch.stack((loss_rgb, loss_rgb, loss_rgb), dim=1)
        #     clr = loss_rgb
        #     # 不考虑ignore和图片外的点
        #     xyz = points[(points[:,6]==0) & (points[:,11]!=-1)][:, :3].numpy().astype(np.float64)

        points = o3d.geometry.PointCloud(
            points=o3d.utility.Vector3dVector(xyz))
        points.colors = o3d.utility.Vector3dVector(clr)

        vis.clear_geometries()
        vis.add_geometry(points, idx == 0)

        if 'gt_bboxes_3d' in data:
            box3d = data['gt_bboxes_3d'][0].data
            names = data['gt_labels_3d'][0].data
            gt_box, gt_vel = renderbox(box3d, names, 0.5)
            vis.add_geometry(gt_box, idx == 0)
            if gt_vel is not None:
                vis.add_geometry(gt_vel, idx == 0)

        if det_box3d is not None and len(det_box3d):
            det_box, det_vel = renderbox(det_box3d, det_names, 1.0)
            vis.add_geometry(det_box, idx == 0)
            if det_vel is not None:
                vis.add_geometry(det_vel, idx == 0)

        progress_bar.update()
        return False

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_key_callback(ord(" "), key_cbk)
    vis.create_window(width=1080, height=720)
    op = vis.get_render_option()
    # op.background_color = np.array([1., 1., 1.])
    op.background_color = np.array([0., 0., 0.])
    op.point_size = 1.0
    if key_cbk(vis):
        return
    else:
        vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    main()