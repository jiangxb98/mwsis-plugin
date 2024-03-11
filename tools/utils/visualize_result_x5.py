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
from mmdet.core import BitmapMasks, PolygonMasks

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def plot_mask(img, mask):
    plt.figure(figsize=(10, 10))
    plt.imshow(img.permute(1, 2, 0).cpu().numpy())
    show_mask(mask.cpu().numpy(), plt.gca())

def plot_segment3d_img(data, pred, img_metas):
    # data[key][0].data
    batch_nums = int(len(data['img'][0]._data.squeeze(0))/len(data['points']))
    # point-pixel mapping
    points = data['points'][0].data.cpu().numpy()
    pts_semantic_mask = data['pts_semantic_mask'][0].data
    scale = img_metas.get('scale_factor', None)
    # if scale is not None:
    #     points[:, 12:16] = points[:, 12:16] / scale[0][0]

    for i in range(batch_nums):
        img_shape = img_metas['img_shape'][i]
        sample_idx = img_metas['sample_idx']
        # get ori_img
        original_img = tensor2imgs(data['img'][0]._data[i].unsqueeze(0), mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0],
                            to_rgb=False)[0]  # bgr
        original_img = original_img[0:img_shape[0],0:img_shape[1],:]
        ori_shape = img_metas['ori_shape'][i]
        original_img = mmcv.imresize(original_img, ori_shape[0:2][::-1])
        # save ori_img
        cv2.imwrite('ori_img_{}.jpeg'.format(i), original_img)
        # plot segment points
        rgb = [[0,255,255], [255,0,0], [0,0,255], [255,204,102]]
        filter_mask = (points[:,5]==0)&(points[:,6]==0)
        img_semantic_mask = pts_semantic_mask[(points[:, 10]==i)&filter_mask].cpu().numpy()
        img_semantic_mask[img_semantic_mask==-1] = 3
        img_semantic_mask = img_semantic_mask.astype(np.int)
        for j, point in enumerate(points[(points[:, 10]==i)&filter_mask]):
            x1 = (point[12]).astype(np.int)
            y1 = (point[14]).astype(np.int)
            cv2.circle(original_img, (x1, y1), 1, rgb[img_semantic_mask[j]], thickness=-1, lineType=2)
        for j, point in enumerate(points[(points[:, 11]==i)&filter_mask]):
            x1 = (point[13]).astype(np.int)
            y1 = (point[15]).astype(np.int)
            cv2.circle(original_img, (x1, y1), 1, rgb[img_semantic_mask[j]], thickness=-1, lineType=2)
        cv2.imwrite('seg3d_img_{}.jpeg'.format(i), original_img)

def plot_multiview_img(data, pred, flag, points=None, img_metas=None):
    path = "/home/jiangguangfeng/桌面/work_dirs/multimodal_model_segment_config/images/ours2/"
    ori_path = "/home/jiangguangfeng/桌面/work_dirs/multimodal_model_segment_config/images/"
    # data [N,C,H,W]
    batch_nums = int(len(data['img'][0]._data.squeeze(0))/len(data['points']))
    sample_idx = img_metas['sample_idx']
    if flag == 0:
        for i in range(batch_nums):
            # get rgb
            original_img = tensor2imgs(data['img'][0]._data[i].unsqueeze(0), mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0],
                                to_rgb=False)[0]  # bgr
            img_shape = img_metas['img_shape'][i]
            original_img = original_img[0:img_shape[0],0:img_shape[1],:]
            ori_shape = img_metas['ori_shape'][i]
            original_img = mmcv.imresize(original_img, ori_shape[0:2][::-1])
            cv2.imwrite('{}ori_img_{}_{}.jpeg'.format(ori_path, sample_idx, i), original_img)
            # original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
            if 'img_bbox' not in pred['img_preds'][i].keys():
                print('\n 这张图片没有预测结果{}'.format(i))
                cv2.imwrite('{}pred_{}_{}.jpeg'.format(path, sample_idx, i), original_img)
                h, w, c = img_metas['ori_shape'][i]
                original_mask = np.zeros((h, w, c))
                cv2.imwrite('{}pred_mask_{}_{}.jpeg'.format(path, sample_idx, i), original_mask)
                continue
            bboxes_list = pred['img_preds'][i]['img_bbox']
            boxes = np.concatenate(bboxes_list) #/img_metas['scale_factor'][i][0]
            valid_mask = boxes[:,4] > 0.15
            boxes = boxes[valid_mask]
            # for gt_bbox in boxes:
            #     cv2.rectangle(original_img, (int(gt_bbox[0]), int(gt_bbox[1])), (int(gt_bbox[2]), int(gt_bbox[3])), (0, 255, 0), 1)
            
            # if points is not None and 'segment3d' in pred.keys():
            #     in_img_mask_1 = points[:,10]==i
            #     bg_points_mask = (pred['segment3d']==2).cpu().numpy()
            #     in_img_mask_1 = in_img_mask_1&bg_points_mask
            #     if (in_img_mask_1).sum() != 0:
            #         for p, point in enumerate(points[in_img_mask_1]):
            #             x1 = (point[12]).astype(np.int)
            #             y1 = (point[14]).astype(np.int)
            #             cv2.circle(original_img,(x1, y1),1,(0,255,0),thickness=-1)
            #     in_img_mask_2 = points[:,11]==i
            #     in_img_mask_2 = in_img_mask_2&bg_points_mask
            #     if (in_img_mask_2).sum() != 0:
            #         for p, point in enumerate(points[in_img_mask_2]):
            #             x2 = (point[13]).astype(np.int)
            #             y2 = (point[15]).astype(np.int)
            #             cv2.circle(original_img,(x2, y2),1,(0,255,0),thickness=-1)

            # 可视化点云到图片
            # id=40
            # in_img_mask_1 = points[:,10]==i
            # pts_instance_mask = data['pts_instance_mask'][0].data
            # if (in_img_mask_1).sum() != 0:
            #     for p, point in enumerate(points[in_img_mask_1]):
            #         if pts_instance_mask[in_img_mask_1][p]>0 and pts_instance_mask[in_img_mask_1][p]==id:
            #             x1 = (point[12]*2).astype(np.int)
            #             y1 = (point[14]*2).astype(np.int)
            #             cv2.circle(original_img,(x1, y1),1,(0,255,255),thickness=-1)
            # in_img_mask_1 = points[:,11]==i
            # pts_instance_mask = data['pts_instance_mask'][0].data
            # if (in_img_mask_1).sum() != 0:
            #     for p, point in enumerate(points[in_img_mask_1]):
            #         if pts_instance_mask[in_img_mask_1][p]>0 and pts_instance_mask[in_img_mask_1][p]==id:
            #             x1 = (point[13]*2).astype(np.int)
            #             y1 = (point[15]*2).astype(np.int)
            #             cv2.circle(original_img,(x1, y1),1,(0,255,255),thickness=-1)

            # cv2.imwrite('{}pred_{}_{}.jpeg'.format(path, sample_idx, i), original_img)

            mask_2d = []
            pred_masks = pred['img_preds'][i]['img_mask']
            boxes = np.concatenate(bboxes_list)
            boxes = boxes[valid_mask]
            for j in range(len(pred_masks)):
                if len(pred_masks[j])!=0:
                    mask_2d.append(mask_utils.decode(pred_masks[j]).transpose(2,0,1))
            if len(mask_2d) != 0:
                instance_mask = np.concatenate(mask_2d)
                instance_mask = instance_mask[valid_mask]
                h, w, c = pred['img_preds'][i]['img_metas']['ori_shape']
                original_mask = np.zeros((h, w, c))
                for j in range(len(instance_mask)):
                    rgb = np.array([np.random.randint(255) for _ in range(3)])
                    instance_mask_rgb = np.repeat(np.expand_dims(instance_mask[j],0),3,axis=0)
                    instance_mask_rgb = instance_mask_rgb.transpose(1,2,0)
                    original_mask = original_mask + instance_mask_rgb*rgb
            #     for gt_bbox in boxes:
            #         cv2.rectangle(original_mask, (int(gt_bbox[0]), int(gt_bbox[1])), (int(gt_bbox[2]), int(gt_bbox[3])), (0, 255, 0), 1)
                
                original_mask = original_img + original_mask*0.5
                cv2.imwrite('{}pred_mask_{}_{}.jpeg'.format(path, sample_idx, i), original_mask)

            #     if points is not None and 'segment3d' in pred.keys():
            #         in_img_mask_1 = points[:,10]==i
            #         bg_points_mask = (pred['segment3d']==2).cpu().numpy()
            #         in_img_mask_1 = in_img_mask_1&bg_points_mask
            #         if (in_img_mask_1).sum() != 0:
            #             for p, point in enumerate(points[in_img_mask_1]):
            #                 x1 = (point[12]).astype(np.int)
            #                 y1 = (point[14]).astype(np.int)
            #                 cv2.circle(original_img,(x1, y1),1,(0,255,0),thickness=-1)
            #         in_img_mask_2 = points[:,11]==i
            #         in_img_mask_2 = in_img_mask_2&bg_points_mask
            #         if (in_img_mask_2).sum() != 0:
            #             for p, point in enumerate(points[in_img_mask_2]):
            #                 x2 = (point[13]).astype(np.int)
            #                 y2 = (point[15]).astype(np.int)
            #                 cv2.circle(original_img,(x2, y2),1,(0,255,0),thickness=-1)
            #     cv2.imwrite('pred_mask_{}.jpeg'.format(i), original_mask)

            # # gt mask
            if 'ori_gt_masks' in img_metas.keys():
                gt_masks = img_metas['ori_gt_masks'][i].masks
                vis_mask = np.zeros((h, w, c))
                for j in range(len(gt_masks)):
                    # if i == 1 and j==4:
                        rgb = np.array([np.random.randint(255) for _ in range(3)])
                        instance_mask_rgb = np.repeat(np.expand_dims(gt_masks[j],0),3,axis=0)
                        instance_mask_rgb = instance_mask_rgb.transpose(1,2,0)
                        vis_mask = vis_mask + instance_mask_rgb*rgb
                vis_mask = original_img + vis_mask * 0.5
                cv2.imwrite('{}gt_mask_{}_{}.jpeg'.format(ori_path, sample_idx, i), vis_mask)

    elif flag == 1:
        bboxes_list = pred
        for i in range(batch_nums):
            boxes = bboxes_list[i].numpy()
            original_img = tensor2imgs(data['img'][0]._data[i].unsqueeze(0), mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0],
                                       to_rgb=False)[0]  # in RGB format. ori_image already rgb, no need to swap dim
            # original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
            for gt_bbox in boxes:
                cv2.rectangle(original_img, (int(gt_bbox[0]), int(gt_bbox[1])), (int(gt_bbox[2]), int(gt_bbox[3])), (0, 255, 0), 1)
            # if points is not None:
            #     for point in points:
            #         x = point[12].astype(np.int)
            #         y = point[13].astype(np.int)
            #         cv2.circle(original_img,(x, y),1,(0,255,0),thickness=-1)
            cv2.imwrite('label_{}.jpeg'.format(i), original_img)

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

        if 'offsets' in pred:
            points[:, 0:3] = points[:, 0:3] + pred['offsets']

        if '3d_instance_id' in pred:
            instance_id = pred['3d_instance_id']

        pts_semantic_mask = None
        if 'pts_semantic_mask' in data:
            pts_semantic_mask = data['pts_semantic_mask'][0].data
            pts_intsance_mask = data['pts_instance_mask'][0].data

        if 'pts_bbox' in pred:
            det = pred['pts_bbox']

        seg_label = None
        det_box3d = None
        det_names = None

        # 2D vis
        # points_ = points.numpy().astype(np.float64)
        # plot_segment3d_img(data, pred, data['img_metas'][0].data)
        # plot_multiview_img(data, pred, 0, points_, data['img_metas'][0].data)
        # plot_multiview_img(data, data['gt_bboxes'][0]._data, 1, points_, data['img_metas'][0].data)
        print('\n', data['img_metas'][0].data['sample_idx'])

        if 'boxes_3d' in pred:
            det_box3d = pred['boxes_3d']
            det_names = pred['labels_3d']
            det_scores = pred['scores_3d']
            v = det_scores.ge(0.4)
            det_box3d = det_box3d[v]
            det_names = det_names[v]
        if 'segment3d' in pred:
            seg_label = pred['segment3d']
            assert points.shape[0] == seg_label.shape[0]
        
        # 观察range image 或 语义分割结果
        # seg_label = pred['segment3d'] * 0
        # seg_label[(points[:,5]==0)&(points[:,8]==32)] = 1
        # seg_label[(points[:,5]==1)&(points[:,8]==32)] = 2
        xyz = points[:, :3].numpy().astype(np.float64)
        # clr = plt.get_cmap('gist_rainbow')(points[:, 3])[:, :3]

        if seg_label is not None:
            clr = plt.get_cmap('tab20')((seg_label % 10) / 10)[:, :3]

        # if pts_intsance_mask is not None:
        #     pts_intsance_mask[(pts_intsance_mask!=0) & (pts_intsance_mask%10==0)] += 1
        #     clr = plt.get_cmap('tab20')((pts_intsance_mask % 10) / 10)[:, :3]

        # if pts_semantic_mask is not None:
        #     pts_semantic_mask[pts_semantic_mask==0] = 0
        #     pts_semantic_mask[pts_semantic_mask==1] = 0
        #     pts_semantic_mask[pts_semantic_mask==2] = 0
        #     pts_semantic_mask[pts_semantic_mask==3] = 0
        #     pts_semantic_mask[pts_semantic_mask==-1] = 0
        #     clr = plt.get_cmap('tab20')((pts_semantic_mask % 10) / 10)[:, :3]

        # 可视化实例分割结果
        clr = plt.get_cmap('tab20')((instance_id % 10) / 10)[:, :3]

        # 可视化run 或 可视化 CCL结果 [19,20](一个点可能落到两个框内) 17 18
        # xyz = points[(points[:,6]==0)][:, :3].numpy().astype(np.float64)
        # points[:, 20][points[:, 20]==-1] = 0
        # points[:, 21][points[:, 21]==-1] = 0
        # clr = plt.get_cmap('tab20')((points[points[:,6]==0][:,20:22].mean(1) % 10) / 10)[:, :3]
        # in 2d box可视化
        # clr = plt.get_cmap('tab20')((points[points[:,6]==0][:, 17] % 10) / 10)[:, :3]

        # if 'seg_logits' in pred:
        #     # 可视化3D分割的loss
        #     seg_logits = pred['seg_logits'].sigmoid()
        #     gt_labels = data['gt_labels'][0].data
        #     mask = (points[:,6]==0) & (points[:,17]!=-1)
        #     targets = torch.zeros((mask.sum(),3))
        #     for i in range(3):
        #         cls_inds = torch.where(gt_labels == i)[0]
        #         for j in range(len(cls_inds)):
        #             targets[:,i][points[:,19][mask]==(cls_inds[j])] = 1.0
        #     seg_logits = seg_logits[mask]
        #     seg_logits[~(targets>0)] = 1 - seg_logits[~(targets>0)]
        #     loss = (-1 * seg_logits.log()).sum(dim=1)
        #     loss_rgb = loss/loss.max()
        #     loss_rgb = torch.stack((loss_rgb, loss_rgb, loss_rgb), dim=1)
        #     clr = loss_rgb
        #     # 不考虑ignore和非主雷达的点
        #     xyz = points[(points[:,6]==0) & (points[:,17]!=-1)][:, :3].numpy().astype(np.float64)

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