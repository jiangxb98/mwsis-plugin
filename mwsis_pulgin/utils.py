import copy
import torch
import random
import numpy as np
from torch import nn
from torch.nn import functional as F
from mmcv.cnn import build_norm_layer
from itertools import accumulate

from mmdet3d.core.post_processing.box3d_nms import nms_bev, nms_normal_bev


def box3d_multiclass_nms(mlvl_bboxes,
                         mlvl_bboxes_for_nms,
                         mlvl_scores,
                         score_thr,
                         max_num,
                         cfg,
                         mlvl_dir_scores=None,
                         mlvl_attr_scores=None,
                         mlvl_bboxes2d=None):
    """Multi-class NMS for 3D boxes. The IoU used for NMS is defined as the 2D
    IoU between BEV boxes.

    Args:
        mlvl_bboxes (torch.Tensor): Multi-level boxes with shape (N, M).
            M is the dimensions of boxes.
        mlvl_bboxes_for_nms (torch.Tensor): Multi-level boxes with shape
            (N, 5) ([x1, y1, x2, y2, ry]). N is the number of boxes.
            The coordinate system of the BEV boxes is counterclockwise.
        mlvl_scores (torch.Tensor): Multi-level boxes with shape
            (N, C + 1). N is the number of boxes. C is the number of classes.
        score_thr (float): Score threshold to filter boxes with low
            confidence.
        max_num (int): Maximum number of boxes will be kept.
        cfg (dict): Configuration dict of NMS.
        mlvl_dir_scores (torch.Tensor, optional): Multi-level scores
            of direction classifier. Defaults to None.
        mlvl_attr_scores (torch.Tensor, optional): Multi-level scores
            of attribute classifier. Defaults to None.
        mlvl_bboxes2d (torch.Tensor, optional): Multi-level 2D bounding
            boxes. Defaults to None.

    Returns:
        tuple[torch.Tensor]: Return results after nms, including 3D
            bounding boxes, scores, labels, direction scores, attribute
            scores (optional) and 2D bounding boxes (optional).
    """
    # do multi class nms
    # the fg class id range: [0, num_classes-1]
    num_classes = mlvl_scores.shape[1] - 1
    bboxes = []
    scores = []
    labels = []
    dir_scores = []
    attr_scores = []
    bboxes2d = []
    for i in range(0, num_classes):
        # get bboxes and scores of this class
        cls_inds = mlvl_scores[:, i] > score_thr
        if not cls_inds.any():
            continue

        _scores = mlvl_scores[cls_inds, i]
        _bboxes_for_nms = mlvl_bboxes_for_nms[cls_inds, :]

        if cfg.use_rotate_nms:
            nms_func = nms_bev
        else:
            nms_func = nms_normal_bev

        # hacking as fsd
        if cfg.nms_thr is None:
            selected = torch.arange(len(_scores), dtype=torch.long, device=_scores.device)
        else:
            selected = nms_func(_bboxes_for_nms, _scores, cfg.nms_thr)
        _mlvl_bboxes = mlvl_bboxes[cls_inds, :]
        bboxes.append(_mlvl_bboxes[selected])
        scores.append(_scores[selected])
        cls_label = mlvl_bboxes.new_full((len(selected), ),
                                         i,
                                         dtype=torch.long)
        labels.append(cls_label)

        if mlvl_dir_scores is not None:
            _mlvl_dir_scores = mlvl_dir_scores[cls_inds]
            dir_scores.append(_mlvl_dir_scores[selected])
        if mlvl_attr_scores is not None:
            _mlvl_attr_scores = mlvl_attr_scores[cls_inds]
            attr_scores.append(_mlvl_attr_scores[selected])
        if mlvl_bboxes2d is not None:
            _mlvl_bboxes2d = mlvl_bboxes2d[cls_inds]
            bboxes2d.append(_mlvl_bboxes2d[selected])

    if bboxes:
        bboxes = torch.cat(bboxes, dim=0)
        scores = torch.cat(scores, dim=0)
        labels = torch.cat(labels, dim=0)
        if mlvl_dir_scores is not None:
            dir_scores = torch.cat(dir_scores, dim=0)
        if mlvl_attr_scores is not None:
            attr_scores = torch.cat(attr_scores, dim=0)
        if mlvl_bboxes2d is not None:
            bboxes2d = torch.cat(bboxes2d, dim=0)
        if bboxes.shape[0] > max_num:
            _, inds = scores.sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            scores = scores[inds]
            if mlvl_dir_scores is not None:
                dir_scores = dir_scores[inds]
            if mlvl_attr_scores is not None:
                attr_scores = attr_scores[inds]
            if mlvl_bboxes2d is not None:
                bboxes2d = bboxes2d[inds]
    else:
        bboxes = mlvl_scores.new_zeros((0, mlvl_bboxes.size(-1)))
        scores = mlvl_scores.new_zeros((0, ))
        labels = mlvl_scores.new_zeros((0, ), dtype=torch.long)
        if mlvl_dir_scores is not None:
            dir_scores = mlvl_scores.new_zeros((0, ))
        if mlvl_attr_scores is not None:
            attr_scores = mlvl_scores.new_zeros((0, ))
        if mlvl_bboxes2d is not None:
            bboxes2d = mlvl_scores.new_zeros((0, 4))

    results = (bboxes, scores, labels)

    if mlvl_dir_scores is not None:
        results = results + (dir_scores, )
    if mlvl_attr_scores is not None:
        results = results + (attr_scores, )
    if mlvl_bboxes2d is not None:
        results = results + (bboxes2d, )

    return results



def build_mlp(in_channel, hidden_dims, norm_cfg, is_head=False, act='relu', bias=False, dropout=0):
    layer_list = []
    last_channel = in_channel
    for i, c in enumerate(hidden_dims):
        act_layer = get_activation_layer(act, c)

        norm_layer = build_norm_layer(norm_cfg, c)[1]
        if i == len(hidden_dims) - 1 and is_head:
            layer_list.append(nn.Linear(last_channel, c, bias=True),)
        else:
            sq = [
                nn.Linear(last_channel, c, bias=bias),
                norm_layer,
                act_layer,
            ]
            if dropout > 0:
                sq.append(nn.Dropout(dropout))
            layer_list.append(
                nn.Sequential(
                    *sq
                )
            )

        last_channel = c
    mlp = nn.Sequential(*layer_list)
    return mlp


def build_norm_act(norm_cfg, act, channel):
    act_layer = get_activation_layer(act, channel)
    norm_layer = build_norm_layer(norm_cfg, channel)[1]
    sq = [
        norm_layer,
        act_layer,
    ]
    return nn.Sequential(*sq)


def get_activation_layer(act, dim=None):
    """Return an activation function given a string"""
    act = act.lower()
    if act == 'relu':
        act_layer = nn.ReLU(inplace=True)
    elif act == 'gelu':
        act_layer = nn.GELU()
    elif act == 'leakyrelu':
        act_layer = nn.LeakyReLU(inplace=True)
    elif act == 'prelu':
        act_layer = nn.PReLU(num_parameters=dim)
    elif act == 'swish' or act == 'silu':
        act_layer = nn.SiLU(inplace=True)
    elif act == 'glu':
        act_layer = nn.GLU()
    elif act == 'elu':
        act_layer = nn.ELU(inplace=True)
    else:
        raise NotImplementedError
    return act_layer


## for multi-task training
## adapted from https://github.com/tomtang110/multi-task_loss_optimizer
def get_gradient(model, loss):
    model.zero_grad()
    loss.backward(retain_graph=True)


def set_gradient(grads, optimizer, shapes):
    shapes_idx = 0
    for group in optimizer.param_groups:
        length = 0
        for i, p in enumerate(group['params']):
            # if p.grad is None: continue
            i_size = np.prod(shapes[shapes_idx])
            get_grad = grads[length:length + i_size]
            length += i_size
            p.grad = get_grad.view(shapes[shapes_idx])
            shapes_idx += 1


def pcgrad_fn(model, losses, optimizer, mode='sum'):
    grad_list = []
    shapes = []
    shares = []
    for i, loss in enumerate(losses):
        get_gradient(model, loss)
        grads = []
        for p in model.parameters():
            if i == 0:
                shapes.append(p.shape)
            if p.grad is not None:
                grads.append(p.grad.view(-1))
            else:
                grads.append(torch.zeros_like(p).view(-1))
        new_grad = torch.cat(grads, dim=0)
        grad_list.append(new_grad)

        if shares == []:
            shares = (new_grad != 0)
        else:
            shares &= (new_grad != 0)
    #clear memory
    loss_all = 0
    for los in losses:
        loss_all += los
    loss_all.backward()
    grad_list2 = copy.deepcopy(grad_list)
    for idx_i, g_i in enumerate(grad_list):
        random.shuffle(grad_list2)
        for idx_j, g_j in enumerate(grad_list2):
            # if idx_i >= idx_j:
            #     continue
            g_i_g_j = torch.dot(g_i, g_j)
            if g_i_g_j < 0:
                g_i -= (g_i_g_j) * g_j / (g_j.norm() ** 2)

    grads = torch.cat(grad_list, dim=0)
    grads = grads.view(len(losses), -1)
    if mode == 'mean':
        grads_share = grads * shares.float()

        grads_share = grads_share.mean(dim=0)
        grads_no_share = grads * (1 - shares.float())
        grads_no_share = grads_no_share.sum(dim=0)

        grads = grads_share + grads_no_share
    else:
        grads = grads.sum(dim=0)

    set_gradient(grads, optimizer, shapes)


## another pcgrad implemention
## adapted from https://github.com/wgchang/PCGrad-pytorch-example
def PCGrad_backward(optimizer, losses):
    grads_task = []
    grad_shapes = [p.shape if p.requires_grad is True else None
                   for group in optimizer.param_groups for p in group['params']]
    grad_numel = [p.numel() if p.requires_grad is True else 0
                  for group in optimizer.param_groups for p in group['params']]
    num_tasks = len(losses)  # T
    assert isinstance(losses, list)
    optimizer.zero_grad()
    
    # calculate gradients for each task
    for i in range(num_tasks):
        loss = losses[i]
        retain = i < (num_tasks - 1)
        # loss.backward(retain_graph=retain)
        if i == 0:
            (losses[0] + losses[1] * 0).backward(retain_graph=retain)
        else:
            (losses[0] * 0 + losses[1]).backward()

        devices = [
            p.device for group in optimizer.param_groups for p in group['params']]

        grad = [p.grad.detach().clone().flatten() if (p.requires_grad is True and p.grad is not None)
                else None for group in optimizer.param_groups for p in group['params']]

        # fill zero grad if grad is None but requires_grad is true
        grads_task.append(torch.cat([g if g is not None else torch.zeros(
            grad_numel[i], device=devices[i]) for i, g in enumerate(grad)]))
        optimizer.zero_grad()
        
    # shuffle gradient order
    random.shuffle(grads_task)

    # gradient projection
    grads_task = torch.stack(grads_task, dim=0)  # (T, # of params)
    proj_grad = grads_task.clone()

    def _proj_grad(grad_task):
        for k in range(num_tasks):
            inner_product = torch.sum(grad_task*grads_task[k])
            proj_direction = inner_product / (torch.sum(
                grads_task[k]*grads_task[k])+1e-12)
            grad_task = grad_task - torch.min(
                proj_direction, torch.zeros_like(proj_direction)) * grads_task[k]
        return grad_task

    proj_grad = torch.sum(torch.stack(
        list(map(_proj_grad, list(proj_grad)))), dim=0)  # (of params, )

    indices = [0, ] + [v for v in accumulate(grad_numel)]
    params = [p for group in optimizer.param_groups for p in group['params']]
    assert len(params) == len(grad_shapes) == len(indices[:-1])
    for param, grad_shape, start_idx, end_idx in zip(params, grad_shapes, indices[:-1], indices[1:]):
        if grad_shape is not None:
            param.grad[...] = proj_grad[start_idx:end_idx].view(grad_shape)  # copy proj grad

    return


def pts_semantic_confusion_matrix(pts_pred, pts_target, num_classes):
    pts_cond = pts_target * num_classes + pts_pred
    pts_cond_count = pts_cond.bincount(minlength=num_classes * num_classes)
    return pts_cond_count[:num_classes * num_classes].reshape(
        num_classes, num_classes).cpu().numpy()


def get_in_2d_box_inds(points, bboxes, img_metas=None):
    inds = (torch.ones(points.shape[0]) * -1).to(points.device)  # -1

    if isinstance(bboxes, list):
        batch_img_nums = int(len(bboxes))
        gt_lens = [len(gt_bbox) for gt_bbox in bboxes]
        gt_nums = torch.tensor([sum(gt_lens[:i]) for i in range(batch_img_nums)],device=points.device)
        # img_id*1000+box_inds
        for i in range(batch_img_nums):
            if gt_lens[i] == 0:
                continue
            mask1 = ((points[:,10] == i)|(points[:,11] == i)) & (points[:,20] != -1)
            mask2 = ((points[:,10] == i)|(points[:,11] == i)) & (points[:,21] != -1)
            if (mask1 | mask2).sum() == 0:
                continue
            inds[mask1] = (points[mask1][:,20]%1000) + gt_nums[(points[mask1][:,20]//1000).long()]
            inds[mask2] = (points[mask2][:,21]%1000) + gt_nums[(points[mask2][:,21]//1000).long()]
    else:
        if points.shape[1] > 3:
            # 这里使用的是对2D box内的点进行CCL后的结果
            for i, gt_bbox in enumerate(bboxes):
                # agt_mask = (((points[:, 8] >= gt_bbox[0]) & (points[:, 8] < gt_bbox[2])) &
                #             ((points[:, 9] >= gt_bbox[1]) & (points[:, 9] < gt_bbox[3]))&
                #             ((points[:, 6] == 1) & (points[:, 7]>=0))
                #             )
                gt_mask = points[:, 15] == i
                inds[gt_mask] = i

        elif points.shape[1] == 3:
            cam_points = lidar2img_fun(points, img_metas['lidar2img'], img_metas['scale_factor'][0])
            for i, gt_bbox in enumerate(bboxes):
                gt_mask = (((cam_points[:, 0] >= gt_bbox[0]) & (cam_points[:, 0] < gt_bbox[2])) &
                        ((cam_points[:, 1] >= gt_bbox[1]) & (cam_points[:, 1] < gt_bbox[3])))

                inds[gt_mask] = i
            
    return inds.long()

def lidar2img_fun(points, lidar2img_matrix, scale=0.5):
    
    if not torch.is_tensor(lidar2img_matrix):
        lidar2img_matrix = torch.tensor(lidar2img_matrix, device=points.device, dtype=torch.float32)

    if points.dim() == 2:
        # 注意：这里图片是否resize了，如果resize了，那么映射到的u,v也需要resize
        points = points[:,:3]
        points = torch.hstack((points, torch.ones((points.shape[0], 1), device=points.device,dtype=torch.float32)))
        cam_uv_z = torch.matmul(lidar2img_matrix, points.T).T
        rec_z = 1 / cam_uv_z[:, 2]
        cam_uv = torch.mul(cam_uv_z.T, rec_z).T
        return (cam_uv[:, :2] * scale).type(torch.float32)

    elif points.dim() == 3:
        batch_cam_uv = torch.zeros((points.shape[0],points.shape[1],2), device=points.device)
        for i in range(len(points)):  # 这里可以直接flatten展开，然后计算后再view回去 points.view(-1, points.shape[-1]) (N*8,3), cam_uv.view(points.shape[0],-1,3)
            cam_uv = lidar2img_fun(points[i], lidar2img_matrix, scale)
            batch_cam_uv[i] = cam_uv
        return batch_cam_uv.type(torch.float32)

def get_bounding_rec_2d(corners):
    """
    Input:
        corners: (N,8,2)
    Return:
        corners_2d: (N,4)
        center_2d: (N,2)
        wh: (N,2)
    """
    x1y1, _ = torch.min(corners, dim=1)  # 注意这里返回值可能超出了像素坐标，比如坐标值＜0或者＞像素边界
    x2y2, _ = torch.max(corners, dim=1)

    corners_2d = torch.cat((x1y1, x2y2), dim=1) # (N,4)

    w = (corners_2d[:,2] - corners_2d[:,0])
    h = (corners_2d[:,3] - corners_2d[:,1])
    wh = torch.stack((w, h), dim=1)  

    center_2d_u = (corners_2d[:,2] + corners_2d[:,0]) * 0.5
    center_2d_v = (corners_2d[:,3] + corners_2d[:,1]) * 0.5
    center_2d = torch.stack((center_2d_u, center_2d_v), dim=1)

    return corners_2d, center_2d, wh

def calc_dis_rect_object_centric(wl, Ry, points, density):
    """基于BEV的计算,注意输入的points坐标也需要转换
    wl 先验anchor的长和宽
    Ry 朝向角
    points 投影在2d box的点,减去预测的中心点(N,2)
    density (N,1) 每个points的密度
    """
    # 转到weakm3d下 orient_new = orient_old - np.pi/2
    PI = torch.tensor([np.pi], device=Ry.device)
    Ry = Ry + PI/2

    init_theta, length = torch.atan(wl[0] / wl[1]), torch.sqrt(wl[0] ** 2 + wl[1] ** 2) / 2  # 0.5:1
    corners = [(length * torch.cos(init_theta + Ry),
                length * torch.sin(init_theta + Ry)),

               (length * torch.cos(np.pi - init_theta + Ry),
                length * torch.sin(np.pi - init_theta + Ry)),

               (length * torch.cos(np.pi + init_theta + Ry),
                length * torch.sin(np.pi + init_theta + Ry)),

               (length * torch.cos(-init_theta + Ry),
                length * torch.sin(-init_theta + Ry))]

    if Ry == np.pi/2:
        Ry -= 1e-4
    if Ry == 0:
        Ry += 1e-4

    k1, k2 = torch.tan(Ry), torch.tan(Ry + np.pi / 2)
    b11 = corners[0][1] - k1 * corners[0][0]
    b12 = corners[2][1] - k1 * corners[2][0]
    b21 = corners[0][1] - k2 * corners[0][0]
    b22 = corners[2][1] - k2 * corners[2][0]

    # line0 = [k1, -1, b11]
    # line1 = [k1, -1, b12]
    # line2 = [k2, -1, b21]
    # line3 = [k2, -1, b22]

    line0 = [k1, -1, b11]
    line1 = [k2, -1, b22]
    line2 = [k1, -1, b12]
    line3 = [k2, -1, b21]

    points[points[:, 0] == 0, 0] = 1e-4
    #################################################
    slope_x = points[:, 1] / points[:, 0]
    intersect0 = torch.stack([line0[2] / (slope_x - line0[0]),
                              line0[2]*slope_x / (slope_x - line0[0])], dim=1)
    intersect1 = torch.stack([line1[2] / (slope_x - line1[0]),
                              line1[2]*slope_x / (slope_x - line1[0])], dim=1)
    intersect2 = torch.stack([line2[2] / (slope_x - line2[0]),
                              line2[2]*slope_x / (slope_x - line2[0])], dim=1)
    intersect3 = torch.stack([line3[2] / (slope_x - line3[0]),
                              line3[2]*slope_x / (slope_x - line3[0])], dim=1)

    # dis0 = torch.sqrt((intersect0[:, 0] - points[:, 0])**2 +
    #                   (intersect0[:, 1] - points[:, 1])**2)
    # dis1 = torch.sqrt((intersect1[:, 0] - points[:, 0])**2 +
    #                   (intersect1[:, 1] - points[:, 1])**2)
    # dis2 = torch.sqrt((intersect2[:, 0] - points[:, 0])**2 +
    #                   (intersect2[:, 1] - points[:, 1])**2)
    # dis3 = torch.sqrt((intersect3[:, 0] - points[:, 0])**2 +
    #                   (intersect3[:, 1] - points[:, 1])**2)

    dis0 = torch.abs(intersect0[:, 0] - points[:, 0]) + torch.abs(intersect0[:, 1] - points[:, 1])
    dis1 = torch.abs(intersect1[:, 0] - points[:, 0]) + torch.abs(intersect1[:, 1] - points[:, 1])
    dis2 = torch.abs(intersect2[:, 0] - points[:, 0]) + torch.abs(intersect2[:, 1] - points[:, 1])
    dis3 = torch.abs(intersect3[:, 0] - points[:, 0]) + torch.abs(intersect3[:, 1] - points[:, 1])


    # dis_inter2center0 = torch.sqrt(intersect0[:, 0]**2 + intersect0[:, 1]**2)
    # dis_inter2center1 = torch.sqrt(intersect1[:, 0]**2 + intersect1[:, 1]**2)
    # dis_inter2center2 = torch.sqrt(intersect2[:, 0]**2 + intersect2[:, 1]**2)
    # dis_inter2center3 = torch.sqrt(intersect3[:, 0]**2 + intersect3[:, 1]**2)
    #
    # dis_point2center = torch.sqrt(points[:, 0]**2 + points[:, 1]**2)
    #################################################

    points_z = torch.cat([points, torch.zeros_like(points[:, :1])], dim=1)
    vec0 = torch.tensor([corners[0][0], corners[0][1], 0], device=points_z.device)
    vec1 = torch.tensor([corners[1][0], corners[1][1], 0], device=points_z.device)
    vec2 = torch.tensor([corners[2][0], corners[2][1], 0], device=points_z.device)
    vec3 = torch.tensor([corners[3][0], corners[3][1], 0], device=points_z.device)

    ''' calc direction of vectors'''
    cross0 = torch.cross(points_z, vec0.unsqueeze(0).repeat(points_z.shape[0], 1))[:, 2]
    cross1 = torch.cross(points_z, vec1.unsqueeze(0).repeat(points_z.shape[0], 1))[:, 2]
    cross2 = torch.cross(points_z, vec2.unsqueeze(0).repeat(points_z.shape[0], 1))[:, 2]
    cross3 = torch.cross(points_z, vec3.unsqueeze(0).repeat(points_z.shape[0], 1))[:, 2]


    ''' calc angle across vectors'''
    norm_p = torch.sqrt(points_z[:, 0] ** 2 + points_z[:, 1] ** 2)
    norm_d = torch.sqrt(corners[0][0] ** 2 + corners[0][1] ** 2).repeat(points_z.shape[0])
    norm = norm_p * norm_d

    dot_vec0 = torch.matmul(points_z, vec0.unsqueeze(0).repeat(points_z.shape[0], 1).t())[:, 0]
    dot_vec1 = torch.matmul(points_z, vec1.unsqueeze(0).repeat(points_z.shape[0], 1).t())[:, 0]
    dot_vec2 = torch.matmul(points_z, vec2.unsqueeze(0).repeat(points_z.shape[0], 1).t())[:, 0]
    dot_vec3 = torch.matmul(points_z, vec3.unsqueeze(0).repeat(points_z.shape[0], 1).t())[:, 0]

    angle0 = torch.acos(dot_vec0/(norm))
    angle1 = torch.acos(dot_vec1/(norm))
    angle2 = torch.acos(dot_vec2/(norm))
    angle3 = torch.acos(dot_vec3/(norm))

    angle_sum0 = (angle0 + angle1) < np.pi
    angle_sum1 = (angle1 + angle2) < np.pi
    angle_sum2 = (angle2 + angle3) < np.pi
    angle_sum3 = (angle3 + angle0) < np.pi

    cross_dot0 = (cross0 * cross1) < 0
    cross_dot1 = (cross1 * cross2) < 0
    cross_dot2 = (cross2 * cross3) < 0
    cross_dot3 = (cross3 * cross0) < 0

    cross_all = torch.stack([cross_dot0, cross_dot1, cross_dot2, cross_dot3], dim=1)
    angle_sum_all = torch.stack([angle_sum0, angle_sum1, angle_sum2, angle_sum3], dim=1)

    choose_ind = cross_all & angle_sum_all
    dis_all = torch.stack([dis0, dis1, dis2, dis3], dim=1) / density.unsqueeze(1)
    choose_dis = dis_all[choose_ind]
    choose_dis[choose_dis != choose_dis] = 0

    dis_error = torch.mean(choose_dis)

    return dis_error

def calc_dis_ray_tracing(wl, Ry, points, density, bev_box_center):
    # 转到weakm3d下 orient_new = orient_old - np.pi/2
    PI = torch.tensor([np.pi], device=Ry.device)
    Ry = Ry + PI/2
 
    init_theta, length = torch.atan(wl[0] / wl[1]), torch.sqrt(wl[0] ** 2 + wl[1] ** 2) / 2  # 0.5:1
    corners = [(length * torch.cos(init_theta + Ry) + bev_box_center[0],
                length * torch.sin(init_theta + Ry) + bev_box_center[1]),

               (length * torch.cos(np.pi - init_theta + Ry) + bev_box_center[0],
                length * torch.sin(np.pi - init_theta + Ry) + bev_box_center[1]),

               (length * torch.cos(np.pi + init_theta + Ry) + bev_box_center[0],
                length * torch.sin(np.pi + init_theta + Ry) + bev_box_center[1]),

               (length * torch.cos(-init_theta + Ry) + bev_box_center[0],
                length * torch.sin(-init_theta + Ry) + bev_box_center[1])]
    if Ry == np.pi/2:
        Ry -= 1e-4
    if Ry == 0:
        Ry += 1e-4
    k1, k2 = torch.tan(Ry), torch.tan(Ry + np.pi / 2)
    b11 = corners[0][1] - k1 * corners[0][0]
    b12 = corners[2][1] - k1 * corners[2][0]
    b21 = corners[0][1] - k2 * corners[0][0]
    b22 = corners[2][1] - k2 * corners[2][0]

    line0 = [k1, -1, b11]
    line1 = [k2, -1, b22]
    line2 = [k1, -1, b12]
    line3 = [k2, -1, b21]

    points[points[:, 0] == 0, 0] = 1e-4 # avoid inf
    #################################################
    slope_x = points[:, 1] / points[:, 0]
    intersect0 = torch.stack([line0[2] / (slope_x - line0[0]),
                              line0[2]*slope_x / (slope_x - line0[0])], dim=1)
    intersect1 = torch.stack([line1[2] / (slope_x - line1[0]),
                              line1[2]*slope_x / (slope_x - line1[0])], dim=1)
    intersect2 = torch.stack([line2[2] / (slope_x - line2[0]),
                              line2[2]*slope_x / (slope_x - line2[0])], dim=1)
    intersect3 = torch.stack([line3[2] / (slope_x - line3[0]),
                              line3[2]*slope_x / (slope_x - line3[0])], dim=1)


    dis0 = torch.abs(intersect0[:, 0] - points[:, 0]) + torch.abs(intersect0[:, 1] - points[:, 1])
    dis1 = torch.abs(intersect1[:, 0] - points[:, 0]) + torch.abs(intersect1[:, 1] - points[:, 1])
    dis2 = torch.abs(intersect2[:, 0] - points[:, 0]) + torch.abs(intersect2[:, 1] - points[:, 1])
    dis3 = torch.abs(intersect3[:, 0] - points[:, 0]) + torch.abs(intersect3[:, 1] - points[:, 1])


    dis_inter2center0 = torch.sqrt(intersect0[:, 0]**2 + intersect0[:, 1]**2)
    dis_inter2center1 = torch.sqrt(intersect1[:, 0]**2 + intersect1[:, 1]**2)
    dis_inter2center2 = torch.sqrt(intersect2[:, 0]**2 + intersect2[:, 1]**2)
    dis_inter2center3 = torch.sqrt(intersect3[:, 0]**2 + intersect3[:, 1]**2)

    intersect0 = torch.round(intersect0*1e4)
    intersect1 = torch.round(intersect1*1e4)
    intersect2 = torch.round(intersect2*1e4)
    intersect3 = torch.round(intersect3*1e4)


    dis0_in_box_edge = ((intersect0[:, 0] > torch.round(min(corners[0][0], corners[1][0])*1e4)) &
                        (intersect0[:, 0] < torch.round(max(corners[0][0], corners[1][0])*1e4))) | \
                       ((intersect0[:, 1] > torch.round(min(corners[0][1], corners[1][1])*1e4)) &
                        (intersect0[:, 1] < torch.round(max(corners[0][1], corners[1][1])*1e4)))
    dis1_in_box_edge = ((intersect1[:, 0] > torch.round(min(corners[1][0], corners[2][0])*1e4)) &
                        (intersect1[:, 0] < torch.round(max(corners[1][0], corners[2][0])*1e4))) | \
                       ((intersect1[:, 1] > torch.round(min(corners[1][1], corners[2][1])*1e4)) &
                        (intersect1[:, 1] < torch.round(max(corners[1][1], corners[2][1])*1e4)))
    dis2_in_box_edge = ((intersect2[:, 0] > torch.round(min(corners[2][0], corners[3][0])*1e4)) &
                        (intersect2[:, 0] < torch.round(max(corners[2][0], corners[3][0])*1e4))) | \
                       ((intersect2[:, 1] > torch.round(min(corners[2][1], corners[3][1])*1e4)) &
                        (intersect2[:, 1] < torch.round(max(corners[2][1], corners[3][1])*1e4)))
    dis3_in_box_edge = ((intersect3[:, 0] > torch.round(min(corners[3][0], corners[0][0])*1e4)) &
                        (intersect3[:, 0] < torch.round(max(corners[3][0], corners[0][0])*1e4))) | \
                       ((intersect3[:, 1] > torch.round(min(corners[3][1], corners[0][1])*1e4)) &
                        (intersect3[:, 1] < torch.round(max(corners[3][1], corners[0][1])*1e4)))


    dis_in_mul = torch.stack([dis0_in_box_edge, dis1_in_box_edge,
                              dis2_in_box_edge, dis3_in_box_edge], dim=1)  # (100,4)
    dis_inter2cen = torch.stack([dis_inter2center0, dis_inter2center1,
                                 dis_inter2center2, dis_inter2center3], dim=1)  # (100,4)
    dis_all = torch.stack([dis0, dis1, dis2, dis3], dim=1)  # (100, 4)

    # dis_in = torch.sum(dis_in_mul, dim=1).type(torch.bool)  和下面的效果一样
    dis_in = (torch.sum(dis_in_mul, dim=1) == 2).type(torch.bool)
    if torch.sum(dis_in.int()) < 3:
        return 0

    dis_in_mul = dis_in_mul[dis_in]
    dis_inter2cen = dis_inter2cen[dis_in]  # 交点到中心点的距离，(100,4)
    dis_all = dis_all[dis_in]
    density = density[dis_in]

    z_buffer_ind = torch.argmin(dis_inter2cen[dis_in_mul].view(-1, 2), dim=1)  # 返回最近距离的索引是0还是1
    # z_buffer_ind_gather = torch.stack([~z_buffer_ind.byte(), z_buffer_ind.byte()],
    #                                   dim=1).type(torch.bool)
    z_buffer_ind_gather = torch.stack([~(z_buffer_ind.type(torch.bool)), z_buffer_ind.type(torch.bool)],
                                      dim=1)

    dis = (dis_all[dis_in_mul].view(-1, 2))[z_buffer_ind_gather] / density

    dis_mean = torch.mean(dis)
    return dis_mean

def ground_segmentation(ori_data):
    """RANSAC remov ground points
    输入：
        data: 一帧完整点云
    输出：
        segmengted_cloud: 删除地面点之后的点云
    """
    #初始化数据
    data = ori_data[:,0:3]
    idx_segmented = []
    segmented_cloud = []
    iters = 100   #最大迭代次数  000002.bin：10
    sigma = 0.4   #数据和模型之间可接受的最大差值
    ##最好模型的参数估计和内点数目,平面表达方程为   aX + bY + cZ +D= 0
    best_a = 0
    best_b = 0
    best_c = 0
    best_d = 0
    pretotal = 0 #上一次inline的点数
    #希望的到正确模型的概率
    P = 0.99
    n = len(data)    #点的数目
    outline_ratio = 0.6   #e :outline_ratio   000002.bin：0.6    000001.bin: 0.5  000000.bin: 0.6   002979.bin：0.6
    for i in range(iters):
        ground_cloud = []
        idx_ground = []
        #step1 选择可以估计出模型的最小数据集，对于平面拟合来说，就是三个点
        sample_index = random.sample(range(n),3)    #重数据集中随机选取3个点
        point1 = data[sample_index[0]]
        point2 = data[sample_index[1]]
        point3 = data[sample_index[2]]
        #step2 求解模型
        ##先求解法向量
        point1_2 = (point1-point2)      #向量 poin1 -> point2
        point1_3 = (point1-point3)      #向量 poin1 -> point3
        N = np.cross(point1_3,point1_2)            #向量叉乘求解 平面法向量
        ##slove model 求解模型的a,b,c,d
        a = N[0]
        b = N[1]
        c = N[2]
        d = -N.dot(point1)
        #step3 将所有数据带入模型，计算出“内点”的数目；(累加在一定误差范围内的适合当前迭代推出模型的数据)
        total_inlier = 0
        pointn_1 = (data - point1)    #sample（三点）外的点 与 sample内的三点其中一点 所构成的向量
        distance = abs(pointn_1.dot(N))/ np.linalg.norm(N)     #求距离
        ##使用距离判断inline
        idx_ground = (distance <= sigma)
        total_inlier = np.sum(idx_ground == True)    #统计inline得点数
        ##判断当前的模型是否比之前估算的模型
        if total_inlier > pretotal:                                           #     log(1 - p)
            iters = np.log(1 - P) / np.log(1 - pow(total_inlier / n, 3))  #N = ------------
            pretotal = total_inlier                                               #log(1-[(1-e)**s])
            #获取最好得 abcd 模型参数
            best_a = a
            best_b = b
            best_c = c
            best_d = d

        # 判断是否当前模型已经符合超过 inline_ratio
        if total_inlier > n*(1-outline_ratio):
            break
    # print("iters = %f" %iters)
    #提取分割后得点
    idx_segmented = np.logical_not(idx_ground)
    ground_cloud = ori_data[idx_ground]
    segmented_cloud = ori_data[idx_segmented]
    return ground_cloud, segmented_cloud

###################################################
# FGR 论文中的RANSAC没法用， 不知道哪里的参数设置有问题
###################################################
def check_parallel(points):
    a = np.linalg.norm(points[0] - points[1])
    b = np.linalg.norm(points[1] - points[2])
    c = np.linalg.norm(points[2] - points[0])
    p = (a + b + c) / 2
    
    area = np.sqrt(p * (p - a) * (p - b) * (p - c))
    if area < 1e-2:
        return True
    else:
        return False

def fitPlane(points):
    if points.shape[0] == points.shape[1]:
        return np.linalg.solve(points, np.ones(points.shape[0]))
    else:
        return np.linalg.lstsq(points, np.ones(points.shape[0]))[0]

def calculate_ground(points, thresh_ransac=0.15,):
    """RANSAC remov ground points"""
    # waymo 坐标系下
    point_cloud = points[:, 0:3]

    planeDiffThreshold = thresh_ransac
    temp = np.sort(point_cloud[:,2])[int(point_cloud.shape[0] * 0.25)]
    cloud = point_cloud[point_cloud[:,2] < temp]
    points_np = point_cloud
    mask_all = np.ones(points_np.shape[0])
    final_sample_points = None
    for i in range(5):
         best_len = 0
         for iteration in range(min(cloud.shape[0], 100)):
             sampledPoints = cloud[np.random.choice(np.arange(cloud.shape[0]), size=(3), replace=False)]
                
             while check_parallel(sampledPoints) == True:
                sampledPoints = cloud[np.random.choice(np.arange(cloud.shape[0]), size=(3), replace=False)]
                continue

             plane = fitPlane(sampledPoints)
             diff = np.abs(np.matmul(points_np, plane) - np.ones(points_np.shape[0])) / np.linalg.norm(plane)
             inlierMask = diff < planeDiffThreshold
             numInliers = inlierMask.sum()
             if numInliers > best_len and np.abs(np.dot(plane/np.linalg.norm(plane),np.array([0,1,0])))>0.9:
                 mask_ground = inlierMask
                 best_len = numInliers
                 best_plane = plane
                 final_sample_points = sampledPoints
         mask_all *= 1 - mask_ground

    return mask_all, final_sample_points

def bbox_flip(bboxes, img_shape, direction):
    """Flip bboxes horizontally.

    Args:
        bboxes (numpy.ndarray): Bounding boxes, shape (..., 4*k)
        img_shape (tuple[int]): Image shape (height, width)
        direction (str): Flip direction. Options are 'horizontal',
            'vertical'.

    Returns:
        numpy.ndarray: Flipped bounding boxes.
    """

    assert bboxes.shape[-1] % 4 == 0
    flipped = bboxes.copy()
    if direction == 'horizontal':
        w = img_shape[1]
        flipped[..., 0::4] = w - bboxes[..., 2::4]
        flipped[..., 2::4] = w - bboxes[..., 0::4]
    elif direction == 'vertical':
        h = img_shape[0]
        flipped[..., 1::4] = h - bboxes[..., 3::4]
        flipped[..., 3::4] = h - bboxes[..., 1::4]
    elif direction == 'diagonal':
        w = img_shape[1]
        h = img_shape[0]
        flipped[..., 0::4] = w - bboxes[..., 2::4]
        flipped[..., 1::4] = h - bboxes[..., 3::4]
        flipped[..., 2::4] = w - bboxes[..., 0::4]
        flipped[..., 3::4] = h - bboxes[..., 1::4]
    else:
        raise ValueError(f"Invalid flipping direction '{direction}'")
    return flipped