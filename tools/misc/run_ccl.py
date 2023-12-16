import numpy as np
from scipy.sparse.csgraph import connected_components  # CCL

def if_connect(pts1, pts2, times=1, max_dist=50):
    # 深度距离判断
    times = np.clip(times, 1, 1.5)
    if np.abs(pts1[7] - pts2[7]) < 0.24 * 1. * max_dist / 50:
        return True
    # 欧几里得距离判断 不加的原因是防止轮胎和地面连一块
    # elif np.sqrt((((pts1[0:3]-pts2[0:3])**2).sum())) < 0.05 * times:
    #     return True
    # elif np.abs(pts1[7] - pts2[7]) >= 0.5 and np.sqrt((((pts1[0:3]-pts2[0:3])**2).sum())) < 0.3:
    #     return True
    else:
        return False

def range_seg_ccl(points):
    range_seg_inds = np.ones((points.shape[0])).astype(np.float32)*-1
    # top_lidar points mask two return all use
    top_points_mask = (points[:,6]==0)
    top_points  = points[top_points_mask]
    points_row = points[top_points_mask][:,8]
    row_set, inv_inds = np.unique(points_row, return_inverse=True)
    row_set = row_set.astype(np.int)
    clusters_id_all = []
    column_nums = 2650
    num_clusters = 0
    for r in range(64):
        if r not in row_set:
            clusters_id_all.append(np.ones((2, column_nums))*-1)
            continue

        # 第r行的点云深度排序(2,2650)
        range_dist = np.ones((2, column_nums))*-1
        # 第一次回波
        fir_return_points = top_points[(top_points[:,5]==0) & (top_points[:,8]==r)]
        for p, pts in enumerate(fir_return_points):
            range_dist[0][int(pts[9])] = pts[7]
            # range_dist[0][int(pts[9])] = np.sqrt((pts[0]**2+pts[1]**2))
        # 第二次回波
        sec_return_points = top_points[(top_points[:,5]==1) & (top_points[:,8]==r)]
        for p, pts in enumerate(sec_return_points):
            range_dist[1][int(pts[9])] = pts[7]
            # range_dist[1][int(pts[9])] = np.sqrt((pts[0]**2+pts[1]**2))
        
        max_dist = np.max(range_dist)

        # 初始化equalTable, num_cluster, clusters_id
        kernel_size = 10 / max_dist * 50
        kernel_size = np.clip(kernel_size, 12, 50)  # 50米看4个点
        equal_tabel = np.array([i for i in range(2*column_nums)]).reshape((2,column_nums))
        # num_clusters = 0
        clusters_id = np.ones((2, column_nums))*-1
        # 1. 建立equaltree
        for i in range(column_nums):
            if range_dist[0][i] == -1 and range_dist[1][i] == -1:
                continue
            # 判断第一次回波
            if range_dist[0][i] != -1:
                for j in range(int(kernel_size/2)):
                    if i >= j and range_dist[0][i-j]!=-1 and (j!=0):
                        dist_flag = if_connect(fir_return_points[(fir_return_points[:,8]==r)&(fir_return_points[:,9]==i)][0],
                        fir_return_points[(fir_return_points[:,8]==r)&(fir_return_points[:,9]==(i-j))][0],
                        j+1, max_dist)
                        if dist_flag:
                            equal_tabel[0][i] = equal_tabel[0][i-j]
                            break
                    # 第一次回波对应位置不与对应位置的第二次回波对比，只有第二次的才与对应第一次的对比
                    if i >= j and range_dist[1][i-j]!=-1 and (j!=0):
                        dist_flag = if_connect(fir_return_points[(fir_return_points[:,8]==r)&(fir_return_points[:,9]==i)][0],
                        sec_return_points[(sec_return_points[:,8]==r)&(sec_return_points[:,9]==(i-j))][0],
                        j+1, max_dist)
                        if dist_flag:
                            equal_tabel[0][i] = equal_tabel[1][i-j]
                            break
            # 判断第二次回波
            if range_dist[1][i] != -1:
                for j in range(int(kernel_size/2)):                 
                    if i >= j and range_dist[0][i-j]!=-1:
                        dist_flag = if_connect(sec_return_points[(sec_return_points[:,8]==r)&(sec_return_points[:,9]==i)][0],
                                                    fir_return_points[(fir_return_points[:,8]==r)&(fir_return_points[:,9]==(i-j))][0],
                                                    j+1, max_dist)
                        if dist_flag:
                            equal_tabel[1][i] = equal_tabel[0][i-j]
                            break
                    if i >= j and range_dist[1][i-j]!=-1 and (j!=0):
                        dist_flag = if_connect(sec_return_points[(sec_return_points[:,8]==r)&(sec_return_points[:,9]==i)][0],
                                                    sec_return_points[(sec_return_points[:,8]==r)&(sec_return_points[:,9]==(i-j))][0],
                                                    j+1, max_dist)
                        if dist_flag:
                            equal_tabel[1][i] = equal_tabel[1][i-j]
                            break
        # 2. 统一label
        for i in range(column_nums):
            if range_dist[0][i] == -1 and range_dist[1][i] == -1:
                continue
            if range_dist[0][i] != -1:
                if equal_tabel[0][i] == i:
                    clusters_id[0][i] = num_clusters
                    num_clusters += 1
            if range_dist[1][i] != -1:
                if equal_tabel[1][i] == i + column_nums:
                    clusters_id[1][i] = num_clusters
                    num_clusters += 1
        # 3. 重新label
        for i in range(column_nums):
            if range_dist[0][i] == -1 and range_dist[1][i] == -1:
                continue
            if range_dist[0][i] != -1:
                label = i
                while label != equal_tabel[label//column_nums][label-(label//column_nums)*column_nums]:
                    batch_id = label//column_nums
                    label = equal_tabel[batch_id][label-(batch_id)*column_nums]
                batch_id = label//column_nums
                clusters_id[0][i] = int(clusters_id[batch_id][label-(batch_id)*column_nums])
            if range_dist[1][i] != -1:
                label = column_nums + i
                while label != equal_tabel[label//column_nums][label-(label//column_nums)*column_nums]:
                    batch_id = label//column_nums
                    label = equal_tabel[batch_id][label-(batch_id)*column_nums]
                batch_id = label//column_nums
                clusters_id[1][i] = int(clusters_id[batch_id][label-(batch_id)*column_nums])

        clusters_id_all.append(clusters_id)
    clusters_id_all = np.stack(clusters_id_all, 0).transpose(1,0,2)  # (64,2,2650)-->(2,64,2650)
    c, h, w = top_points[:,5].astype(np.int), top_points[:,8].astype(np.int), top_points[:,9].astype(np.int)
    range_seg_inds[top_points_mask] = clusters_id_all[(c,h,w)]

    points = np.concatenate((points, range_seg_inds.reshape(points.shape[0],1)), axis=1)

    return points

def find_connected_componets_single_batch(points, dist):

    this_points = points
    dist_mat = this_points[:, None, :2] - this_points[None, :, :2] # only care about xy
    dist_mat = (dist_mat ** 2).sum(2) ** 0.5
    adj_mat = dist_mat < dist
    adj_mat = adj_mat
    c_inds = connected_components(adj_mat, directed=False)[1]

    return c_inds

def get_in_2d_box_points(points_, gt_bboxes, labels):
    dist=(0.6,0.1,0.4)
    assert len(gt_bboxes)==len(labels)
    gt_mask_all = np.zeros((points_.shape[0])).astype(np.bool)
    for i in range(len(gt_bboxes)):
        if len(labels[i])==0:
            continue
        for j, gt_bbox in enumerate(gt_bboxes[i]):
            # 过滤掉空的
            gt_mask1 = (((points_[:, 12] >= gt_bbox[0]) & (points_[:, 12] < gt_bbox[2])) &
                        ((points_[:, 14] >= gt_bbox[1]) & (points_[:, 14] < gt_bbox[3])  &
                        (points_[:,10]==i)))
            gt_mask2 = (((points_[:, 13] >= gt_bbox[0]) & (points_[:, 13] < gt_bbox[2])) &
                        ((points_[:, 15] >= gt_bbox[1]) & (points_[:, 13] < gt_bbox[3])  &
                        (points_[:,11]==i)))
            gt_mask = gt_mask1 | gt_mask2
            gt_mask_all = gt_mask_all | gt_mask
    points_[:,17][gt_mask_all] = 1  # 在2D box内的点大部分是前景点，后面进行筛选

    box_flag = np.zeros((points_.shape[0],3)).astype(np.float32) # box_flag[:,2]是个标志位，表示第一个是否被填充
    out_box_points = points_[~gt_mask_all]
    points_index = np.array(range(0,len(points_))).astype(np.int)
    # img
    dis = 0.05
    dis_h = 0.1
    for i in range(len(gt_bboxes)):
        if len(labels[i])==0:
            continue
        for j, gt_bbox in enumerate(gt_bboxes[i]):
            w,h = gt_bbox[2]-gt_bbox[0], gt_bbox[3]-gt_bbox[1]
            x1, y1 = gt_bbox[0]-dis*w, gt_bbox[1]-dis_h*h
            x2, y2 =  gt_bbox[2]+dis*w, gt_bbox[3]+dis*h
            gt_mask1 = (((points_[:, 12] >= x1) & (points_[:, 12] < x2)) &
                        ((points_[:, 14] >= y1) & (points_[:, 14] < y2)  &
                            (points_[:,10]==i)))
            gt_mask2 = (((points_[:, 13] >= x1) & (points_[:, 13] < x2)) &
                        ((points_[:, 15] >= y1) & (points_[:, 13] < y2)  &
                            (points_[:,11]==i)))
            gt_mask = gt_mask1 | gt_mask2
            in_box_points = points_[gt_mask]
            # 进行run过滤计算
            in_box_points
            run_sets, inv_inds, counts = np.unique(in_box_points[:,18], return_inverse=True, return_counts=True)
            for s in range(len(run_sets)):
                prop = (out_box_points[:,18]==run_sets[s]).sum()/(points_[:,18]==run_sets[s]).sum()
                if run_sets[s] == -1:
                    points_[:,17][(gt_mask)&(points_[:,18]==run_sets[s])] = -1
                elif prop >= 0.5:
                    points_[:,17][(gt_mask)&(points_[:,18]==run_sets[s])] = 0
                elif prop >= 0.05 and prop < 0.5:
                    points_[:,17][(gt_mask)&(points_[:,18]==run_sets[s])] = -1
                else:
                    points_[:,17][(gt_mask)&(points_[:,18]==run_sets[s])] = 1
            # 更新gt_mask
            in_box_mask = gt_mask & (points_[:,17]>0)
            # 聚类
            # Car=0.6, Pedestrian=0.1, Cyclist=0.4
            # labels[i][j]
            if in_box_mask.sum()==0:
                continue
            c_inds = find_connected_componets_single_batch(points_[in_box_mask][:,0:3], dist[labels[i][j]])
            set_c_inds = list(set(c_inds))
            c_ind = np.argmax([np.sum(c_inds == i) for i in set_c_inds])
            c_mask = c_inds == set_c_inds[c_ind]
            # 对box flag 位赋值
            # 一个点可能投射到两个box内
            # c_mask_1表示能够往第一个格子放box索引的位置mask
            c_mask_1 = box_flag[:,2][in_box_mask] == 0
            max_in_box_index = points_index[in_box_mask][c_mask&c_mask_1]
            box_flag[:,2][max_in_box_index] = 1
            box_flag[:,0][max_in_box_index] = i*1000+j+1
            # 第一个位置有box id的地方不能赋值，往第二个里放
            c_mask_2 = box_flag[:,2][in_box_mask] == 0
            max_in_box_index_2 = points_index[in_box_mask][c_mask&c_mask_2]
            box_flag[:,1][max_in_box_index_2] = i*1000+j+1

            # c_mask_1 = box_flag[:,2][in_box_mask] == 0
            # box_flag[:,2][in_box_mask] = (c_mask&c_mask_1).astype(np.float32)
            # box_flag[:,0][in_box_mask] = (c_mask&c_mask_1).astype(np.float32) * (i*1000+j+1)
            # # 第一个位置又box id的地方不能赋值，往第二个里放
            # c_mask_2 = box_flag[:,2][in_box_mask] != 0
            # box_flag[:,1][in_box_mask] = (c_mask&(~c_mask_2)).astype(np.float32) * (i*1000+j+1)
    box_flag[:,0:2] = box_flag[:,0:2] - 1
    points_ = np.concatenate((points_, box_flag[:,0:2]),axis=1)

    return points_

def drop_arrays_by_name(gt_names, drop_classes):
    inds = [i for i, x in enumerate(gt_names) if x not in drop_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds

CLASSES = ('Car', 'Pedestrian', 'Cyclist')
def class_name_to_label_index(gt_names):
    gt_labels = []
    for cat in gt_names:
        if cat in CLASSES:
            gt_labels.append(CLASSES.index(cat))
        else:
            gt_labels.append(-1)
    return np.array(gt_labels).astype(np.int64)

def filter_ann(info):
    annos = info['annos']
    # we need other objects to avoid collision when sample
    gt_names = [np.array(n) for n in annos['name']]  # 2d
    gt_bboxes = [np.array(b, dtype=np.float32) for b in annos['bbox']]  # 2d
    selected = [drop_arrays_by_name(n, ['DontCare', 'Sign',]) for n in  # filter obejcts which we not need #'Cyclist'
                gt_names]
    gt_names = [n[s] for n, s in zip(gt_names, selected)]  # select the need objects
    gt_bboxes = [b[s] for b, s in zip(gt_bboxes, selected)]
    gt_labels = [class_name_to_label_index(n) for n in gt_names]  # encode label

    return gt_labels, gt_bboxes