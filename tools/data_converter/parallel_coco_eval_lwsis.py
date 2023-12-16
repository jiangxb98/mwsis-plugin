# 在生成rle编码时,没有将实例id归一化是没有影响的,
# 所以输入的mask矩阵是0,1,0,1...与0,222,0,222...是无关的
# 是仿照lwsis的处理方式来处理数据，过滤mask小于200像素，且3D Box内小于10个点的点
# 相较于parallel_coco_lwsis添加了点云过滤的处理

import json
import numpy as np
import torch
from pycocotools import mask
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import numpy as np
from io import BytesIO
import minio
import pymongo
import time
import mmcv
import os
from datetime import datetime
from tqdm import tqdm
import pickle
from mmdet3d.core.points import get_points_type, BasePoints
from mmdet3d.core.bbox import box_np_ops, Coord3DMode, Box3DMode, get_box_type, LiDARInstance3DBoxes

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf-8');
        return json.JSONEncoder.default(self, obj)

class ConvertToCOCO:
    def __init__(self,
                 minio_cfg=dict(endpoint='ossapi.cowarobot.cn:9000',
                                access_key='abcdef',
                                secret_key='12345678',
                                region='shjd-oss',
                                secure=False,),
                 mongo_cfg=dict(host="mongodb://root:root@172.16.110.100:27017/"),
                 bucket='ai-waymo-v1.4',
                 database='ai-waymo-v1_4',
                 workers=8,
                 collection='training/infos',
                 index_filename='/root/3D/work_dirs/dataset_infos/training_infos_panseg.pkl',
                 json_path=None,
                 result_path=None,):
        self.minio_cfg = minio_cfg
        self.mongo_cfg = mongo_cfg
        self.bucket = bucket
        self.database =database
        self.workers = workers

        self.pred_test = []
        # self.img_prefix = 'training/image/'
        self.car = [2,3,4,5,7,8,11] # 1 is ego car
        self.ped = [9,16]  # 9,16
        self.cyc = [6,10]  # 6 is bicycle not cyclist but temporary use
        self.semantic_label = [[2,3,4,5,7,8,11],[9,16],[6,10]]  # [2,3,4,5,7,8,11]
        self.num_classes = 3
        self.instance_id = 0
        self.collection = collection
        self.index_filename = index_filename
        self.json_path = json_path
        self.result_path = result_path
        self.mapping = {0:"FRONT",1:"FRONT_LEFT",2:"FRONT_RIGHT",3:"SIDE_LEFT",4:"SIDE_RIGHT"}

        # dataset
        self.filter_classes = ['DontCare', 'Sign']
        classes = ['Car', 'Pedestrian', 'Cyclist']
        self.CLASSES = self.get_classes(classes)


    def query_index(self, collection='training/infos', filter=dict(), filter_sem='panseg_info'):
        _client = pymongo.MongoClient(**self.mongo_cfg)[self.database]
        # client = minio.Minio(**self.minio_cfg)
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
            f = open(self.index_filename, 'wb')
            pickle.dump(ret, f)
            f.close()    
            return ret

    def load_index(self):
        f = open(self.index_filename, "rb+")
        ret = pickle.load(f)
        f.close()
        return ret


    def get_all(self):
        # json文件格式
        dataset_dict = {
            'info':{},
            'images':[],
            'annotations': [],
            'categories': [],
            'licenses': ['']
        }
        dataset_dict['info'] = {
            'description': 'Waymo Panseg',
            'version': '1.4.0',
            'year': 2023,
            'contributor': 'jiangxb',
            'data_created': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        dataset_dict['categories'] = [
            {'id':0,
            'name':'Car'},
            {'id':1,
            'name':'Pedestrian'},
            {'id':2,
            'name':'Cyclist'}
        ]
        pred_list = []
        self.ret = self.load_index()
        if self.workers == 0:
            results = []
            for i in mmcv.track_iter_progress(self.ret):
                results.append(self.get_one(i))
        else:
            results = mmcv.track_parallel_progress(self.get_one, self.ret, self.workers)
        print("完成读取")
        for image, annotations, pred in mmcv.track_iter_progress(results):
            dataset_dict['images'].extend(image)
            dataset_dict['annotations'].extend(annotations)
            pred_list.extend(pred)
        for i in range(len(dataset_dict['annotations'])):
            dataset_dict['annotations'][i]['id'] = int(i+1)
        print("实例数量{}".format(len(dataset_dict['annotations'])))
        with open(self.json_path,'w') as f:
            json.dump(dataset_dict, f, indent=4, ensure_ascii=False, cls=MyEncoder)
        print('\n保存路径: {}'.format(self.json_path))
        # with open(self.result_path,'w') as f:
        #     json.dump(pred_list, f, indent=4, ensure_ascii=False, cls=MyEncoder)

    def get_one(self, ind):
        _client = pymongo.MongoClient(**self.mongo_cfg)[self.database]
        client = minio.Minio(**self.minio_cfg)
        out_images = []
        out_annotations = []
        out_preds = []

        # torch.cuda.set_device(int(ind%4))

        info = _client.get_collection(self.collection).find_one(ind)
        img_path = info['images']  # list len=5
        pts_path = info['pts_info']['path']
        # 获得点云
        pts_bytes = client.get_object(self.bucket, pts_path).read()
        points_ = np.load(BytesIO(pts_bytes))
        points = np.stack([ points_['x'].astype('f4'),
                            points_['y'].astype('f4'),
                            points_['z'].astype('f4'),
                            points_['lidar_idx'].astype('i2'),   # 3
                            points_['cam_idx_0'].astype('i2'),   # 4
                            points_['cam_idx_1'].astype('i2'),   # 5
                            points_['cam_column_0'].astype('i2'),
                            points_['cam_column_1'].astype('i2'),
                            points_['cam_row_0'].astype('i2'),
                            points_['cam_row_1'].astype('i2'),   # 9
                            ], axis=-1)
        points_class = get_points_type('LIDAR')
        points = points_class(points, points_dim=points.shape[-1])
        # pts annos\
        annos = info['annos']
        gt_names_3d = np.array(annos['name_3d'])
        gt_bboxes_3d = np.array(annos['bbox_3d']).astype('float32').reshape(-1, 7)
        selected_3d = self.drop_arrays_by_name(gt_names_3d, self.filter_classes)
        gt_names_3d = gt_names_3d[selected_3d]
        gt_bboxes_3d = gt_bboxes_3d[selected_3d]
        # exchange width&length
        # gt_bboxes_3d = gt_bboxes_3d[:, [0, 1, 2, 4, 3, 5, 6]]
        gt_bboxes_3d = LiDARInstance3DBoxes(gt_bboxes_3d)  # 定义3Dbox实例，gt3d的相对位置关系是(0.5,0.5,0)
        gt_labels_3d = self.class_name_to_label_index(gt_names_3d)
        device = 'cuda:{}'.format(0)
        gt_bboxes_3d = gt_bboxes_3d.to(device)
        points = points.to(device)
        inbox_inds = gt_bboxes_3d.points_in_boxes_part(points.tensor[:, :3]).long()
        sets, counts = torch.unique(inbox_inds, return_counts=True)
        sets = sets[counts > 10]
        points_mask = torch.zeros(points.tensor.shape[0]).type(torch.bool).to(device)
        for p in range(len(sets)):
            if sets[p] == -1:
                continue
            tmp_mask = inbox_inds == sets[p]
            points_mask = tmp_mask | points_mask
        points = points[points_mask]
        points = points.tensor.cpu().numpy()
        for j in range(5):
            # 获得图片 
            img_bytes = client.get_object(self.bucket, img_path[j]['path']).read()
            img = mmcv.imfrombytes(img_bytes, flag='color', channel_order='rgb')
            # 获得图片id
            split_path = img_path[j]['path'].split('/')[-2:]
            for p in range(len(split_path)):
                split_path[p] = int(split_path[p])
            image_id = split_path[0]*10+split_path[1]
            height, width, channel = img.shape
            file_name = image_id
            license = 'license'
            coco_url = 'waymo'
            data_captured = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            out_images.append({
                'id': image_id,
                'width': width,
                'height': height,
                'file_name': '{}/{}_{}.png'.format(info['record'], info['frame_idx'], self.mapping[j]),
                'license': license,
                'flickr_url': coco_url,
                'coco_url': coco_url,
                'date_captured': data_captured
            })

            # 获取实例分割结果
            panseg_pth = info['panseg_info'][j]['path']
            panseg_bytes = client.get_object(self.bucket, panseg_pth).read()
            panseg = np.load(BytesIO(panseg_bytes))
            semantic = panseg['panseg_cls'].squeeze().astype(np.int8)
            instance = panseg['panseg_instance_id'].squeeze().astype(np.int)

            tmp = np.zeros_like(semantic)
            for cls in range(len(self.semantic_label)):
                for cls_id in self.semantic_label[cls]:
                    tmp += np.where(semantic==cls_id, cls+1, 0).astype(np.int8)
            new_semantic = tmp-1
            new_instance = instance * ((tmp-1)>=0)
            instance_inds, counts = np.unique(new_instance, return_counts=True)
            # 将每个实例都存到annotation中
            for ind in range(len(instance_inds)):
                if instance_inds[ind] == 0 or counts[ind] <= 200:  # filter small object
                    continue
                tmp_instance = (new_instance*(new_instance==instance_inds[ind])).astype(np.int8)
                category_id = tmp*(new_instance==instance_inds[ind])
                if len(np.unique(category_id)) >= 3:  # 这里会出现三个instance
                    print('\n一个实例出现多个语义标签{}\n'.format(image_id))
                    for abn_cls in np.unique(category_id):
                        if abn_cls == 0:
                            continue
                        category_id_ = abn_cls-1
                        tmp_instance_ = tmp_instance*((new_semantic==category_id_)&(tmp_instance==instance_inds[ind]))
                        tmp_instance_ = (tmp_instance_/tmp_instance_.max()).astype(np.uint8)
                        tmp_instance_ = np.asfortranarray(tmp_instance_)
                        rle = mask.encode(tmp_instance_)
                        area = float(mask.area(rle))
                        bbox = mask.toBbox(rle)
                        x1,y1,x2,y2 = bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]
                        in_mask1 = (points[:, 4]==j) & (points[:, 6] >= x1) & (points[:, 6] < x2) & (points[:, 8] >= y1) & (points[:, 8]< y2)
                        in_mask2 = (points[:, 5]==j) & (points[:, 7] >= x1) & (points[:, 7] < x2) & (points[:, 9] >= y1) & (points[:, 9]< y2)
                        if (in_mask1 | in_mask2).sum() < 20:
                            continue
                        out_annotations.append({
                            'id': 0,  # 这里后面重新赋值，并行化存在问题，导致id不唯一
                            'image_id': int(image_id),
                            'category_id': int(category_id_),
                            'segmentation': rle,
                            'area': area,
                            'bbox': bbox,
                            'iscrowd': 0,
                        })
                        out_preds.append({
                            'image_id': int(image_id),
                            'category_id': int(category_id_),
                            "segmentation": rle,
                            'score': 1.0
                        })
                    continue
                category_id = category_id.max()-1
                tmp_instance = (tmp_instance/tmp_instance.max()).astype(np.uint8)
                tmp_instance = np.asfortranarray(tmp_instance)
                rle = mask.encode(tmp_instance)
                area = float(mask.area(rle))
                bbox = mask.toBbox(rle)  # [xmin ymin w h]
                x1,y1,x2,y2 = bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]
                in_mask1 = (points[:, 4]==j) & (points[:, 6] >= x1) & (points[:, 6] < x2) & (points[:, 8] >= y1) & (points[:, 8]< y2)
                in_mask2 = (points[:, 5]==j) & (points[:, 7] >= x1) & (points[:, 7] < x2) & (points[:, 9] >= y1) & (points[:, 9]< y2)
                if (in_mask1 | in_mask2).sum() < 10:
                    continue
                out_annotations.append({
                    'id': 0,  # id应该全局唯一，但并行存在问题，所以在得到所有数据后再重新赋值
                    'image_id': int(image_id),
                    'category_id': int(category_id),
                    'segmentation': rle,
                    'area': area,
                    'bbox': bbox,
                    'iscrowd': 0,
                })
                out_preds.append({
                    'image_id': int(image_id),
                    'category_id': int(category_id),
                    "segmentation": rle,
                    'score': 1.0
                })
                # self.instance_id += 1
        return out_images, out_annotations, out_preds

    @staticmethod
    def drop_arrays_by_name(gt_names, drop_classes):
        inds = [i for i, x in enumerate(gt_names) if x not in drop_classes]
        inds = np.array(inds, dtype=np.int64)
        return inds

    def class_name_to_label_index(self, gt_names):
        gt_labels = []
        for cat in gt_names:
            if cat in self.CLASSES:
                gt_labels.append(self.CLASSES.index(cat))
            else:
                gt_labels.append(-1)
        return np.array(gt_labels).astype(np.int64)
    
    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Return:
            list[str]: A list of class names.
        """
        if classes is None:
            return cls.CLASSES

        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    # index_filename = '/root/3D/work_dirs/dataset_infos/training_infos_panseg.pkl'
    index_filename = '/root/3D/work_dirs/dataset_infos/validation_infos_panseg.pkl'
    # json_path = '/root/3D/work_dirs/dataset_infos/training_instance_infos_front.json'
    json_path = '/root/3D/work_dirs/dataset_infos/validation_instance_infos_lwsis_299.json'
    # result_path = '/root/3D/work_dirs/dataset_infos/training_pred_results_front.json'
    result_path = '/root/3D/work_dirs/dataset_infos/validation_pred_results_299.json'
    # collection = 'training/infos'
    collection = 'validation/infos'

    # index_filename = '/home/jiangguangfeng/桌面/codebase/validation_infos_panseg.pkl'
    # json_path = '/home/jiangguangfeng/桌面/codebase/validation_instance_infos_lwsis_299.json'

    ConvertToCOCO(workers=16,
                  json_path=json_path,
                  index_filename=index_filename,
                  result_path=result_path,
                  collection=collection).get_all()
    # annType = ['segm','bbox','keypoints']
    # annType = annType[0]  # specify type here
    # anno_json = json_path
    # pred_json = result_path
    # cocoGt = COCO(anno_json)
    # cocoDt = cocoGt.loadRes(pred_json)
    # cocoEval = COCOeval(cocoGt,cocoDt,annType)
    # # cocoEval.params.imgIds  = imgIds
    # cocoEval.evaluate()
    # cocoEval.accumulate()
    # cocoEval.summarize()

# with open('training_panseg_instance.json','w') as f:
#     json.dump(dataset_dict, f, indent=4, ensure_ascii=False, cls=MyEncoder)

# annType = ['segm','bbox','keypoints']
# annType = annType[1]  # specify type here

# # gt  = [{"image_id":231,"category_id":0,"segmentation":{"size":[427,640],"counts":"UUl0o0\\<01O0O10000000000001O1O000000000000001O00000000000001O00000000i[Q7"},"score":0.097}]

# # json_str = json.dumps(pred_test, ensure_ascii=False, cls=MyEncoder, indent=4)
# # with open('pred_results.json','w') as json_file:
# #     json_file.write(json_str)
# with open('pred_results.json','w') as f:
#     json.dump(pred_test, f, indent=4, ensure_ascii=False, cls=MyEncoder)

# anno_json = r'/home/jiangguangfeng/桌面/codebase/training_panseg_instance.json'
# pred_json = r'/home/jiangguangfeng/桌面/codebase/pred_results.json'
# cocoGt = COCO(anno_json)
# cocoDt = cocoGt.loadRes(pred_json)

# cocoEval = COCOeval(cocoGt,cocoDt,annType)
# cocoEval.params.imgIds  = imgIds
# cocoEval.evaluate()
# cocoEval.accumulate()
# cocoEval.summarize()

