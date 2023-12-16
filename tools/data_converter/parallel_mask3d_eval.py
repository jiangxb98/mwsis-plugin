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
import cv2
from mmdet3d.core.points import get_points_type, BasePoints

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
        self.img_prefix = 'training/image/'
        self.car = [1,2,3,4,5,13]
        self.ped = [7]
        self.cyc = [6]
        self.semantic_label = [[1,2,3,4,5,13],[7],[6]]
        self.num_classes = 3
        self.instance_id = 1
        self.collection = collection
        self.index_filename = index_filename
        self.json_path = json_path
        self.result_path = result_path
        self.pcd_range = [-80, -80, -2, 80, 80, 4]
        self.coord_type = 'LIDAR'
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
            'description': 'Waymo Semseg',
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
        with open(self.json_path,'w') as f:
            json.dump(dataset_dict, f, indent=4, ensure_ascii=False, cls=MyEncoder)
        print('\n保存路径: {}'.format(self.json_path))
        with open(self.result_path,'w') as f:
            json.dump(pred_list, f, indent=4, ensure_ascii=False, cls=MyEncoder)

    def get_one(self, ind):
        _client = pymongo.MongoClient(**self.mongo_cfg)[self.database]
        client = minio.Minio(**self.minio_cfg)
        out_images = []
        out_annotations = []
        out_preds = []
        info = _client.get_collection(self.collection).find_one(ind)
        semseg_path = info['semseg_info']['path']
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
                            points_['cam_row_1'].astype('i2'),   # 6
                            ], axis=-1)
        points_class = get_points_type(self.coord_type)
        points = points_class(points, points_dim=points.shape[-1])  # 实例化，LiDARPoints
        points_mask = points.in_range_3d(self.pcd_range).numpy()  # 去除超过范围的点
        points = points.tensor.numpy()
        # 指定获取的点云
        in_front_mask = (points[:,4]!=-1) | (points[:,5]!=-1)
        top_mask = points[:,3]==0
        out_mask = top_mask & in_front_mask & points_mask
        # 获得语义标签
        annos_bytes = client.get_object(self.bucket, semseg_path).read()
        pts_annos = np.load(BytesIO(annos_bytes))
        semantic = pts_annos['semseg_cls'][out_mask].reshape(1,-1).astype(np.int8)
        instance = pts_annos['instance_id'][out_mask].reshape(1,-1).astype(np.int)
        # 获得图片id
        split_path = semseg_path.split('/')[-1]
        image_id = int(split_path)
        height, width = instance.shape
        file_name = image_id
        license = 'license'
        coco_url = 'waymo'
        data_captured = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        out_images.append({
            'id': image_id,
            'width': width,
            'height': height,
            'file_name': file_name,
            'license': license,
            'flickr_url': coco_url,
            'coco_url': coco_url,
            'date_captured': data_captured
        })

        tmp = np.zeros_like(semantic)
        for cls in range(len(self.semantic_label)):
            for cls_id in self.semantic_label[cls]:
                tmp += np.where(semantic==cls_id, cls+1, 0).astype(np.int8)
        new_semantic = tmp-1
        new_instance = instance * (new_semantic>=0)  # 过滤非目标类别实例
        instance_inds, counts = np.unique(new_instance, return_counts=True)
        # 将每个实例都存到annotation中
        for ind in range(len(instance_inds)):
            if instance_inds[ind] == 0 or instance_inds[ind] == -1:
                continue
            tmp_instance = (new_instance*(new_instance==instance_inds[ind])).astype(np.int)
            category_id = tmp*(tmp_instance==instance_inds[ind])
            if len(np.unique(category_id)) >= 3:  # 这里会出现三个或更多的instance，下面就是要区分开
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
                    # out_ = (tmp_instance_/tmp_instance_.max()).astype(np.uint8)
                    # out_ = mask.encode(out_)
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
                        'score': 0.5
                    })
                    # self.instance_id += 1
                continue
            category_id = category_id.max()-1
            tmp_instance = (tmp_instance/tmp_instance.max()).astype(np.uint8)
            tmp_instance = np.asfortranarray(tmp_instance)
            rle = mask.encode(tmp_instance)
            area = float(mask.area(rle))
            bbox = mask.toBbox(rle)
            # out_ = (tmp_instance/tmp_instance.max()).astype(np.uint8)
            # out_ = mask.encode(out_)
            out_annotations.append({
                'id': 0,  # 这里后面重新赋值，并行化存在问题，导致id不唯一
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
                'score': 0.5
            })
            self.instance_id += 1

        return out_images, out_annotations, out_preds

if __name__ == '__main__':
    # index_filename = '/root/3D/work_dirs/dataset_infos/training_infos_senseg.pkl'
    index_filename = '/root/3D/work_dirs/dataset_infos/validation_infos_semseg.pkl'
    # json_path = '/root/3D/work_dirs/dataset_infos/training_3dmask_infos_front.json'
    json_path = '/root/3D/work_dirs/dataset_infos/validation_3dmask_infos.json'
    # result_path = '/root/3D/work_dirs/dataset_infos/training_pred_results_front.json'
    result_path = '/root/3D/work_dirs/dataset_infos/validation_3dpred_results.json'
    # collection = 'training/infos'
    collection = 'validation/infos'
    ConvertToCOCO(workers=16,
                  json_path=json_path,
                  index_filename=index_filename,
                  result_path=result_path,
                  collection=collection).get_all()
    annType = ['segm','bbox','keypoints']
    annType = annType[0]  # specify type here
    anno_json = json_path
    pred_json = result_path
    cocoGt = COCO(anno_json)
    cocoDt = cocoGt.loadRes(pred_json)
    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    # cocoEval.params.imgIds  = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

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