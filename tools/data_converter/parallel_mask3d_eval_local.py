import json
import numpy as np
import torch
from pycocotools import mask
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import numpy as np
from io import BytesIO
import time
import mmcv
import os
from datetime import datetime
from tqdm import tqdm
import pickle
import cv2
from mmdet3d.core.points import get_points_type

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf-8');
        return json.JSONEncoder.default(self, obj)

class ConvertToCOCO:
    def __init__(self,
                 workers=1,
                 index_filename=None,
                 json_path=None,
                 result_path=None,
                 panseg_label_path= None,
                 frame_infos_path=None):
        self.workers = workers
        self.pred_test = []
        self.car = [1, 2, 3, 4, 5, 13]
        self.ped = [7]
        self.cyc = [6]
        self.semantic_label = [[1 ,2, 3, 4, 5, 13], [7], [6]]
        self.num_classes = 3
        self.instance_id = 1
        self.result_path = result_path
        self.pcd_range = [-80, -80, -2, 80, 80, 4]
        self.coord_type = 'LIDAR'
        self.panseg_label_path = panseg_label_path
        self.frame_infos_path = frame_infos_path
        self.index_filename = index_filename
        self.json_path = json_path
        self.file_client_args = dict(backend='disk')
        self.file_client = None

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

    def load_index(self):
        f = open(self.index_filename, "rb+")
        ret = pickle.load(f)
        f.close()
        return ret

    def get_all(self):
        # json format
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
        
        print("Finished load")

        for image, annotations, pred in mmcv.track_iter_progress(results):
            dataset_dict['images'].extend(image)
            dataset_dict['annotations'].extend(annotations)
            pred_list.extend(pred)
        for i in range(len(dataset_dict['annotations'])):
            dataset_dict['annotations'][i]['id'] = int(i+1)
        with open(self.json_path,'w') as f:
            json.dump(dataset_dict, f, indent=4, ensure_ascii=False, cls=MyEncoder)
        print('\nSave path: {}'.format(self.json_path))
        with open(self.result_path,'w') as f:
            json.dump(pred_list, f, indent=4, ensure_ascii=False, cls=MyEncoder)

    def load_local_pkl(self, index_path):
        with open(index_path, 'rb') as f:
            index = pickle.load(f)
        index.sort()
        return index

    def load_local_infos(self, info_path, index):
        info_path = os.path.join(info_path, str(index).zfill(7)+'.npy')
        while True:
            try:
                data_infos = np.load(info_path, allow_pickle=True)
            except:
                time.sleep(0.1)
                continue
            return data_infos[()]  # Dict

    def get_one(self, index):
        out_images = []
        out_annotations = []
        out_preds = []

        info = self.load_local_infos(self.frame_infos_path, index)

        semseg_path_old = info['semseg_info']['path']
        pts_path = info['pts_info']['path']

        # obtain point cloud
        points = np.fromfile(pts_path, dtype=np.float32, count=-1).reshape([-1, 16])
        # have a bug need to fix
        # There is a problem with the order of the point cloud array
        points[:, 3] = np.tanh(points[:, 3])
        points[:, 5:8] = points[:, [6, 7, 5]]
        
        points_class = get_points_type(self.coord_type)
        points = points_class(points, points_dim=points.shape[-1])
        points_mask = points.in_range_3d(self.pcd_range).numpy()  # Remove points that are out of range

        points = points.tensor.numpy()

        # Get the point cloud that falls on the image and the top LiDAR
        in_front_mask = (points[:, 10]!=-1) | (points[:, 11]!=-1)
        top_mask = points[:, 6] == 0
        out_mask = top_mask & in_front_mask & points_mask

        # Get the panseg label
        if not semseg_path_old.endswith('.npy'):
            semseg_path = semseg_path_old + '.npy'

        annos_bytes = self.file_client.get(semseg_path)
        pts_annos = np.load(BytesIO(annos_bytes))
        semantic = pts_annos['semseg_cls'][out_mask].reshape(1, -1).astype(np.int8)
        instance = pts_annos['instance_id'][out_mask].reshape(1, -1).astype(np.int)
        # Get the image id
        split_path = semseg_path_old.split('/')[-1]
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
    index_filename = 'data/waymo/kitti_format/training/pts_panseg_index_val.pkl'
    json_path = './data/waymo/kitti_format/validation_3dmask_infos.json'
    result_path = './data/waymo/kitti_format/validation_3dpred_results.json'
    panseg_label_path = './data/waymo/kitti_format/training/lidar_panseg_label'
    frame_infos_path = './data/waymo/kitti_format/training/frame_infos'

    ConvertToCOCO(workers=8,
                  json_path=json_path,
                  index_filename=index_filename,
                  result_path=result_path,
                  panseg_label_path= panseg_label_path,
                  frame_infos_path=frame_infos_path).get_all()
    
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
