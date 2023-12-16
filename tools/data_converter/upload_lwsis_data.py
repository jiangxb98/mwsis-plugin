# 配合clean_lwsis_data.py转一下命名方式，然后上传到minio
import io
import time
import minio
import pymongo
import json
from io import BytesIO
import numpy as np
import pickle
from tqdm import tqdm
import open3d as o3d
import matplotlib.pyplot as plt
from mmdet3d.core.points import get_points_type, BasePoints
import numpy as np
import mmcv
from pycocotools import mask as mask_utils

class UploadLabel:
    def __init__(self,
                bucket,  # minio
                new_bucket,
                database,# mongodb
                minio_cfg,
                mongo_cfg,
                workers=4,
                collection='training/infos',
                lwsis_path='/root/3D/work_dirs/dataset_infos/waymo_lwsis_train1.1_my.json'
                ):
        self.bucket = bucket
        self.new_bucket = new_bucket
        self.database = database
        self.minio_cfg = minio_cfg
        self.mongo_cfg = mongo_cfg
        self.workers = workers
        self.collection = collection
        self.points_idx = np.array(self.query_index()).astype(np.int)
        self.records = np.unique((self.points_idx//1000).astype(np.int))[18:]

        # f = open(lwsis_path, 'r')
        # content = f.read()
        # self.lwsis_data = json.loads(content)
        # # lwsis_val_data_filename = [a['file_name'].split('/')[0] for a in self.lwsis_val_data['images']]
        # f.close()

        from pycocotools.coco import COCO
        self.coco_api = COCO(lwsis_path)

    def upload(self):
        new_client = minio.Minio(**self.minio_cfg)
        found = new_client.bucket_exists(self.new_bucket)

        if not found:
            raise ValueError("不可新建")
            new_client.make_bucket(self.new_bucket)

        self.img_ids = self.coco_api.getImgIds()

        if self.workers == 0:
            result = []
            for idx in mmcv.track_iter_progress(range(len(self.img_ids))):
                result.append(self.upload_one(idx))
        else:
            result = mmcv.track_parallel_progress(self.upload_one,
                                                  range(len(self.img_ids)),
                                                  self.workers)
        print('上传完成')
        print('\n上传地址:{}'.format(self.new_bucket))
        
    def upload_one(self, idx):
        mongo_client = pymongo.MongoClient(**self.mongo_cfg)[self.database]
        minio_client = minio.Minio(**self.minio_cfg)

        img_id = self.img_ids[idx]

        ann_ids = self.coco_api.getAnnIds(imgIds=img_id)
        upload_annos = []
        for j, ann_id in enumerate(ann_ids):
            anns = self.coco_api.loadAnns(ann_id)
            upload_annos.append(anns)
        img_path = "{}/{}/{}".format(collection.split('/')[0], "img", img_id)
        while True:
            try:
                buf = io.BytesIO()
                np.save(buf, upload_annos)
                buf.seek(0, 2)
                buf_size = buf.tell()
                buf.seek(0, 0)
                minio_client.put_object(self.new_bucket, img_path, buf, buf_size)
            except Exception as e:
                print(e)
                time.sleep(0.1)
                continue
            break     

    def query_index(self, filter=dict(), filter_sem=None):
        _client = pymongo.MongoClient(**self.mongo_cfg)[self.database]
        while True:
            try:
                if filter_sem is None:
                    # ret = [o['_id'] for o in _client.get_collection(self.collection).find(filter, projection=[])]
                    if 'training' in self.collection:
                        f = open('/root/3D/work_dirs/dataset_infos/training_infos_panseg.pkl', "rb+")
                    elif 'validation' in self.collection:
                        f = open('/root/3D/work_dirs/dataset_infos/validation_infos_panseg.pkl', "rb+")
                    ret = pickle.load(f)
                    f.close()
                else:
                    raise ValueError("filter_sem need None")
                    ret = [o['_id'] for o in _client.get_collection(self.collection).find() if filter_sem in o.keys()]
            except Exception as e:
                print(e)
                time.sleep(0.1)
                continue
            return ret

if __name__=='__main__':
    # torch.multiprocessing.set_start_method('spawn')
    bucket = 'ai-waymo-v1.4'
    new_bucket = 'ai-waymo-v1.4-pts-pseudo-labels'
    database = 'ai-waymo-v1_4'
    workers = 32
    collection = 'training/infos'
    minio_cfg = dict(endpoint='ossapi.cowarobot.cn:9000',
            access_key='abcdef',
            secret_key='12345678',
            region='shjd-oss',
            secure=False,)
    mongo_cfg = dict(host="mongodb://root:root@172.16.110.100:27017/")
    lwsis_path = '/root/3D/work_dirs/dataset_infos/waymo_lwsis_train1.1_my.json'
    UploadLabel(
        bucket=bucket,
        new_bucket=new_bucket,
        database=database,
        minio_cfg=minio_cfg,
        mongo_cfg=mongo_cfg,
        workers=workers,
        collection=collection,
        lwsis_path=lwsis_path,
    ).upload()