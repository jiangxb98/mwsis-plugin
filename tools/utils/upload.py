import io
import time
import minio
import pymongo
from io import BytesIO
import numpy as np
import numpy as np
import mmcv
import os

class UploadLabel:
    def __init__(self,
                bucket,    # minio
                new_bucket,
                database,  # mongodb
                minio_cfg,
                mongo_cfg,
                workers=4,
                ):
        self.bucket = bucket
        self.new_bucket = new_bucket
        self.database = database
        self.minio_cfg = minio_cfg
        self.mongo_cfg = mongo_cfg
        self.workers = workers
        self.points_idx = self.query_index()

    def upload(self):

        new_client = minio.Minio(**self.minio_cfg)
        found = new_client.bucket_exists(self.new_bucket)
        if not found:
            new_client.make_bucket(self.new_bucket)

        if self.workers == 0:
            result = []
            for idx in mmcv.track_iter_progress(range(len(self.points_idx))):
                result.append(self.upload_one(idx))
        else:
            result = mmcv.track_parallel_progress(self.upload_one,
                                                  range(len(self.points_idx)),
                                                  self.workers)
        print('上传完成')
        print('\n上传地址:{}'.format(self.new_bucket))

    def upload_one(self, ind):
        # load points
        minio_client = minio.Minio(**self.minio_cfg)

        upload_results = images  # 本地读取图片

        upload_path = 'nuscenes/nuimages'

        while True:
            try:
                buf = io.BytesIO()
                np.save(buf, upload_results)
                buf.seek(0, 2)
                buf_size = buf.tell()
                buf.seek(0, 0)
                minio_client.put_object(self.new_bucket, upload_path, buf, buf_size)
            except Exception as e:
                print(e)
                time.sleep(0.1)
                continue
            break
        return ind

    def exists(self, file_client, path):
        try:
            file_client.stat_object(self.new_bucket, path)
            return True
        except:
            return False

        return c_inds
if __name__=='__main__':
    bucket = 'ai-waymo-v1.4'
    new_bucket = 'ai-nuscenes-v1.0'
    database = 'ai-waymo-v1_4'
    workers = 4
    minio_cfg=dict(endpoint='ossapi.cowarobot.cn:9000',
            access_key='abcdef',
            secret_key='12345678',
            region='shjd-oss',
            secure=False,)
    mongo_cfg=dict(host="mongodb://root:root@172.16.110.100:27017/")
    UploadLabel(
        bucket=bucket,
        new_bucket=new_bucket,
        database=database,
        minio_cfg=minio_cfg,
        mongo_cfg=mongo_cfg,
        workers=workers,
    ).upload()