import random
from mmcv.fileio import FileClient, BaseStorageBackend
import time
import os

@FileClient.register_backend('MyMONGODB')
class MONGODBBackend(BaseStorageBackend):
    def __init__(self, database, path_mapping=None, scope=None,
                 **mongodb_cfg):
        self.database = database
        try:
            import pymongo
        except ImportError:
            raise ImportError(
                'Please install pymongo to enable MONGODBBackend.')
        self._client = pymongo.MongoClient(**mongodb_cfg)[database]
        assert isinstance(path_mapping, dict) or path_mapping is None
        self.path_mapping = path_mapping
        self.collections = dict()

    def get_collection(self, name):
        if name in self.collections:
            return self.collections[name]
        collection = self._client.get_collection(name)
        self.collections[name] = collection
        return collection

    def get(self, arg):
        collection, index = arg
        while True:
            try:
                ret = self.get_collection(collection).find_one(index)
            except Exception as e:
                print(e)
                time.sleep(0.1)
                continue
            return ret

    def query(self, collection, filter=dict(), projection=[]):
        while True:
            try:
                ret = [*self.get_collection(collection).find(
                    filter, projection=projection)]
            except Exception as e:
                print(e)
                time.sleep(0.1)
                continue
            return ret

    def query_index(self, collection, filter=dict(), filter_sem=None, test_model=False):
        while True:
            try:
                if filter_sem is None:
                    ret = [o['_id'] for o in self.get_collection(collection).find(filter, projection=[])]
                elif filter_sem == 'panseg_info':
                    is_local = True
                    # ret = [o['_id'] for o in self.get_collection(collection).find() if filter_sem in o.keys()]
                    if test_model:
                        import pickle
                        if os.path.exists('/root/3D/work_dirs/dataset_infos/validation_infos_panseg.pkl'):
                            f = open('/root/3D/work_dirs/dataset_infos/validation_infos_panseg.pkl', "rb+")
                        elif os.path.exists('/home/jiangguangfeng/桌面/codebase/validation_infos_panseg.pkl'):
                            f = open('/home/jiangguangfeng/桌面/codebase/validation_infos_panseg.pkl', "rb+")
                        else:
                            ret = [o['_id'] for o in self.get_collection(collection).find() if filter_sem in o.keys()]
                            is_local = False
                    else:
                        import pickle
                        if os.path.exists('/root/3D/work_dirs/dataset_infos/training_infos_panseg.pkl'):
                            f = open('/root/3D/work_dirs/dataset_infos/training_infos_panseg.pkl', "rb+")
                        elif os.path.exists('/home/jiangguangfeng/桌面/codebase/training_infos_panseg.pkl'):
                            f = open('/home/jiangguangfeng/桌面/codebase/training_infos_panseg.pkl', "rb+")
                        else:
                            ret = [o['_id'] for o in self.get_collection(collection).find() if filter_sem in o.keys()]
                            is_local = False
                    if is_local:
                        ret = pickle.load(f)
                        f.close()
                    print("--------load {} dataset index------".format(filter_sem))
                else:
                    is_local = True
                    # ret = [o['_id'] for o in self.get_collection(collection).find() if filter_sem in o.keys()]
                    if test_model:
                        import pickle
                        if os.path.exists('/root/3D/work_dirs/dataset_infos/validation_infos_semseg.pkl'):
                            f = open('/root/3D/work_dirs/dataset_infos/validation_infos_semseg.pkl', "rb+")
                        elif os.path.exists('/home/jiangguangfeng/桌面/codebase/validation_infos_semseg.pkl'):
                            f = open('/home/jiangguangfeng/桌面/codebase/validation_infos_semseg.pkl', "rb+")
                        else:
                            ret = [o['_id'] for o in self.get_collection(collection).find() if filter_sem in o.keys()]
                            is_local = False
                    else:
                        import pickle
                        if os.path.exists('/root/3D/work_dirs/dataset_infos/training_infos_semseg.pkl'):
                            f = open('/root/3D/work_dirs/dataset_infos/training_infos_semseg.pkl', "rb+")
                        elif os.path.exists('/home/jiangguangfeng/桌面/codebase/training_infos_semseg.pkl'):
                            f = open('/home/jiangguangfeng/桌面/codebase/training_infos_semseg.pkl', "rb+")
                        else:
                            ret = [o['_id'] for o in self.get_collection(collection).find() if filter_sem in o.keys()]
                    if is_local:
                        ret = pickle.load(f)
                        f.close()
                    print("--------load {} dataset index------".format(filter_sem))
            except Exception as e:
                print(e)
                time.sleep(0.1)
                continue
            # cyc sample
            if test_model:
                pass
            else:
                if filter_sem == 'panseg_info':
                    # training
                    import pickle
                    if os.path.exists('/root/3D/work_dirs/dataset_infos/cyc_frame_id.pkl'):
                        f = open('/root/3D/work_dirs/dataset_infos/cyc_frame_id.pkl', "rb+")
                    elif os.path.exists('/home/jiangguangfeng/桌面/codebase/cyc_frame_id.pkl'):
                        f = open('/home/jiangguangfeng/桌面/codebase/cyc_frame_id.pkl', "rb+")
                    cyc_ret = pickle.load(f)
                    cyc_ret = list(set(ret).intersection(cyc_ret))
                    cyc_ret = cyc_ret[::4]
                    res_ret = list(set(ret).difference(cyc_ret))
                    # ret = random.sample(res_ret, 128)
                    # ret = res_ret[::15] + cyc_ret + res_ret[::847]
                    # ret = random.sample(res_ret[::40], 297) + cyc_ret
                    # ret = ret[::2]
                else:
                    # training
                    import pickle
                    if os.path.exists('/root/3D/work_dirs/dataset_infos/cyc_frame_id.pkl'):
                        f = open('/root/3D/work_dirs/dataset_infos/cyc_frame_id.pkl', "rb+")
                    elif os.path.exists('/home/jiangguangfeng/桌面/codebase/cyc_frame_id.pkl'):
                        f = open('/home/jiangguangfeng/桌面/codebase/cyc_frame_id.pkl', "rb+")
                    cyc_ret = pickle.load(f)
                    cyc_ret = list(set(ret).intersection(cyc_ret))
                    cyc_ret = cyc_ret[::10]
                    res_ret = list(set(ret).difference(cyc_ret))
                    # ret = random.sample(res_ret[::10], 2310) + cyc_ret # 196 23495
                    # ret = res_ret[::108] + cyc_ret # 196 23495
                #     print("--------training load cyc dataset index------")
                    # ret = ret + cyc_ret*10
            return [1030148, 1108025,1150054,1155074,1178065] #[1098029,1108045,1108075,1108155,1155084,1155114] # sorted(random.sample(ret, 500)) ret[881:883] 4900 行人 车3262 [16090,24139] 周视图bug[751177,50138,693079,267021] [500::100]

    def get_text(self, filepath, encoding=None):
        raise NotImplementedError

    def list_dir_or_file(self, dir_path, list_dir, list_file, suffix,
                         recursive):
        raise NotImplementedError
