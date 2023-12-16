# 将lwsis的waymo数据转为自己的命名方式（image_id命名方式不同）
import json
import mmcv
import numpy as np

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf-8');
        return json.JSONEncoder.default(self, obj)


f = open('/root/3D/work_dirs/dataset_infos/waymo_lwsis_train1.1_my.json', 'r')
content = f.read()
lwsis_val_data = json.loads(content)
lwsis_val_data_filename = [a['file_name'].split('/')[0] for a in lwsis_val_data['images']]
print(type(lwsis_val_data))
f.close()
f2 = open('/root/3D/work_dirs/dataset_infos/training_instance_infos_lwsis_299.json', 'r')
content = f2.read()
b = json.loads(content)
b_filename = [bb['file_name'].split('/')[0] for bb in b['images']]
print(type(b))
f2.close()

# compare filename
lwsis_val_data_filename = sorted(list(set(lwsis_val_data_filename)))
b_filename = sorted(list(set(b_filename)))
assert b_filename == lwsis_val_data_filename
diff_nums = 0
for i in range(len(b_filename)):
    if b_filename[i] != lwsis_val_data_filename[i]:
        diff_nums += 1
print("文件名不同的数量{}".format(diff_nums))
    

lwsis_map = {}
lwsis_images = lwsis_val_data['images']
for i in range(len(lwsis_images)):
    lwsis_map[lwsis_images[i]['id']] = lwsis_images[i]['file_name']
my_map = {}
my_images = b['images']
for i in range(len(my_images)):
    my_map[my_images[i]['file_name']] = my_images[i]['id']

for i in range(len(lwsis_images)):
    lwsis_images[i]['id'] = my_map[lwsis_images[i]['file_name']]

lwsis_annotations = lwsis_val_data['annotations']
for i in range(len(lwsis_annotations)):
    lwsis_annotations[i]['image_id'] = my_map[lwsis_map.get(lwsis_annotations[i]['image_id'])]

# json_path = '/root/3D/work_dirs/dataset_infos/waymo_lwsis_train1.1_my.json'
# a = lwsis_val_data.copy()
# for i in range(len(a['annotations'])):
#     # point_coords point_labels 3din_coords 3dout_coords
#     del a['annotations'][i]['point_coords']
#     del a['annotations'][i]['point_labels']
#     del a['annotations'][i]['3din_coords']
#     del a['annotations'][i]['3dout_coords']

# print("saving")
# with open(json_path, 'w') as f:
#     json.dump(a, f, indent=4, ensure_ascii=False, cls=MyEncoder)
# print("saved")
