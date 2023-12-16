# 参考如何多进程数据处理
import cv2
import sys
import time
import argparse
import numpy as np

from tqdm import tqdm
from loguru import logger
from multiprocessing import Pool, Manager
from minio_helper import MinioHelper
from shapely.geometry import Polygon
from ..utils.utils import mask2polygon

minio_helper = MinioHelper()
classes_dict = {
    1: ('自车', 'Ego Vehicle'),
    2: ('轿车', 'Car'),
    3: ('卡车', 'Truck'),
    4: ('公共汽车', 'Bus'),
    5: ('其他大型车辆', 'Other Large Vehicle'),
    6: ('自行车', 'Bicycle'),
    7: ('摩托车', 'Motorcycle'),
    8: ('挂车', 'Trailer'),
    9: ('行人', 'Pedestrian'),
    10: ('骑自行车的人', 'Cyclist'),
    11: ('摩托车手', 'Motorcyclist'),
    12: ('鸟', 'Bird'),
    13: ('地面动物', 'Ground Animal'),
    14: ('建筑锥', 'Construction Cone'),
    15: ('杆子', 'Pole'),
    16: ('行人物体', 'Pedestrian Object'),
    17: ('警示牌', 'Sign'),
    18: ('红绿灯', 'Traffic Light'),
    19: ('建筑物', 'Building'),
    20: ('道路', 'Road'),
    21: ('车道标记', 'Lane Marker'),
    22: ('路标', 'Road Marker'),
    23: ('人行道', 'Sidewalk'),
    24: ('植被', 'Vegetation'),
    25: ('天空', 'Sky'),
    26: ('地面', 'Ground'),
    27: ('其他动态目标', 'Dynamic'),
    28: ('其他静态目标', 'Static')
}

def convert_single_waymo_mask_to_polygon(mask, panseg_instance_id, classes_dict):
    # image = cv2.imread('./images/0_origin.jpg')
    # image = np.zeros_like(panseg_instance_id)
    class_labels = set(mask.flatten())
    instance_labels = set(panseg_instance_id.flatten())
    
    output_per_sample = []
    for class_label in class_labels:
        for instance_label in instance_labels:
            if class_label == 0 or instance_label == 0:
                continue
            polygons = mask2polygon(np.array((mask == class_label) & (panseg_instance_id == instance_label), dtype=np.uint8))
            if len(polygons) == 0:
                continue
            keep = []
            for i, polygon in enumerate(polygons):
                if len(polygon) < 3:
                    continue
                p = Polygon(polygon.reshape((-1, 2)))
                bbox = p.bounds
                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                area = p.area
                if area < 25 or h < 5 or w < 5:
                    # logger.info('area = {}'.format(area))
                    # logger.info('h = {}'.format(h))
                    # logger.info('w = {}'.format(w))
                    continue
                keep.append(polygon)
            if len(keep) == 0:
                continue
            if len(keep) < len(polygons):
                polygons = keep
            
            output_per_sample.append([classes_dict[class_label][1], [x.tolist() for x in polygons]])
            # for polygon in polygons:
            #     pts = polygon.reshape((-1, 1, 2))
                # cv2.polylines(image, [pts], True, 255, 2)
    return output_per_sample

# @logger.catch
# def func(i, line, classes_dict, num_left):
#     mask, panseg_instance_id = minio_helper.read_waymo_panseg_from_minio(line, bucket_name='ai-waymo-v1.4')
#     polygon_per_sample = convert_single_waymo_mask_to_polygon(mask, panseg_instance_id, classes_dict)
#     num_left.value -= 1
#     logger.info('Num of images left: {}                 \r'.format(num_left.value))
#     # sys.stdout.write('Num of images left: {}, output = {}                 \r'.format(num_left.value, len(output)))
#     # sys.stdout.flush()
#     return i, polygon_per_sample

@logger.catch
def func(i, line, classes_dict, output, num_left):
    mask, panseg_instance_id = minio_helper.read_waymo_panseg_from_minio(line, bucket_name='ai-waymo-v1.4')
    polygon_per_sample = convert_single_waymo_mask_to_polygon(mask, panseg_instance_id, classes_dict)
    output.append((i, polygon_per_sample))
    num_left.value -= 1
    # logger.info('Num of images left: {}, output = {}                 \r'.format(num_left.value, len(output)))
    sys.stdout.write('Num of images left: {}                 \r'.format(num_left.value))
    sys.stdout.flush()

def convert_waymo_mask_to_polygon(image_set_file):
    logger.info('Start')
    start = time.time()
    pool = Pool(100)
    manager = Manager()
    output = manager.list()
    # output = []
    with open(image_set_file, 'r') as f:
        lines = [x.split(' ')[-1] for x in f.read().splitlines()]
    logger.info('Num samples = {}'.format(len(lines)))
    # lines = lines[:200]
    num_left = manager.Value('d', len(lines))
    for i in tqdm(range(len(lines))):
        # output.append(pool.apply_async(func=func, args=(i, lines[i], classes_dict, num_left)))
        pool.apply_async(func=func, args=(i, lines[i], classes_dict, output, num_left))
        # func(i, lines[i], classes_dict, output, num_left)
    pool.close()
    pool.join()
    logger.info('Num output = {}'.format(len(output)))
    # output = [x.get()[0] for x in output]
    output = sorted(output, key=lambda x: x[0])
    output = [x[1] for x in output]
    # logger.info('output = {}'.format(output))
    logger.info('Finished. ({:.4f}s)'.format(time.time() - start))
    return np.array(output, dtype=object)

'''
python make_waymo_dataset.py --split train
python make_waymo_dataset.py --split valid
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.description = 'Convert waymo masks to polygons'
    parser.add_argument('--split', default='train', type=str, help='The data split to generate pollygon data')
    args = parser.parse_args()

    dataset_file = '/disk/deepdata/zjc_workspace/scripts/waymo_tools/waymo_{}_set.txt'.format(args.split)
    polygons = convert_waymo_mask_to_polygon(dataset_file)
    np.save('waymo_{}_label.npy'.format(args.split), polygons)

    # train_set_file = '/disk/deepdata/zjc_workspace/scripts/utils/waymo_train_set.txt'
    # valid_set_file = '/disk/deepdata/zjc_workspace/scripts/utils/waymo_valid_set.txt'
    
    # train_polygons = convert_waymo_mask_to_polygon(train_set_file)
    # np.save('waymo_train_label.npy', train_polygons)
    
    # valid_polygons = convert_waymo_mask_to_polygon(valid_set_file)
    # np.save('waymo_valid_label.npy', valid_polygons)