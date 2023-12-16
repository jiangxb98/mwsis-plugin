import mmcv
import torch
import pickle
import argparse
import numpy as np
import open3d as o3d
import os.path as osp
import matplotlib.pyplot as plt
import pickle
from mmcv import Config
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.apis import init_model
from mmdet3d.models import build_model
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmcv.image import tensor2imgs
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--pth', help='train checkpoint path', default=None)
    parser.add_argument('--out', help='inference result', default=None)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    dataset = build_dataset(cfg.data.test)
    results_list = []

    if args.pth:
        model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
        checkpoint = load_checkpoint(model, args.pth, map_location='cpu')
        # model = init_model(cfg, args.pth)
        model = revert_sync_batchnorm(model)
        model = MMDataParallel(model, device_ids=[0])
        model.eval()
        loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=0,
            dist=False,
            shuffle=False)
        progress_bar = mmcv.ProgressBar(len(dataset))
        for i, data in enumerate(loader):
            with torch.no_grad():
                results = model(return_loss=False, rescale=True, **data)
            out = []
            for key, val in data.items():
                out.append({key: val[0]._data[0][0]})
            for key, val in results[0].items():
                out.append({key: val[0]})
            results_list.append(out)
            progress_bar.update()
            if i == 100:
                break

        filename = "/home/jiangguangfeng/桌面/work_dirs/results/ema_debug_epoch_24_results.pkl"
        print("save results to {}".format(filename))
        f = open(filename, "wb")
        pickle.dump(results_list)
        f.close()
        # f = open(filename, "rb+")
        # data = pickle.load(f)
        # f.close()
if __name__ == '__main__':
    main()