# <font size=5>MWSIS: Multimodal Weakly Supervised Instance Segmentation with 2D Box Annotations for Autonomous Driving</font>

## <font size=4.5>Get the paper [arxiv](https://arxiv.org/abs/2312.06988).</font>

## <font size=4.5>ToDo</font>

- [x] Release the MWSIS-pts code.

- [ ] Improve data processing
- [ ] Reorganize the current code (pts).
- [ ] Replace the CCL.
- [ ] Release the MWSIS-img code which is same as the MWSIS-pts.

## <font size=4.5>News</font>

- [23-12-16] The code for MWSIS-pts is released here.

## <font size=4.5>Instructions for MWSIS.</font>

```bash
# python version is 3.7

# install torch
pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111

# install mmdet3d==1.0.0rc5 dependency
pip install mmcv-full==1.6.0 mmsegmentation==0.27.0 mmdet==2.25.1 spconv-cu111 open3d minio==7.1.11 pymongo==3.12.3 waymo-open-dataset-tf-2-6-0==1.4.9

# install pairwise_loss
cd boxinst-plugin-main
pip install -e .
mv boxinst_plugin/ops/ /mwsis_plugin/ops

# install torchex https://github.com/tusen-ai/SST
cd torchex
pip install -e .

# install torch scatter
pip install torch-scatter==2.1.0

# install easydict
pip install easydict

# download the resnet50 checkpoint
wget https://download.openmmlab.com/pretrain/third_party/resnet50_msra-5891d200.pth
mv .pth .cache/torch/hub/checkpoints/
```

## <font size=4.5>Data convert.</font>

**Train data**

Please follow the tutorial given by [mmdet3d](https://github.com/open-mmlab/mmdetection3d) to process the data, and then use our [waymo conversion code](./tools/data_converter/waymo_converter_local.py) to process the data. (Note: You should mv `./mwsis_plugin/tools/data_converter/waymo_converter_local.py` `./tools/data_converter`)

**Val data**

Please use `mwsis_plugin/tools/data_converter/parallel_mask3d_eval_local.py`

## <font size=4.5>How to train MWSIS.</font>

**Note:**

+ **About CCL**

  Please use funcition `connected_components()` to replace function `voxel_spccl3d()`. Later, we will change the CCL operator to the operator in the torchex library. Currently, sicpy library is used instead.



```bash
# signle gpu
python tools/train.py mwsis/config/mwsis_pts_model_config.py --work-dir <your filepath>

# multi gpu
bash tools/dist_train.sh ...
```

![image-20231203171901541](./images/framwork.png)



## <font size=4.5>Acknowledgments</font>

This project is based on the following codebases.  

* [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)
* [SST](https://github.com/tusen-ai/SST)
