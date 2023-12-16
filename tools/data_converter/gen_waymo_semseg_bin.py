import os
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import numpy as np

tf.enable_eager_execution()

from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.protos import segmentation_metrics_pb2
from waymo_open_dataset.protos import segmentation_submission_pb2


import zlib


def convert_range_image_to_point_cloud_semseg(frame,
                                              range_images,
                                              segmentation_labels,
                                              ri_index=0):
    """Convert segmentation labels from range images to point clouds.
    Args:
        frame: open dataset frame
        range_images: A dict of {laser_name, [range_image_first_return,
        range_image_second_return]}.
        segmentation_labels: A dict of {laser_name, [range_image_first_return,
        range_image_second_return]}.
        ri_index: 0 for the first return, 1 for the second return.
    Returns:
        point_labels: {[N, 2]} list of 3d lidar points's segmentation labels. 0 for
        points that are not labeled.
    """
    filter_no_label_zone_points = True
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    point_labels = []
    for c in calibrations:
        ## only open_dataset.LaserName.TOP
        if c.name != open_dataset.LaserName.TOP:
            continue

        range_image = range_images[c.name][ri_index]
        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(range_image.data), range_image.shape.dims)
        range_image_mask = range_image_tensor[..., 0] > 0

        if filter_no_label_zone_points:
            nlz_mask = range_image_tensor[..., 3] != 1.0  # 1.0: in NLZ
            range_image_mask = range_image_mask & nlz_mask

        if c.name in segmentation_labels:
            sl = segmentation_labels[c.name][ri_index]
            sl_tensor = tf.reshape(tf.convert_to_tensor(sl.data), sl.shape.dims)
            sl_points_tensor = tf.gather_nd(sl_tensor, tf.where(range_image_mask))
        else:
            num_valid_point = tf.math.reduce_sum(tf.cast(range_image_mask, tf.int32))
            sl_points_tensor = tf.zeros([num_valid_point, 2], dtype=tf.int32)

        point_labels.append(sl_points_tensor.numpy())
    return point_labels


def compress_array(array: np.ndarray, is_int32: bool = False):
  """Compress a numpy array to ZLIP compressed serialized MatrixFloat/Int32.

  Args:
    array: A numpy array.
    is_int32: If true, use MatrixInt32, otherwise use MatrixFloat.

  Returns:
    The compressed bytes.
  """
  if is_int32:
    m = open_dataset.MatrixInt32()
  else:
    m = open_dataset.MatrixFloat()
  m.shape.dims.extend(list(array.shape))
  m.data.extend(array.reshape([-1]).tolist())
  return zlib.compress(m.SerializeToString())

def decompress_array(array_compressed: bytes, is_int32: bool = False):
  """Decompress bytes (of serialized MatrixFloat/Int32) to a numpy array.

  Args:
    array_compressed: bytes.
    is_int32: If true, use MatrixInt32, otherwise use MatrixFloat.

  Returns:
    The decompressed numpy array.
  """
  decompressed = zlib.decompress(array_compressed)
  if is_int32:
    m = open_dataset.MatrixInt32()
    dtype = np.int32
  else:
    m = open_dataset.MatrixFloat()
    dtype = np.float32
  m.ParseFromString(decompressed)
  return np.array(m.data, dtype=dtype).reshape(m.shape.dims)


TOP_LIDAR_ROW_NUM = 64
TOP_LIDAR_COL_NUM = 2650


def get_range_image_point_indexing(range_images, ri_index=0):
  """Get the indices of the valid points (of the TOP lidar) in the range image.

  The order of the points match those from convert_range_image_to_point_cloud
  and convert_range_image_to_point_cloud_labels.

  Args:
    range_images: A dict of {laser_name, [range_image_first_return,
       range_image_second_return]}.
    ri_index: 0 for the first return, 1 for the second return.

  Returns:
    points_indexing_top: (N, 2) col and row indices of the points in the
      TOP lidar.
  """
  points_indexing_top = None
  xgrid, ygrid = np.meshgrid(range(TOP_LIDAR_COL_NUM), range(TOP_LIDAR_ROW_NUM))
  col_row_inds_top = np.stack([xgrid, ygrid], axis=-1)
  range_image = range_images[open_dataset.LaserName.TOP][ri_index]
  range_image_tensor = tf.reshape(
      tf.convert_to_tensor(range_image.data), range_image.shape.dims)
  range_image_mask = range_image_tensor[..., 0] > 0
  if True:   # filter_no_label_zone_points
    nlz_mask = range_image_tensor[..., 3] != 1.0  # 1.0: in NLZ
    range_image_mask = range_image_mask & nlz_mask
  points_indexing_top = col_row_inds_top[np.where(range_image_mask)]
  return points_indexing_top


def dummy_semseg_for_one_frame(frame, dummy_class=14):
  """Assign all valid points to a single dummy class.

  Args:
    frame: An Open Dataset Frame proto.
    dummy_class: The class to assign to. Default is 14 (building).

  Returns:
    segmentation_frame: A SegmentationFrame proto.
  """
  (range_images, camera_projections, segmentation_labels,
   range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(
       frame)
  # Get the col, row indices of the valid points.
  points_indexing_top = get_range_image_point_indexing(range_images, ri_index=0)
  points_indexing_top_ri2 = get_range_image_point_indexing(
      range_images, ri_index=1)
  
  # get seg label
  point_labels = convert_range_image_to_point_cloud_semseg(
      frame, range_images, segmentation_labels)
  point_labels_ri2 = convert_range_image_to_point_cloud_semseg(
      frame, range_images, segmentation_labels, ri_index=1)

  # Point labels.
  # [:, 1] -> for semantic segmentation; [:, 0] -> for instance segmentation -1 is 植被建筑物等
  point_labels_all = np.concatenate(point_labels, axis=0)
  point_labels_all_ri2 = np.concatenate(point_labels_ri2, axis=0)
  if point_labels_all.shape[0] != points_indexing_top.shape[0]:
    import pdb;pdb.set_trace()

  tmp_point_labels_all = np.copy(point_labels_all)
  point_labels_all[:, 1] = 0
  need_cls = [[1,2,3,4,5],[7],[6]]  # car ped cyc
  # 过滤掉不相关的点 [[1,2,3,4,5],[7],[6]]  # car ped cyc
  for i in range(len(need_cls)):
    mask_cls = np.zeros_like(tmp_point_labels_all[:,1]).astype(np.bool)
    for j in range(len(need_cls[i])):
      tmp_mask = point_labels_all[:,1] == need_cls[i][j]
      mask_cls = mask_cls | tmp_mask
    point_labels_all[:,1][mask_cls] = i   # 语义 1 is car ped=2
    point_labels_all[:,0][~mask_cls] = 0  # 实例

  tmp_point_labels_all_ri2 = np.copy(point_labels_all_ri2)
  point_labels_all_ri2[:, 1] = 0
  for i in range(len(need_cls)):
    mask_cls = np.zeros_like(tmp_point_labels_all_ri2[:,1]).astype(np.bool)
    for j in range(len(need_cls[i])):
      tmp_mask = point_labels_all_ri2[:,1] == need_cls[i][j]
      mask_cls = mask_cls | tmp_mask
    point_labels_all_ri2[:,1][mask_cls] = i   # 语义
    point_labels_all_ri2[:,0][~mask_cls] = 0  # 实例

  # semseg_labels['semseg_cls'] = point_labels_all[:, 1]

  # Assign the dummy class to all valid points (in the range image)
  range_image_pred = np.zeros(
      (TOP_LIDAR_ROW_NUM, TOP_LIDAR_COL_NUM, 2), dtype=np.int32)
  range_image_pred[points_indexing_top[:, 1],
                   points_indexing_top[:, 0], 1] = point_labels_all[:, 1]
  range_image_pred_ri2 = np.zeros(
      (TOP_LIDAR_ROW_NUM, TOP_LIDAR_COL_NUM, 2), dtype=np.int32)
  range_image_pred_ri2[points_indexing_top_ri2[:, 1],
                       points_indexing_top_ri2[:, 0], 1] = point_labels_all_ri2[:, 1]
  # Construct the SegmentationFrame proto.
  segmentation_frame = segmentation_metrics_pb2.SegmentationFrame()
  segmentation_frame.context_name = frame.context.name
  segmentation_frame.frame_timestamp_micros = frame.timestamp_micros
  laser_semseg = open_dataset.Laser()
  laser_semseg.name = open_dataset.LaserName.TOP
  laser_semseg.ri_return1.segmentation_label_compressed = compress_array(
      range_image_pred, is_int32=True)
  laser_semseg.ri_return2.segmentation_label_compressed = compress_array(
      range_image_pred_ri2, is_int32=True)
  segmentation_frame.segmentation_labels.append(laser_semseg)
  return segmentation_frame



# Create the dummy pred file for the validation set run segments.

# Replace this path with the real path to the WOD validation set folder.
folder_name = '/deepdata/dataset/waymo_v1.4/validation'

filenames = [os.path.join(folder_name, x) for x in os.listdir(
    folder_name) if 'tfrecord' in x]
assert(len(filenames) == 202)

segmentation_frame_list = segmentation_metrics_pb2.SegmentationFrameList()
for idx, filename in enumerate(filenames):
  if idx % 10 == 0:
    print('Processing %d/%d run segments...' % (idx, len(filenames)))
  dataset = tf.data.TFRecordDataset(filename, compression_type='')
  for data in dataset:
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(data.numpy()))
    if frame.lasers[0].ri_return1.segmentation_label_compressed:
      segmentation_frame = dummy_semseg_for_one_frame(frame)
      segmentation_frame_list.frames.append(segmentation_frame)
print('Total number of frames: ', len(segmentation_frame_list.frames))


# Create the submission file, which can be uploaded to the eval server.
submission = segmentation_submission_pb2.SemanticSegmentationSubmission()
submission.account_name = 'joe@gmail.com'
submission.unique_method_name = 'JoeNet'
submission.affiliation = 'Smith Inc.'
submission.authors.append('Joe Smith')
submission.description = "A dummy method by Joe (val set)."
submission.method_link = 'NA'
submission.sensor_type = 1
submission.number_past_frames_exclude_current = 2
submission.number_future_frames_exclude_current = 0
submission.inference_results.CopyFrom(segmentation_frame_list)


# output_filename = './wod_semseg_val_set_gt_submission.bin'
# f = open(output_filename, 'wb')
# f.write(submission.SerializeToString())
# f.close()

output_filename = '/root/3D/work_dirs/dataset_infos/waymo_semseg_val_set_gt.bin'
f = open(output_filename, 'wb')
f.write(segmentation_frame_list.SerializeToString())
f.close()