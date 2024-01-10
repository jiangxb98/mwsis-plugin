import os
import pickle

def get_npy_filenames(folder_path):
    npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
    return npy_files

folder_path = "/dataset/waymo_v1.4.2/waymo/kitti_format/training/img_panseg_label_0/"  # 替换为你的文件夹路径
npy_filenames = get_npy_filenames(folder_path)

# 提取数字并保存在数组中
numbers = []
for filename in npy_filenames:
    number = int(filename.replace('.npz', ''))
    numbers.append(number)

# 排序
numbers.sort()

# 打印整数数组
print("提取的整数数组:")
print(numbers)
print(len(numbers))

train_numbers = numbers[0:13206]
val_numbers = numbers[13206:]

# 保存到.pkl文件
output_file = "/dataset/waymo_v1.4.2/waymo/kitti_format/training/img_panseg_index_train.pkl"  # 替换为输出文件的路径
with open(output_file, 'wb') as f:
    pickle.dump(train_numbers, f)

output_file = "/dataset/waymo_v1.4.2/waymo/kitti_format/training/img_panseg_index_val.pkl"  # 替换为输出文件的路径
with open(output_file, 'wb') as f:
    pickle.dump(val_numbers, f)

print(f"整数数组已保存到 {output_file}")