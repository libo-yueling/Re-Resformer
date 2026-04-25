import os
import glob
import re 

data_root = '/root/autodl-tmp/mmcls/dataset/niejihejing/pingxi'
train_dir = os.path.join(data_root, 'train')
val_dir = os.path.join(data_root, 'val')
test_dir = os.path.join(data_root, 'test')

train_file = os.path.join(data_root, 'train.txt')
val_file = os.path.join(data_root, 'val.txt')
test_file = os.path.join(data_root, 'test.txt')

train_data = []
val_data = []
test_data = []

def process_dataset(directory, data_list):
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            image_files = glob.glob(os.path.join(class_path, '*.*'))
            match = re.search(r'(\d+\.\d+)', class_name) 
            if match:
                roughness_value = float(match.group(1)) 
            else:
                continue

            # 只保存粗糙度值
            for image_file in image_files:
                rel_path = image_file.replace('\\', '/')
                data_list.append(f"{rel_path} {roughness_value}\n")

process_dataset(train_dir, train_data)
process_dataset(val_dir, val_data)
process_dataset(test_dir, test_data)

with open(train_file, 'w') as f:
    f.writelines(train_data)

with open(val_file, 'w') as f:
    f.writelines(val_data)

with open(test_file, 'w') as f:
    f.writelines(test_data)
