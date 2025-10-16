import os
import glob
import re  # 引入正则表达式模块

# 设置路径
data_root = 'E:/classiyf-module/Re-Resformer/dataset/2Q235/'
train_dir = os.path.join(data_root, 'train')
val_dir = os.path.join(data_root, 'val')
test_dir = os.path.join(data_root, 'test')

# 保存标签文件的路径（保存在当前文件夹中）
train_file = os.path.join(data_root, 'train.txt')
val_file = os.path.join(data_root, 'val.txt')
test_file = os.path.join(data_root, 'test.txt')

# 初始化标签索引和文件内容
class_labels = {}
train_data = []
val_data = []
test_data = []
label_index = 0

# 处理数据集的函数
def process_dataset(directory, data_list):
    global label_index  # 声明使用全局变量
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            image_files = glob.glob(os.path.join(class_path, '*.*'))

            # 提取粗糙度值
            match = re.search(r'(\d+\.\d+)', class_name)  # 正则表达式提取类似 2.989 这样的浮动数值
            if match:
                roughness_value = float(match.group(1))  # 提取并转换为浮动数值
            else:
                continue  # 如果没有匹配到粗糙度值，跳过这个文件夹

            # 统一类别名称
            category_name = class_name.split('/')[0]  # 提取文件夹的前缀部分作为类别
            category_name = category_name.strip()  # 去除可能的空格

            if category_name not in class_labels:
                class_labels[category_name] = label_index
                label_index += 1

            # 只保存类别标签和粗糙度值
            for image_file in image_files:
                label = class_labels[category_name]
                rel_path = image_file.replace('\\', '/')  # 替换为正斜杠

                # 添加到数据列表，格式为 "相对路径 类别标签 粗糙度值"
                data_list.append(f"{rel_path} {label} {roughness_value}\n")

# 处理训练集、验证集和测试集
process_dataset(train_dir, train_data)
process_dataset(val_dir, val_data)
process_dataset(test_dir, test_data)

# 将训练数据写入文件
with open(train_file, 'w') as f:
    f.writelines(train_data)

# 将验证数据写入文件
with open(val_file, 'w') as f:
    f.writelines(val_data)

# 将测试数据写入文件
with open(test_file, 'w') as f:
    f.writelines(test_data)

print("标签文件已生成：")
print(f"训练集标签文件: {train_file}")
print(f"验证集标签文件: {val_file}")
print(f"测试集标签文件: {test_file}")
