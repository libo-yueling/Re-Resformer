import os
import glob
import re 

# road file
data_root = 'E:/classiyf-module/Re-Resformer/dataset/2Q235/'
train_dir = os.path.join(data_root, 'train')
val_dir = os.path.join(data_root, 'val')
test_dir = os.path.join(data_root, 'test')

# save the label
train_file = os.path.join(data_root, 'train.txt')
val_file = os.path.join(data_root, 'val.txt')
test_file = os.path.join(data_root, 'test.txt')

class_labels = {}
train_data = []
val_data = []
test_data = []
label_index = 0

def process_dataset(directory, data_list):
    global label_index 
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            image_files = glob.glob(os.path.join(class_path, '*.*'))

           
            match = re.search(r'(\d+\.\d+)', class_name)  
            if match:
                roughness_value = float(match.group(1)) 
            else:
                continue 

            category_name = class_name.split('/')[0]  
            category_name = category_name.strip()

            if category_name not in class_labels:
                class_labels[category_name] = label_index
                label_index += 1

            for image_file in image_files:
                label = class_labels[category_name]
                rel_path = image_file.replace('\\', '/') 

                data_list.append(f"{rel_path} {label} {roughness_value}\n")

process_dataset(train_dir, train_data)
process_dataset(val_dir, val_data)
process_dataset(test_dir, test_data)

with open(train_file, 'w') as f:
    f.writelines(train_data)

with open(val_file, 'w') as f:
    f.writelines(val_data)

with open(test_file, 'w') as f:
    f.writelines(test_data)

print("label is already：")
print(f"train label: {train_file}")
print(f"val label: {val_file}")
print(f"test label: {test_file}")
