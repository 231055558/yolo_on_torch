import json
import os

# 读取 JSON 文件路径
json_file_path = './annotations/instances_val2017.json'
output_folder = './annfiles'  # 定义输出目录

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 加载 JSON 文件
with open(json_file_path, 'r') as f:
    data = json.load(f)

# 获取各部分数据
images = {image['id']: image['file_name'] for image in data['images']}
annotations = data['annotations']

# 遍历标注信息并过滤 id >= 80 的类别
for annotation in annotations:
    category_id = annotation['category_id']
    if category_id >= 80:
        continue  # 跳过 id >= 80 的目标

    # 提取相关信息
    image_id = annotation['image_id']
    bbox = annotation['bbox']  # x_min, y_min, width, height

    # 获取图像名称并创建对应的 txt 文件
    image_name = images[image_id].split('.')[0]  # 去掉文件扩展名
    output_path = os.path.join(output_folder, f"{image_name}.txt")

    # 转换 bbox 为期望格式，并保存至文件
    with open(output_path, 'a') as txt_file:
        x_min, y_min, width, height = bbox
        x_max = x_min + width
        y_max = y_min + height
        # 写入格式: image_id category_id x_min y_min x_max y_max
        txt_file.write(f"{image_id} {category_id} {x_min} {y_min} {x_max} {y_max}\n")