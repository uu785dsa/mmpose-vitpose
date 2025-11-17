import os
import cv2
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from mmpose.apis import init_model, inference_topdown
from mmpose.registry import VISUALIZERS
import torch
# ----------------------------
# 利用voc格式数据集确认需要进行姿态估计的范围，再使用指定模型对该范围内的人进行姿态估计，并输出csv格式的数据集
# ----------------------------
# ----------------------------
# 配置参数
# ----------------------------
# 模型配置和权重文件路径
config_file = r'configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py'
checkpoint_file = r'G:\vitpose\mmpose\td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth'

# 数据集路径（图片和xml文件在同一目录）fen
person_dir = '015'  # 请替换为实际数据集路径
angle_dir = '135-4'  # 请替换为实际角度类型
temp = r'G:\gait_dataset\newname'+ '/' + person_dir + '/nm/' + angle_dir
dataset_dir = temp  # 请替换为实际数据集路径
output_dir = dataset_dir         # 结果输出目录
output_dir2 = r'G:\gait_dataset\newname'+ '/' + person_dir  # 结果输出目录
target_class = 'person'                   # 需要处理的目标类别

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)
pred_img_dir = os.path.join(output_dir, 'predictions')
os.makedirs(pred_img_dir, exist_ok=True)

# 关键点名称（COCO数据集17个关键点）
keypoint_names = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# 设备配置
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# ----------------------------
# 初始化模型和可视化工具
# ----------------------------
print("正在加载模型...")
model = init_model(config_file, checkpoint_file, device=device)
visualizer = VISUALIZERS.build(
    dict(
        type='PoseLocalVisualizer',
        name='visualizer',
        save_dir=pred_img_dir
    )
)
visualizer.set_dataset_meta(model.dataset_meta)

# ----------------------------
# XML解析函数
# ----------------------------
def parse_xml(xml_path):
    """解析VOC格式的XML文件，返回指定类别的边界框"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    bboxes = []
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls == target_class:
            bndbox = obj.find('bndbox')
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))
            bboxes.append((xmin, ymin, xmax, ymax))
    
    return bboxes

# ----------------------------
# 处理数据集
# ----------------------------
results = []

# 获取所有图片文件
image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
image_files = [f for f in os.listdir(dataset_dir) 
              if f.lower().endswith(image_extensions)]

for img_filename in image_files:
    # 构建文件路径
    img_basename = os.path.splitext(img_filename)[0]
    img_path = os.path.join(dataset_dir, img_filename)
    xml_path = os.path.join(dataset_dir, f'{img_basename}.xml')
    
    # 检查XML文件是否存在
    if not os.path.exists(xml_path):
        print(f"⚠️ 未找到XML文件: {xml_path}，跳过图片")
        continue
    
    print(f"处理图片: {img_filename}")
    
    # 读取图片
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ 无法读取图片: {img_path}，跳过")
        continue
    
    # 解析XML获取边界框
    bboxes = parse_xml(xml_path)
    if not bboxes:
        print(f"⚠️ 未在XML中找到{target_class}类别，跳过图片")
        continue
    
    # 对每个边界框进行姿态估计
    for i, (xmin, ymin, xmax, ymax) in enumerate(bboxes):
        # 裁剪边界框区域
        bbox_width = xmax - xmin
        bbox_height = ymax - ymin
        
        # 确保边界框有效
        if bbox_width <= 0 or bbox_height <= 0:
            print(f"⚠️ 无效的边界框，跳过第{i+1}个框")
            continue
            
        cropped_img = img[ymin:ymax, xmin:xmax]
        
        # 姿态估计
        try:
            pred_results = inference_topdown(model, cropped_img)
        except Exception as e:
            print(f"❌ 第{i+1}个框推理出错: {e}")
            continue
        
        if not pred_results:
            print(f"⚠️ 第{i+1}个框未检测到姿态")
            continue
        
        # 处理姿态结果
        pose_result = pred_results[0]
        keypoints = pose_result.pred_instances.keypoints[0]  # (17, 2)
        keypoint_scores = pose_result.pred_instances.keypoint_scores[0]  # (17,)
        
        # 转换为整图坐标（加上边界框偏移量）
        full_img_keypoints = keypoints.copy()
        full_img_keypoints[:, 0] += xmin  # x坐标加上xmin偏移
        full_img_keypoints[:, 1] += ymin  # y坐标加上ymin偏移
        
        # 更新姿态结果中的关键点坐标为整图坐标
        pose_result.pred_instances.keypoints[0] = full_img_keypoints
        
        # 保存结果到列表
        row = {
            'image_name': dataset_dir +'/'+ img_filename,
            'bbox_id': i + 1,
            'bbox_xmin': xmin,
            'bbox_ymin': ymin,
            'bbox_xmax': xmax,
            'bbox_ymax': ymax
        }
        
        for j, name in enumerate(keypoint_names):
            row[f'{name}_x'] = full_img_keypoints[j, 0]
            row[f'{name}_y'] = full_img_keypoints[j, 1]
            row[f'{name}_conf'] = float(keypoint_scores[j])
        
        results.append(row)
        
        # 可视化（在原图上绘制，已使用整图坐标）
        visualizer.add_datasample(
            name=f'result_{img_basename}_{i}',
            image=img.copy(),
            data_sample=pose_result,
            draw_gt=False,
            draw_bbox=True,
            kpt_thr=0.3,
            show=False,
            out_file=os.path.join(pred_img_dir, f'{img_basename}_{i}.png')
        )

# -----------------------------

# ----------------------------
# 保存结果到CSV
# ----------------------------
if results:
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir2, person_dir + '--' + angle_dir + '.keypoints_with_bbox.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✅ 完成！结果已保存至: {csv_path}")
    print(f"   可视化结果保存至: {pred_img_dir}")
else:
    print("\n⚠️ 未处理到任何有效数据")