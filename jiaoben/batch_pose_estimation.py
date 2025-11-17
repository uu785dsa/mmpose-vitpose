import os
import cv2
import pandas as pd
import numpy as np
from mmpose.apis import init_model, inference_topdown
from mmpose.registry import VISUALIZERS
import torch

# ----------------------------
# 配置参数
# ----------------------------
config_file = r'configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py'
checkpoint_file = r'G:\vitpose\mmpose\td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth'
output_dir = "G:/gait_dataset/newname/001/nm/135-1"
input_dir = output_dir + '/image_crop'

pred_img_dir = os.path.join(output_dir, 'predictions')
os.makedirs(pred_img_dir, exist_ok=True)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

keypoint_names = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# ----------------------------
# 初始化模型
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
# 处理所有图片（按数字排序）
# ----------------------------
results = []

def get_file_number(filename):
    """提取文件名中的数字用于排序"""
    name = os.path.splitext(filename)[0]
    try:
        return int(name)
    except ValueError:
        return float('inf')

# 筛选图片并按数字排序
image_files = [f for f in os.listdir(input_dir) 
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
image_files.sort(key=get_file_number)

for filename in image_files:
    img_path = os.path.join(input_dir, filename)
    print(f"处理: {filename}")

    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ 无法读取图像: {img_path}")
        continue

    # 移除归一化相关的宽高获取（虽然保留也不影响，但可以简化）
    # h, w = img.shape[:2]  # 此行可删除，因为不再用于归一化

    try:
        pred_results = inference_topdown(model, img)
    except Exception as e:
        print(f"❌ 推理出错 {filename}: {e}")
        continue

    if len(pred_results) == 0:
        print(f"⚠️ 未检测到姿态: {filename}")
        row = {'image_name': f"{output_dir}/{filename}"}
        for name in keypoint_names:
            row[f'{name}_x'] = pd.NA
            row[f'{name}_y'] = pd.NA
            row[f'{name}_conf'] = pd.NA
        results.append(row)
        continue

    pose_result = pred_results[0]
    keypoints = pose_result.pred_instances.keypoints[0]  # (17, 2) 原始像素坐标
    keypoint_scores = pose_result.pred_instances.keypoint_scores[0]  # (17,)

    # 移除归一化操作，直接使用原始坐标
    # 原代码：x_norm = keypoints[:, 0] / w
    # 原代码：y_norm = keypoints[:, 1] / h

    row = {'image_name': f"{output_dir}/{filename}"}
    for i, name in enumerate(keypoint_names):
        # 直接使用keypoints的原始值（像素坐标）
        row[f'{name}_x'] = keypoints[i, 0]  # 原始x像素坐标
        row[f'{name}_y'] = keypoints[i, 1]  # 原始y像素坐标
        row[f'{name}_conf'] = float(keypoint_scores[i])
    results.append(row)

    # 可视化代码（可选）
    visualizer.add_datasample(
        name='result',
        image=img,
        data_sample=pose_result,
        draw_gt=False,
        draw_bbox=False,
        kpt_thr=0.3,
        show=False,
        out_file=os.path.join(pred_img_dir, filename)
    )

# ----------------------------
# 保存CSV
# ----------------------------
df = pd.DataFrame(results)
csv_path = os.path.join(output_dir, 'keypoints.csv')
df.to_csv(csv_path, index=False)
print(f"\n✅ 完成！结果已保存至:")
print(f"   📊 CSV: {csv_path}")