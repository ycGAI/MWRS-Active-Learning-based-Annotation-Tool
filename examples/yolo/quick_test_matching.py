#!/usr/bin/env python3
# 快速测试类别匹配问题

import os
import sys
from PIL import Image
from ultralytics import YOLO

def test_single_file(model_path, img_file, txt_file):
    """测试单个文件的匹配"""
    
    print(f"\n=== 测试文件 ===")
    print(f"图像: {img_file}")
    print(f"标注: {txt_file}")
    
    # 1. 读取ground truth
    print("\n1. Ground Truth内容:")
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) >= 5:
            print(f"  GT{i}: 类别ID={parts[0]}, 中心=({parts[1]}, {parts[2]}), 尺寸=({parts[3]}, {parts[4]})")
    
    # 2. 加载模型并检查类别映射
    print("\n2. 模型类别映射:")
    model = YOLO(model_path)
    class_names = model.names
    
    for class_id, class_name in sorted(class_names.items()):
        print(f"  {class_id}: {class_name}")
    
    # 3. 运行预测
    print("\n3. 预测结果:")
    results = model(img_file, conf=0.1)
    
    predictions = []
    for result in results:
        for i in range(len(result.boxes)):
            cls_id = int(result.boxes.cls[i].item())
            cls_name = result.names[cls_id]
            conf = result.boxes.conf[i].item()
            bbox = result.boxes.xyxy[i].tolist()
            
            print(f"  Pred{i}: 类别ID={cls_id}, 类别名称='{cls_name}', 置信度={conf:.3f}")
            predictions.append({
                'class_id': cls_id,
                'class_name': cls_name,
                'confidence': conf,
                'bbox': bbox
            })
    
    # 4. 分析匹配问题
    print("\n4. 匹配分析:")
    
    # 解析GT中的类别ID
    gt_class_ids = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            gt_class_ids.append(int(parts[0]))
    
    print(f"GT中的类别ID: {gt_class_ids}")
    print(f"预测中的类别ID: {[p['class_id'] for p in predictions]}")
    
    # 检查是否有匹配
    for pred in predictions:
        if pred['class_id'] in gt_class_ids:
            print(f"✓ 类别 {pred['class_id']} ({pred['class_name']}) 可以匹配")
        else:
            print(f"✗ 类别 {pred['class_id']} ({pred['class_name']}) 无法匹配")
    
    return class_names, predictions, gt_class_ids

def main():
    if len(sys.argv) < 2:
        print("用法: python quick_test_matching.py <model_path> [image_file] [txt_file]")
        print("如果不提供image_file和txt_file，将使用默认测试路径")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    # 默认测试文件
    if len(sys.argv) >= 4:
        img_file = sys.argv[2]
        txt_file = sys.argv[3]
    else:
        # 使用默认路径中的第一个文件
        img_dir = "/media/gyc/Backup Plus6/gyc/project_MWRS/Data/images/test"
        txt_dir = "/media/gyc/Backup Plus6/gyc/project_MWRS/Data/labels/test"
        
        # 找一个示例文件
        import glob
        img_files = glob.glob(os.path.join(img_dir, "*.jpg"))[:1]
        if img_files:
            img_file = img_files[0]
            img_name = os.path.splitext(os.path.basename(img_file))[0]
            txt_file = os.path.join(txt_dir, img_name + '.txt')
        else:
            print("找不到测试图像")
            sys.exit(1)
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"模型不存在: {model_path}")
        sys.exit(1)
    
    if not os.path.exists(img_file):
        print(f"图像不存在: {img_file}")
        sys.exit(1)
        
    if not os.path.exists(txt_file):
        print(f"标注不存在: {txt_file}")
        sys.exit(1)
    
    # 运行测试
    test_single_file(model_path, img_file, txt_file)

if __name__ == "__main__":
    main()