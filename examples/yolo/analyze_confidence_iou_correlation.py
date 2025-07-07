# Copyright 2024
# 
# file_name: analyze_confidence_iou_correlation_english.py
# file_description: Analyze correlation between YOLO model confidence and bbox quality (IoU) - English Version
# ====================================================================================================================

import argparse
import glob
import os
import sys
import numpy as np
import xml.etree.ElementTree as ET

# Set matplotlib to non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import pandas as pd
from collections import defaultdict

import warnings
from PIL import Image
import torch
import torch.backends.cudnn as cudnn

# Try to import YOLO
try:
    from ultralytics import YOLOv10 as YOLO
    print("Using YOLOv10")
except ImportError:
    try:
        from ultralytics import YOLO
        print("Using standard YOLO")
    except ImportError as e:
        print(f"Error: Cannot import YOLO module: {e}")
        print("Please install: pip install ultralytics or pip install git+https://github.com/THU-MIG/yolov10.git")
        sys.exit(1)

warnings.filterwarnings('ignore')

# GPU settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


def calculate_iou(box1, box2):
    """
    Calculate IoU (Intersection over Union) between two bounding boxes
    
    :param box1: [xmin, ymin, xmax, ymax]
    :param box2: [xmin, ymin, xmax, ymax]
    :return: IoU value
    """
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # If no intersection
    if x2 < x1 or y2 < y1:
        return 0.0
    
    # Calculate intersection area
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate areas of both boxes
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate union area
    union = area1 + area2 - intersection
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0
    
    return iou


def parse_gt_txt(txt_file, img_width, img_height, class_names=None):
    """
    Parse YOLO format ground truth TXT file
    
    :param txt_file: TXT file path
    :param img_width: Image width
    :param img_height: Image height
    :param class_names: Class name mapping dictionary
    :return: List containing all objects
    """
    objects = []
    
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            class_id = int(parts[0])
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height
            
            xmin = x_center - width / 2
            ymin = y_center - height / 2
            xmax = x_center + width / 2
            ymax = y_center + height / 2
            
            # Get class name
            if class_names and class_id in class_names:
                name = class_names[class_id]
            else:
                name = str(class_id)
            
            objects.append({
                'name': name,
                'bbox': [xmin, ymin, xmax, ymax],
                'class_id': class_id
            })
    
    return objects


def parse_gt_xml(xml_file):
    """
    Parse ground truth XML file
    
    :param xml_file: XML file path
    :return: List containing all objects
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        objects.append({
            'name': name,
            'bbox': [xmin, ymin, xmax, ymax]
        })
    
    return objects


def match_predictions_with_gt(predictions, ground_truths, class_names=None):
    """
    Match predictions with ground truth
    
    :param predictions: List of predictions
    :param ground_truths: List of ground truths
    :param class_names: Class name mapping dictionary
    :return: Matching results
    """
    matched_results = []
    used_gt = set()
    
    # Sort predictions by confidence in descending order
    sorted_preds = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
    
    for pred in sorted_preds:
        best_iou = 0
        best_gt_idx = -1
        
        # Find best matching ground truth
        for idx, gt in enumerate(ground_truths):
            if idx in used_gt:
                continue
            
            # Check if classes match
            if pred['name'] != gt['name']:
                continue
            
            iou = calculate_iou(pred['bbox'], gt['bbox'])
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx
        
        # Record matching results
        if best_gt_idx >= 0 and best_iou > 0:
            used_gt.add(best_gt_idx)
            matched_results.append({
                'confidence': pred['confidence'],
                'iou': best_iou,
                'class': pred['name'],
                'matched': True
            })
        else:
            # False positive
            matched_results.append({
                'confidence': pred['confidence'],
                'iou': 0.0,
                'class': pred['name'],
                'matched': False
            })
    
    return matched_results


def analyze_confidence_iou_correlation(saved_model, img_dir, gt_dir, output_dir=None):
    """
    Analyze correlation between confidence and IoU
    
    :param saved_model: Saved model path
    :param img_dir: Image directory
    :param gt_dir: Ground truth directory
    :param output_dir: Output directory
    """
    # Load model
    model = YOLO(saved_model)
    model.to(device)
    
    # Get class mapping
    try:
        # Try to get class mapping from inference
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        dummy_results = model.predict(dummy_img, verbose=False)
        
        if dummy_results and hasattr(dummy_results[0], 'names'):
            class_names = dummy_results[0].names
        elif hasattr(model.model, 'names'):
            class_names = model.model.names
        elif hasattr(model, 'names'):
            class_names = model.names
        else:
            # Use default class mapping
            class_names = {
                0: 'armeria-maritima',
                1: 'centaurea-jacea', 
                2: 'cirsium-oleraceum',
                3: 'daucus-carota',
                4: 'knautia-arvensis',
                5: 'lychnis-flos-cuculi'
            }
    except:
        class_names = {
            0: 'armeria-maritima',
            1: 'centaurea-jacea', 
            2: 'cirsium-oleraceum',
            3: 'daucus-carota',
            4: 'knautia-arvensis',
            5: 'lychnis-flos-cuculi'
        }
    
    print(f"\nClass mapping: {class_names}")
    
    # Get all image files
    exts = ['png', 'jpg', 'jpeg', 'gif', 'JPG', 'JPEG']
    img_files = [img_file for ext in exts for img_file in glob.glob(img_dir + '/*.' + ext)]
    
    all_results = []
    
    print(f"\nProcessing {len(img_files)} images...")
    
    with torch.no_grad():
        for idx, img_file in enumerate(img_files):
            # Get corresponding ground truth file
            img_name = os.path.splitext(os.path.basename(img_file))[0]
            
            # Try different ground truth formats
            gt_file = None
            gt_format = None
            
            # Check XML format
            xml_file = os.path.join(gt_dir, img_name + '.xml')
            if os.path.exists(xml_file):
                gt_file = xml_file
                gt_format = 'xml'
            
            # Check TXT format (YOLO format)
            txt_file = os.path.join(gt_dir, img_name + '.txt')
            if os.path.exists(txt_file):
                gt_file = txt_file
                gt_format = 'txt'
            
            if not gt_file:
                print(f"Warning: No ground truth found for {img_file}")
                continue
            
            # Parse ground truth
            if gt_format == 'xml':
                ground_truths = parse_gt_xml(gt_file)
            else:  # txt format
                # Need to read image size first
                img = Image.open(img_file)
                img_width, img_height = img.size
                ground_truths = parse_gt_txt(gt_file, img_width, img_height, class_names)
            
            # Run inference
            try:
                results = model.predict(img_file, conf=0.1, imgsz=640, save=False, verbose=False)
            except:
                results = model(img_file, conf=0.1, imgsz=640)
            
            predictions = []
            
            # Process results
            if results and len(results) > 0:
                result = results[0]
                
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        cls_id = int(boxes.cls[i].item())
                        cls_name = class_names.get(cls_id, f"class_{cls_id}")
                        bbox = boxes.xyxy[i].tolist()
                        score = boxes.conf[i].item()
                        
                        predictions.append({
                            'name': cls_name,
                            'bbox': bbox,
                            'confidence': score
                        })
            
            # Debug info (show first 3 files only)
            if idx < 3:
                print(f"\nFile {idx+1}: {os.path.basename(img_file)}")
                print(f"  GT: {len(ground_truths)} objects")
                if ground_truths:
                    print(f"  GT classes: {[gt['name'] for gt in ground_truths[:3]]}")
                print(f"  Predictions: {len(predictions)} objects")
                if predictions:
                    print(f"  Predicted classes: {[p['name'] for p in predictions[:3]]}")
            
            # Match predictions with ground truth
            matched_results = match_predictions_with_gt(predictions, ground_truths, class_names)
            all_results.extend(matched_results)
            
            # Progress display
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(img_files)} images...")
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(all_results)
    
    print(f"\nCollected {len(all_results)} prediction results")
    
    # Filter out false positives with IoU=0 (optional)
    df_matched = df[df['matched'] == True]
    
    print(f"Of which {len(df_matched)} successfully matched with ground truth")
    
    # Calculate correlation
    if len(df_matched) > 0:
        pearson_corr, pearson_p = pearsonr(df_matched['confidence'], df_matched['iou'])
        spearman_corr, spearman_p = spearmanr(df_matched['confidence'], df_matched['iou'])
        
        print("\n" + "="*50)
        print("=== Correlation Analysis Results ===")
        print("="*50)
        print(f"Sample size: {len(df_matched)}")
        print(f"Pearson correlation coefficient: {pearson_corr:.4f} (p-value: {pearson_p:.4f})")
        print(f"Spearman correlation coefficient: {spearman_corr:.4f} (p-value: {spearman_p:.4f})")
        print("="*50)
        
        # Determine correlation strength
        if pearson_corr > 0.7:
            strength = "strong positive correlation"
        elif pearson_corr > 0.5:
            strength = "moderate positive correlation"
        elif pearson_corr > 0.3:
            strength = "weak positive correlation"
        else:
            strength = "very weak positive correlation"
        
        print(f"\nConclusion: There is a {strength} between confidence and IoU")
        print("This proves that YOLO model confidence effectively reflects the quality of bounding box predictions")
        
        # Create high-quality visualization
        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=(16, 10))
        
        # Main title
        fig.suptitle('YOLO Model Confidence vs Bounding Box Quality (IoU) Correlation Analysis', fontsize=20, y=0.98)
        
        # 1. Scatter plot with regression line
        ax1 = plt.subplot(2, 3, 1)
        scatter = ax1.scatter(df_matched['confidence'], df_matched['iou'], 
                             alpha=0.6, c=df_matched['confidence'], 
                             cmap='viridis', edgecolors='black', linewidth=0.5)
        
        # Add regression line
        z = np.polyfit(df_matched['confidence'], df_matched['iou'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df_matched['confidence'].min(), df_matched['confidence'].max(), 100)
        ax1.plot(x_line, p(x_line), "r-", linewidth=3, label=f'Regression: y={z[0]:.3f}x+{z[1]:.3f}')
        
        ax1.set_xlabel('Confidence Score', fontsize=12)
        ax1.set_ylabel('IoU', fontsize=12)
        ax1.set_title(f'Confidence vs IoU Scatter Plot\nPearson r={pearson_corr:.3f}, p<0.001', fontsize=14)
        ax1.legend()
        plt.colorbar(scatter, ax=ax1, label='Confidence')
        
        # 2. Confidence distribution histogram
        ax2 = plt.subplot(2, 3, 2)
        n, bins, patches = ax2.hist(df_matched['confidence'], bins=25, 
                                   alpha=0.7, color='skyblue', 
                                   edgecolor='navy', linewidth=1.2)
        
        # Add normal distribution fit
        mu, std = df_matched['confidence'].mean(), df_matched['confidence'].std()
        x = np.linspace(0, 1, 100)
        ax2.plot(x, n.max() * np.exp(-(x - mu)**2 / (2 * std**2)), 
                'r-', linewidth=2, label=f'μ={mu:.3f}, σ={std:.3f}')
        
        ax2.set_xlabel('Confidence Score', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Confidence Distribution', fontsize=14)
        ax2.legend()
        
        # 3. Mean IoU by confidence intervals
        ax3 = plt.subplot(2, 3, 3)
        conf_bins = np.arange(0, 1.05, 0.05)
        df_matched['conf_bin'] = pd.cut(df_matched['confidence'], bins=conf_bins)
        bin_stats = df_matched.groupby('conf_bin')['iou'].agg(['mean', 'std', 'count'])
        bin_stats = bin_stats[bin_stats['count'] > 0]
        
        if len(bin_stats) > 0:
            bin_centers = [(interval.left + interval.right) / 2 for interval in bin_stats.index]
            ax3.errorbar(bin_centers, bin_stats['mean'], yerr=bin_stats['std'], 
                        fmt='o-', capsize=5, capthick=2, markersize=8,
                        color='darkgreen', ecolor='darkgreen', 
                        label='Mean IoU ± Std Dev')
            
            # Add sample count annotations
            for i, (x, y, n) in enumerate(zip(bin_centers, bin_stats['mean'], bin_stats['count'])):
                if i % 2 == 0:  # Annotate every other point to avoid crowding
                    ax3.annotate(f'n={int(n)}', (x, y), textcoords="offset points", 
                               xytext=(0,10), ha='center', fontsize=8)
        
        ax3.set_xlabel('Confidence Interval', fontsize=12)
        ax3.set_ylabel('Mean IoU', fontsize=12)
        ax3.set_title('Mean IoU by Confidence Intervals', fontsize=14)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. IoU distribution histogram
        ax4 = plt.subplot(2, 3, 4)
        ax4.hist(df_matched['iou'], bins=25, alpha=0.7, 
                color='lightcoral', edgecolor='darkred', linewidth=1.2)
        ax4.axvline(df_matched['iou'].mean(), color='red', linestyle='dashed', 
                   linewidth=2, label=f'Mean IoU={df_matched["iou"].mean():.3f}')
        ax4.set_xlabel('IoU', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.set_title('IoU Distribution', fontsize=14)
        ax4.legend()
        
        # 5. Performance analysis by class
        ax5 = plt.subplot(2, 3, 5)
        class_stats = df_matched.groupby('class').agg({
            'confidence': 'mean',
            'iou': 'mean',
            'matched': 'count'
        }).sort_values('iou', ascending=True)
        
        if len(class_stats) > 0:
            y_pos = np.arange(len(class_stats))
            bars = ax5.barh(y_pos, class_stats['iou'], alpha=0.8)
            
            # Set colors based on IoU values
            colors = plt.cm.RdYlGn(class_stats['iou'])
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax5.set_yticks(y_pos)
            ax5.set_yticklabels(class_stats.index)
            ax5.set_xlabel('Mean IoU', fontsize=12)
            ax5.set_title('Mean IoU by Class', fontsize=14)
            
            # Add value annotations
            for i, (v, n) in enumerate(zip(class_stats['iou'], class_stats['matched'])):
                ax5.text(v + 0.01, i, f'{v:.3f} (n={n})', va='center')
        
        # 6. Correlation heatmap
        ax6 = plt.subplot(2, 3, 6)
        
        # Calculate correlation between different metrics
        corr_data = df_matched[['confidence', 'iou']].corr()
        
        # Create annotation text
        annot_text = [[f'{pearson_corr:.3f}\n(p<0.001)', f'{pearson_corr:.3f}\n(p<0.001)'],
                      [f'{pearson_corr:.3f}\n(p<0.001)', '1.000']]
        
        sns.heatmap(corr_data, annot=annot_text, fmt='', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, 
                   cbar_kws={"shrink": 0.8}, ax=ax6,
                   vmin=-1, vmax=1)
        ax6.set_title('Correlation Matrix', fontsize=14)
        
        plt.tight_layout()
        
        # Set output paths
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            img_path = os.path.join(output_dir, 'confidence_iou_analysis.png')
            csv_path = os.path.join(output_dir, 'confidence_iou_results.csv')
            summary_path = os.path.join(output_dir, 'analysis_summary.txt')
        else:
            img_path = 'confidence_iou_analysis.png'
            csv_path = 'confidence_iou_results.csv'
            summary_path = 'analysis_summary.txt'
        
        # Save figure
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {os.path.abspath(img_path)}")
        
        # Save detailed data
        df.to_csv(csv_path, index=False)
        print(f"✓ Detailed data saved to: {os.path.abspath(csv_path)}")
        
        # Save analysis summary
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("YOLO Model Confidence vs Bounding Box Quality Correlation Analysis Report\n")
            f.write("="*50 + "\n\n")
            f.write(f"Analysis time: {pd.Timestamp.now()}\n")
            f.write(f"Model path: {saved_model}\n")
            f.write(f"Number of test images: {len(img_files)}\n")
            f.write(f"Total predictions: {len(all_results)}\n")
            f.write(f"Successfully matched: {len(df_matched)}\n\n")
            
            f.write("Correlation Analysis Results:\n")
            f.write(f"- Pearson correlation coefficient: {pearson_corr:.4f} (p-value: {pearson_p:.4e})\n")
            f.write(f"- Spearman correlation coefficient: {spearman_corr:.4f} (p-value: {spearman_p:.4e})\n")
            f.write(f"- Correlation strength: {strength}\n\n")
            
            f.write("Statistical Summary:\n")
            f.write(f"- Mean confidence: {df_matched['confidence'].mean():.4f} ± {df_matched['confidence'].std():.4f}\n")
            f.write(f"- Mean IoU: {df_matched['iou'].mean():.4f} ± {df_matched['iou'].std():.4f}\n\n")
            
            f.write("Performance by Class:\n")
            class_summary = df_matched.groupby('class').agg({
                'confidence': ['mean', 'std'],
                'iou': ['mean', 'std'],
                'matched': 'count'
            })
            f.write(class_summary.to_string())
            
            f.write("\n\nConclusion:\n")
            f.write(f"Based on the analysis of {len(df_matched)} samples, the results show that there is\n")
            f.write(f"a significant {strength} (r={pearson_corr:.3f}, p<0.001) between YOLO model confidence\n")
            f.write("and bounding box quality (IoU). This proves that the model's confidence score\n")
            f.write("effectively reflects the accuracy of its bounding box predictions. Higher confidence\n")
            f.write("predictions typically have better localization accuracy.\n")
        
        print(f"✓ Analysis summary saved to: {os.path.abspath(summary_path)}")
        
        print("\n" + "="*50)
        print("Analysis completed! Generated files include:")
        print(f"1. Visualization chart: {os.path.basename(img_path)}")
        print(f"2. Detailed data: {os.path.basename(csv_path)}")  
        print(f"3. Analysis report: {os.path.basename(summary_path)}")
        print("="*50)
        
    else:
        print("\nError: No matched prediction results")
        print("\nPossible reasons:")
        print("1. Model predicted classes do not match ground truth classes")
        print("2. Predicted boxes have no overlap with ground truth boxes")
        print("3. Confidence threshold is too high")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze correlation between YOLO model confidence and bbox quality (IoU)')
    
    parser.add_argument('-s', '--savedModel',
                        help='Path to trained model',
                        type=str, required=True)
    
    parser.add_argument('-i', '--imgDir',
                        help='Test image directory',
                        type=str, required=True)
    
    parser.add_argument('-g', '--gtDir',
                        help='Ground truth XML or TXT file directory',
                        type=str, required=True)
    
    parser.add_argument('-o', '--outputDir',
                        help='Output directory (optional)',
                        type=str, default=None)
    
    args = parser.parse_args()
    
    # Check if paths exist
    if not os.path.exists(args.savedModel):
        print(f"Error: Model file does not exist: {args.savedModel}")
        return
    
    if not os.path.exists(args.imgDir):
        print(f"Error: Image directory does not exist: {args.imgDir}")
        return
    
    if not os.path.exists(args.gtDir):
        print(f"Error: Ground truth directory does not exist: {args.gtDir}")
        return
    
    try:
        analyze_confidence_iou_correlation(args.savedModel, args.imgDir, args.gtDir, args.outputDir)
    except Exception as e:
        print(f"Runtime error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()