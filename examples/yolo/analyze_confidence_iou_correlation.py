# Copyright 2024
# 
# file_name: analyze_confidence_iou_correlation_enhanced.py
# file_description: Analyze YOLO model with confidence-IoU correlation, F1 score, and confusion matrix
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
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, classification_report

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
    """Calculate IoU between two bounding boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    iou = intersection / union if union > 0 else 0
    
    return iou


def parse_gt_txt(txt_file, img_width, img_height, class_names=None):
    """Parse YOLO format ground truth TXT file"""
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
    """Parse ground truth XML file"""
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


def match_predictions_with_gt_detailed(predictions, ground_truths, iou_threshold=0.5):
    """
    Enhanced matching with detailed metrics for F1 and confusion matrix
    """
    matched_results = []
    used_gt = set()
    
    # For confusion matrix
    true_positives = defaultdict(int)
    false_positives = defaultdict(int)
    false_negatives = defaultdict(int)
    confusion_pairs = []  # (true_label, pred_label) pairs
    
    # Sort predictions by confidence
    sorted_preds = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
    
    for pred in sorted_preds:
        best_iou = 0
        best_gt_idx = -1
        best_gt = None
        
        for idx, gt in enumerate(ground_truths):
            if idx in used_gt:
                continue
            
            iou = calculate_iou(pred['bbox'], gt['bbox'])
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx
                best_gt = gt
        
        # Match found with IoU above threshold
        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            used_gt.add(best_gt_idx)
            
            # True positive if classes match
            if pred['name'] == best_gt['name']:
                matched_results.append({
                    'confidence': pred['confidence'],
                    'iou': best_iou,
                    'class': pred['name'],
                    'matched': True,
                    'type': 'TP'
                })
                true_positives[pred['name']] += 1
                confusion_pairs.append((best_gt['name'], pred['name']))
            else:
                # Wrong class prediction
                matched_results.append({
                    'confidence': pred['confidence'],
                    'iou': best_iou,
                    'class': pred['name'],
                    'matched': False,
                    'type': 'FP_wrong_class'
                })
                false_positives[pred['name']] += 1
                false_negatives[best_gt['name']] += 1
                confusion_pairs.append((best_gt['name'], pred['name']))
        else:
            # False positive (no match or IoU too low)
            matched_results.append({
                'confidence': pred['confidence'],
                'iou': best_iou if best_gt_idx >= 0 else 0,
                'class': pred['name'],
                'matched': False,
                'type': 'FP'
            })
            false_positives[pred['name']] += 1
    
    # Count false negatives (unmatched ground truths)
    for idx, gt in enumerate(ground_truths):
        if idx not in used_gt:
            false_negatives[gt['name']] += 1
            confusion_pairs.append((gt['name'], 'background'))
    
    return matched_results, true_positives, false_positives, false_negatives, confusion_pairs


def calculate_metrics_per_class(tp, fp, fn):
    """Calculate precision, recall, and F1 score per class"""
    metrics = {}
    
    all_classes = set(tp.keys()) | set(fp.keys()) | set(fn.keys())
    
    for cls in all_classes:
        tp_count = tp.get(cls, 0)
        fp_count = fp.get(cls, 0)
        fn_count = fn.get(cls, 0)
        
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[cls] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': tp_count + fn_count
        }
    
    return metrics


def plot_confusion_matrix(confusion_pairs, class_names, output_path):
    """Create and save confusion matrix visualization"""
    # Get all unique classes including 'background'
    all_classes = list(class_names.values()) + ['background']
    
    # Create label mapping
    label_to_idx = {label: idx for idx, label in enumerate(all_classes)}
    
    # Convert pairs to indices
    y_true = []
    y_pred = []
    
    for true_label, pred_label in confusion_pairs:
        y_true.append(label_to_idx.get(true_label, len(all_classes)-1))
        y_pred.append(label_to_idx.get(pred_label, len(all_classes)-1))
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(all_classes)))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized[np.isnan(cm_normalized)] = 0
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=all_classes, yticklabels=all_classes,
                cbar_kws={'label': 'Count'}, ax=ax1)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_title('Confusion Matrix (Raw Counts)', fontsize=14)
    
    # Plot normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=all_classes, yticklabels=all_classes,
                cbar_kws={'label': 'Proportion'}, ax=ax2)
    ax2.set_xlabel('Predicted Label', fontsize=12)
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm


def plot_f1_scores(metrics, output_path):
    """Create F1 score visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Prepare data
    classes = list(metrics.keys())
    f1_scores = [metrics[cls]['f1'] for cls in classes]
    precisions = [metrics[cls]['precision'] for cls in classes]
    recalls = [metrics[cls]['recall'] for cls in classes]
    supports = [metrics[cls]['support'] for cls in classes]
    
    # 1. F1 Score Bar Chart
    ax1 = axes[0, 0]
    bars = ax1.bar(classes, f1_scores, color='skyblue', edgecolor='navy')
    ax1.set_xlabel('Class', fontsize=12)
    ax1.set_ylabel('F1 Score', fontsize=12)
    ax1.set_title('F1 Score by Class', fontsize=14)
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    # Rotate x labels if needed
    if len(classes) > 6:
        ax1.set_xticklabels(classes, rotation=45, ha='right')
    
    # 2. Precision vs Recall Scatter
    ax2 = axes[0, 1]
    scatter = ax2.scatter(recalls, precisions, s=[s*10 for s in supports], 
                         alpha=0.6, c=f1_scores, cmap='viridis')
    
    # Add class labels
    for i, cls in enumerate(classes):
        ax2.annotate(cls, (recalls[i], precisions[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision vs Recall (size=support, color=F1)', fontsize=14)
    ax2.set_xlim([-0.05, 1.05])
    ax2.set_ylim([-0.05, 1.05])
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('F1 Score', fontsize=10)
    
    # 3. Metrics Comparison
    ax3 = axes[1, 0]
    x = np.arange(len(classes))
    width = 0.25
    
    bars1 = ax3.bar(x - width, precisions, width, label='Precision', color='lightcoral')
    bars2 = ax3.bar(x, recalls, width, label='Recall', color='lightgreen')
    bars3 = ax3.bar(x + width, f1_scores, width, label='F1 Score', color='skyblue')
    
    ax3.set_xlabel('Class', fontsize=12)
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_title('Precision, Recall, and F1 Score Comparison', fontsize=14)
    ax3.set_xticks(x)
    ax3.set_xticklabels(classes)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    if len(classes) > 6:
        ax3.set_xticklabels(classes, rotation=45, ha='right')
    
    # 4. Macro/Micro Average Summary
    ax4 = axes[1, 1]
    
    # Calculate macro averages
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1_scores)
    
    # Calculate micro averages (weighted by support)
    total_support = sum(supports)
    micro_precision = sum(p * s for p, s in zip(precisions, supports)) / total_support if total_support > 0 else 0
    micro_recall = sum(r * s for r, s in zip(recalls, supports)) / total_support if total_support > 0 else 0
    micro_f1 = sum(f * s for f, s in zip(f1_scores, supports)) / total_support if total_support > 0 else 0
    
    # Create summary table
    summary_data = {
        'Metric': ['Macro Avg', 'Weighted Avg'],
        'Precision': [macro_precision, micro_precision],
        'Recall': [macro_recall, micro_recall],
        'F1 Score': [macro_f1, micro_f1]
    }
    
    # Plot as table
    ax4.axis('tight')
    ax4.axis('off')
    
    table_data = []
    table_data.append(['', 'Precision', 'Recall', 'F1 Score'])
    table_data.append(['Macro Avg', f'{macro_precision:.3f}', f'{macro_recall:.3f}', f'{macro_f1:.3f}'])
    table_data.append(['Weighted Avg', f'{micro_precision:.3f}', f'{micro_recall:.3f}', f'{micro_f1:.3f}'])
    
    # Add per-class details
    table_data.append(['', '', '', ''])
    table_data.append(['Per-Class:', 'Precision', 'Recall', 'F1 Score'])
    for cls in classes:
        table_data.append([cls, 
                          f'{metrics[cls]["precision"]:.3f}',
                          f'{metrics[cls]["recall"]:.3f}',
                          f'{metrics[cls]["f1"]:.3f}'])
    
    table = ax4.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Style header rows
    for i in [0, 4]:
        for j in range(4):
            table[(i, j)].set_facecolor('#40466e')
            table[(i, j)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Performance Summary', fontsize=14, pad=20)
    
    plt.suptitle('Model Performance Metrics Analysis', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def analyze_confidence_iou_correlation(saved_model, img_dir, gt_dir, output_dir=None, conf_threshold=0.25, iou_threshold=0.5):
    """
    Enhanced analysis with F1 score and confusion matrix
    """
    # Load model
    model = YOLO(saved_model)
    model.to(device)
    
    # Get class mapping
    try:
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        dummy_results = model.predict(dummy_img, verbose=False)
        
        if dummy_results and hasattr(dummy_results[0], 'names'):
            class_names = dummy_results[0].names
        elif hasattr(model.model, 'names'):
            class_names = model.model.names
        elif hasattr(model, 'names'):
            class_names = model.names
        else:
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
    all_tp = defaultdict(int)
    all_fp = defaultdict(int)
    all_fn = defaultdict(int)
    all_confusion_pairs = []
    
    print(f"\nProcessing {len(img_files)} images...")
    print(f"Using confidence threshold: {conf_threshold}")
    print(f"Using IoU threshold: {iou_threshold}")
    
    with torch.no_grad():
        for idx, img_file in enumerate(img_files):
            img_name = os.path.splitext(os.path.basename(img_file))[0]
            
            # Find ground truth file
            gt_file = None
            gt_format = None
            
            xml_file = os.path.join(gt_dir, img_name + '.xml')
            if os.path.exists(xml_file):
                gt_file = xml_file
                gt_format = 'xml'
            
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
            else:
                img = Image.open(img_file)
                img_width, img_height = img.size
                ground_truths = parse_gt_txt(gt_file, img_width, img_height, class_names)
            
            # Run inference
            try:
                results = model.predict(img_file, conf=conf_threshold, imgsz=640, save=False, verbose=False)
            except:
                results = model(img_file, conf=conf_threshold, imgsz=640)
            
            predictions = []
            
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
            
            # Enhanced matching with detailed metrics
            matched_results, tp, fp, fn, confusion_pairs = match_predictions_with_gt_detailed(
                predictions, ground_truths, iou_threshold
            )
            
            all_results.extend(matched_results)
            
            # Accumulate metrics
            for cls in tp:
                all_tp[cls] += tp[cls]
            for cls in fp:
                all_fp[cls] += fp[cls]
            for cls in fn:
                all_fn[cls] += fn[cls]
            
            all_confusion_pairs.extend(confusion_pairs)
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(img_files)} images...")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    print(f"\nCollected {len(all_results)} prediction results")
    
    # Filter matched results for IoU analysis
    df_matched = df[df['matched'] == True]
    
    print(f"Of which {len(df_matched)} successfully matched with ground truth")
    
    # Calculate per-class metrics
    class_metrics = calculate_metrics_per_class(all_tp, all_fp, all_fn)
    
    # Set output paths
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_path = output_dir
    else:
        base_path = '.'
    
    # Create main visualization (original confidence-IoU analysis)
    if len(df_matched) > 0:
        pearson_corr, pearson_p = pearsonr(df_matched['confidence'], df_matched['iou'])
        spearman_corr, spearman_p = spearmanr(df_matched['confidence'], df_matched['iou'])
        
        print("\n" + "="*50)
        print("=== Analysis Results ===")
        print("="*50)
        
        # Correlation results
        print("\n1. Confidence-IoU Correlation:")
        print(f"   - Pearson correlation: {pearson_corr:.4f} (p={pearson_p:.4f})")
        print(f"   - Spearman correlation: {spearman_corr:.4f} (p={spearman_p:.4f})")
        
        # F1 scores
        print("\n2. F1 Scores by Class:")
        for cls in sorted(class_metrics.keys()):
            m = class_metrics[cls]
            print(f"   - {cls}: F1={m['f1']:.3f}, Precision={m['precision']:.3f}, Recall={m['recall']:.3f}")
        
        # Overall metrics
        macro_f1 = np.mean([m['f1'] for m in class_metrics.values()])
        print(f"\n3. Overall Performance:")
        print(f"   - Macro F1 Score: {macro_f1:.3f}")
        
        # Create original confidence-IoU visualization
        create_confidence_iou_plot(df, df_matched, pearson_corr, pearson_p, spearman_corr, spearman_p,
                                  class_names, os.path.join(base_path, 'confidence_iou_analysis.png'))
        
        # Create F1 score visualization
        plot_f1_scores(class_metrics, os.path.join(base_path, 'f1_scores_analysis.png'))
        print(f"✓ F1 score visualization saved")
        
        # Create confusion matrix
        cm = plot_confusion_matrix(all_confusion_pairs, class_names, 
                                   os.path.join(base_path, 'confusion_matrix.png'))
        print(f"✓ Confusion matrix saved")
        
        # Save detailed results
        df.to_csv(os.path.join(base_path, 'confidence_iou_results.csv'), index=False)
        
        # Save metrics summary
        metrics_df = pd.DataFrame(class_metrics).T
        metrics_df.to_csv(os.path.join(base_path, 'class_metrics.csv'))
        
        # Save comprehensive report
        save_comprehensive_report(base_path, saved_model, img_files, all_results, df_matched,
                                pearson_corr, pearson_p, spearman_corr, spearman_p,
                                class_metrics, all_tp, all_fp, all_fn, cm, class_names)
        
        print("\n" + "="*50)
        print("Analysis completed! Generated files:")
        print(f"1. Confidence-IoU visualization: confidence_iou_analysis.png")
        print(f"2. F1 scores visualization: f1_scores_analysis.png")
        print(f"3. Confusion matrix: confusion_matrix.png")
        print(f"4. Detailed results: confidence_iou_results.csv")
        print(f"5. Class metrics: class_metrics.csv")
        print(f"6. Comprehensive report: analysis_summary.txt")
        print("="*50)
        
    else:
        print("\nError: No matched prediction results")


def create_confidence_iou_plot(df, df_matched, pearson_corr, pearson_p, spearman_corr, spearman_p, 
                              class_names, output_path):
    """Create the original confidence-IoU visualization"""
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(16, 10))
    
    fig.suptitle('YOLO Model Confidence vs Bounding Box Quality (IoU) Analysis', fontsize=20, y=0.98)
    
    # 1. Scatter plot with regression
    ax1 = plt.subplot(2, 3, 1)
    scatter = ax1.scatter(df_matched['confidence'], df_matched['iou'], 
                         alpha=0.6, c=df_matched['confidence'], 
                         cmap='viridis', edgecolors='black', linewidth=0.5)
    
    z = np.polyfit(df_matched['confidence'], df_matched['iou'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df_matched['confidence'].min(), df_matched['confidence'].max(), 100)
    ax1.plot(x_line, p(x_line), "r-", linewidth=3, label=f'y={z[0]:.3f}x+{z[1]:.3f}')
    
    ax1.set_xlabel('Confidence Score', fontsize=12)
    ax1.set_ylabel('IoU', fontsize=12)
    ax1.set_title(f'Confidence vs IoU\nPearson r={pearson_corr:.3f}', fontsize=14)
    ax1.legend()
    plt.colorbar(scatter, ax=ax1, label='Confidence')
    
    # 2. Confidence distribution
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(df_matched['confidence'], bins=25, alpha=0.7, color='skyblue', edgecolor='navy')
    ax2.axvline(df_matched['confidence'].mean(), color='red', linestyle='dashed', linewidth=2)
    ax2.set_xlabel('Confidence Score', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Confidence Distribution', fontsize=14)
    
    # 3. Mean IoU by confidence intervals
    ax3 = plt.subplot(2, 3, 3)
    conf_bins = np.arange(0, 1.05, 0.1)
    df_matched['conf_bin'] = pd.cut(df_matched['confidence'], bins=conf_bins)
    bin_stats = df_matched.groupby('conf_bin')['iou'].agg(['mean', 'std', 'count'])
    bin_stats = bin_stats[bin_stats['count'] > 0]
    
    if len(bin_stats) > 0:
        bin_centers = [(interval.left + interval.right) / 2 for interval in bin_stats.index]
        ax3.errorbar(bin_centers, bin_stats['mean'], yerr=bin_stats['std'], 
                    fmt='o-', capsize=5, markersize=8, color='darkgreen')
    
    ax3.set_xlabel('Confidence Interval', fontsize=12)
    ax3.set_ylabel('Mean IoU', fontsize=12)
    ax3.set_title('Mean IoU by Confidence', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # 4. IoU distribution
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(df_matched['iou'], bins=25, alpha=0.7, color='lightcoral', edgecolor='darkred')
    ax4.axvline(df_matched['iou'].mean(), color='red', linestyle='dashed', linewidth=2)
    ax4.set_xlabel('IoU', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('IoU Distribution', fontsize=14)
    
    # 5. Performance by class
    ax5 = plt.subplot(2, 3, 5)
    class_stats = df_matched.groupby('class').agg({
        'confidence': 'mean',
        'iou': 'mean',
        'matched': 'count'
    }).sort_values('iou', ascending=True)
    
    if len(class_stats) > 0:
        y_pos = np.arange(len(class_stats))
        bars = ax5.barh(y_pos, class_stats['iou'], alpha=0.8)
        colors = plt.cm.RdYlGn(class_stats['iou'])
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(class_stats.index)
        ax5.set_xlabel('Mean IoU', fontsize=12)
        ax5.set_title('Mean IoU by Class', fontsize=14)
        
        for i, (v, n) in enumerate(zip(class_stats['iou'], class_stats['matched'])):
            ax5.text(v + 0.01, i, f'{v:.3f} (n={n})', va='center')
    
    # 6. Correlation heatmap
    ax6 = plt.subplot(2, 3, 6)
    corr_data = df_matched[['confidence', 'iou']].corr()
    sns.heatmap(corr_data, annot=True, fmt='.3f', cmap='coolwarm', 
               center=0, square=True, linewidths=1, ax=ax6)
    ax6.set_title('Correlation Matrix', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_comprehensive_report(base_path, saved_model, img_files, all_results, df_matched,
                             pearson_corr, pearson_p, spearman_corr, spearman_p,
                             class_metrics, all_tp, all_fp, all_fn, cm, class_names):
    """Save comprehensive analysis report"""
    summary_path = os.path.join(base_path, 'analysis_summary.txt')
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("YOLO Model Comprehensive Analysis Report\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Analysis timestamp: {pd.Timestamp.now()}\n")
        f.write(f"Model path: {saved_model}\n")
        f.write(f"Number of test images: {len(img_files)}\n")
        f.write(f"Total predictions: {len(all_results)}\n")
        f.write(f"Successfully matched: {len(df_matched)}\n\n")
        
        f.write("="*70 + "\n")
        f.write("1. CONFIDENCE-IoU CORRELATION ANALYSIS\n")
        f.write("="*70 + "\n")
        f.write(f"Pearson correlation: {pearson_corr:.4f} (p={pearson_p:.4e})\n")
        f.write(f"Spearman correlation: {spearman_corr:.4f} (p={spearman_p:.4e})\n")
        
        if pearson_corr > 0.7:
            strength = "strong positive"
        elif pearson_corr > 0.5:
            strength = "moderate positive"
        elif pearson_corr > 0.3:
            strength = "weak positive"
        else:
            strength = "very weak"
        f.write(f"Correlation strength: {strength}\n\n")
        
        f.write("="*70 + "\n")
        f.write("2. CLASSIFICATION PERFORMANCE METRICS\n")
        f.write("="*70 + "\n\n")
        
        # Per-class metrics table
        f.write("Per-Class Performance:\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'Support':<10}\n")
        f.write("-"*70 + "\n")
        
        for cls in sorted(class_metrics.keys()):
            m = class_metrics[cls]
            f.write(f"{cls:<20} {m['precision']:<12.3f} {m['recall']:<12.3f} "
                   f"{m['f1']:<12.3f} {m['support']:<10}\n")
        
        f.write("-"*70 + "\n")
        
        # Overall metrics
        macro_precision = np.mean([m['precision'] for m in class_metrics.values()])
        macro_recall = np.mean([m['recall'] for m in class_metrics.values()])
        macro_f1 = np.mean([m['f1'] for m in class_metrics.values()])
        
        f.write(f"\nMacro Average:\n")
        f.write(f"  Precision: {macro_precision:.3f}\n")
        f.write(f"  Recall: {macro_recall:.3f}\n")
        f.write(f"  F1 Score: {macro_f1:.3f}\n\n")
        
        # Detection statistics
        total_tp = sum(all_tp.values())
        total_fp = sum(all_fp.values())
        total_fn = sum(all_fn.values())
        
        f.write("Detection Statistics:\n")
        f.write(f"  True Positives: {total_tp}\n")
        f.write(f"  False Positives: {total_fp}\n")
        f.write(f"  False Negatives: {total_fn}\n\n")
        
        f.write("="*70 + "\n")
        f.write("3. CONFUSION MATRIX SUMMARY\n")
        f.write("="*70 + "\n")
        
        if cm is not None and cm.size > 0:
            f.write("\nTop Confusions:\n")
            # Find top confusions
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized[np.isnan(cm_normalized)] = 0
            
            all_classes = list(class_names.values()) + ['background']
            confusions = []
            
            for i, true_cls in enumerate(all_classes):
                for j, pred_cls in enumerate(all_classes):
                    if i != j and cm[i, j] > 0:
                        confusions.append((true_cls, pred_cls, cm[i, j], cm_normalized[i, j]))
            
            confusions.sort(key=lambda x: x[2], reverse=True)
            
            for true_cls, pred_cls, count, proportion in confusions[:5]:
                f.write(f"  {true_cls} → {pred_cls}: {int(count)} times ({proportion:.1%})\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("4. CONCLUSIONS\n")
        f.write("="*70 + "\n")
        
        f.write(f"\nThe analysis of {len(df_matched)} matched predictions shows:\n\n")
        f.write(f"• Confidence-IoU Correlation: {strength} correlation (r={pearson_corr:.3f})\n")
        f.write(f"  This indicates that model confidence {'' if pearson_corr > 0.5 else 'moderately '}reflects\n")
        f.write(f"  the quality of bounding box predictions.\n\n")
        
        f.write(f"• Classification Performance: Macro F1 Score of {macro_f1:.3f}\n")
        
        best_class = max(class_metrics.keys(), key=lambda x: class_metrics[x]['f1'])
        worst_class = min(class_metrics.keys(), key=lambda x: class_metrics[x]['f1'])
        
        f.write(f"  Best performing class: {best_class} (F1={class_metrics[best_class]['f1']:.3f})\n")
        f.write(f"  Worst performing class: {worst_class} (F1={class_metrics[worst_class]['f1']:.3f})\n\n")
        
        if total_fp > total_fn:
            f.write("• The model tends to over-detect (more false positives than false negatives)\n")
        else:
            f.write("• The model tends to under-detect (more false negatives than false positives)\n")
    
    print(f"✓ Comprehensive report saved to: {os.path.abspath(summary_path)}")


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced YOLO analysis with confidence-IoU correlation, F1 scores, and confusion matrix')
    
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
    
    parser.add_argument('-c', '--confThreshold',
                        help='Confidence threshold for predictions (default: 0.25)',
                        type=float, default=0.25)
    
    parser.add_argument('-t', '--iouThreshold',
                        help='IoU threshold for matching (default: 0.5)',
                        type=float, default=0.5)
    
    args = parser.parse_args()
    
    # Check paths
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
        analyze_confidence_iou_correlation(
            args.savedModel, 
            args.imgDir, 
            args.gtDir, 
            args.outputDir,
            args.confThreshold,
            args.iouThreshold
        )
    except Exception as e:
        print(f"Runtime error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()