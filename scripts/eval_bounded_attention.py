import argparse
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms.functional import to_tensor
import json

def load_clip():
    """Load CLIP model for semantic similarity scoring."""
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

def load_object_detector():
    """Load FasterRCNN for object detection."""
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

def compute_clip_score(model, processor, image_path, text):
    """Compute CLIP score between image and text."""
    image = Image.open(image_path)
    inputs = processor(images=image, text=text, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    return outputs.logits_per_image.item()

def compute_clip_count_accuracy(detector, image_path, target_classes, expected_counts, confidence_threshold=0.7):
    """Compute counting accuracy for specific object classes."""
    image = Image.open(image_path)
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        predictions = detector(image_tensor)
    
    pred_classes = predictions[0]['labels'].numpy()
    pred_scores = predictions[0]['scores'].numpy()
    
    # Filter predictions by confidence threshold
    confident_mask = pred_scores > confidence_threshold
    confident_preds = pred_classes[confident_mask]
    
    # Count occurrences of each target class
    actual_counts = {}
    for cls in target_classes:
        actual_counts[str(cls)] = np.sum(confident_preds == cls)
    
    # Calculate counting accuracy
    count_accuracies = {}
    for cls_str, expected in expected_counts.items():
        actual = actual_counts.get(cls_str, 0)
        accuracy = 1.0 - abs(expected - actual) / max(expected, actual) if max(expected, actual) > 0 else 1.0
        count_accuracies[cls_str] = {
            'expected': expected,
            'actual': actual,
            'accuracy': accuracy
        }
    
    return count_accuracies

def compute_object_metrics(detector, image_path, target_classes, confidence_threshold=0.7):
    """Compute precision, recall, F1 for specific object classes."""
    image = Image.open(image_path)
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        predictions = detector(image_tensor)
    
    pred_classes = predictions[0]['labels'].numpy()
    pred_scores = predictions[0]['scores'].numpy()
    
    # Filter predictions by confidence threshold
    confident_preds = pred_classes[pred_scores > confidence_threshold]
    
    # Count predictions for target classes
    true_positives = sum(1 for c in confident_preds if c in target_classes)
    false_positives = len(confident_preds) - true_positives
    false_negatives = len(target_classes) - true_positives
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }

def convert_to_json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    return obj

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples_dir', type=str, required=True, help='Directory containing generated samples')
    parser.add_argument('--prompt', type=str, required=True, help='Original prompt used for generation')
    parser.add_argument('--target_objects', type=str, required=True, help='Comma-separated list of target object classes')
    parser.add_argument('--expected_counts', type=str, required=True, help='Comma-separated list of expected counts (same order as target_objects)')
    args = parser.parse_args()
    
    # Load models
    print("Loading CLIP model...")
    clip_model, clip_processor = load_clip()
    
    print("Loading object detector...")
    detector = load_object_detector()
    
    # Parse target objects and expected counts
    target_objects = [obj.strip() for obj in args.target_objects.split(',')]
    expected_counts_list = [int(c.strip()) for c in args.expected_counts.split(',')]
    
    # Map target objects to COCO class indices
    coco_class_mapping = {
        'cup': 47,
        'plate': 48,
        # Add more mappings as needed
    }
    target_classes = [coco_class_mapping[obj] for obj in target_objects if obj in coco_class_mapping]
    expected_counts = {str(coco_class_mapping[obj]): count 
                      for obj, count in zip(target_objects, expected_counts_list) 
                      if obj in coco_class_mapping}
    
    # Evaluate all samples
    results = []
    clip_scores = []
    count_accuracies = []
    
    print("Evaluating samples...")
    for image_file in tqdm(os.listdir(args.samples_dir)):
        if not image_file.endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        image_path = os.path.join(args.samples_dir, image_file)
        
        # Compute CLIP score
        clip_score = compute_clip_score(clip_model, clip_processor, image_path, args.prompt)
        clip_scores.append(clip_score)
        
        # Compute object detection metrics
        metrics = compute_object_metrics(detector, image_path, target_classes)
        
        # Compute counting accuracy
        count_acc = compute_clip_count_accuracy(detector, image_path, target_classes, expected_counts)
        count_accuracies.append(count_acc)
        
        results.append({
            'image': image_file,
            'clip_score': clip_score,
            'count_accuracy': convert_to_json_serializable(count_acc),
            **convert_to_json_serializable(metrics)
        })
    
    # Compute average metrics
    avg_count_acc = {}
    for cls in target_classes:
        cls_str = str(cls)
        accuracies = [r['count_accuracy'][cls_str]['accuracy'] for r in results]
        avg_count_acc[cls_str] = float(np.mean(accuracies))
    
    avg_metrics = {
        'avg_clip_score': float(np.mean(clip_scores)),
        'avg_precision': float(np.mean([r['precision'] for r in results])),
        'avg_recall': float(np.mean([r['recall'] for r in results])),
        'avg_f1': float(np.mean([r['f1'] for r in results])),
        'avg_count_accuracy': avg_count_acc
    }
    
    # Save results
    output_file = os.path.join(args.samples_dir, 'evaluation_results.json')
    with open(output_file, 'w') as f:
        json.dump({
            'prompt': args.prompt,
            'target_objects': target_objects,
            'expected_counts': convert_to_json_serializable(expected_counts),
            'individual_results': results,
            'average_metrics': avg_metrics
        }, f, indent=2)
    
    # Print summary
    print("\nEvaluation Results:")
    print(f"Average CLIP Score: {avg_metrics['avg_clip_score']:.4f}")
    print(f"Average Precision: {avg_metrics['avg_precision']:.4f}")
    print(f"Average Recall: {avg_metrics['avg_recall']:.4f}")
    print(f"Average F1 Score: {avg_metrics['avg_f1']:.4f}")
    print("\nCounting Accuracy per Class:")
    for cls in target_classes:
        cls_str = str(cls)
        obj_name = [k for k, v in coco_class_mapping.items() if str(v) == cls_str][0]
        print(f"  {obj_name}: {avg_metrics['avg_count_accuracy'][cls_str]:.4f}")
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main() 