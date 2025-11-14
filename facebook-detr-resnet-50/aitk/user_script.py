# user_script.py
from logging import getLogger
from pathlib import Path
import numpy as np
import torch
import transformers
from torch import from_numpy, tensor
from torch.utils.data import Dataset
from olive.data.registry import Registry
from datasets import load_dataset

logger = getLogger(__name__)

def get_coco_label_map():
    """Get COCO dataset label mapping for DETR"""
    import json
    cache_file = Path(f"./cache/data/coco_detr_class_index.json")
    if not cache_file.exists():
        # DETR uses COCO's 91 classes (including background at index 91)
        coco_to_detr = {
            1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
            11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19,
            22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29,
            35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39,
            46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49,
            56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59,
            67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69,
            80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79
        }
        
        cache_file.parent.resolve().mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(coco_to_detr, f)
    else:
        with open(cache_file) as f:
            coco_to_detr = json.loads(f.read())
    
    return {int(k): v for k, v in coco_to_detr.items()}

def convert_bbox_to_detr_format(bbox, image_width, image_height):
    """Convert COCO bbox [x, y, width, height] to DETR format [cx, cy, w, h] normalized"""
    x, y, w, h = bbox
    # Convert to center coordinates and normalize
    cx = (x + w / 2) / image_width
    cy = (y + h / 2) / image_height
    w_norm = w / image_width
    h_norm = h / image_height
    return [cx, cy, w_norm, h_norm]

def prepare_detr_targets(annotations, image_width=800, image_height=600, max_objects=100):
    """Prepare target annotations in DETR format"""
    if not annotations or len(annotations) == 0:
        return {
            'class_labels': torch.full((max_objects,), 91, dtype=torch.long),
            'boxes': torch.zeros((max_objects, 4), dtype=torch.float32)
        }
    
    label_map = get_coco_label_map()
    labels = []
    boxes = []
    
    for ann in annotations[:max_objects]:
        if isinstance(ann, dict):
            category_id = ann.get('category_id', 1)
            bbox = ann.get('bbox', [0, 0, 1, 1])
            
            detr_label = label_map.get(category_id, 79)
            labels.append(detr_label)
            
            detr_bbox = convert_bbox_to_detr_format(bbox, image_width, image_height)
            boxes.append(detr_bbox)
    
    while len(labels) < max_objects:
        labels.append(91)
        boxes.append([0.0, 0.0, 0.0, 0.0])
    
    return {
        'class_labels': torch.tensor(labels[:max_objects], dtype=torch.long),
        'boxes': torch.tensor(boxes[:max_objects], dtype=torch.float32)
    }

class CocoDetrDataset(Dataset):
    def __init__(self, images, targets):
        """
        Initialize with lists of tensors instead of numpy arrays
        """
        self.images = images  # List of torch tensors
        self.targets = targets  # List of target dictionaries
        
    def __len__(self):
        return min(len(self.images), len(self.targets))

    def __getitem__(self, idx):
        return {"pixel_values": self.images[idx]}, self.targets[idx]

# Load DETR ResNet-50 processor with fixed size
from transformers import AutoImageProcessor
processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50", use_fast=True)

# Override the processor's size to ensure 800x800 output
processor.size = {"height": 800, "width": 800}
processor.do_resize = True
processor.do_rescale = True

@Registry.register_pre_process()
def detr_dataset_pre_process(output_data, **kwargs):
    """
    Pre-process COCO dataset for DETR with fixed 800x800 input size
    """
    shuffle = kwargs.get("shuffle", True)
    size = kwargs.get("size", 500)
    max_objects = kwargs.get("max_objects", 100)
    cache_key = kwargs.get("cache_key")
    
    # Check if we need to load the dataset directly
    if output_data is None or not hasattr(output_data, '__iter__'):
        logger.info("Loading COCO dataset directly...")
        try:
            dataset = load_dataset("detection-datasets/coco", split="validation")
            if shuffle:
                seed = kwargs.get("seed", 42)
                dataset = dataset.shuffle(seed=seed)
            output_data = dataset
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    # Handle shuffling if not already done
    if shuffle and hasattr(output_data, 'shuffle'):
        seed = kwargs.get("seed", 42)
        output_data = output_data.shuffle(seed=seed)
    
    # Check cache
    cache_file = None
    if cache_key:
        cache_file = Path(f"./cache/data/{cache_key}_detr_{size}.pt")
        if cache_file.exists():
            logger.info(f"Loading cached data from {cache_file}")
            try:
                cached_data = torch.load(cache_file)
                return CocoDetrDataset(cached_data['images'], cached_data['targets'])
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}, regenerating...")

    targets = []
    images = []
    
    logger.info(f"Processing {size} samples from COCO dataset...")
    
    processed_count = 0
    for i, sample in enumerate(output_data):
        if processed_count >= size:
            break
            
        try:
            # Handle detection-datasets/coco format
            image = sample["image"]
            
            # Get original image dimensions
            if hasattr(image, 'size'):
                image_width, image_height = image.size
            else:
                # Fallback if size is not available
                image_width, image_height = 800, 600
            
            # Process image with DETR processor
            if not hasattr(image, 'convert'):
                # Handle numpy arrays or other formats
                from PIL import Image
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                else:
                    image = Image.fromarray(np.array(image))
            
            image = image.convert("RGB")
            
            # Process with DETR processor - force 800x800 size
            processed = processor(
                images=image, 
                return_tensors="pt",
                size={"height": 800, "width": 800},
                do_resize=True
            )
            image_tensor = processed["pixel_values"][0]  # Shape: [3, 800, 800]
            
            # Verify tensor shape is exactly [3, 800, 800]
            expected_shape = (3, 800, 800)
            if image_tensor.shape != expected_shape:
                logger.warning(f"Unexpected image tensor shape: {image_tensor.shape}, expected {expected_shape}")
                # Resize manually if needed
                import torch.nn.functional as F
                if len(image_tensor.shape) == 3:
                    image_tensor = F.interpolate(
                        image_tensor.unsqueeze(0), 
                        size=(800, 800), 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze(0)
                else:
                    logger.warning(f"Skipping sample {i} due to invalid tensor shape")
                    continue
            
            # Handle annotations from detection-datasets/coco
            objects = sample.get('objects', {})
            annotations = []
            
            if isinstance(objects, dict):
                # Format: {'bbox': [...], 'category': [...]}
                bboxes = objects.get('bbox', [])
                categories = objects.get('category', [])
                
                for bbox, category in zip(bboxes, categories):
                    annotations.append({
                        'bbox': bbox,
                        'category_id': category
                    })
            elif isinstance(objects, list):
                # Format: [{'bbox': [...], 'category_id': ...}, ...]
                annotations = objects
            
            # Prepare DETR targets
            target = prepare_detr_targets(
                annotations, 
                image_width=image_width, 
                image_height=image_height,
                max_objects=max_objects
            )
            
            images.append(image_tensor)
            targets.append(target)
            processed_count += 1
            
            if processed_count % 50 == 0:
                logger.info(f"Processed {processed_count}/{size} samples, last image shape: {image_tensor.shape}")
                
        except Exception as e:
            logger.warning(f"Skipping sample {i} due to error: {e}")
            continue

    logger.info(f"Successfully processed {len(images)} samples")
    
    # Verify all images have the same shape
    if images:
        shapes = [img.shape for img in images]
        unique_shapes = set(shapes)
        if len(unique_shapes) > 1:
            logger.warning(f"Found multiple image shapes: {unique_shapes}")
        else:
            logger.info(f"All images have consistent shape: {shapes[0]}")
    
    # Create dataset with tensor lists
    result_data = CocoDetrDataset(images, targets)

    # Cache the processed data using torch.save
    if cache_file and len(images) > 0:
        logger.info(f"Caching processed data to {cache_file}")
        cache_file.parent.resolve().mkdir(parents=True, exist_ok=True)
        try:
            torch.save({
                'images': images,
                'targets': targets
            }, cache_file)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    return result_data

@Registry.register_post_process()
def detr_dataset_post_process(output):
    """Post-process DETR model outputs"""
    if hasattr(output, 'logits') and hasattr(output, 'pred_boxes'):
        class_probs = torch.softmax(output.logits, dim=-1)
        pred_classes = class_probs[..., :-1].argmax(dim=-1)
        confidence_scores = class_probs[..., :-1].max(dim=-1)[0]
        
        return {
            'pred_logits': output.logits,
            'pred_boxes': output.pred_boxes,
            'pred_classes': pred_classes,
            'confidence_scores': confidence_scores
        }
    else:
        return output.logits.argmax(axis=-1) if hasattr(output, 'logits') else output