"""
Data loading and preprocessing utilities for SAR images
Handles image loading, augmentation, and batch generation
Supports both synthetic data and COCO format datasets (e.g., SARDet-100K)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'np_mlp_autograd'))

import numpy as np
from typing import Tuple, List, Optional, Callable, Dict, Any, Union
import glob
import json
import random
from pathlib import Path

# Optional: PIL for image loading
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Optional: scipy for image resizing
try:
    from scipy.ndimage import zoom as scipy_zoom
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class COCOSARDataset:
    """
    Dataset class for COCO-format SAR detection datasets (e.g., SARDet-100K)
    
    Loads images and annotations in COCO JSON format, generates segmentation
    masks from bounding boxes, and supports multi-class segmentation.
    
    Expected directory structure:
        dataset_root/
            images/
                train/
                    *.jpg
                val/
                    *.jpg
                test/
                    *.jpg
            annotations/
                instances_train.json
                instances_val.json
                instances_test.json
    """
    
    def __init__(self, 
                 annotation_file: str, 
                 image_dir: str,
                 image_size: Tuple[int, int] = (512, 512),
                 num_classes: int = 6,
                 augment: bool = True,
                 normalize: bool = True,
                 binary_segmentation: bool = False):
        """
        Args:
            annotation_file: Path to COCO JSON annotation file
            image_dir: Directory containing images
            image_size: Target image size (height, width)
            num_classes: Number of object categories (default: 6 for SARDet-100K)
            augment: Whether to apply data augmentation
            normalize: Whether to normalize images
            binary_segmentation: If True, treat all objects as foreground (binary segmentation)
        """
        self.annotation_file = annotation_file
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.num_classes = num_classes
        self.augment = augment
        self.normalize = normalize
        self.binary_segmentation = binary_segmentation
        
        # Load COCO annotations
        print(f"Loading COCO annotations from {annotation_file}...")
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Parse annotations(as dictionaries)
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        
        # Create category ID mapping: COCO category IDs -> class labels
        # Background is always class 0, object categories start from 1
        sorted_cat_ids = sorted(self.categories.keys())
        self.category_id_to_class = {cat_id: cat_id + 1 for cat_id in sorted_cat_ids}
        # Note: If category_ids are 0,1,2,3,4,5 -> classes become 1,2,3,4,5,6 (with background=0)
        
        print(f"Category mapping: {self.category_id_to_class}")
        print(f"Categories: {self.categories}")
        
        # Group annotations by image_id
        self.annotations_by_image = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations_by_image:
                self.annotations_by_image[img_id] = []
            self.annotations_by_image[img_id].append(ann)
        
        # Create list of valid image IDs (images that have annotations)
        self.image_ids = list(self.annotations_by_image.keys())
        
        print(f"Loaded {len(self.image_ids)} images with {len(self.coco_data['annotations'])} annotations")
        
        # Validate num_classes configuration
        if not self.binary_segmentation:
            actual_num_classes = len(self.categories) + 1  # +1 for background
            if num_classes != actual_num_classes:
                print(f"WARNING: num_classes={num_classes} but dataset has {len(self.categories)} categories.")
                print(f"         Expected num_classes={actual_num_classes} (including background).")
                print(f"         The model may fail if num_classes is incorrect!")
                print(f"         Using dataset's actual classes: {actual_num_classes}")
                self.num_classes = actual_num_classes
        else:
            # Binary segmentation: background + foreground
            if num_classes != 2:
                print(f"WARNING: binary_segmentation=True but num_classes={num_classes}")
                print(f"         For binary segmentation, num_classes should be 2.")
                self.num_classes = 2
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get image and mask pair
        
        Returns:
            image: (C, H, W) grayscale image in channel-first format
            mask: (H, W) with class indices [0, num_classes-1]
        """
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        annotations = self.annotations_by_image[img_id] # annotations are grouped by image id for easy reference
        
        # Load image
        img_path = self.image_dir / img_info['file_name']
        image = self._load_image(str(img_path))
        
        # Generate mask from annotations
        mask = self._generate_mask(img_info, annotations)
        
        # Apply augmentation
        if self.augment:
            image, mask = self._augment(image, mask)
        
        # Normalize
        if self.normalize:
            image = self._normalize(image)
        
        # Convert to channel-first format (C, H, W) to match U-Net expectations
        if len(image.shape) == 2:
            image = image[np.newaxis, :, :]  # (H, W) -> (1, H, W)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            image = image.transpose(2, 0, 1)  # (H, W, 1) -> (1, H, W)
        
        return image, mask
    
    def _load_image(self, img_path: str) -> np.ndarray:
        """Load and preprocess image"""
        if HAS_PIL:
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            img = img.resize((self.image_size[1], self.image_size[0]), Image.Resampling.BILINEAR)
            image = np.array(img, dtype=np.float32)
        else:
            # Fallback to numpy if PIL not available
            # Assume images are already stored as .npy files
            image = np.load(img_path.replace('.jpg', '.npy'))
            if len(image.shape) == 3:
                image = np.mean(image, axis=2)  # Convert to grayscale
            
            # Resize if needed
            if image.shape != self.image_size:
                if HAS_SCIPY:
                    zoom_factors = (self.image_size[0] / image.shape[0], 
                                   self.image_size[1] / image.shape[1])
                    image = scipy_zoom(image, zoom_factors, order=1)
                else:
                    # Simple nearest neighbor resize without scipy
                    image = self._simple_resize(image, self.image_size)
        
        # Add channel dimension: (H, W) -> (H, W, 1)
        image = np.expand_dims(image, axis=-1)
        return image
    
    def _simple_resize(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Simple nearest neighbor resize without external dependencies"""
        old_h, old_w = image.shape[:2]
        new_h, new_w = target_size
        
        # Create coordinate arrays
        row_indices = (np.arange(new_h) * old_h / new_h).astype(int)
        col_indices = (np.arange(new_w) * old_w / new_w).astype(int)
        
        # Index into original image
        resized = image[row_indices[:, np.newaxis], col_indices]
        return resized
    
    def _generate_mask(self, img_info: Dict[str, Any], annotations: List[Dict[str, Any]]) -> np.ndarray:
        """
        Generate segmentation mask from COCO annotations
        
        For detection datasets with bounding boxes, creates masks by filling
        the bounding box regions with the corresponding category ID.
        """
        orig_height = img_info['height']
        orig_width = img_info['width']
        
        # Initialize mask
        if self.binary_segmentation:
            # Binary: background (0) and foreground (1)
            mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
        else:
            # Multi-class
            mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
        
        # Fill mask with bounding boxes
        for ann in annotations:
            if ann.get('ignore', 0) == 1 or ann.get('iscrowd', 0) == 1:
                continue  # Skip ignored annotations
            
            bbox = ann['bbox']  # [x, y, width, height]
            category_id = ann['category_id']
            
            # Convert to integer coordinates
            x, y, w, h = [int(v) for v in bbox]
            x2, y2 = x + w, y + h
            
            # Clip to image bounds
            x = max(0, min(x, orig_width - 1))
            y = max(0, min(y, orig_height - 1))
            x2 = max(0, min(x2, orig_width))
            y2 = max(0, min(y2, orig_height))
            
            if self.binary_segmentation:
                # All objects are class 1 (foreground)
                mask[y:y2, x:x2] = 1
            else:
                # Map COCO category_id to 0-indexed class label
                class_label = self.category_id_to_class.get(category_id, 0)
                mask[y:y2, x:x2] = class_label
        
        # Resize mask to target size
        if (orig_height, orig_width) != self.image_size:
            # Use nearest neighbor to preserve class labels
            if HAS_SCIPY:
                zoom_factors = (self.image_size[0] / orig_height, 
                               self.image_size[1] / orig_width)
                mask = scipy_zoom(mask, zoom_factors, order=0)  # order=0 for nearest neighbor
            else:
                mask = self._simple_resize(mask, self.image_size)
        
        # Convert to one-hot encoding
        # if self.binary_segmentation:
        #     num_output_classes = 2  # background + foreground
        # else:
        #     num_output_classes = self.num_classes + 1  # +1 for background
        
        # mask_one_hot = np.zeros((*self.image_size, num_output_classes), dtype=np.float32)
        # for c in range(num_output_classes):
        #     mask_one_hot[:, :, c] = (mask == c).astype(np.float32)
        
        # return mask_one_hot
        return mask
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range"""
        image = image.astype(np.float32)
        if image.max() > 0:
            image = image / 255.0
        return image
    
    def _augment(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation"""
        # Horizontal flip
        if random.random() > 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
        
        # Vertical flip
        if random.random() > 0.5:
            image = np.flipud(image)
            mask = np.flipud(mask)
        
        # 90-degree rotations
        if random.random() > 0.5:
            k = random.randint(1, 3)  # Number of 90-degree rotations
            image = np.rot90(image, k)
            mask = np.rot90(mask, k)
        
        # Brightness adjustment
        if random.random() > 0.5:
            factor = 0.8 + random.random() * 0.4  # [0.8, 1.2]
            image = np.clip(image * factor, 0, 255)
        
        # Gaussian noise
        if random.random() > 0.7:
            noise = np.random.normal(0, 5, image.shape)
            image = np.clip(image + noise, 0, 255)
        
        return image, mask


class SARDataset:
    """
    Dataset class for SAR images and segmentation masks
    
    Handles loading, preprocessing, and augmentation of SAR images
    for training U-Net segmentation models.
    """
    
    def __init__(self, image_dir: str, mask_dir: str, 
                 image_size: Tuple[int, int] = (256, 256),
                 augment: bool = True,
                 normalize: bool = True):
        """
        Args:
            image_dir: Directory containing SAR images
            mask_dir: Directory containing segmentation masks
            image_size: Target image size (height, width)
            augment: Whether to apply data augmentation
            normalize: Whether to normalize images
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.augment = augment
        self.normalize = normalize
        
        # Load file paths
        self.image_paths = self._load_file_paths(image_dir)
        self.mask_paths = self._load_file_paths(mask_dir)
        
        # Verify matching number of images and masks
        if len(self.image_paths) != len(self.mask_paths):
            print(f"Warning: Number of images ({len(self.image_paths)}) doesn't match masks ({len(self.mask_paths)})")
        
        print(f"Loaded {len(self.image_paths)} images from {image_dir}")
    
    def _load_file_paths(self, directory: str) -> List[str]:
        """Load all image file paths from directory"""
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} does not exist")
            return []
        
        # Support common image formats
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.npy']
        file_paths = []
        
        for ext in extensions:
            pattern = os.path.join(directory, ext)
            file_paths.extend(glob.glob(pattern))
        
        return sorted(file_paths)
    
    def __len__(self) -> int:
        """Return number of samples in dataset"""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a single sample
        
        Returns:
            image: Preprocessed image (channels, height, width)
            mask: Segmentation mask (height, width)
        """
        # Load image and mask
        image = self._load_image(self.image_paths[idx])
        mask = self._load_mask(self.mask_paths[idx])
        
        # Resize
        image = self._resize(image, self.image_size)
        mask = self._resize(mask, self.image_size, is_mask=True)
        
        # Apply augmentation
        if self.augment:
            image, mask = self._augment(image, mask)
        
        # Normalize image
        if self.normalize:
            image = self._normalize(image)
        
        # Convert to channel-first format (C, H, W)
        if len(image.shape) == 2:
            image = image[np.newaxis, :, :]  # Add channel dimension
        elif len(image.shape) == 3 and image.shape[2] in [1, 3]:
            image = image.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
        
        return image, mask
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load image from file"""
        if path.endswith('.npy'):
            image = np.load(path)
        else:
            try:
                from PIL import Image
                image = np.array(Image.open(path))
            except ImportError:
                # Fallback: try loading as raw data
                print("PIL not available, attempting to load as numpy array")
                image = np.load(path)
        
        return image.astype(np.float32)
    
    def _load_mask(self, path: str) -> np.ndarray:
        """Load segmentation mask from file"""
        if path.endswith('.npy'):
            mask = np.load(path)
        else:
            try:
                from PIL import Image
                mask = np.array(Image.open(path))
            except ImportError:
                mask = np.load(path)
        
        return mask.astype(np.int32)
    
    def _resize(self, image: np.ndarray, size: Tuple[int, int], is_mask: bool = False) -> np.ndarray:
        """Resize image to target size"""
        if image.shape[:2] == size:
            return image
        
        try:
            from PIL import Image as PILImage
            if is_mask:
                # Use nearest neighbor for masks to preserve labels
                resized = PILImage.fromarray(image).resize((size[1], size[0]), PILImage.NEAREST)
            else:
                # Use bilinear for images
                resized = PILImage.fromarray(image).resize((size[1], size[0]), PILImage.BILINEAR)
            return np.array(resized)
        except ImportError:
            # Fallback: simple nearest neighbor
            return self._resize_nearest(image, size)
    
    def _resize_nearest(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Simple nearest neighbor resize (fallback)"""
        h, w = image.shape[:2]
        new_h, new_w = size
        
        # Calculate scale factors
        scale_h = h / new_h
        scale_w = w / new_w
        
        # Create output array
        if len(image.shape) == 3:
            resized = np.zeros((new_h, new_w, image.shape[2]), dtype=image.dtype)
        else:
            resized = np.zeros((new_h, new_w), dtype=image.dtype)
        
        # Nearest neighbor interpolation
        for i in range(new_h):
            for j in range(new_w):
                src_i = int(i * scale_h)
                src_j = int(j * scale_w)
                resized[i, j] = image[src_i, src_j]
        
        return resized
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] or standardize
        
        For SAR images, apply log transform to reduce speckle noise effect
        """
        # Apply log transform for SAR images (reduces dynamic range)
        image = np.log1p(np.abs(image))  # log(1 + x) to handle zeros
        
        # Normalize to [0, 1]
        min_val = np.min(image)
        max_val = np.max(image)
        
        if max_val > min_val:
            image = (image - min_val) / (max_val - min_val)
        
        return image.astype(np.float32)
    
    def _augment(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply data augmentation
        
        Augmentations:
        - Random horizontal flip
        - Random vertical flip
        - Random rotation (90, 180, 270 degrees)
        - Random brightness adjustment (for images)
        """
        # Random horizontal flip
        if np.random.rand() > 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
        
        # Random vertical flip
        if np.random.rand() > 0.5:
            image = np.flipud(image)
            mask = np.flipud(mask)
        
        # Random rotation (90 degree increments)
        k = np.random.randint(0, 4)
        if k > 0:
            image = np.rot90(image, k)
            mask = np.rot90(mask, k)
        
        # Random brightness adjustment (only for images)
        if np.random.rand() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 255)
        
        return image, mask


class DataLoader:
    """
    Data loader for batch generation
    
    Handles batching, shuffling, and iteration over dataset
    """
    
    def __init__(self, dataset: COCOSARDataset, batch_size: int = 8, 
                 shuffle: bool = True, drop_last: bool = False):
        """
        Args:
            dataset: SARDataset instance
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data each epoch
            drop_last: Whether to drop last incomplete batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        self.num_samples = len(dataset)
        self.num_batches = self.num_samples // batch_size
        if not drop_last and self.num_samples % batch_size != 0:
            self.num_batches += 1
        
        self.indices = np.arange(self.num_samples)
        if shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self) -> int:
        """Return number of batches"""
        return self.num_batches
    
    def __iter__(self):
        """Create iterator"""
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        for i in range(self.num_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, self.num_samples)
            
            batch_indices = self.indices[start_idx:end_idx]
            
            # Load batch
            images = []
            masks = []
            
            for idx in batch_indices:
                image, mask = self.dataset[idx]
                images.append(image)
                masks.append(mask)
            
            # Stack into batch tensors
            images_batch = np.stack(images, axis=0).astype(np.float64)
            masks_batch = np.stack(masks, axis=0).astype(np.int32)
            
            yield images_batch, masks_batch


def create_synthetic_sar_data(num_samples: int = 100, 
                              image_size: Tuple[int, int] = (256, 256),
                              num_classes: int = 2,
                              output_dir: str = './synthetic_data') -> Tuple[str, str]:
    """
    Create synthetic SAR data for testing
    
    Generates synthetic images with simple geometric shapes as targets
    
    Args:
        num_samples: Number of samples to generate
        image_size: Image size (height, width)
        num_classes: Number of segmentation classes
        output_dir: Directory to save synthetic data
    
    Returns:
        Tuple of (image_dir, mask_dir)
    """
    image_dir = os.path.join(output_dir, 'images')
    mask_dir = os.path.join(output_dir, 'masks')
    
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    print(f"Generating {num_samples} synthetic SAR images...")
    
    for i in range(num_samples):
        # Generate background with speckle noise (characteristic of SAR)
        image = np.random.rayleigh(50, image_size).astype(np.float32)
        
        # Add some coherent scattering (bright spots)
        num_targets = np.random.randint(1, 5)
        mask = np.zeros(image_size, dtype=np.int32)
        
        for _ in range(num_targets):
            # Random position and size
            cx = np.random.randint(20, image_size[1] - 20)
            cy = np.random.randint(20, image_size[0] - 20)
            radius = np.random.randint(10, 30)
            
            # Create circular target
            y, x = np.ogrid[:image_size[0], :image_size[1]]
            dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
            
            target_mask = dist_from_center <= radius
            
            # Add bright return to image
            image[target_mask] += np.random.uniform(100, 200)
            
            # Update segmentation mask
            mask[target_mask] = 1 if num_classes == 2 else np.random.randint(1, num_classes)
        
        # Add noise
        image += np.random.normal(0, 10, image_size)
        image = np.clip(image, 0, 255)
        
        # Save image and mask
        np.save(os.path.join(image_dir, f'image_{i:04d}.npy'), image)
        np.save(os.path.join(mask_dir, f'mask_{i:04d}.npy'), mask)
    
    print(f"Synthetic data saved to {output_dir}")
    return image_dir, mask_dir


def split_dataset(dataset: COCOSARDataset, train_ratio: float = 0.8) -> Tuple[SARDataset, SARDataset]:
    """
    Split dataset into training and validation sets
    
    Args:
        dataset: Original dataset
        train_ratio: Ratio of training samples
    
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    num_samples = len(dataset)   #where __len__ is defined
    num_train = int(num_samples * train_ratio)
    
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    
    # Create subset datasets
    train_dataset = SARDatasetSubset(dataset, train_indices)
    val_dataset = SARDatasetSubset(dataset, val_indices)
    
    return train_dataset, val_dataset


class SARDatasetSubset:
    """Subset of a dataset"""
    
    def __init__(self, dataset: SARDataset, indices: np.ndarray):
        self.dataset = dataset
        self.indices = indices
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.dataset[self.indices[idx]]
