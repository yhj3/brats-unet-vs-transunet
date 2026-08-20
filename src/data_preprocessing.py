#data_preprocessing.py
"""
Data Preprocessing Module for Brain MRI Segmentation
"""

import numpy as np
import pandas as pd
import nibabel as nib
import cv2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import glob
from typing import Tuple, List, Dict, Optional, Union
import qma

plotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class MRIDataPreprocessor:
    """
    Comprehensive MRI data preprocessing for brain segmentation
    """
    
    def __init__(self, target_size: Tuple[int, int, int] = (128, 128, 128)):
        """
        Initialize MRI data preprocessor
        
        Args:
            target_size: Target size for MRI volumes (depth, height, width)
        """
        self.target_size = target_size
        self.scaler = StandardScaler()
        self.intensity_stats = {}
        
    def load_nifti_volume(self, filepath: str) -> np.ndarray:
        """
        Load NIfTI volume from file
        
        Args:
            filepath: Path to NIfTI file
            
        Returns:
            3D numpy array
        """
        try:
            nii_img = nib.load(filepath)
            volume = nii_img.get_fdata()
            return volume.astype(np.float32)
        except Exception as e:
            raise ValueError(f"Error loading NIfTI file {filepath}: {str(e)}")
    
    def load_dicom_series(self, dicom_dir: str) -> np.ndarray:
        """
        Load DICOM series from directory
        
        Args:
            dicom_dir: Directory containing DICOM files
            
        Returns:
            3D numpy array
        """
        try:
            import pydicom
            
            # Get all DICOM files
            dicom_files = glob.glob(os.path.join(dicom_dir, "*.dcm"))
            if not dicom_files:
                raise ValueError(f"No DICOM files found in {dicom_dir}")
            
            # Sort files by instance number
            dicom_data = []
            for file in dicom_files:
                ds = pydicom.dcmread(file)
                dicom_data.append((ds.InstanceNumber, ds.pixel_array))
            
            dicom_data.sort(key=lambda x: x[0])
            
            # Stack slices
            volume = np.stack([data[1] for data in dicom_data], axis=0)
            return volume.astype(np.float32)
            
        except ImportError:
            raise ImportError("pydicom is required for DICOM loading. Install with: pip install pydicom")
        except Exception as e:
            raise ValueError(f"Error loading DICOM series from {dicom_dir}: {str(e)}")
    
    def normalize_intensity(self, volume: np.ndarray, method: str = 'z_score') -> np.ndarray:
        """
        Normalize volume intensity
        
        Args:
            volume: Input volume
            method: Normalization method ('z_score', 'min_max', 'percentile')
            
        Returns:
            Normalized volume
        """
        if method == 'z_score':
            # Z-score normalization
            mean_val = np.mean(volume)
            std_val = np.std(volume)
            if std_val > 0:
                normalized = (volume - mean_val) / std_val
            else:
                normalized = volume - mean_val
                
        elif method == 'min_max':
            # Min-max normalization to [0, 1]
            min_val = np.min(volume)
            max_val = np.max(volume)
            if max_val > min_val:
                normalized = (volume - min_val) / (max_val - min_val)
            else:
                normalized = np.zeros_like(volume)
                
        elif method == 'percentile':
            # Percentile-based normalization (robust to outliers)
            p1, p99 = np.percentile(volume, [1, 99])
            if p99 > p1:
                normalized = np.clip((volume - p1) / (p99 - p1), 0, 1)
            else:
                normalized = np.zeros_like(volume)
                
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized.astype(np.float32)
    
    def skull_stripping(self, volume: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """
        Simple skull stripping using thresholding
        
        Args:
            volume: Input volume
            threshold: Intensity threshold for brain tissue
            
        Returns:
            Skull-stripped volume
        """
        # Normalize volume first
        normalized = self.normalize_intensity(volume, method='percentile')
        
        # Create brain mask
        brain_mask = normalized > threshold
        
        # Apply morphological operations to clean up mask
        for slice_idx in range(brain_mask.shape[0]):
            slice_mask = brain_mask[slice_idx].astype(np.uint8)
            
            # Morphological closing to fill holes
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            slice_mask = cv2.morphologyEx(slice_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find largest connected component (brain)
            num_labels, labels = cv2.connectedComponents(slice_mask)
            if num_labels > 1:
                # Find largest component (excluding background)
                largest_component = 1
                largest_size = 0
                for label in range(1, num_labels):
                    size = np.sum(labels == label)
                    if size > largest_size:
                        largest_size = size
                        largest_component = label
                
                slice_mask = (labels == largest_component).astype(np.uint8)
            
            brain_mask[slice_idx] = slice_mask
        
        # Apply mask to original volume
        skull_stripped = volume * brain_mask
        
        return skull_stripped
    
    def bias_field_correction(self, volume: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Simple bias field correction using polynomial fitting
        
        Args:
            volume: Input volume
            mask: Brain mask (optional)
            
        Returns:
            Bias-corrected volume
        """
        if mask is None:
            # Create simple mask
            mask = volume > (np.mean(volume) * 0.1)
        
        corrected_volume = volume.copy()
        
        # Apply correction slice by slice
        for slice_idx in range(volume.shape[0]):
            slice_data = volume[slice_idx]
            slice_mask = mask[slice_idx]
            
            if np.sum(slice_mask) > 100:  # Enough voxels for correction
                # Get coordinates of brain voxels
                y_coords, x_coords = np.where(slice_mask)
                intensities = slice_data[slice_mask]
                
                # Fit 2nd order polynomial
                try:
                    from sklearn.preprocessing import PolynomialFeatures
                    from sklearn.linear_model import LinearRegression
                    
                    # Create polynomial features
                    coords = np.column_stack([x_coords, y_coords])
                    poly = PolynomialFeatures(degree=2)
                    poly_coords = poly.fit_transform(coords)
                    
                    # Fit model
                    model = LinearRegression()
                    model.fit(poly_coords, intensities)
                    
                    # Predict bias field for entire slice
                    y_grid, x_grid = np.meshgrid(
                        np.arange(slice_data.shape[0]),
                        np.arange(slice_data.shape[1]),
                        indexing='ij'
                    )
                    all_coords = np.column_stack([x_grid.ravel(), y_grid.ravel()])
                    all_poly_coords = poly.transform(all_coords)
                    bias_field = model.predict(all_poly_coords).reshape(slice_data.shape)
                    
                    # Correct slice
                    mean_intensity = np.mean(intensities)
                    correction_factor = mean_intensity / (bias_field + 1e-8)
                    corrected_volume[slice_idx] = slice_data * correction_factor
                    
                except Exception:
                    # Fallback: no correction
                    corrected_volume[slice_idx] = slice_data
        
        return corrected_volume
    
    def resize_volume(self, volume: np.ndarray, target_size: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        """
        Resize volume to target size
        
        Args:
            volume: Input volume
            target_size: Target size (depth, height, width)
            
        Returns:
            Resized volume
        """
        if target_size is None:
            target_size = self.target_size
        
        current_size = volume.shape
        
        # Calculate scaling factors
        scale_factors = [
            target_size[i] / current_size[i] for i in range(3)
        ]
        
        # Resize using scipy
        from scipy.ndimage import zoom
        resized_volume = zoom(volume, scale_factors, order=1)
        
        return resized_volume.astype(np.float32)
    
    def augment_volume(self, volume: np.ndarray, mask: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        Apply data augmentation to volume
        
        Args:
            volume: Input volume
            mask: Segmentation mask (optional)
            
        Returns:
            List of augmented volumes (and masks if provided)
        """
        augmented_data = []
        
        # Original
        if mask is not None:
            augmented_data.append((volume, mask))
        else:
            augmented_data.append(volume)
        
        # Horizontal flip
        flipped_volume = np.flip(volume, axis=2)
        if mask is not None:
            flipped_mask = np.flip(mask, axis=2)
            augmented_data.append((flipped_volume, flipped_mask))
        else:
            augmented_data.append(flipped_volume)
        
        # Rotation (small angles)
        from scipy.ndimage import rotate
        for angle in [5, -5]:
            rotated_volume = rotate(volume, angle, axes=(1, 2), reshape=False, order=1)
            if mask is not None:
                rotated_mask = rotate(mask, angle, axes=(1, 2), reshape=False, order=0)
                augmented_data.append((rotated_volume, rotated_mask))
            else:
                augmented_data.append(rotated_volume)
        
        # Intensity scaling
        for scale in [0.9, 1.1]:
            scaled_volume = volume * scale
            if mask is not None:
                augmented_data.append((scaled_volume, mask))
            else:
                augmented_data.append(scaled_volume)
        
        return augmented_data
    
    def create_patches(self, volume: np.ndarray, mask: Optional[np.ndarray] = None,
                      patch_size: Tuple[int, int, int] = (64, 64, 64),
                      stride: Tuple[int, int, int] = (32, 32, 32)) -> List[Tuple[np.ndarray, ...]]:
        """
        Extract patches from volume
        
        Args:
            volume: Input volume
            mask: Segmentation mask (optional)
            patch_size: Size of patches
            stride: Stride for patch extraction
            
        Returns:
            List of patches
        """
        patches = []
        
        d, h, w = volume.shape
        pd, ph, pw = patch_size
        sd, sh, sw = stride
        
        for z in range(0, d - pd + 1, sd):
            for y in range(0, h - ph + 1, sh):
                for x in range(0, w - pw + 1, sw):
                    patch_volume = volume[z:z+pd, y:y+ph, x:x+pw]
                    
                    if mask is not None:
                        patch_mask = mask[z:z+pd, y:y+ph, x:x+pw]
                        patches.append((patch_volume, patch_mask))
                    else:
                        patches.append(patch_volume)
        
        return patches
    
    def preprocess_dataset(self, data_dir: str, output_dir: str,
                          include_masks: bool = True) -> Dict[str, List[str]]:
        """
        Preprocess entire dataset
        
        Args:
            data_dir: Directory containing raw data
            output_dir: Directory to save processed data
            include_masks: Whether to process segmentation masks
            
        Returns:
            Dictionary with processed file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        if include_masks:
            os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)
        
        processed_files = {'images': [], 'masks': []}
        
        # Find all image files
        image_patterns = ['*.nii', '*.nii.gz', '*.nrrd']
        image_files = []
        for pattern in image_patterns:
            image_files.extend(glob.glob(os.path.join(data_dir, 'images', pattern)))
        
        for i, image_file in enumerate(image_files):
            try:
                print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_file)}")
                
                # Load volume
                volume = self.load_nifti_volume(image_file)
                
                # Preprocessing pipeline
                volume = self.normalize_intensity(volume, method='percentile')
                volume = self.skull_stripping(volume)
                volume = self.bias_field_correction(volume)
                volume = self.resize_volume(volume)
                
                # Save processed volume
                base_name = os.path.splitext(os.path.basename(image_file))[0]
                if base_name.endswith('.nii'):
                    base_name = os.path.splitext(base_name)[0]
                
                output_file = os.path.join(output_dir, 'images', f'{base_name}_processed.npy')
                np.save(output_file, volume)
                processed_files['images'].append(output_file)
                
                # Process corresponding mask if exists
                if include_masks:
                    mask_file = os.path.join(data_dir, 'masks', os.path.basename(image_file))
                    if os.path.exists(mask_file):
                        mask = self.load_nifti_volume(mask_file)
                        mask = self.resize_volume(mask)
                        mask = (mask > 0.5).astype(np.uint8)  # Binarize
                        
                        mask_output_file = os.path.join(output_dir, 'masks', f'{base_name}_mask.npy')
                        np.save(mask_output_file, mask)
                        processed_files['masks'].append(mask_output_file)
                
            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")
                continue
        
        # Save preprocessing statistics
        stats_file = os.path.join(output_dir, 'preprocessing_stats.txt')
        with open(stats_file, 'w') as f:
            f.write(f"Processed {len(processed_files['images'])} images\n")
            f.write(f"Target size: {self.target_size}\n")
            f.write(f"Normalization: percentile-based\n")
            f.write(f"Skull stripping: applied\n")
            f.write(f"Bias correction: applied\n")
        
        return processed_files
    
    def load_preprocessed_data(self, data_dir: str, test_size: float = 0.2) -> Tuple[np.ndarray, ...]:
        """
        Load preprocessed data and split into train/test
        
        Args:
            data_dir: Directory containing processed data
            test_size: Fraction of data for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test) if masks available,
            otherwise (X_train, X_test)
        """
        # Load images
        image_dir = os.path.join(data_dir, 'images')
        image_files = glob.glob(os.path.join(image_dir, '*.npy'))
        
        if not image_files:
            raise ValueError(f"No processed images found in {image_dir}")
        
        # Load all images
        images = []
        for image_file in sorted(image_files):
            volume = np.load(image_file)
            images.append(volume)
        
        X = np.array(images)
        
        # Load masks if available
        mask_dir = os.path.join(data_dir, 'masks')
        if os.path.exists(mask_dir):
            mask_files = glob.glob(os.path.join(mask_dir, '*.npy'))
            
            if len(mask_files) == len(image_files):
                masks = []
                for mask_file in sorted(mask_files):
                    mask = np.load(mask_file)
                    masks.append(mask)
                
                y = np.array(masks)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                return X_train, X_test, y_train, y_test
        
        # No masks available, return only images
        X_train, X_test = train_test_split(X, test_size=test_size, random_state=42)
        
        return X_train, X_test
    
    def visualize_preprocessing(self, original: np.ndarray, processed: np.ndarray,
                              slice_idx: Optional[int] = None):
        """
        Visualize preprocessing results
        
        Args:
            original: Original volume
            processed: Processed volume
            slice_idx: Slice index to visualize (middle slice if None)
        """
        if slice_idx is None:
            slice_idx = original.shape[0] // 2
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original
        axes[0].imshow(original[slice_idx], cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Processed
        axes[1].imshow(processed[slice_idx], cmap='gray')
        axes[1].set_title('Processed')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()


class SyntheticMRIGenerator:
    """
    Generate synthetic MRI data for testing
    """
    
    @staticmethod
    def generate_synthetic_brain(size: Tuple[int, int, int] = (128, 128, 128),
                               num_classes: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic brain MRI with segmentation
        
        Args:
            size: Volume size
            num_classes: Number of segmentation classes
            
        Returns:
            Tuple of (volume, segmentation_mask)
        """
        d, h, w = size
        
        # Create synthetic brain structure
        volume = np.zeros(size, dtype=np.float32)
        mask = np.zeros(size, dtype=np.uint8)
        
        # Create brain outline (ellipsoid)
        center_z, center_y, center_x = d // 2, h // 2, w // 2
        
        for z in range(d):
            for y in range(h):
                for x in range(w):
                    # Distance from center
                    dz = (z - center_z) / (d * 0.4)
                    dy = (y - center_y) / (h * 0.4)
                    dx = (x - center_x) / (w * 0.4)
                    
                    dist = dz*dz + dy*dy + dx*dx
                    
                    if dist < 1.0:  # Inside brain
                        # Add noise and structure
                        intensity = 0.5 + 0.3 * np.exp(-dist * 2)
                        intensity += 0.1 * np.random.normal()
                        
                        volume[z, y, x] = max(0, intensity)
                        
                        # Create segmentation classes
                        if dist < 0.3:
                            mask[z, y, x] = 3  # White matter
                        elif dist < 0.6:
                            mask[z, y, x] = 2  # Gray matter
                        elif dist < 0.8:
                            mask[z, y, x] = 1  # CSF
                        else:
                            mask[z, y, x] = 0  # Background
        
        # Add some random structures
        np.random.seed(42)
        for _ in range(5):
            cx = np.random.randint(w//4, 3*w//4)
            cy = np.random.randint(h//4, 3*h//4)
            cz = np.random.randint(d//4, 3*d//4)
            
            radius = np.random.randint(3, 8)
            intensity = np.random.uniform(0.8, 1.2)
            
            for z in range(max(0, cz-radius), min(d, cz+radius)):
                for y in range(max(0, cy-radius), min(h, cy+radius)):
                    for x in range(max(0, cx-radius), min(w, cx+radius)):
                        if (x-cx)**2 + (y-cy)**2 + (z-cz)**2 < radius**2:
                            volume[z, y, x] = intensity
        
        return volume, mask