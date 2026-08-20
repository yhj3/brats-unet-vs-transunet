#basic_inference.py
"""
Basic Brain MRI Segmentation Inference Example

This example demonstrates how to use the brain MRI segmentation model
for inference on medical images.
"""

import sys
import os
import numpy as np
import torch
import nibabel as nib
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from GANNET_seg import GANUNet, dice_coeff
    from visualization import visualize_segmentation
except ImportError:
    print("Could not import modules. Make sure you're in the correct directory.")
    sys.exit(1)


class BrainMRIInference:
    """Brain MRI segmentation inference pipeline."""
    
    def __init__(self, model_path=None, device='auto'):
        """
        Initialize the inference pipeline.
        
        Args:
            model_path: Path to trained model weights
            device: Device to run inference on ('auto', 'cpu', 'cuda')
        """
        self.device = self._setup_device(device)
        self.model = None
        self.model_path = model_path
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def _setup_device(self, device):
        """Setup computation device."""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def create_sample_model(self):
        """Create a sample model for demonstration."""
        print("Creating sample model for demonstration...")
        
        # Initialize model architecture
        model = GANUNet(
            in_channels=4,  # T1, T1-CE, T2, FLAIR
            num_classes=4,  # Background + 3 tumor classes
            features=[64, 128, 256, 512]
        )
        
        model.to(self.device)
        model.eval()
        
        self.model = model
        print(f"Sample model created on {self.device}")
        return model
    
    def load_model(self, model_path):
        """Load pre-trained model from checkpoint."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Initialize model
            model = GANUNet(
                in_channels=4,
                num_classes=4,
                features=[64, 128, 256, 512]
            )
            
            # Load weights
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(self.device)
            model.eval()
            
            self.model = model
            print(f"Model loaded from {model_path}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating sample model instead...")
            self.create_sample_model()
    
    def preprocess_image(self, image_data):
        """Preprocess MRI image for inference."""
        # Normalize intensity values
        image_data = (image_data - image_data.mean()) / (image_data.std() + 1e-8)
        
        # Clip extreme values
        image_data = np.clip(image_data, -3, 3)
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.FloatTensor(image_data).unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def predict_sample_data(self):
        """Run inference on synthetic sample data."""
        print("\n" + "="*60)
        print("BRAIN MRI SEGMENTATION INFERENCE DEMO")
        print("="*60)
        
        if self.model is None:
            self.create_sample_model()
        
        # Create synthetic brain MRI data (4 modalities)
        print("\n1. Creating synthetic brain MRI data...")
        
        # Simulate 4-channel brain MRI (T1, T1-CE, T2, FLAIR)
        height, width, depth = 240, 240, 155  # Typical brain MRI dimensions
        
        # Create realistic brain-like patterns
        np.random.seed(42)
        
        # Base brain structure
        center_h, center_w = height // 2, width // 2
        y, x = np.ogrid[:height, :width]
        brain_mask = ((y - center_h)**2 + (x - center_w)**2) < (100**2)
        
        sample_data = np.zeros((4, height, width, depth))
        
        for i in range(4):  # 4 modalities
            for z in range(depth):
                # Create brain-like intensity patterns
                base_signal = np.random.normal(0.5, 0.1, (height, width))
                
                # Add brain structure
                base_signal[brain_mask] += 0.3
                
                # Add some tumor-like structures (high intensity regions)
                if 50 <= z <= 100:  # Middle slices
                    tumor_x, tumor_y = center_w + 20, center_h - 10
                    tumor_mask = ((y - tumor_y)**2 + (x - tumor_x)**2) < (15**2)
                    base_signal[tumor_mask] += 0.5
                
                # Different modalities have different contrasts
                if i == 0:  # T1
                    base_signal *= 0.8
                elif i == 1:  # T1-CE (enhanced)
                    base_signal[tumor_mask] *= 1.5
                elif i == 2:  # T2
                    base_signal *= 1.2
                elif i == 3:  # FLAIR
                    base_signal *= 0.9
                
                sample_data[i, :, :, z] = base_signal
        
        print(f"   - Created synthetic 4D MRI volume: {sample_data.shape}")
        print(f"   - Modalities: T1, T1-CE, T2, FLAIR")
        print(f"   - Volume dimensions: {height}x{width}x{depth}")
        
        # Take a middle slice for demonstration
        slice_idx = depth // 2
        sample_slice = sample_data[:, :, :, slice_idx]
        
        print(f"\n2. Processing slice {slice_idx} for segmentation...")
        
        # Preprocess
        input_tensor = self.preprocess_image(sample_slice)
        print(f"   - Input tensor shape: {input_tensor.shape}")
        print(f"   - Device: {input_tensor.device}")
        
        # Run inference
        print(f"\n3. Running segmentation inference...")
        with torch.no_grad():
            output = self.model(input_tensor)
            prediction = torch.softmax(output, dim=1)
            segmentation = torch.argmax(prediction, dim=1)
        
        # Convert to numpy
        segmentation_np = segmentation.cpu().numpy().squeeze()
        probabilities = prediction.cpu().numpy().squeeze()
        
        print(f"   - Output shape: {output.shape}")
        print(f"   - Segmentation shape: {segmentation_np.shape}")
        print(f"   - Unique labels: {np.unique(segmentation_np)}")
        
        # Calculate some metrics
        print(f"\n4. Segmentation Results:")
        for label in range(4):
            pixel_count = np.sum(segmentation_np == label)
            percentage = (pixel_count / segmentation_np.size) * 100
            label_names = ['Background', 'Tumor Core', 'Edema', 'Enhancing Tumor']
            print(f"   - {label_names[label]}: {pixel_count} pixels ({percentage:.1f}%)")
        
        # Calculate confidence
        max_probs = np.max(probabilities, axis=0)
        mean_confidence = np.mean(max_probs)
        print(f"   - Mean prediction confidence: {mean_confidence:.3f}")
        
        # Simulate Dice scores (for demonstration)
        simulated_dice_scores = {
            'Whole Tumor': np.random.uniform(0.85, 0.92),
            'Tumor Core': np.random.uniform(0.75, 0.85),
            'Enhancing Tumor': np.random.uniform(0.70, 0.80)
        }
        
        print(f"\n5. Simulated Performance Metrics:")
        for region, dice in simulated_dice_scores.items():
            print(f"   - {region} Dice Score: {dice:.3f}")
        
        print(f"\n6. Clinical Insights:")
        tumor_volume = np.sum(segmentation_np > 0)  # All non-background
        print(f"   - Estimated tumor volume: {tumor_volume} voxels")
        print(f"   - Tumor burden: {(tumor_volume/segmentation_np.size)*100:.1f}% of slice")
        
        if tumor_volume > 1000:
            print(f"   - Recommendation: Significant tumor burden detected")
        else:
            print(f"   - Recommendation: Small or no tumor detected")
        
        print(f"\n" + "="*60)
        print("INFERENCE COMPLETED SUCCESSFULLY")
        print("="*60)
        
        return {
            'segmentation': segmentation_np,
            'probabilities': probabilities,
            'input_data': sample_slice,
            'metrics': simulated_dice_scores
        }
    
    def predict_nifti(self, image_paths):
        """
        Run inference on actual NIfTI files.
        
        Args:
            image_paths: Dict with keys 't1', 't1ce', 't2', 'flair'
        """
        if self.model is None:
            print("No model loaded. Please load a model first.")
            return None
        
        print("Loading NIfTI files...")
        
        images = []
        for modality in ['t1', 't1ce', 't2', 'flair']:
            if modality in image_paths:
                img = nib.load(image_paths[modality])
                data = img.get_fdata()
                images.append(data)
                print(f"Loaded {modality}: {data.shape}")
        
        if len(images) != 4:
            print("Error: Need all 4 modalities (T1, T1-CE, T2, FLAIR)")
            return None
        
        # Stack modalities
        multi_modal = np.stack(images, axis=0)
        
        # Process slice by slice or full volume (depending on memory)
        print("Running segmentation...")
        
        # For demonstration, process middle slice
        middle_slice = multi_modal.shape[-1] // 2
        slice_data = multi_modal[:, :, :, middle_slice]
        
        input_tensor = self.preprocess_image(slice_data)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            segmentation = torch.argmax(output, dim=1)
        
        return segmentation.cpu().numpy().squeeze()


def main():
    """Main demonstration function."""
    
    print("Brain MRI Segmentation Inference Demo")
    print("====================================")
    
    # Initialize inference pipeline
    inference = BrainMRIInference()
    
    # Run demonstration with synthetic data
    results = inference.predict_sample_data()
    
    print("\nDemo completed! In a real scenario, you would:")
    print("1. Load actual brain MRI NIfTI files")
    print("2. Use a trained model checkpoint")
    print("3. Process full 3D volumes")
    print("4. Apply post-processing and clinical validation")
    
    # Example of how to use with real data
    print("\nExample usage with real NIfTI files:")
    print("""
    # Load real model
    inference = BrainMRIInference('path/to/trained/model.pth')
    
    # Define paths to patient data
    patient_data = {
        't1': 'patient_001_t1.nii.gz',
        't1ce': 'patient_001_t1ce.nii.gz', 
        't2': 'patient_001_t2.nii.gz',
        'flair': 'patient_001_flair.nii.gz'
    }
    
    # Run segmentation
    segmentation = inference.predict_nifti(patient_data)
    """)


if __name__ == "__main__":
    main()