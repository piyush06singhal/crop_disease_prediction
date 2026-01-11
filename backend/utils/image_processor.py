# utils/image_processor.py - Advanced image preprocessing utilities
"""
Image Processor - Handles image loading, preprocessing, and augmentation.

Features:
- Image loading and validation
- Preprocessing for ML models (resize, normalize)
- Data augmentation for training
- GPU-accelerated processing
- Real-time camera/video processing
- Advanced segmentation and leaf isolation
- Format conversion and optimization
"""

import os
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import cv2
from typing import Optional, Tuple, Dict, List, Any
import logging
import base64
import io
from concurrent.futures import ThreadPoolExecutor
import torch
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

class ImageProcessor:
    """
    Advanced utility class for image preprocessing and augmentation.
    Handles all image-related operations with GPU acceleration and real-time processing.
    """

    def __init__(self, target_size: Tuple[int, int] = (224, 224),
                 use_gpu: bool = True, batch_size: int = 1):
        """
        Initialize image processor.

        Args:
            target_size: Target image size for model input
            use_gpu: Whether to use GPU acceleration
            batch_size: Batch size for processing
        """
        self.target_size = target_size
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.batch_size = batch_size
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')

        # Initialize GPU transforms if available
        self._init_transforms()

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)

        logger.info(f"ImageProcessor initialized - GPU: {self.use_gpu}, Device: {self.device}")

    def _init_transforms(self):
        """Initialize GPU-accelerated transforms"""
        if self.use_gpu:
            try:
                self.gpu_transform = transforms.Compose([
                    transforms.Resize(self.target_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])
            except Exception as e:
                logger.warning(f"GPU transforms failed: {str(e)}")
                self.use_gpu = False

    def preprocess(self, image_path: str, enhance: bool = True,
                  segment_leaf: bool = False) -> np.ndarray:
        """
        Preprocess image for model inference with advanced options.

        Args:
            image_path: Path to input image
            enhance: Whether to apply enhancement
            segment_leaf: Whether to isolate leaf regions

        Returns:
            Preprocessed image array ready for model input
        """
        try:
            # Load image
            image = self._load_image(image_path)

            # Apply advanced preprocessing pipeline
            processed = self._apply_advanced_preprocessing(image, enhance, segment_leaf)

            # Convert to model input format
            model_input = self._to_model_input(processed)

            return model_input

        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise

    def preprocess_batch(self, image_paths: List[str]) -> np.ndarray:
        """
        Preprocess multiple images in batch for efficiency.

        Args:
            image_paths: List of paths to input images

        Returns:
            Batch of preprocessed images
        """
        try:
            # Process images in parallel
            futures = [self.executor.submit(self.preprocess, path)
                      for path in image_paths]

            # Collect results
            batch_images = []
            for future in futures:
                try:
                    batch_images.append(future.result())
                except Exception as e:
                    logger.warning(f"Failed to process image: {str(e)}")
                    continue

            if not batch_images:
                raise ValueError("No images could be processed")

            # Stack into batch
            batch = np.stack(batch_images, axis=0)
            return batch

        except Exception as e:
            logger.error(f"Batch preprocessing failed: {str(e)}")
            raise

    def _load_image(self, image_path: str) -> Image.Image:
        """
        Load image from file path with validation.

        Args:
            image_path: Path to image file

        Returns:
            PIL Image object
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            image = Image.open(image_path)

            # Convert to RGB if necessary
            if image.mode not in ['RGB', 'L']:
                image = image.convert('RGB')

            # Validate image quality
            self._validate_image_quality(image)

            return image

        except Exception as e:
            raise ValueError(f"Failed to load image: {str(e)}")

    def _apply_advanced_preprocessing(self, image: Image.Image,
                                    enhance: bool = True,
                                    segment_leaf: bool = False) -> Image.Image:
        """
        Apply advanced preprocessing transformations.

        Args:
            image: PIL Image object
            enhance: Whether to apply enhancement
            segment_leaf: Whether to segment leaf

        Returns:
            Preprocessed PIL Image
        """
        # Resize with high-quality resampling
        image = image.resize(self.target_size, Image.Resampling.LANCZOS)

        if segment_leaf:
            image = self._segment_leaf(image)

        if enhance:
            image = self._apply_enhancement(image)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        image = self._apply_clahe(image)

        return image

    def _segment_leaf(self, image: Image.Image) -> Image.Image:
        """
        Segment and isolate leaf regions using advanced techniques.

        Args:
            image: PIL Image object

        Returns:
            Image with isolated leaf regions
        """
        try:
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Convert to HSV for better segmentation
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

            # Define range for green colors (typical leaf color)
            lower_green = np.array([25, 40, 40])
            upper_green = np.array([80, 255, 255])

            # Create mask
            mask = cv2.inRange(hsv, lower_green, upper_green)

            # Apply morphological operations to clean mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Apply mask to original image
            segmented = cv2.bitwise_and(cv_image, cv_image, mask=mask)

            # Convert back to PIL
            return Image.fromarray(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))

        except Exception as e:
            logger.warning(f"Leaf segmentation failed: {str(e)}")
            return image

    def _apply_enhancement(self, image: Image.Image) -> Image.Image:
        """
        Apply advanced image enhancement techniques.

        Args:
            image: PIL Image object

        Returns:
            Enhanced PIL Image
        """
        # Auto contrast
        image = ImageOps.autocontrast(image)

        # Sharpen
        image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))

        # Color enhancement
        image = ImageEnhance.Color(image).enhance(1.1)

        # Brightness adjustment
        image = ImageEnhance.Brightness(image).enhance(1.05)

        return image

    def _apply_clahe(self, image: Image.Image) -> Image.Image:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Args:
            image: PIL Image object

        Returns:
            Enhanced PIL Image
        """
        try:
            # Convert to OpenCV
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB)

            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cv_image[:, :, 0] = clahe.apply(cv_image[:, :, 0])

            # Convert back
            enhanced = cv2.cvtColor(cv_image, cv2.COLOR_LAB2RGB)
            return Image.fromarray(enhanced)

        except Exception as e:
            logger.warning(f"CLAHE failed: {str(e)}")
            return image

    def _to_model_input(self, image: Image.Image) -> np.ndarray:
        """
        Convert PIL image to model input format with GPU acceleration.

        Args:
            image: PIL Image object

        Returns:
            Numpy array in model input format
        """
        if self.use_gpu and hasattr(self, 'gpu_transform'):
            try:
                # Use GPU-accelerated transform
                tensor = self.gpu_transform(image)
                return tensor.numpy()
            except Exception as e:
                logger.warning(f"GPU transform failed: {str(e)}")

        # Fallback to CPU processing
        # Convert to numpy array
        img_array = np.array(image)

        # Normalize to [0, 1]
        img_array = img_array.astype(np.float32) / 255.0

        # Standard ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def preprocess_for_training(self, image_path: str,
                              augment: bool = False) -> np.ndarray:
        """
        Preprocess image for training with optional augmentation.

        Args:
            image_path: Path to image file
            augment: Whether to apply data augmentation

        Returns:
            Preprocessed image array
        """
        image = self._load_image(image_path)

        if augment:
            image = self._apply_advanced_augmentation(image)

        return self._to_model_input(self._apply_advanced_preprocessing(image))

    def _apply_advanced_augmentation(self, image: Image.Image) -> Image.Image:
        """
        Apply advanced data augmentation techniques.

        Args:
            image: PIL Image object

        Returns:
            Augmented PIL Image
        """
        # Convert to OpenCV format for augmentation
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Random rotation
        if np.random.random() > 0.5:
            angle = np.random.uniform(-45, 45)
            height, width = cv_image.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
            cv_image = cv2.warpAffine(cv_image, rotation_matrix, (width, height),
                                    borderMode=cv2.BORDER_REFLECT)

        # Random horizontal/vertical flip
        if np.random.random() > 0.5:
            cv_image = cv2.flip(cv_image, 1)  # Horizontal
        if np.random.random() > 0.3:
            cv_image = cv2.flip(cv_image, 0)  # Vertical

        # Random brightness/contrast/saturation adjustment
        if np.random.random() > 0.5:
            alpha = np.random.uniform(0.7, 1.3)  # Contrast
            beta = np.random.uniform(-30, 30)    # Brightness
            cv_image = cv2.convertScaleAbs(cv_image, alpha=alpha, beta=beta)

        # Random Gaussian noise
        if np.random.random() > 0.7:
            noise = np.random.normal(0, 5, cv_image.shape).astype(np.uint8)
            cv_image = cv2.add(cv_image, noise)

        # Random blur/sharpen
        if np.random.random() > 0.8:
            if np.random.random() > 0.5:
                cv_image = cv2.GaussianBlur(cv_image, (3, 3), 0)
            else:
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                cv_image = cv2.filter2D(cv_image, -1, kernel)

        # Convert back to PIL
        return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

    def process_camera_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process real-time camera frame for live disease detection.

        Args:
            frame: Camera frame as numpy array

        Returns:
            Tuple of (processed_frame, detection_info)
        """
        try:
            # Convert to PIL
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Apply preprocessing
            processed = self._apply_advanced_preprocessing(image, enhance=True, segment_leaf=True)

            # Convert back to numpy for display
            processed_frame = cv2.cvtColor(np.array(processed), cv2.COLOR_RGB2BGR)

            # Detection info (placeholder for actual detection)
            detection_info = {
                'processed': True,
                'leaf_detected': True,
                'quality_score': self._assess_image_quality(processed)
            }

            return processed_frame, detection_info

        except Exception as e:
            logger.error(f"Camera frame processing failed: {str(e)}")
            return frame, {'error': str(e)}

    def _assess_image_quality(self, image: Image.Image) -> float:
        """
        Assess image quality for disease detection.

        Args:
            image: PIL Image object

        Returns:
            Quality score (0.0 to 1.0)
        """
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

            # Sharpness assessment using Laplacian variance
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Brightness assessment
            brightness = np.mean(gray)

            # Contrast assessment
            contrast = gray.std()

            # Combine metrics (normalized)
            sharpness_score = min(1.0, sharpness / 500.0)
            brightness_score = 1.0 - abs(brightness - 128) / 128  # Optimal around 128
            contrast_score = min(1.0, contrast / 50.0)

            # Weighted average
            quality = (sharpness_score * 0.4 + brightness_score * 0.3 + contrast_score * 0.3)

            return max(0.0, min(1.0, quality))

        except Exception:
            return 0.5

    def _validate_image_quality(self, image: Image.Image):
        """
        Validate image quality and raise warnings if needed.

        Args:
            image: PIL Image object
        """
        quality = self._assess_image_quality(image)

        if quality < 0.3:
            logger.warning(f"Low quality image detected (score: {quality:.2f})")
        elif quality < 0.5:
            logger.info(f"Moderate quality image (score: {quality:.2f})")

    def validate_image(self, image_path: str) -> Dict:
        """
        Validate image file for processing with detailed feedback.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with validation results
        """
        try:
            with Image.open(image_path) as img:
                # Check basic properties
                if img.size[0] < 64 or img.size[1] < 64:
                    return {'valid': False, 'reason': 'Image too small'}

                # Check format
                supported_formats = ['JPEG', 'PNG', 'BMP', 'TIFF', 'WEBP']
                if img.format not in supported_formats:
                    return {'valid': False, 'reason': f'Unsupported format: {img.format}'}

                # Check file size (max 10MB)
                file_size = os.path.getsize(image_path)
                if file_size > 10 * 1024 * 1024:
                    return {'valid': False, 'reason': 'File too large (>10MB)'}

                # Assess quality
                quality = self._assess_image_quality(img)

                return {
                    'valid': True,
                    'size': img.size,
                    'format': img.format,
                    'file_size': file_size,
                    'quality_score': quality,
                    'warnings': [] if quality > 0.5 else ['Low quality image']
                }

        except Exception as e:
            return {'valid': False, 'reason': f'Error: {str(e)}'}

    def get_image_info(self, image_path: str) -> Dict:
        """
        Get comprehensive information about an image file.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with image information
        """
        try:
            with Image.open(image_path) as img:
                info = {
                    'size': img.size,
                    'mode': img.mode,
                    'format': img.format,
                    'valid': True,
                    'file_size': os.path.getsize(image_path),
                    'quality_score': self._assess_image_quality(img)
                }

                # Add EXIF data if available
                if hasattr(img, '_getexif') and img._getexif():
                    exif = img._getexif()
                    info['exif'] = {k: v for k, v in exif.items()}

                return info

        except Exception as e:
            return {
                'error': str(e),
                'valid': False
            }

    def save_processed_image(self, image_array: np.ndarray,
                           output_path: str, format: str = 'JPEG') -> None:
        """
        Save processed image array to file with optimization.

        Args:
            image_array: Processed image array
            output_path: Output file path
            format: Output format
        """
        try:
            # Remove batch dimension if present
            if len(image_array.shape) == 4:
                image_array = image_array[0]

            # Denormalize from ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_array = image_array * std + mean

            # Clip to [0, 1] and convert to [0, 255]
            image_array = np.clip(image_array, 0, 1)
            image_array = (image_array * 255).astype(np.uint8)

            # Convert to PIL Image and save
            image = Image.fromarray(image_array)

            # Optimize based on format
            if format.upper() == 'JPEG':
                image.save(output_path, format, quality=95, optimize=True)
            else:
                image.save(output_path, format)

        except Exception as e:
            logger.error(f"Failed to save image: {str(e)}")
            raise

    def image_to_base64(self, image_path: str) -> str:
        """
        Convert image to base64 string for web transmission.

        Args:
            image_path: Path to image file

        Returns:
            Base64 encoded image string
        """
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()

            # Get image format
            with Image.open(image_path) as img:
                format = img.format.lower()

            return f"data:image/{format};base64,{base64.b64encode(image_data).decode()}"

        except Exception as e:
            logger.error(f"Base64 conversion failed: {str(e)}")
            return ""

    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        image = ImageOps.autocontrast(image)

        # Optional: Apply additional enhancements
        # image = ImageEnhance.Sharpness(image).enhance(1.2)

        return image

    def _to_model_input(self, image: Image.Image) -> np.ndarray:
        """
        Convert PIL image to model input format.

        Args:
            image: PIL Image object

        Returns:
            Numpy array in model input format
        """
        # Convert to numpy array
        img_array = np.array(image)

        # Normalize to [0, 1]
        img_array = img_array.astype(np.float32) / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def preprocess_for_training(self, image_path: str,
                              augment: bool = False) -> np.ndarray:
        """
        Preprocess image for training with optional augmentation.

        Args:
            image_path: Path to image file
            augment: Whether to apply data augmentation

        Returns:
            Preprocessed image array
        """
        image = self._load_image(image_path)

        if augment:
            image = self._apply_augmentation(image)

        return self._to_model_input(self._apply_preprocessing(image))

    def _apply_augmentation(self, image: Image.Image) -> Image.Image:
        """
        Apply data augmentation techniques.

        Args:
            image: PIL Image object

        Returns:
            Augmented PIL Image
        """
        # Convert to OpenCV format for augmentation
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Random rotation
        if np.random.random() > 0.5:
            angle = np.random.uniform(-30, 30)
            height, width = cv_image.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
            cv_image = cv2.warpAffine(cv_image, rotation_matrix, (width, height))

        # Random horizontal flip
        if np.random.random() > 0.5:
            cv_image = cv2.flip(cv_image, 1)

        # Random brightness/contrast adjustment
        if np.random.random() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)  # Contrast
            beta = np.random.uniform(-20, 20)    # Brightness
            cv_image = cv2.convertScaleAbs(cv_image, alpha=alpha, beta=beta)

        # Convert back to PIL
        return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

    def validate_image(self, image_path: str) -> bool:
        """
        Validate image file for processing.

        Args:
            image_path: Path to image file

        Returns:
            True if image is valid for processing
        """
        try:
            with Image.open(image_path) as img:
                # Check basic properties
                if img.size[0] < 32 or img.size[1] < 32:
                    return False

                # Check format
                supported_formats = ['JPEG', 'PNG', 'BMP', 'TIFF']
                if img.format not in supported_formats:
                    return False

                return True

        except Exception:
            return False

    def get_image_info(self, image_path: str) -> Dict:
        """
        Get information about an image file.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with image information
        """
        try:
            with Image.open(image_path) as img:
                return {
                    'size': img.size,
                    'mode': img.mode,
                    'format': img.format,
                    'valid': True
                }
        except Exception as e:
            return {
                'error': str(e),
                'valid': False
            }

    def save_processed_image(self, image_array: np.ndarray,
                           output_path: str) -> None:
        """
        Save processed image array to file.

        Args:
            image_array: Processed image array
            output_path: Output file path
        """
        # Remove batch dimension if present
        if len(image_array.shape) == 4:
            image_array = image_array[0]

        # Denormalize from [0,1] to [0,255]
        image_array = (image_array * 255).astype(np.uint8)

        # Convert to PIL Image and save
        image = Image.fromarray(image_array)
        image.save(output_path)