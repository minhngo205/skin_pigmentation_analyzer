import os
import cv2
import numpy as np
import pandas as pd
import requests
import zipfile
import tarfile
from tqdm import tqdm
import shutil
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import albumentations as A
import matplotlib.pyplot as plt
import sys

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.image_preprocessing import ImagePreprocessor

class DatasetDownloader:
    """
    Class for downloading and preparing datasets for skin pigmentation analysis
    """
    def __init__(self, data_dir='data'):
        """
        Initialize the dataset downloader
        
        Parameters:
        ----------
        data_dir : str
            Directory to store downloaded datasets
        """
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, 'raw')
        self.processed_dir = os.path.join(data_dir, 'processed')
        self.train_dir = os.path.join(self.processed_dir, 'train')
        self.val_dir = os.path.join(self.processed_dir, 'val')
        self.test_dir = os.path.join(self.processed_dir, 'test')
        
        # Create directories if they don't exist
        for dir_path in [self.data_dir, self.raw_dir, self.processed_dir, 
                         self.train_dir, self.val_dir, self.test_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def download_file(self, url, dest_path):
        """
        Download a file from a URL
        
        Parameters:
        ----------
        url : str
            URL to download from
        dest_path : str
            Path to save the file
            
        Returns:
        -------
        bool
            True if download was successful, False otherwise
        """
        try:
            if os.path.exists(dest_path):
                print(f"File {dest_path} already exists. Skipping download.")
                return True
            
            print(f"Downloading {url} to {dest_path}")
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 KB
            
            with open(dest_path, 'wb') as f, tqdm(
                total=total_size, unit='B', unit_scale=True, desc=os.path.basename(dest_path)
            ) as progress_bar:
                for data in response.iter_content(block_size):
                    f.write(data)
                    progress_bar.update(len(data))
            
            return True
        except Exception as e:
            print(f"Error downloading file: {e}")
            return False
    
    def extract_archive(self, archive_path, extract_dir):
        """
        Extract a zip or tar archive
        
        Parameters:
        ----------
        archive_path : str
            Path to the archive file
        extract_dir : str
            Directory to extract to
            
        Returns:
        -------
        bool
            True if extraction was successful, False otherwise
        """
        try:
            if archive_path.endswith('.zip'):
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
                with tarfile.open(archive_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(extract_dir)
            elif archive_path.endswith('.tar'):
                with tarfile.open(archive_path, 'r') as tar_ref:
                    tar_ref.extractall(extract_dir)
            else:
                print(f"Unsupported archive format: {archive_path}")
                return False
            
            return True
        except Exception as e:
            print(f"Error extracting archive: {e}")
            return False
    
    def download_fitzpatrick17k(self):
        """
        Download the Fitzpatrick17k dataset
        
        Returns:
        -------
        bool
            True if download and extraction were successful, False otherwise
        """
        # Fitzpatrick17k dataset URL
        url = "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Training_Input.zip"
        
        # Download the dataset
        dest_path = os.path.join(self.raw_dir, 'fitzpatrick17k.zip')
        if not self.download_file(url, dest_path):
            return False
        
        # Extract the dataset
        extract_dir = os.path.join(self.raw_dir, 'fitzpatrick17k')
        os.makedirs(extract_dir, exist_ok=True)
        if not self.extract_archive(dest_path, extract_dir):
            return False
        
        print("Fitzpatrick17k dataset downloaded and extracted successfully.")
        return True
    
    def download_face_skin_dataset(self):
        """
        Download a face skin dataset.
        Note: This is a placeholder. In practice, you might need to access
        specific medical datasets through proper channels.
        
        Returns:
        -------
        bool
            True if download and extraction were successful, False otherwise
        """
        # Placeholder for face skin dataset URL
        # In reality, you would need to get proper access to dermatological datasets
        url = "https://www.cs.sfu.ca/~hamarneh/ecopy/isbi2019_dataset.zip"
        
        # Download the dataset
        dest_path = os.path.join(self.raw_dir, 'face_skin_dataset.zip')
        if not self.download_file(url, dest_path):
            return False
        
        # Extract the dataset
        extract_dir = os.path.join(self.raw_dir, 'face_skin')
        os.makedirs(extract_dir, exist_ok=True)
        if not self.extract_archive(dest_path, extract_dir):
            return False
        
        print("Face skin dataset downloaded and extracted successfully.")
        return True
    
    def preprocess_images(self, source_dir, target_dir, n_samples=None):
        """
        Preprocess images from source directory and save to target directory
        
        Parameters:
        ----------
        source_dir : str
            Source directory containing raw images
        target_dir : str
            Target directory to save preprocessed images
        n_samples : int, optional
            Number of samples to process (default: None, processes all)
            
        Returns:
        -------
        int
            Number of images successfully processed
        """
        # Create the image preprocessor
        preprocessor = ImagePreprocessor(
            target_size=(224, 224),
            face_percent=80,
            gamma_correction=True,
            enhance_contrast=True,
            normalize=False  # Don't normalize yet, as we'll save as images
        )
        
        # Create subdirectories for images and masks
        images_dir = os.path.join(target_dir, 'images')
        masks_dir = os.path.join(target_dir, 'masks')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)
        
        # Get all image files
        image_files = []
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))
        
        # Limit the number of samples if specified
        if n_samples is not None and n_samples < len(image_files):
            image_files = random.sample(image_files, n_samples)
        
        # Process each image
        n_processed = 0
        for img_path in tqdm(image_files, desc="Preprocessing images"):
            try:
                # Read image
                img = cv2.imread(img_path)
                
                if img is None:
                    print(f"Failed to read image: {img_path}")
                    continue
                
                # Preprocess image
                processed_img, original_face, face_detected = preprocessor.preprocess_image(img)
                
                if not face_detected:
                    continue
                
                # Generate mask (for demonstration purposes, we'll create synthetic masks)
                # In a real application, you would use actual masks from datasets or annotations
                spots_mask = self._generate_synthetic_mask(processed_img)
                
                # Save processed image and mask
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                cv2.imwrite(os.path.join(images_dir, f"{base_name}.png"), processed_img * 255)
                cv2.imwrite(os.path.join(masks_dir, f"{base_name}.png"), spots_mask)
                
                n_processed += 1
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        print(f"Processed {n_processed} images.")
        return n_processed
    
    def _generate_synthetic_mask(self, image):
        """
        Generate a synthetic mask for pigmentation spots
        
        Parameters:
        ----------
        image : numpy.ndarray
            Input image
            
        Returns:
        -------
        mask : numpy.ndarray
            Binary mask with synthetic spots
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image.astype(np.uint8) * 255, cv2.COLOR_BGR2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Apply morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Add some random noise to make it more realistic
        for _ in range(np.random.randint(5, 20)):
            # Add random spots
            x = np.random.randint(10, image.shape[1] - 10)
            y = np.random.randint(10, image.shape[0] - 10)
            radius = np.random.randint(2, 10)
            cv2.circle(mask, (x, y), radius, 255, -1)
        
        return mask
    
    def split_dataset(self, test_size=0.2, val_size=0.2, seed=42):
        """
        Split the processed dataset into train, validation, and test sets
        
        Parameters:
        ----------
        test_size : float
            Fraction of dataset to use for testing
        val_size : float
            Fraction of training data to use for validation
        seed : int
            Random seed for reproducibility
            
        Returns:
        -------
        dict
            Dictionary with counts of images in each split
        """
        processed_images_dir = os.path.join(self.processed_dir, 'images')
        processed_masks_dir = os.path.join(self.processed_dir, 'masks')
        
        # Check if directories exist
        if not os.path.exists(processed_images_dir) or not os.path.exists(processed_masks_dir):
            print(f"Processed data directories not found.")
            return None
        
        # Get all image files
        image_files = [f for f in os.listdir(processed_images_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Split into train+val and test sets
        train_val_files, test_files = train_test_split(
            image_files, test_size=test_size, random_state=seed
        )
        
        # Split train+val into train and val sets
        train_files, val_files = train_test_split(
            train_val_files, test_size=val_size, random_state=seed
        )
        
        # Create directories for each split
        for dir_name in ['images', 'masks']:
            os.makedirs(os.path.join(self.train_dir, dir_name), exist_ok=True)
            os.makedirs(os.path.join(self.val_dir, dir_name), exist_ok=True)
            os.makedirs(os.path.join(self.test_dir, dir_name), exist_ok=True)
        
        # Copy files to respective directories
        for file_list, target_dir in [
            (train_files, self.train_dir),
            (val_files, self.val_dir),
            (test_files, self.test_dir)
        ]:
            for file_name in file_list:
                # Copy image
                shutil.copy(
                    os.path.join(processed_images_dir, file_name),
                    os.path.join(target_dir, 'images', file_name)
                )
                
                # Copy mask
                shutil.copy(
                    os.path.join(processed_masks_dir, file_name),
                    os.path.join(target_dir, 'masks', file_name)
                )
        
        # Return the counts
        return {
            'train': len(train_files),
            'val': len(val_files),
            'test': len(test_files)
        }
    
    def create_image_generators(self, batch_size=16, augment=True):
        """
        Create image generators for training, validation, and testing
        
        Parameters:
        ----------
        batch_size : int
            Batch size for generators
        augment : bool
            Whether to apply data augmentation to training set
            
        Returns:
        -------
        dict
            Dictionary containing generators for train, val, and test sets
        """
        if augment:
            # Data augmentation for training
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator(rescale=1./255)
        
        # No augmentation for validation and test sets
        val_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            os.path.join(self.train_dir, 'images'),
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode=None,
            seed=42
        )
        
        val_generator = val_datagen.flow_from_directory(
            os.path.join(self.val_dir, 'images'),
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode=None,
            seed=42
        )
        
        test_generator = test_datagen.flow_from_directory(
            os.path.join(self.test_dir, 'images'),
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode=None,
            seed=42
        )
        
        return {
            'train': train_generator,
            'val': val_generator,
            'test': test_generator
        }
    
    def create_segmentation_generators(self, batch_size=16, augment=True):
        """
        Create paired image and mask generators for segmentation tasks
        
        Parameters:
        ----------
        batch_size : int
            Batch size for generators
        augment : bool
            Whether to apply data augmentation to training set
            
        Returns:
        -------
        dict
            Dictionary containing paired generators for train, val, and test sets
        """
        def _get_mask_generator(img_generator, mask_dir):
            """Helper to create a mask generator that follows the image generator"""
            for batch in img_generator:
                batch_masks = []
                for i in range(len(batch)):
                    img_name = os.path.basename(img_generator.filenames[i * batch_size + i])
                    mask_path = os.path.join(mask_dir, img_name)
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    mask = cv2.resize(mask, (224, 224))
                    mask = mask / 255.0
                    batch_masks.append(mask)
                yield np.array(batch_masks)
        
        # Create image generators
        image_generators = self.create_image_generators(batch_size, augment)
        
        # Create paired generators
        train_mask_dir = os.path.join(self.train_dir, 'masks')
        val_mask_dir = os.path.join(self.val_dir, 'masks')
        test_mask_dir = os.path.join(self.test_dir, 'masks')
        
        train_mask_generator = _get_mask_generator(image_generators['train'], train_mask_dir)
        val_mask_generator = _get_mask_generator(image_generators['val'], val_mask_dir)
        test_mask_generator = _get_mask_generator(image_generators['test'], test_mask_dir)
        
        # Return paired generators
        return {
            'train': (image_generators['train'], train_mask_generator),
            'val': (image_generators['val'], val_mask_generator),
            'test': (image_generators['test'], test_mask_generator)
        }
    
    def prepare_complete_dataset(self, n_samples=None):
        """
        Prepare a complete dataset by downloading, preprocessing, and splitting
        
        Parameters:
        ----------
        n_samples : int, optional
            Number of samples to process (default: None, processes all)
            
        Returns:
        -------
        dict
            Dictionary with dataset statistics
        """
        # Download datasets
        print("Downloading datasets...")
        self.download_fitzpatrick17k()
        self.download_face_skin_dataset()
        
        # Preprocess images
        print("Preprocessing images...")
        self.preprocess_images(
            os.path.join(self.raw_dir, 'fitzpatrick17k'),
            self.processed_dir,
            n_samples=n_samples
        )
        
        # Split dataset
        print("Splitting dataset...")
        split_counts = self.split_dataset()
        
        return {
            'split_counts': split_counts
        }


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    downloader = DatasetDownloader(data_dir=data_dir)
    
    # Prepare dataset with 1000 samples for demonstration
    stats = downloader.prepare_complete_dataset(n_samples=1000)
    print("Dataset preparation completed.")
    print(f"Split counts: {stats['split_counts']}") 