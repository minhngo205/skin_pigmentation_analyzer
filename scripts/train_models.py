import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys

# Add parent directory to path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.pigmentation_model import PigmentationSegmentationModel, PigmentationClassifier
from scripts.data_preparation import DatasetDownloader

def train_segmentation_model(args):
    """
    Train the segmentation model for pigmentation spot detection
    
    Parameters:
    ----------
    args : argparse.Namespace
        Command line arguments
    """
    print("Training pigmentation segmentation model...")
    
    # Set up paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models/checkpoints')
    os.makedirs(model_dir, exist_ok=True)
    
    # Create dataset generator
    downloader = DatasetDownloader(data_dir=data_dir)
    
    # Check if dataset exists, otherwise prepare it
    if not os.path.exists(os.path.join(data_dir, 'processed', 'train', 'images')):
        print("Dataset not found. Preparing dataset...")
        stats = downloader.prepare_complete_dataset(n_samples=args.n_samples)
        print(f"Dataset preparation completed. Split counts: {stats['split_counts']}")
    
    # Create data generators
    gen_dict = downloader.create_segmentation_generators(batch_size=args.batch_size)
    train_gen, val_gen = gen_dict['train'], gen_dict['val']
    
    # Create and build the model
    segmentation_model = PigmentationSegmentationModel(
        input_shape=(224, 224, 3),
        encoder_backbone=args.backbone,
        num_classes=1,  # Binary segmentation
        learning_rate=args.learning_rate,
        batch_size=args.batch_size
    )
    
    model = segmentation_model.build_model()
    print(f"Model created with {args.backbone} backbone.")
    
    # Calculate steps per epoch
    steps_per_epoch = len(train_gen[0]) // args.batch_size
    validation_steps = len(val_gen[0]) // args.batch_size
    
    # Train the model
    history = segmentation_model.train(
        train_generator=train_gen,
        validation_generator=val_gen,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        fine_tune_encoder=args.fine_tune,
        model_dir=model_dir
    )
    
    # Plot training history
    plot_training_history(history, os.path.join(model_dir, 'segmentation_training_history.png'))
    
    print(f"Segmentation model training completed. Model saved to {model_dir}")
    
def train_classification_model(args):
    """
    Train the classification model for pigmentation severity assessment
    
    Parameters:
    ----------
    args : argparse.Namespace
        Command line arguments
    """
    print("Training pigmentation severity classification model...")
    
    # Set up paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models/checkpoints')
    os.makedirs(model_dir, exist_ok=True)
    
    # Create dataset generator
    downloader = DatasetDownloader(data_dir=data_dir)
    
    # Check if dataset exists, otherwise prepare it
    if not os.path.exists(os.path.join(data_dir, 'processed', 'train', 'images')):
        print("Dataset not found. Preparing dataset...")
        stats = downloader.prepare_complete_dataset(n_samples=args.n_samples)
        print(f"Dataset preparation completed. Split counts: {stats['split_counts']}")
    
    # Create data generators
    gen_dict = downloader.create_image_generators(batch_size=args.batch_size)
    train_gen, val_gen = gen_dict['train'], gen_dict['val']
    
    # Create and build the model
    classification_model = PigmentationClassifier(
        input_shape=(224, 224, 3),
        base_model=args.backbone,
        num_classes=args.num_classes,
        learning_rate=args.learning_rate
    )
    
    model = classification_model.build_model()
    print(f"Model created with {args.backbone} backbone.")
    
    # Calculate steps per epoch
    steps_per_epoch = len(train_gen) // args.batch_size
    validation_steps = len(val_gen) // args.batch_size
    
    # Train the model
    history = classification_model.train(
        train_generator=train_gen,
        validation_generator=val_gen,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        fine_tune=args.fine_tune,
        model_dir=model_dir
    )
    
    # Plot training history
    plot_training_history(history, os.path.join(model_dir, 'classification_training_history.png'))
    
    print(f"Classification model training completed. Model saved to {model_dir}")

def plot_training_history(history, output_path):
    """
    Plot training history and save to file
    
    Parameters:
    ----------
    history : tensorflow.keras.callbacks.History
        Training history
    output_path : str
        Path to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy or IoU
    plt.subplot(1, 2, 2)
    if 'accuracy' in history.history:
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
    elif 'iou' in history.history:  # IoU metric for segmentation
        plt.plot(history.history['iou'], label='Training IoU')
        plt.plot(history.history['val_iou'], label='Validation IoU')
        plt.title('Model IoU')
        plt.ylabel('IoU')
    
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Training history plot saved to {output_path}")

def main():
    """Main function to parse arguments and train models"""
    parser = argparse.ArgumentParser(description='Train models for skin pigmentation analysis')
    
    # Common arguments
    parser.add_argument('--model', type=str, choices=['segmentation', 'classification', 'both'], 
                        default='both', help='Model to train')
    parser.add_argument('--backbone', type=str, choices=['mobilenetv2', 'resnet50v2'], 
                        default='mobilenetv2', help='Backbone architecture')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--fine_tune', action='store_true', help='Fine-tune the backbone')
    parser.add_argument('--n_samples', type=int, default=1000, 
                        help='Number of samples to use (for dataset preparation)')
    
    # Classification specific arguments
    parser.add_argument('--num_classes', type=int, default=5, 
                        help='Number of severity classes for classification')
    
    args = parser.parse_args()
    
    # Set memory growth for GPUs
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"Using {len(physical_devices)} GPU(s) for training.")
    else:
        print("No GPU found. Using CPU for training.")
    
    # Train selected models
    if args.model in ['segmentation', 'both']:
        train_segmentation_model(args)
    
    if args.model in ['classification', 'both']:
        train_classification_model(args)
    
if __name__ == '__main__':
    main() 