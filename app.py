import os
import cv2
import numpy as np
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import sys

# Add the parent directory to the path to import from other modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.image_preprocessing import ImagePreprocessor
from utils.feature_extraction import PigmentationFeatureExtractor
from utils.pigmentation_analyzer import PigmentationAnalyzer
from models.pigmentation_model import PigmentationSegmentationModel, PigmentationClassifier

class SkinPigmentationApp:
    """
    Main application for skin pigmentation detection and analysis
    """
    def __init__(self,
                 segmentation_model_path=None,
                 classification_model_path=None,
                 use_ml_models=True,
                 use_traditional_cv=True):
        """
        Initialize the skin pigmentation analysis application
        
        Parameters:
        ----------
        segmentation_model_path : str, optional
            Path to the pretrained segmentation model
        classification_model_path : str, optional
            Path to the pretrained classification model
        use_ml_models : bool
            Whether to use machine learning models for analysis
        use_traditional_cv : bool
            Whether to use traditional computer vision methods for analysis
        """
        self.use_ml_models = use_ml_models
        self.use_traditional_cv = use_traditional_cv
        
        # Initialize image preprocessor
        self.preprocessor = ImagePreprocessor(
            target_size=(224, 224),
            face_percent=60,
            gamma_correction=True,
            enhance_contrast=True,
            normalize=True
        )
        
        # Initialize feature extractor
        self.feature_extractor = PigmentationFeatureExtractor(color_space='lab')
        
        # Initialize pigmentation analyzer
        self.analyzer = PigmentationAnalyzer(severity_levels=5)
        
        # Load ML models if requested
        self.segmentation_model = None
        self.classification_model = None
        
        if use_ml_models:
            # Try to find models in default locations if not specified
            if segmentation_model_path is None:
                default_seg_path = os.path.join('models', 'checkpoints', 'pigmentation_model_final.h5')
                if os.path.exists(default_seg_path):
                    segmentation_model_path = default_seg_path
                else:
                    print("No segmentation model found. Will use traditional CV methods only.")
            
            if classification_model_path is None:
                default_cls_path = os.path.join('models', 'checkpoints', 'pigmentation_classifier_final.h5')
                if os.path.exists(default_cls_path):
                    classification_model_path = default_cls_path
                else:
                    print("No classification model found. Will use traditional CV methods for severity estimation.")
            
            # Load segmentation model if available
            if segmentation_model_path and os.path.exists(segmentation_model_path):
                try:
                    print(f"Loading segmentation model from {segmentation_model_path}")
                    self.segmentation_model = PigmentationSegmentationModel()
                    self.segmentation_model.load_model(segmentation_model_path)
                except Exception as e:
                    print(f"Error loading segmentation model: {e}")
                    self.segmentation_model = None
            
            # Load classification model if available
            if classification_model_path and os.path.exists(classification_model_path):
                try:
                    print(f"Loading classification model from {classification_model_path}")
                    self.classification_model = PigmentationClassifier()
                    self.classification_model.load_model(classification_model_path)
                except Exception as e:
                    print(f"Error loading classification model: {e}")
                    self.classification_model = None
    
    def process_image(self, image_path):
        """
        Process an image for pigmentation analysis
        
        Parameters:
        ----------
        image_path : str
            Path to the input image
            
        Returns:
        -------
        dict
            Analysis results
        """
        # Read image
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error reading image: {image_path}")
            return None
        
        # Preprocess image
        processed_img, original_face, face_detected = self.preprocessor.preprocess_image(image)
        
        if not face_detected:
            print("No face detected in the image")
            return None
        
        # Initialize results dictionary
        results = {
            'original_image': image,
            'processed_image': processed_img,
            'original_face': original_face,
            'face_detected': face_detected
        }
        
        # Apply segmentation to detect spots
        if self.use_ml_models and self.segmentation_model is not None:
            # Use ML model for segmentation
            spots_mask = self.segmentation_model.predict(processed_img)
            results['spots_mask'] = spots_mask
            results['detection_method'] = 'deep_learning'
        else:
            # Use traditional CV methods for segmentation
            spots_mask, spots_labeled = self.feature_extractor.extract_pigmentation_spots(processed_img)
            results['spots_mask'] = spots_mask
            results['spots_labeled'] = spots_labeled
            results['detection_method'] = 'traditional_cv'
        
        # Extract features
        if self.use_traditional_cv:
            features = self.feature_extractor.extract_features(processed_img)
            results['features'] = features
            
            # Analyze pigmentation
            analysis_results = self.analyzer.analyze(processed_img, features)
            results['analysis'] = analysis_results
            
            # Set severity level
            severity_level = analysis_results['severity_level']
            severity_label = analysis_results['severity_label']
            severity_score = analysis_results['severity_score']
        
        # Use ML model for classification if available
        if self.use_ml_models and self.classification_model is not None:
            # Use ML model for severity classification
            severity_class, confidence = self.classification_model.predict(processed_img)
            
            # Map class index to level and label
            labels = {
                0: "Minimal",
                1: "Mild",
                2: "Moderate",
                3: "Significant",
                4: "Severe"
            }
            
            results['ml_severity_class'] = severity_class
            results['ml_severity_confidence'] = confidence
            results['ml_severity_label'] = labels.get(severity_class, f"Class {severity_class}")
            
            # Use ML prediction as primary if traditional CV not used
            if not self.use_traditional_cv:
                severity_level = severity_class + 1  # Convert 0-based index to 1-based level
                severity_label = labels.get(severity_class, f"Level {severity_level}")
                severity_score = confidence * 100
        
        # Set final severity results
        results['severity_level'] = severity_level
        results['severity_label'] = severity_label
        results['severity_score'] = severity_score
        
        return results
    
    def visualize_results(self, results):
        """
        Visualize the pigmentation analysis results
        
        Parameters:
        ----------
        results : dict
            Results from process_image method
            
        Returns:
        -------
        None
        """
        if results is None:
            print("No results to visualize")
            return
        
        # Get images from results
        original_image = results['original_image']
        processed_image = results['processed_image']
        original_face = results['original_face']
        spots_mask = results['spots_mask']
        
        # Create a figure
        plt.figure(figsize=(15, 10))
        
        # Original image
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        # Detected face
        plt.subplot(2, 3, 2)
        plt.imshow(cv2.cvtColor(original_face, cv2.COLOR_BGR2RGB))
        plt.title('Detected Face')
        plt.axis('off')
        
        # Processed image
        plt.subplot(2, 3, 3)
        if isinstance(processed_image, np.ndarray) and processed_image.max() <= 1.0:
            # If normalized, scale to 0-255
            plt.imshow(cv2.cvtColor((processed_image * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        plt.title('Processed Image')
        plt.axis('off')
        
        # Spots mask
        plt.subplot(2, 3, 4)
        if len(spots_mask.shape) == 3 and spots_mask.shape[2] > 1:
            # If mask is multi-channel, convert to grayscale
            plt.imshow(cv2.cvtColor(spots_mask, cv2.COLOR_BGR2GRAY), cmap='gray')
        else:
            plt.imshow(spots_mask, cmap='gray')
        plt.title('Spots Mask')
        plt.axis('off')
        
        # Overlay spots on face
        plt.subplot(2, 3, 5)
        face_overlay = original_face.copy()
        
        # Create spot overlay
        if len(spots_mask.shape) == 2:
            # If mask is single channel, create RGB overlay
            spot_overlay = np.zeros_like(face_overlay)
            if spots_mask.max() <= 1.0:
                binary_mask = (spots_mask > 0.5).astype(np.uint8)
            else:
                binary_mask = (spots_mask > 127).astype(np.uint8)
            
            spot_overlay[binary_mask > 0] = [0, 0, 255]  # Red color for spots
            face_overlay = cv2.addWeighted(face_overlay, 0.7, spot_overlay, 0.3, 0)
        else:
            # If mask is already RGB
            face_overlay = cv2.addWeighted(face_overlay, 0.7, spots_mask, 0.3, 0)
        
        plt.imshow(cv2.cvtColor(face_overlay, cv2.COLOR_BGR2RGB))
        plt.title('Detected Spots')
        plt.axis('off')
        
        # Severity info
        plt.subplot(2, 3, 6)
        plt.axis('off')
        
        # Get severity info from results
        severity_level = results.get('severity_level', 0)
        severity_label = results.get('severity_label', 'Unknown')
        severity_score = results.get('severity_score', 0)
        
        # Create text for severity information
        severity_text = f"Severity Level: {severity_level}\n"
        severity_text += f"Severity Label: {severity_label}\n"
        severity_text += f"Severity Score: {severity_score:.1f}"
        
        # Add detection method info
        detection_method = results.get('detection_method', 'Unknown')
        if detection_method == 'deep_learning':
            method_text = "Detection Method: Deep Learning"
        else:
            method_text = "Detection Method: Computer Vision"
            
        # Add features info if available
        features_text = ""
        if 'features' in results and 'global' in results['features']:
            global_features = results['features']['global']
            features_text += f"\nSpot Count: {len(results['features']['spots'])}\n"
            features_text += f"Coverage: {global_features['spot_percentage']:.2f}%\n"
            
        # Add ML info if available
        ml_text = ""
        if 'ml_severity_class' in results:
            ml_text = f"\nML Classification:\n"
            ml_text += f"Class: {results['ml_severity_class']}\n"
            ml_text += f"Label: {results['ml_severity_label']}\n"
            ml_text += f"Confidence: {results['ml_severity_confidence'] * 100:.1f}%"
            
        # Display all the text
        plt.text(0.5, 0.9, severity_text, ha='center', va='top', fontsize=12,
                bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
        
        plt.text(0.5, 0.5, method_text + features_text + ml_text, ha='center', va='center', fontsize=10,
                bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
        
        plt.title('Analysis Results')
        
        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()
        
        # If traditional CV analysis was performed, also show the analyzer visualization
        if self.use_traditional_cv and 'features' in results and 'analysis' in results:
            self.analyzer.visualize_results(
                original_face, 
                results['features'], 
                results['analysis']
            )
    
    def batch_process(self, image_dir, output_dir=None):
        """
        Process all images in a directory
        
        Parameters:
        ----------
        image_dir : str
            Directory containing images to process
        output_dir : str, optional
            Directory to save analysis results (default: None, don't save)
            
        Returns:
        -------
        list
            List of analysis results for each image
        """
        # Create output directory if needed
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_files = []
        for file in os.listdir(image_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(image_dir, file))
        
        # Process each image
        results_list = []
        
        for img_path in image_files:
            print(f"Processing {img_path}...")
            results = self.process_image(img_path)
            
            if results is not None:
                results_list.append(results)
                
                # Save results if output directory provided
                if output_dir is not None:
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    
                    # Save processed face
                    face_output_path = os.path.join(output_dir, f"{base_name}_face.png")
                    cv2.imwrite(face_output_path, results['original_face'])
                    
                    # Save spots mask
                    mask_output_path = os.path.join(output_dir, f"{base_name}_mask.png")
                    cv2.imwrite(mask_output_path, results['spots_mask'])
                    
                    # Save overlay
                    overlay = results['original_face'].copy()
                    spots_mask = results['spots_mask']
                    
                    if len(spots_mask.shape) == 2:
                        spot_overlay = np.zeros_like(overlay)
                        if spots_mask.max() <= 1.0:
                            binary_mask = (spots_mask > 0.5).astype(np.uint8)
                        else:
                            binary_mask = (spots_mask > 127).astype(np.uint8)
                        
                        spot_overlay[binary_mask > 0] = [0, 0, 255]  # Red for spots
                        overlay = cv2.addWeighted(overlay, 0.7, spot_overlay, 0.3, 0)
                    else:
                        overlay = cv2.addWeighted(overlay, 0.7, spots_mask, 0.3, 0)
                    
                    overlay_output_path = os.path.join(output_dir, f"{base_name}_overlay.png")
                    cv2.imwrite(overlay_output_path, overlay)
                    
                    # Save analysis text
                    text_output_path = os.path.join(output_dir, f"{base_name}_analysis.txt")
                    with open(text_output_path, 'w') as f:
                        f.write(f"Image: {img_path}\n")
                        f.write(f"Severity Level: {results['severity_level']}\n")
                        f.write(f"Severity Label: {results['severity_label']}\n")
                        f.write(f"Severity Score: {results['severity_score']:.1f}\n")
                        
                        if 'features' in results and 'global' in results['features']:
                            global_features = results['features']['global']
                            f.write(f"Spot Count: {len(results['features']['spots'])}\n")
                            f.write(f"Coverage: {global_features['spot_percentage']:.2f}%\n")
                        
                        if 'ml_severity_class' in results:
                            f.write(f"ML Class: {results['ml_severity_class']}\n")
                            f.write(f"ML Label: {results['ml_severity_label']}\n")
                            f.write(f"ML Confidence: {results['ml_severity_confidence'] * 100:.1f}%\n")
        
        return results_list


def main():
    """Main function to run the application from the command line"""
    parser = argparse.ArgumentParser(description='Skin Pigmentation Analysis')
    
    # Input arguments
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--image_dir', type=str, help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, help='Directory to save output')
    
    # Model arguments
    parser.add_argument('--segmentation_model', type=str, help='Path to segmentation model')
    parser.add_argument('--classification_model', type=str, help='Path to classification model')
    parser.add_argument('--no_ml', action='store_true', help='Disable ML models')
    parser.add_argument('--no_cv', action='store_true', help='Disable traditional CV')
    
    # Visualization
    parser.add_argument('--no_vis', action='store_true', help='Disable visualization')
    
    args = parser.parse_args()
    
    # Check if either image or image_dir is provided
    if args.image is None and args.image_dir is None:
        parser.error('Either --image or --image_dir must be provided')
        
    # Create application
    app = SkinPigmentationApp(
        segmentation_model_path=args.segmentation_model,
        classification_model_path=args.classification_model,
        use_ml_models=not args.no_ml,
        use_traditional_cv=not args.no_cv
    )
    
    # Process single image
    if args.image is not None:
        results = app.process_image(args.image)
        
        if results is not None:
            if not args.no_vis:
                app.visualize_results(results)
                
            if args.output_dir is not None:
                os.makedirs(args.output_dir, exist_ok=True)
                
                # Save processed face
                base_name = os.path.splitext(os.path.basename(args.image))[0]
                face_output_path = os.path.join(args.output_dir, f"{base_name}_face.png")
                cv2.imwrite(face_output_path, results['original_face'])
                
                # Save spots mask
                mask_output_path = os.path.join(args.output_dir, f"{base_name}_mask.png")
                cv2.imwrite(mask_output_path, results['spots_mask'])
                
                # Save analysis text
                text_output_path = os.path.join(args.output_dir, f"{base_name}_analysis.txt")
                with open(text_output_path, 'w') as f:
                    f.write(f"Image: {args.image}\n")
                    f.write(f"Severity Level: {results['severity_level']}\n")
                    f.write(f"Severity Label: {results['severity_label']}\n")
                    f.write(f"Severity Score: {results['severity_score']:.1f}\n")
    
    # Process directory of images
    elif args.image_dir is not None:
        results_list = app.batch_process(args.image_dir, args.output_dir)
        print(f"Processed {len(results_list)} images.")
    
if __name__ == '__main__':
    main() 