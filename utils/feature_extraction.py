import cv2
import numpy as np
from skimage import feature, color, morphology, measure
import matplotlib.pyplot as plt

class PigmentationFeatureExtractor:
    """
    A class to extract features from skin pigmentation images
    """
    def __init__(self, color_space='rgb'):
        """
        Initialize the feature extractor
        
        Parameters:
        ----------
        color_space : str
            Color space to use for feature extraction ('rgb', 'hsv', 'lab')
        """
        self.color_space = color_space.lower()
        self.valid_color_spaces = ['rgb', 'hsv', 'lab']
        
        if self.color_space not in self.valid_color_spaces:
            raise ValueError(f"Color space {color_space} not supported. Use one of {self.valid_color_spaces}")
    
    def convert_color_space(self, image):
        """
        Convert image to the selected color space
        
        Parameters:
        ----------
        image : numpy.ndarray
            Input BGR image
            
        Returns:
        -------
        converted : numpy.ndarray
            Image in the selected color space
        """
        # Convert BGR to RGB first
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.color_space == 'rgb':
            return rgb
        elif self.color_space == 'hsv':
            return color.rgb2hsv(rgb)
        elif self.color_space == 'lab':
            return color.rgb2lab(rgb)
    
    def extract_pigmentation_spots(self, image, threshold=0.6):
        """
        Extract pigmentation spots from the image
        
        Parameters:
        ----------
        image : numpy.ndarray
            Preprocessed face image
        threshold : float
            Threshold for spot detection
            
        Returns:
        -------
        spots_mask : numpy.ndarray
            Binary mask of detected spots
        spots_labeled : numpy.ndarray
            Labeled spots image
        """
        # Convert to selected color space
        converted = self.convert_color_space(image)
        
        # Different processing based on color space
        if self.color_space == 'rgb':
            # Use grayscale for RGB
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif self.color_space == 'hsv':
            # Use value channel (brightness) for HSV
            gray = converted[:,:,2]
        elif self.color_space == 'lab':
            # Use L channel (lightness) for LAB
            gray = converted[:,:,0]
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred.astype(np.uint8), 255, 
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Apply morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        spots_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Label the spots
        spots_labeled = measure.label(spots_mask)
        
        return spots_mask, spots_labeled
    
    def calculate_spot_features(self, image, spots_labeled):
        """
        Calculate features for each detected spot
        
        Parameters:
        ----------
        image : numpy.ndarray
            Original image
        spots_labeled : numpy.ndarray
            Labeled spots image
            
        Returns:
        -------
        spot_features : list
            List of dictionaries containing features for each spot
        """
        # Get region properties
        regions = measure.regionprops(spots_labeled, intensity_image=image)
        
        spot_features = []
        
        # Convert to selected color space
        converted = self.convert_color_space(image)
        
        for region in regions:
            # Skip very small regions (likely noise)
            if region.area < 10:
                continue
                
            # Basic shape features
            features = {
                'area': region.area,
                'perimeter': region.perimeter,
                'eccentricity': region.eccentricity,
                'circularity': 4 * np.pi * region.area / (region.perimeter ** 2) if region.perimeter > 0 else 0,
                'bbox': region.bbox,
                'centroid': region.centroid
            }
            
            # Get the coordinates of all pixels in the region
            coords = region.coords
            
            # Calculate color features based on the selected color space
            if self.color_space == 'rgb':
                # RGB color features
                region_pixels = np.array([converted[y, x] for y, x in coords])
                features['mean_r'] = np.mean(region_pixels[:, 0])
                features['mean_g'] = np.mean(region_pixels[:, 1])
                features['mean_b'] = np.mean(region_pixels[:, 2])
                features['std_r'] = np.std(region_pixels[:, 0])
                features['std_g'] = np.std(region_pixels[:, 1])
                features['std_b'] = np.std(region_pixels[:, 2])
                
            elif self.color_space == 'hsv':
                # HSV color features
                region_pixels = np.array([converted[y, x] for y, x in coords])
                features['mean_h'] = np.mean(region_pixels[:, 0])
                features['mean_s'] = np.mean(region_pixels[:, 1])
                features['mean_v'] = np.mean(region_pixels[:, 2])
                features['std_h'] = np.std(region_pixels[:, 0])
                features['std_s'] = np.std(region_pixels[:, 1])
                features['std_v'] = np.std(region_pixels[:, 2])
                
            elif self.color_space == 'lab':
                # LAB color features
                region_pixels = np.array([converted[y, x] for y, x in coords])
                features['mean_l'] = np.mean(region_pixels[:, 0])
                features['mean_a'] = np.mean(region_pixels[:, 1])
                features['mean_b'] = np.mean(region_pixels[:, 2])
                features['std_l'] = np.std(region_pixels[:, 0])
                features['std_a'] = np.std(region_pixels[:, 1])
                features['std_b'] = np.std(region_pixels[:, 2])
            
            spot_features.append(features)
            
        return spot_features
    
    def calculate_global_features(self, image, spots_mask):
        """
        Calculate global features for the entire image
        
        Parameters:
        ----------
        image : numpy.ndarray
            Original image
        spots_mask : numpy.ndarray
            Binary mask of detected spots
            
        Returns:
        -------
        global_features : dict
            Dictionary containing global features
        """
        # Calculate area of the image
        total_area = image.shape[0] * image.shape[1]
        
        # Calculate area covered by spots
        spots_area = np.sum(spots_mask > 0)
        
        # Calculate spot density
        spot_density = spots_area / total_area
        
        # Calculate average color of the image
        converted = self.convert_color_space(image)
        avg_color = np.mean(converted, axis=(0, 1))
        
        global_features = {
            'image_size': (image.shape[0], image.shape[1]),
            'total_area': total_area,
            'spots_area': spots_area,
            'spot_density': spot_density,
            'spot_percentage': spot_density * 100
        }
        
        # Add color features based on the selected color space
        if self.color_space == 'rgb':
            global_features['avg_r'] = avg_color[0]
            global_features['avg_g'] = avg_color[1]
            global_features['avg_b'] = avg_color[2]
        elif self.color_space == 'hsv':
            global_features['avg_h'] = avg_color[0]
            global_features['avg_s'] = avg_color[1]
            global_features['avg_v'] = avg_color[2]
        elif self.color_space == 'lab':
            global_features['avg_l'] = avg_color[0]
            global_features['avg_a'] = avg_color[1]
            global_features['avg_b'] = avg_color[2]
        
        return global_features
    
    def extract_features(self, image):
        """
        Extract features from the image for pigmentation analysis
        
        Parameters:
        ----------
        image : numpy.ndarray
            Preprocessed face image
            
        Returns:
        -------
        features : dict
            Dictionary containing both global and spot-specific features
        """
        # Extract spots
        spots_mask, spots_labeled = self.extract_pigmentation_spots(image)
        
        # Calculate features for individual spots
        spot_features = self.calculate_spot_features(image, spots_labeled)
        
        # Calculate global features
        global_features = self.calculate_global_features(image, spots_mask)
        
        # Combine all features
        features = {
            'global': global_features,
            'spots': spot_features
        }
        
        return features
    
    def visualize_spots(self, image, spots_mask, features=None):
        """
        Visualize detected spots on the original image
        
        Parameters:
        ----------
        image : numpy.ndarray
            Original BGR image
        spots_mask : numpy.ndarray
            Binary mask of detected spots
        features : dict, optional
            Dictionary containing extracted features
            
        Returns:
        -------
        None
        """
        # Create a copy of the image for visualization
        viz_img = image.copy()
        
        # Create a colormap for spots
        spots_colormap = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        spots_colormap[spots_mask > 0] = [0, 0, 255]  # Red color for spots
        
        # Blend the original image with the spots colormap
        alpha = 0.7
        viz_img = cv2.addWeighted(viz_img, alpha, spots_colormap, 1-alpha, 0)
        
        # Display the results
        plt.figure(figsize=(16, 8))
        
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(spots_mask, cmap='gray')
        plt.title('Spots Mask')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(viz_img, cv2.COLOR_BGR2RGB))
        plt.title('Detected Spots')
        plt.axis('off')
        
        # Add text with global features if available
        if features and 'global' in features:
            global_features = features['global']
            info_text = f"Spots: {len(features['spots'])}\n"
            info_text += f"Spot Area: {global_features['spots_area']:.0f} pixels\n"
            info_text += f"Coverage: {global_features['spot_percentage']:.2f}%"
            
            plt.figtext(0.5, 0.01, info_text, ha='center', fontsize=12, 
                        bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
        
        plt.tight_layout()
        plt.show() 