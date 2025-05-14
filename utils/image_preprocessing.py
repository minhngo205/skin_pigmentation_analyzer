import cv2
import numpy as np
import face_recognition
from sklearn.preprocessing import StandardScaler
from skimage import color, exposure, feature, filters, morphology
import matplotlib.pyplot as plt

class ImagePreprocessor:
    """
    A class to preprocess facial images for skin pigmentation analysis
    """
    def __init__(self, 
                 target_size=(224, 224), 
                 face_percent=60, 
                 gamma_correction=False,
                 enhance_contrast=True,
                 normalize=True):
        """
        Initialize the image preprocessor
        
        Parameters:
        ----------
        target_size : tuple
            Target size for the output image (height, width)
        face_percent : int
            Percentage of the face to keep (enlarges the detected face rectangle)
        gamma_correction : bool
            Whether to apply gamma correction
        enhance_contrast : bool
            Whether to enhance contrast using CLAHE
        normalize : bool
            Whether to normalize pixel values
        """
        self.target_size = target_size
        self.face_percent = face_percent
        self.gamma_correction = gamma_correction
        self.enhance_contrast = enhance_contrast
        self.normalize = normalize
        
    def detect_face(self, image):
        """
        Detect face in the image and return the face region
        
        Parameters:
        ----------
        image : numpy.ndarray
            Input image
            
        Returns:
        -------
        face_img : numpy.ndarray
            Cropped face region
        face_detected : bool
            Whether a face was detected
        """
        # Convert BGR to RGB (face_recognition uses RGB)
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
            
        # Detect face locations
        face_locations = face_recognition.face_locations(rgb_image)
        
        if not face_locations:
            return image, False
        
        # Use the first face found
        top, right, bottom, left = face_locations[0]
        
        # Enlarge the face region by face_percent
        if self.face_percent != 100:
            height, width = bottom - top, right - left
            center_y, center_x = (top + bottom) // 2, (left + right) // 2
            
            # Calculate new dimensions
            new_height = int(height * (100 + self.face_percent) / 100)
            new_width = int(width * (100 + self.face_percent) / 100)
            
            # Calculate new boundaries
            top = max(0, center_y - new_height // 2)
            bottom = min(rgb_image.shape[0], center_y + new_height // 2)
            left = max(0, center_x - new_width // 2)
            right = min(rgb_image.shape[1], center_x + new_width // 2)
        
        # Extract face region
        face_img = rgb_image[top:bottom, left:right]
        
        # Convert back to BGR for OpenCV functions
        if len(image.shape) == 3 and image.shape[2] == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
            
        return face_img, True
    
    def apply_gamma_correction(self, image, gamma=1.0):
        """
        Apply gamma correction to enhance details
        
        Parameters:
        ----------
        image : numpy.ndarray
            Input image
        gamma : float
            Gamma value
            
        Returns:
        -------
        corrected : numpy.ndarray
            Gamma-corrected image
        """
        # Convert to float
        image_float = image.astype(np.float32) / 255.0
        
        # Apply gamma correction
        corrected = np.power(image_float, gamma)
        
        # Convert back to 8-bit uint
        corrected = (corrected * 255).astype(np.uint8)
        
        return corrected
    
    def extract_skin_mask(self, image):
        """
        Extract skin regions from the image
        
        Parameters:
        ----------
        image : numpy.ndarray
            Input BGR image
            
        Returns:
        -------
        skin_mask : numpy.ndarray
            Binary mask of skin regions
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define range for skin color in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create binary mask
        mask1 = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Define second range for skin color (handling hue wraparound)
        lower_skin2 = np.array([170, 20, 70], dtype=np.uint8)
        upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
        
        # Create second binary mask
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        
        # Combine masks
        skin_mask = cv2.bitwise_or(mask1, mask2)
        
        # Apply morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        return skin_mask
    
    def enhance_image_contrast(self, image):
        """
        Enhance the contrast of the image using CLAHE
        
        Parameters:
        ----------
        image : numpy.ndarray
            Input image
            
        Returns:
        -------
        enhanced : numpy.ndarray
            Contrast-enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Split LAB channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge channels back
        merged = cv2.merge([cl, a, b])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def preprocess_image(self, image):
        """
        Preprocess an image for skin pigmentation analysis
        
        Parameters:
        ----------
        image : numpy.ndarray
            Input image
            
        Returns:
        -------
        processed_img : numpy.ndarray
            Preprocessed image
        original_face : numpy.ndarray
            Original cropped face image
        face_detected : bool
            Whether a face was detected
        """
        # Make a copy of the original image
        img = image.copy()
        
        # Detect and crop face
        face_img, face_detected = self.detect_face(img)
        
        if not face_detected:
            print("No face detected in the image")
            return None, None, False
        
        # Save the original face image for reference
        original_face = face_img.copy()
        
        # Apply gamma correction if enabled
        if self.gamma_correction:
            face_img = self.apply_gamma_correction(face_img, gamma=1.2)
        
        # Apply contrast enhancement if enabled
        if self.enhance_contrast:
            face_img = self.enhance_image_contrast(face_img)
        
        # Extract skin mask
        skin_mask = self.extract_skin_mask(face_img)
        
        # Apply skin mask to isolate skin regions
        skin_only = cv2.bitwise_and(face_img, face_img, mask=skin_mask)
        
        # Resize to target size
        processed_img = cv2.resize(skin_only, self.target_size)
        
        # Normalize if enabled
        if self.normalize:
            processed_img = processed_img.astype(np.float32) / 255.0
        
        return processed_img, original_face, True
        
    def visualize_preprocessing(self, image):
        """
        Visualize steps of the preprocessing pipeline
        
        Parameters:
        ----------
        image : numpy.ndarray
            Input image
            
        Returns:
        -------
        None
        """
        img = image.copy()
        
        # Detect and crop face
        face_img, face_detected = self.detect_face(img)
        
        if not face_detected:
            print("No face detected in the image")
            return
        
        # Apply gamma correction
        gamma_img = self.apply_gamma_correction(face_img, gamma=1.2)
        
        # Apply contrast enhancement
        enhanced_img = self.enhance_image_contrast(face_img)
        
        # Extract skin mask
        skin_mask = self.extract_skin_mask(face_img)
        
        # Apply skin mask
        skin_only = cv2.bitwise_and(face_img, face_img, mask=skin_mask)
        
        # Display results
        plt.figure(figsize=(20, 10))
        
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        plt.title('Detected Face')
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.imshow(cv2.cvtColor(gamma_img, cv2.COLOR_BGR2RGB))
        plt.title('Gamma Correction')
        plt.axis('off')
        
        plt.subplot(2, 3, 4)
        plt.imshow(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))
        plt.title('Contrast Enhanced')
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
        plt.imshow(skin_mask, cmap='gray')
        plt.title('Skin Mask')
        plt.axis('off')
        
        plt.subplot(2, 3, 6)
        plt.imshow(cv2.cvtColor(skin_only, cv2.COLOR_BGR2RGB))
        plt.title('Skin Only')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
def detect_pigmentation_spots(image, threshold=0.7):
    """
    Detect pigmentation spots in a preprocessed facial image
    
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
    """
    # Convert to grayscale if it's not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Apply morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    spots_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    return spots_mask 