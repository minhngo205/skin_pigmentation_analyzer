import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as mcolors

class PigmentationAnalyzer:
    """
    A class to analyze skin pigmentation, classify spots, and determine severity
    """
    def __init__(self, severity_levels=5):
        """
        Initialize the pigmentation analyzer
        
        Parameters:
        ----------
        severity_levels : int
            Number of severity levels for classification (default: 5)
        """
        self.severity_levels = severity_levels
        self.severity_thresholds = None
        
    def _calculate_severity_score(self, features):
        """
        Calculate a severity score based on extracted features
        
        Parameters:
        ----------
        features : dict
            Dictionary of extracted features from the image
            
        Returns:
        -------
        score : float
            Severity score between 0 and 100
        """
        # Extract relevant features for scoring
        global_features = features['global']
        spots = features['spots']
        
        # Count of spots
        num_spots = len(spots)
        
        # Calculate total area of spots
        spot_percentage = global_features['spot_percentage']
        
        # Calculate average spot size
        avg_spot_size = np.mean([spot['area'] for spot in spots]) if spots else 0
        
        # Calculate average spot darkness
        # This depends on the color space used in feature extraction
        # We'll assume RGB space and check for color feature keys
        if spots and 'mean_r' in spots[0]:
            # RGB space
            avg_darkness = np.mean([
                255 - (spot['mean_r'] + spot['mean_g'] + spot['mean_b'])/3 
                for spot in spots
            ])
        elif spots and 'mean_l' in spots[0]:
            # LAB space, lower L means darker
            avg_darkness = np.mean([
                100 - spot['mean_l'] for spot in spots
            ])
        else:
            avg_darkness = 0
            
        # Normalize the metrics to 0-100 scale
        # These normalization factors are based on empirical observations
        # and may need to be adjusted based on your dataset
        norm_spot_count = min(100, num_spots * 2.5)  # Assuming max ~40 spots
        norm_coverage = min(100, spot_percentage * 10)  # Assuming max ~10% coverage
        norm_size = min(100, avg_spot_size / 20)  # Assuming max size ~2000 pixels
        norm_darkness = min(100, avg_darkness)  # Already in 0-100 scale
        
        # Weight the metrics to calculate final score
        # These weights can be adjusted based on medical guidance
        weights = {
            'count': 0.3,
            'coverage': 0.4,
            'size': 0.2,
            'darkness': 0.1
        }
        
        score = (
            weights['count'] * norm_spot_count +
            weights['coverage'] * norm_coverage +
            weights['size'] * norm_size +
            weights['darkness'] * norm_darkness
        )
        
        return min(100, score)
    
    def _classify_severity(self, score):
        """
        Classify the severity level based on the score
        
        Parameters:
        ----------
        score : float
            Severity score between 0 and 100
            
        Returns:
        -------
        level : int
            Severity level (1 to severity_levels)
        label : str
            Descriptive label for the severity level
        """
        if self.severity_thresholds is None:
            # Define thresholds for severity levels
            self.severity_thresholds = np.linspace(0, 100, self.severity_levels + 1)
            
        # Determine severity level
        for i in range(1, len(self.severity_thresholds)):
            if score <= self.severity_thresholds[i]:
                level = i
                break
        else:
            level = self.severity_levels
            
        # Generate severity label
        if self.severity_levels == 5:
            labels = {
                1: "Minimal",
                2: "Mild",
                3: "Moderate",
                4: "Significant",
                5: "Severe"
            }
        else:
            labels = {i+1: f"Level {i+1}" for i in range(self.severity_levels)}
            
        return level, labels[level]
    
    def classify_spots(self, features, n_clusters=3):
        """
        Classify spots into different types based on their features
        
        Parameters:
        ----------
        features : dict
            Dictionary of extracted features from the image
        n_clusters : int
            Number of clusters for spot classification
            
        Returns:
        -------
        cluster_labels : list
            Cluster label for each spot
        cluster_info : dict
            Information about each cluster
        """
        spots = features['spots']
        
        if not spots:
            return [], {}
            
        # Extract relevant features for clustering
        feature_vectors = []
        
        for spot in spots:
            # Basic shape features
            vec = [
                spot['area'],
                spot['perimeter'],
                spot['eccentricity'],
                spot['circularity']
            ]
            
            # Add color features if available
            if 'mean_r' in spot:  # RGB
                vec.extend([
                    spot['mean_r'],
                    spot['mean_g'],
                    spot['mean_b']
                ])
            elif 'mean_l' in spot:  # LAB
                vec.extend([
                    spot['mean_l'],
                    spot['mean_a'],
                    spot['mean_b']
                ])
            elif 'mean_h' in spot:  # HSV
                vec.extend([
                    spot['mean_h'],
                    spot['mean_s'],
                    spot['mean_v']
                ])
                
            feature_vectors.append(vec)
            
        # Convert to array
        X = np.array(feature_vectors)
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply clustering (KMeans)
        kmeans = KMeans(n_clusters=min(n_clusters, len(spots)), random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Calculate cluster information
        cluster_info = {}
        
        for cluster_id in range(n_clusters):
            # Get spots in this cluster
            cluster_spots = [spots[i] for i in range(len(spots)) if cluster_labels[i] == cluster_id]
            
            if not cluster_spots:
                continue
                
            # Calculate average features for the cluster
            avg_area = np.mean([spot['area'] for spot in cluster_spots])
            avg_circularity = np.mean([spot['circularity'] for spot in cluster_spots])
            
            # Color info depends on color space
            if 'mean_r' in cluster_spots[0]:  # RGB
                color_info = {
                    'mean_r': np.mean([spot['mean_r'] for spot in cluster_spots]),
                    'mean_g': np.mean([spot['mean_g'] for spot in cluster_spots]),
                    'mean_b': np.mean([spot['mean_b'] for spot in cluster_spots])
                }
            elif 'mean_l' in cluster_spots[0]:  # LAB
                color_info = {
                    'mean_l': np.mean([spot['mean_l'] for spot in cluster_spots]),
                    'mean_a': np.mean([spot['mean_a'] for spot in cluster_spots]),
                    'mean_b': np.mean([spot['mean_b'] for spot in cluster_spots])
                }
            else:  # HSV
                color_info = {
                    'mean_h': np.mean([spot['mean_h'] for spot in cluster_spots]),
                    'mean_s': np.mean([spot['mean_s'] for spot in cluster_spots]),
                    'mean_v': np.mean([spot['mean_v'] for spot in cluster_spots])
                }
                
            # Store cluster info
            cluster_info[cluster_id] = {
                'count': len(cluster_spots),
                'avg_area': avg_area,
                'avg_circularity': avg_circularity,
                'color': color_info
            }
            
        return cluster_labels, cluster_info
    
    def analyze(self, image, features):
        """
        Analyze skin pigmentation and determine severity
        
        Parameters:
        ----------
        image : numpy.ndarray
            Original image
        features : dict
            Dictionary of extracted features from the image
            
        Returns:
        -------
        results : dict
            Dictionary containing analysis results
        """
        # Calculate severity score
        score = self._calculate_severity_score(features)
        
        # Classify severity level
        level, label = self._classify_severity(score)
        
        # Classify spots into types
        cluster_labels, cluster_info = self.classify_spots(features)
        
        # Combine results
        results = {
            'severity_score': score,
            'severity_level': level,
            'severity_label': label,
            'spot_count': len(features['spots']),
            'coverage_percentage': features['global']['spot_percentage'],
            'spot_clusters': cluster_info,
            'spot_cluster_labels': cluster_labels
        }
        
        return results
    
    def visualize_results(self, image, features, results):
        """
        Visualize the analysis results
        
        Parameters:
        ----------
        image : numpy.ndarray
            Original image
        features : dict
            Dictionary of extracted features from the image
        results : dict
            Dictionary containing analysis results
            
        Returns:
        -------
        None
        """
        # Create figure
        plt.figure(figsize=(18, 10))
        
        # Original image
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        # Extract spots mask
        if 'spots_mask' in features:
            spots_mask = features['spots_mask']
        else:
            # If mask is not provided, create one from labeled regions
            spots_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            for spot in features['spots']:
                # Use bounding box to approximate spot location
                y1, x1, y2, x2 = spot['bbox']
                cv2.rectangle(spots_mask, (x1, y1), (x2, y2), 255, -1)
        
        # Spots visualization
        plt.subplot(2, 3, 2)
        plt.imshow(spots_mask, cmap='gray')
        plt.title('Detected Spots')
        plt.axis('off')
        
        # Visualize spots on original image
        plt.subplot(2, 3, 3)
        viz_img = image.copy()
        
        # If we have cluster labels, use them to colorize spots
        if results['spot_cluster_labels']:
            cluster_labels = results['spot_cluster_labels']
            
            # Create colormap for clusters
            colors = plt.cm.viridis(np.linspace(0, 1, len(set(cluster_labels))))
            
            # Draw spots with cluster colors
            for i, spot in enumerate(features['spots']):
                if i < len(cluster_labels):
                    # Get cluster color
                    color = colors[cluster_labels[i]][:3]
                    color = tuple([int(c * 255) for c in color])
                    
                    # Draw spot
                    y1, x1, y2, x2 = spot['bbox']
                    cv2.rectangle(viz_img, (x1, y1), (x2, y2), color, 2)
        else:
            # Create a binary overlay
            spot_overlay = np.zeros_like(image)
            spot_overlay[spots_mask > 0] = [0, 0, 255]  # Red for spots
            
            # Blend with original image
            viz_img = cv2.addWeighted(image, 0.7, spot_overlay, 0.3, 0)
            
        plt.imshow(cv2.cvtColor(viz_img, cv2.COLOR_BGR2RGB))
        plt.title('Spots Classification')
        plt.axis('off')
        
        # Severity gauge
        plt.subplot(2, 3, 4)
        self._plot_severity_gauge(results['severity_score'], results['severity_label'])
        
        # Spot statistics
        plt.subplot(2, 3, 5)
        self._plot_spot_statistics(features, results)
        
        # Cluster information
        plt.subplot(2, 3, 6)
        self._plot_cluster_info(results['spot_clusters'])
        
        plt.tight_layout()
        plt.show()
        
    def _plot_severity_gauge(self, score, label):
        """
        Plot a gauge showing the severity score
        """
        # Create gauge-like display
        fig, ax = plt.subplot_mosaic([['gauge']], subplot_kw={'projection': 'polar'})
        gauge_ax = ax['gauge']
        
        # Define colors for severity levels
        colors = plt.cm.RdYlGn_r(np.linspace(0, 1, self.severity_levels))
        
        # Normalize score to 0-1 range
        norm_score = score / 100.0
        
        # Plot severity gauge (semi-circle)
        theta = np.linspace(np.pi, 0, 100)
        r = np.ones_like(theta)
        
        # Create color segments for severity levels
        for i in range(self.severity_levels):
            idx = (theta <= np.pi * (1 - i/self.severity_levels)) & (theta > np.pi * (1 - (i+1)/self.severity_levels))
            gauge_ax.fill_between(theta[idx], 0, r[idx], color=colors[i], alpha=0.7)
        
        # Add needle pointer
        needle_angle = np.pi * (1 - norm_score)
        gauge_ax.plot([needle_angle, needle_angle], [0, 0.8], color='black', linewidth=2)
        gauge_ax.scatter([needle_angle], [0.8], color='black', s=100)
        
        # Add score text
        gauge_ax.text(np.pi/2, 0.5, f"{score:.1f}", fontsize=24, ha='center', va='center')
        gauge_ax.text(np.pi/2, 0.2, f"{label}", fontsize=16, ha='center', va='center')
        
        # Clean up the look
        gauge_ax.set_theta_zero_location('N')
        gauge_ax.set_theta_direction(-1)
        gauge_ax.set_rlim(0, 1)
        gauge_ax.set_rticks([])
        gauge_ax.set_xticks([])
        gauge_ax.set_title("Pigmentation Severity", pad=20)
        
    def _plot_spot_statistics(self, features, results):
        """
        Plot statistics about the detected spots
        """
        # Create a table of statistics
        plt.axis('off')
        plt.title('Pigmentation Statistics')
        
        # Calculate percentiles of spot sizes
        if features['spots']:
            areas = [spot['area'] for spot in features['spots']]
            area_p25 = np.percentile(areas, 25)
            area_p50 = np.percentile(areas, 50)
            area_p75 = np.percentile(areas, 75)
        else:
            area_p25, area_p50, area_p75 = 0, 0, 0
            
        # Prepare statistics
        stats = [
            ('Total Spots', f"{results['spot_count']}"),
            ('Coverage', f"{results['coverage_percentage']:.2f}%"),
            ('Median Spot Size', f"{area_p50:.1f} px²"),
            ('Severity Score', f"{results['severity_score']:.1f}"),
            ('Severity Level', f"{results['severity_label']}")
        ]
        
        # Create table
        table = plt.table(
            cellText=[[stat[1]] for stat in stats],
            rowLabels=[stat[0] for stat in stats],
            loc='center',
            cellLoc='center',
            colWidths=[0.5]
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)
        
    def _plot_cluster_info(self, cluster_info):
        """
        Plot information about the spot clusters
        """
        plt.axis('off')
        plt.title('Spot Types Analysis')
        
        if not cluster_info:
            plt.text(0.5, 0.5, "No spots detected", ha='center', va='center', fontsize=14)
            return
            
        # Prepare cluster data
        cluster_names = [f"Type {i+1}" for i in range(len(cluster_info))]
        counts = [info['count'] for info in cluster_info.values()]
        areas = [info['avg_area'] for info in cluster_info.values()]
        
        # Create a color for each cluster based on its color info
        cluster_colors = []
        for cluster_id, info in cluster_info.items():
            if 'mean_r' in info['color']:  # RGB
                r = info['color']['mean_r'] / 255
                g = info['color']['mean_g'] / 255
                b = info['color']['mean_b'] / 255
                cluster_colors.append((r, g, b))
            elif 'mean_l' in info['color']:  # LAB (approximate RGB)
                # Simple approximation of LAB to RGB
                l = info['color']['mean_l'] / 100
                cluster_colors.append((l, l, l))
            else:  # Default
                cluster_colors.append(plt.cm.viridis(cluster_id / len(cluster_info)))
        
        # Create table with cluster info
        rows = []
        row_labels = []
        
        for i, (cluster_id, info) in enumerate(cluster_info.items()):
            row = [
                f"{info['count']}",
                f"{info['avg_area']:.1f}",
                f"{info['avg_circularity']:.2f}"
            ]
            rows.append(row)
            row_labels.append(f"Type {i+1}")
            
        # Create table
        table = plt.table(
            cellText=rows,
            rowLabels=row_labels,
            colLabels=['Count', 'Avg Size', 'Circularity'],
            loc='center',
            cellLoc='center'
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)
        
        # Color the row labels according to cluster color
        for i, color in enumerate(cluster_colors):
            cell = table[i+1, -1]
            cell.set_facecolor(color)
            
            # Adjust text color for visibility
            brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            text_color = 'white' if brightness < 0.7 else 'black'
            cell.get_text().set_color(text_color) 