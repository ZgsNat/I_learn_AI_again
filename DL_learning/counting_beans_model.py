import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy import ndimage
import argparse
import os
class BeanCounter:
    def __init__(self, 
                 blur_kernel=5,
                 canny_low=50,
                 canny_high=150,
                 min_contour_area=100,
                 max_contour_area=2000,
                 circularity_threshold=0.3,
                 dbscan_eps=30,
                 dbscan_min_samples=1):
        """
        Initialize the Bean Counter with customizable parameters
        
        Args:
            blur_kernel: Gaussian blur kernel size
            canny_low: Lower threshold for Canny edge detection
            canny_high: Upper threshold for Canny edge detection
            min_contour_area: Minimum area for valid bean contours
            max_contour_area: Maximum area for valid bean contours
            circularity_threshold: Minimum circularity for bean detection
            dbscan_eps: DBSCAN clustering epsilon parameter
            dbscan_min_samples: DBSCAN minimum samples parameter
        """
        self.blur_kernel = blur_kernel
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.min_contour_area = min_contour_area
        self.max_contour_area = max_contour_area
        self.circularity_threshold = circularity_threshold
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        
    def preprocess_image(self, image):
        """
        Preprocess the image for better bean detection
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(blurred)
        
        return enhanced
    
    def detect_bean_contours(self, image):
        """
        Detect bean contours using edge detection and contour analysis
        """
        # Apply Canny edge detection
        edges = cv2.Canny(image, self.canny_low, self.canny_high)
        
        # Apply morphological operations to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return contours, edges
    
    def filter_bean_contours(self, contours):
        """
        Filter contours based on area and shape characteristics
        """
        valid_contours = []
        
        for contour in contours:
            # Calculate area
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_contour_area or area > self.max_contour_area:
                continue
            
            # Calculate circularity (4π*area/perimeter²)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Filter by circularity (beans are roughly circular/oval)
            if circularity > self.circularity_threshold:
                valid_contours.append(contour)
        
        return valid_contours
    
    def get_contour_centers(self, contours):
        """
        Get the center points of contours
        """
        centers = []
        
        for contour in contours:
            # Calculate moments
            M = cv2.moments(contour)
            
            # Calculate center coordinates
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy))
        
        return centers
    
    def cluster_nearby_detections(self, centers):
        """
        Use DBSCAN clustering to merge nearby detections of the same bean
        """
        if len(centers) == 0:
            return centers
        
        # Convert to numpy array
        points = np.array(centers)
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        clusters = clustering.fit_predict(points)
        
        # Get cluster centers
        unique_clusters = np.unique(clusters)
        cluster_centers = []
        
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Noise points
                noise_points = points[clusters == cluster_id]
                cluster_centers.extend([tuple(point) for point in noise_points])
            else:
                cluster_points = points[clusters == cluster_id]
                center = tuple(np.mean(cluster_points, axis=0).astype(int))
                cluster_centers.append(center)
        
        return cluster_centers
    
    def count_beans(self, image_path):
        """
        Main method to count beans in an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            tuple: (count, annotated_image)
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Preprocess image
        processed = self.preprocess_image(image)
        
        # Detect contours
        contours, edges = self.detect_bean_contours(processed)
        
        # Filter valid bean contours
        valid_contours = self.filter_bean_contours(contours)
        
        # Get contour centers
        centers = self.get_contour_centers(valid_contours)
        
        # Cluster nearby detections
        final_centers = self.cluster_nearby_detections(centers)
        
        # Create annotated image
        annotated_image = self.annotate_image(image, final_centers)
        
        return len(final_centers), annotated_image, edges
    
    def annotate_image(self, image, centers, circle_color=(0, 255, 0), circle_thickness=3):
        """
        Draw green circles around detected beans
        """
        annotated = image.copy()
        
        for i, (cx, cy) in enumerate(centers):
            # Draw green circle around each detected bean
            cv2.circle(annotated, (cx, cy), 25, circle_color, circle_thickness)
            
            # Add number label
            cv2.putText(annotated, str(i+1), (cx-10, cy-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, circle_color, 2)
        
        return annotated
    
    def display_results(self, original_image, annotated_image, edges, count):
        """
        Display the results using matplotlib
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Edge detection result
        axes[1].imshow(edges, cmap='gray')
        axes[1].set_title('Edge Detection')
        axes[1].axis('off')
        
        # Annotated result
        axes[2].imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f'Detected Beans: {count}')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    """
    Main function to run the bean counter
    """
    parser = argparse.ArgumentParser(description='Count beans in an image')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--blur-kernel', type=int, default=5, help='Gaussian blur kernel size')
    parser.add_argument('--canny-low', type=int, default=50, help='Canny low threshold')
    parser.add_argument('--canny-high', type=int, default=150, help='Canny high threshold')
    parser.add_argument('--min-area', type=int, default=100, help='Minimum contour area')
    parser.add_argument('--max-area', type=int, default=2000, help='Maximum contour area')
    parser.add_argument('--circularity', type=float, default=0.3, help='Minimum circularity threshold')
    parser.add_argument('--save-result', help='Path to save the annotated image')
    
    args = parser.parse_args()
    
    # Initialize bean counter
    counter = BeanCounter(
        blur_kernel=args.blur_kernel,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        min_contour_area=args.min_area,
        max_contour_area=args.max_area,
        circularity_threshold=args.circularity
    )
    
    try:
        # Count beans
        count, annotated_image, edges = counter.count_beans(args.image_path)
        
        # Load original for display
        original_image = cv2.imread(args.image_path)
        
        # Display results
        counter.display_results(original_image, annotated_image, edges, count)
        
        print(f"Total beans detected: {count}")
        
        # Save result if requested
        if args.save_result:
            cv2.imwrite(args.save_result, annotated_image)
            print(f"Annotated image saved to: {args.save_result}")
            
    except Exception as e:
        print(f"Error: {e}")

# Example usage
if __name__ == "__main__":
    # --- BẮT ĐẦU PHẦN CHỈNH SỬA ---

    # 1. Xác định đường dẫn tuyệt đối đến ảnh của bạn
    # Đảm bảo file ảnh "CEO-2-5-scaled.jpg" nằm cùng thư mục với file Python này
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_dir, "CEO-2-5-scaled.jpg")
    except NameError:
        # Xử lý khi chạy trong môi trường không có __file__ (ví dụ: notebook)
        image_path = "CEO-2-5-scaled.jpg"


    # 2. Khởi tạo và chạy bộ đếm
    print(f"Đang xử lý ảnh: {image_path}")
    counter = BeanCounter() # Sử dụng các tham số mặc định
    
    try:
        # Thực hiện đếm
        count, annotated_image, edges = counter.count_beans(image_path)
        
        # Tải lại ảnh gốc để hiển thị
        original_image = cv2.imread(image_path)
        
        # Hiển thị kết quả
        counter.display_results(original_image, annotated_image, edges, count)
        
        print(f"Hoàn thành! Tổng số hạt đếm được: {count}")
        
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")

# Additional utility function for batch processing
def batch_count_beans(image_folder, output_folder=None):
    """
    Count beans in multiple images
    
    Args:
        image_folder: Folder containing images
        output_folder: Optional folder to save annotated images
    """
    import os
    from pathlib import Path
    
    counter = BeanCounter()
    results = []
    
    for image_file in Path(image_folder).glob('*.jpg'):
        try:
            count, annotated, _ = counter.count_beans(str(image_file))
            results.append((image_file.name, count))
            
            if output_folder:
                output_path = Path(output_folder) / f"annotated_{image_file.name}"
                cv2.imwrite(str(output_path), annotated)
                
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
    
    return results