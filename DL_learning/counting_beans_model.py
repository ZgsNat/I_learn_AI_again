import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, "CEO-2-5-scaled.jpg")

def count_beans_watershed_with_labels(image_path):
    """
    Hàm này thực hiện đếm hạt đậu bằng thuật toán Watershed và hiển thị kết quả
    với đường viền và số thứ tự cho từng hạt.
    """
    # 1. Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        print(f"Lỗi: Không thể đọc ảnh từ đường dẫn {image_path}")
        return

    # Tạo bản sao của ảnh gốc để vẽ kết quả lên
    img_labeled = img.copy()

    # --- Các bước xử lý ảnh (như code trước) ---
    # 2. Tiền xử lý
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Phân ngưỡng thích ứng
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # 4. Dọn dẹp nhiễu
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 5. Chuẩn bị cho Watershed
    sure_bg = cv2.dilate(closing, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 6. Tạo Markers
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # 7. Áp dụng Watershed
    markers = cv2.watershed(img, markers)

    # --- Phần đếm và hiển thị chi tiết ---
    # 8. Đếm, vẽ đường viền và đánh số
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4  # Giảm kích thước font một chút để đỡ rối
    font_thickness = 1
    text_color = (0, 0, 0) # Chữ màu đen

    # Lấy các nhãn (labels) duy nhất mà watershed đã tạo ra
    unique_labels = np.unique(markers)
    # Loại bỏ nhãn -1 (ranh giới) và 1 (nền) để chỉ còn lại các nhãn của hạt đậu
    bean_labels = [label for label in unique_labels if label > 1]
    bean_count = len(bean_labels)

    # Lặp qua từng nhãn của hạt đậu để xử lý
    for i, label in enumerate(bean_labels):
        # Tạo một "mặt nạ" chỉ chứa hạt đậu hiện tại
        mask = np.zeros(markers.shape, dtype=np.uint8)
        mask[markers == label] = 255

        # Tìm đường viền (contour) của hạt đậu đó
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Lấy contour lớn nhất (để tránh các nhiễu nhỏ nếu có)
            cnt = max(contours, key=cv2.contourArea)

            # Vẽ đường viền màu xanh lá cây lên ảnh kết quả
            cv2.drawContours(img_labeled, [cnt], -1, (0, 255, 0), 1)

            # Tìm tọa độ tâm của contour để đặt số thứ tự
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # Vẽ số thứ tự (bắt đầu từ 1) lên ảnh
                cv2.putText(img_labeled, str(i + 1), (cX - 10, cY + 5), font,
                            font_scale, text_color, font_thickness, cv2.LINE_AA)

    # In ra tổng số lượng
    print(f"Tổng số lượng hạt đậu đếm được: {bean_count}")

    # --- Hiển thị ảnh kết quả cuối cùng ---
    # Sử dụng Matplotlib để hiển thị ảnh trong các môi trường như Jupyter Notebook
    plt.figure(figsize=(12, 12)) # Kích thước cửa sổ hiển thị
    plt.imshow(cv2.cvtColor(img_labeled, cv2.COLOR_BGR2RGB)) # Chuyển BGR (OpenCV) sang RGB (Matplotlib)
    plt.title(f'Kết quả đếm hạt đậu: {bean_count} hạt', fontsize=16)
    plt.axis('off') # Ẩn các trục tọa độ
    plt.show()

# --- Chạy chương trình ---
# Thay 'bean_image.jpg' bằng đường dẫn đến ảnh của bạn
# Ví dụ: count_beans_watershed_with_labels('C:/Users/Admin/Desktop/bean_image.jpg') 
count_beans_watershed_with_labels(image_path)