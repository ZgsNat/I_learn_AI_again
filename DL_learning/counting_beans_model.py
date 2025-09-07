import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, "CEO-2-5-scaled-removebg-preview.png ")

def count_beans_watershed_with_labels(image_path):
    """
    Đếm hạt đậu bằng thuật toán Watershed và hiển thị kết quả
    với đường viền và số thứ tự cho từng hạt, đồng thời hiển thị
    tất cả các bước xử lý ảnh trên cùng một figure.
    """
    # 1. Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        print(f"Lỗi: Không thể đọc ảnh từ đường dẫn {image_path}")
        return

    img_labeled = img.copy()

    # Danh sách lưu các bước trung gian để hiển thị
    step_imgs = []
    step_titles = []

    # 2. Tiền xử lý
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    step_imgs.append(gray)
    step_titles.append("1. Grayscale")

    # --- BƯỚC CẢI TIẾN: XỬ LÝ NỀN ĐỂ CHUẨN HÓA ÁNH SÁNG ---

    # 2a. Ước tính nền bằng cách làm mờ ảnh với kernel rất lớn
    kernel_size = 151  # Phải là số lẻ, đủ lớn để xóa các hạt đậu
    background = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    step_imgs.append(background)
    step_titles.append("2. Estimated Background")

    # 2b. Trừ nền để lấy tiền cảnh, giúp loại bỏ ánh sáng không đều
    foreground = cv2.subtract(gray, background)
    foreground = cv2.bitwise_not(foreground)
    step_imgs.append(foreground)
    step_titles.append("3. Normalized Foreground")


    # 3. Phân ngưỡng trên ảnh đã được chuẩn hóa ánh sáng
    # Otsu giờ sẽ hoạt động hiệu quả hơn rất nhiều
    ret, thresh = cv2.threshold(foreground, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    step_imgs.append(thresh)
    step_titles.append("4. Otsu's Threshold")

    # 4. Dọn dẹp nhiễu
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    step_imgs.append(opening)
    step_titles.append("5. Opening (Noise Removed)")

    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
    step_imgs.append(closing)
    step_titles.append("6. Closing (Holes Filled)")


    # 5. Chuẩn bị cho Watershed
    sure_bg = cv2.dilate(closing, kernel, iterations=3)
    step_imgs.append(sure_bg)
    step_titles.append("7. Sure Background")

    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
    step_imgs.append(dist_transform)
    step_titles.append("8. Distance Transform")

    # Điều chỉnh ngưỡng này để tách hạt tốt hơn, giảm xuống một chút
    ret, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    step_imgs.append(sure_fg)
    step_titles.append("9. Sure Foreground")

    unknown = cv2.subtract(sure_bg, sure_fg)
    step_imgs.append(unknown)
    step_titles.append("10. Unknown Region")

    # 6. Tạo Markers
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Hiển thị markers trước khi áp dụng Watershed
    markers_vis = cv2.applyColorMap(np.uint8(255 * markers / (markers.max() + 1)), cv2.COLORMAP_JET)
    step_imgs.append(markers_vis)
    step_titles.append("11. Markers")


    # 7. Áp dụng Watershed
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0] # Vẽ đường biên màu đỏ

    # --- Phần đếm và hiển thị chi tiết ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    unique_labels = np.unique(markers)
    # Bỏ qua label 1 (nền) và -1 (đường biên)
    bean_labels = [label for label in unique_labels if label > 1]
    bean_count = len(bean_labels)

    for i, label in enumerate(bean_labels):
        mask = np.zeros(markers.shape, dtype=np.uint8)
        mask[markers == label] = 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            # Vẽ viền xanh quanh mỗi hạt được đếm
            cv2.drawContours(img_labeled, [cnt], -1, (0, 255, 0), 2)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # Viết số thứ tự
                cv2.putText(img_labeled, str(i + 1), (cX - 10, cY + 5), font,
                            0.5, (255, 0, 0), 2, cv2.LINE_AA)

    print(f"Tổng số lượng hạt đậu đếm được: {bean_count}")

    # Thêm kết quả cuối cùng vào danh sách hiển thị
    step_imgs.append(cv2.cvtColor(img_labeled, cv2.COLOR_BGR2RGB))
    step_titles.append(f'12. Final Result ({bean_count} beans)')

    # --- Hiển thị tất cả các bước trên cùng một figure ---
    n_steps = len(step_imgs)
    n_cols = 4
    n_rows = int(np.ceil(n_steps / n_cols))
    plt.figure(figsize=(4 * n_cols, 4 * n_rows))

    for i, (img_disp, title) in enumerate(zip(step_imgs, step_titles)):
        plt.subplot(n_rows, n_cols, i + 1)
        if len(img_disp.shape) == 2:
            cmap = 'gray'
            if "Distance Transform" in title:
                cmap = 'jet'
            plt.imshow(img_disp, cmap=cmap)
        else:
            plt.imshow(img_disp)
        plt.title(title, fontsize=12)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def count_beans_optimized(image_path):
    """
    Đếm hạt đậu trên ảnh đã xóa nền bằng quy trình được tối ưu hóa.
    """
    # 1. Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        print(f"Lỗi: Không thể đọc ảnh từ đường dẫn {image_path}")
        return
    
    img_labeled = img.copy()
    step_imgs = []
    step_titles = []

    # 2. Chuyển sang ảnh xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    step_imgs.append(gray)
    step_titles.append("1. Grayscale")

    # 3. Phân ngưỡng trực tiếp (BỎ QUA BƯỚC ƯỚC TÍNH NỀN)
    # Vì nền đã sạch, Otsu sẽ hoạt động rất tốt
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    step_imgs.append(thresh)
    step_titles.append("2. Otsu's Threshold")

    # 4. Dọn dẹp nhiễu
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    step_imgs.append(opening)
    step_titles.append("3. Opening (Noise Removed)")

    # 5. Chuẩn bị cho Watershed
    # Xác định vùng nền chắc chắn
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    step_imgs.append(sure_bg)
    step_titles.append("4. Sure Background")

    # Xác định vùng tiền cảnh chắc chắn
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    step_imgs.append(dist_transform)
    step_titles.append("5. Distance Transform")
    
    # THAM SỐ QUAN TRỌNG: Điều chỉnh giá trị này (0.2 -> 0.4) để tách hoặc gộp các hạt
    threshold_ratio = 0.3 
    ret, sure_fg = cv2.threshold(dist_transform, threshold_ratio * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    step_imgs.append(sure_fg)
    step_titles.append(f"6. Sure Foreground (Thresh={threshold_ratio})")

    # Vùng không xác định
    unknown = cv2.subtract(sure_bg, sure_fg)
    step_imgs.append(unknown)
    step_titles.append("7. Unknown Region")

    # 6. Tạo Markers
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers_vis = cv2.applyColorMap(np.uint8(markers * 255 / (ret + 1)), cv2.COLORMAP_JET)
    step_imgs.append(markers_vis)
    step_titles.append("8. Markers")

    # 7. Áp dụng Watershed
    markers = cv2.watershed(img, markers)
    
    # --- Phần đếm và hiển thị chi tiết ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    unique_labels = np.unique(markers)
    # Bỏ qua label 1 (nền) và -1 (đường biên)
    bean_labels = [label for label in unique_labels if label > 1]
    bean_count = len(bean_labels)

    for i, label in enumerate(bean_labels):
        # Tạo mask cho từng hạt đậu
        mask = np.zeros(markers.shape, dtype=np.uint8)
        mask[markers == label] = 255
        # Tìm contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Vẽ viền xanh quanh mỗi hạt
            cv2.drawContours(img_labeled, contours, -1, (0, 255, 0), 1)

    print(f"Tổng số lượng hạt đậu đếm được: {bean_count}")
    
    # Thêm text số lượng vào ảnh kết quả
    cv2.putText(img_labeled, f"Count: {bean_count}", (20, 40), font, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
    step_imgs.append(cv2.cvtColor(img_labeled, cv2.COLOR_BGR2RGB))
    step_titles.append(f'9. Final Result ({bean_count} beans)')

    # --- Hiển thị tất cả các bước ---
    n_steps = len(step_imgs)
    n_cols = 3
    n_rows = int(np.ceil(n_steps / n_cols))
    plt.figure(figsize=(4 * n_cols, 4 * n_rows))

    for i, (img_disp, title) in enumerate(zip(step_imgs, step_titles)):
        plt.subplot(n_rows, n_cols, i + 1)
        cmap = 'gray' if len(img_disp.shape) == 2 else None
        if "Distance Transform" in title or "Markers" in title:
            cmap = 'jet'
        plt.imshow(img_disp, cmap=cmap)
        plt.title(title, fontsize=12)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# --- Chạy chương trình ---
def count_beans_adaptive_final(image_path):
    """
    Sử dụng Phân ngưỡng Thích ứng (Adaptive Thresholding) để tách các hạt đậu
    ở mật độ cao, là phương pháp mạnh mẽ nhất cho bài toán này.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Lỗi: Không thể đọc ảnh từ {image_path}")
        return

    img_labeled = img.copy()
    step_imgs, step_titles = [], []

    # --- Bước 1-3: Giữ nguyên phần chuẩn hóa ánh sáng hiệu quả ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    step_imgs.append(gray); step_titles.append("1. Grayscale")

    background = cv2.GaussianBlur(gray, (151, 151), 0)
    step_imgs.append(background); step_titles.append("2. Estimated Background")

    foreground = cv2.subtract(gray, background)
    foreground = cv2.bitwise_not(foreground)
    step_imgs.append(foreground); step_titles.append("3. Normalized Foreground")

    # --- Bước 4: NÂNG CẤP LỚN - SỬ DỤNG ADAPTIVE THRESHOLDING ---
    # Thay thế Otsu bằng Adaptive Thresholding để xử lý các hạt dính nhau
    # blockSize: Kích thước vùng lân cận để tính ngưỡng (phải là số lẻ)
    # C: Hằng số trừ đi từ giá trị trung bình, giúp tinh chỉnh kết quả
    blockSize = 25 # Có thể thử các giá trị như 15, 21, 25, 31...
    C = 4          # Có thể thử các giá trị như 2, 3, 4, 5...
    thresh = cv2.adaptiveThreshold(foreground, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, blockSize, C)
    step_imgs.append(thresh); step_titles.append(f"4. Adaptive Threshold (Block={blockSize}, C={C})")

    # --- Các bước sau giữ nguyên, nhưng sẽ hoạt động trên ảnh nhị phân tốt hơn nhiều ---
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    step_imgs.append(opening); step_titles.append("5. Opening (Noise Removed)")

    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    step_imgs.append(sure_bg); step_titles.append("6. Sure Background")

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    step_imgs.append(dist_transform); step_titles.append("7. Distance Transform")

    threshold_ratio = 0.4 # Có thể cần tăng nhẹ ratio này lên một chút (0.3 -> 0.5)
    ret, sure_fg = cv2.threshold(dist_transform, threshold_ratio * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    step_imgs.append(sure_fg); step_titles.append(f"8. Sure Foreground (Ratio={threshold_ratio})")
    
    unknown = cv2.subtract(sure_bg, sure_fg)
    step_imgs.append(unknown); step_titles.append("9. Unknown Region")
    
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)

    # --- Đếm và hiển thị kết quả ---
    unique_labels = np.unique(markers)
    bean_count = len([label for label in unique_labels if label > 1])

    # Vẽ viền và hiển thị
    for label in np.unique(markers):
        if label <= 1: continue # Bỏ qua nền và đường biên
        mask = np.zeros(markers.shape, dtype=np.uint8)
        mask[markers == label] = 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_labeled, contours, -1, (0, 255, 0), 1) # Viền mỏng hơn

    print(f"Tổng số lượng hạt đậu đếm được: {bean_count}")
    cv2.putText(img_labeled, f"Count: {bean_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    step_imgs.append(cv2.cvtColor(img_labeled, cv2.COLOR_BGR2RGB)); step_titles.append(f'10. Final Result ({bean_count} beans)')
    
    # Hiển thị
    plt.figure(figsize=(16, 12))
    for i, (img_disp, title) in enumerate(zip(step_imgs, step_titles)):
        plt.subplot(3, 4, i + 1)
        cmap = 'gray' if len(img_disp.shape) == 2 else None
        if "Distance Transform" in title: cmap = 'jet'
        plt.imshow(img_disp, cmap=cmap); plt.title(title, fontsize=10); plt.axis('off')
    plt.tight_layout(); plt.show()

# --- Chạy chương trình ---
count_beans_adaptive_final(image_path)
