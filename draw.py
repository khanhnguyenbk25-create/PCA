import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =====================================================================
# HÀM CỦA BẠN: CHIẾU DỮ LIỆU VÀ VẼ BIỂU ĐỒ KÉP
# =====================================================================
# [ĐIỂM SỬA 1]: Thêm tham số raw_data vào đầu hàm, đổi tên scaled_data thành standardized_matrix cho rõ nghĩa.
def plot_pca_results(raw_data, standardized_matrix, sorted_eigenvectors, eigenvalues, num_components=2):
    """
    Hàm thực hiện giảm chiều dữ liệu và vẽ so sánh:
    - raw_data: Ma trận dữ liệu thô gốc (Dùng để hiển thị lên bảng bên trái).
    - standardized_matrix: Ma trận dữ liệu đã chuẩn hóa (Dùng để tính phép chiếu đồ thị).
    - sorted_eigenvectors: Ma trận vector riêng đã sắp xếp.
    - eigenvalues: Mảng trị riêng để tính phần trăm phương sai.
    - num_components: Số chiều muốn giảm xuống (mặc định là 2).
    """
    
    # ---------------------------------------------------------
    # BƯỚC 1: THỰC HIỆN PHÉP CHIẾU DỮ LIỆU (PROJECTION)
    # ---------------------------------------------------------
    # Trích xuất k vector riêng đầu tiên tạo thành Ma trận chiếu W
    W = sorted_eigenvectors[:, :num_components]
    
    # [ĐIỂM SỬA 2]: Phép nhân ma trận BẮT BUỘC dùng standardized_matrix
    projection = standardized_matrix @ W

    if projection.shape[1] < 2:
        print("LỖI: Dữ liệu sau khi chiếu chỉ có 1 chiều, không thể vẽ mặt phẳng 2D.")
        return

    # ---------------------------------------------------------
    # BƯỚC 2: VẼ ĐỒ THỊ
    # ---------------------------------------------------------
    # Tạo một khung Figure chứa 2 vùng (1 hàng, 2 cột)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- BÊN TRÁI: BẢNG SỐ LIỆU TRƯỚC PCA ---
    axes[0].axis('off')   # Tắt các trục tọa độ X, Y
    axes[0].axis('tight') # Ép khung hiển thị vừa vặn
    
    display_rows = 15 
    
    # [ĐIỂM SỬA 3]: Dùng raw_data để cắt dữ liệu đưa lên bảng, thay vì scaled_data
    table_data = np.round(raw_data[:display_rows, :], 2)
    col_labels = [f"Đặc trưng {i+1}" for i in range(raw_data.shape[1])]
    
    table = axes[0].table(cellText=table_data, 
                          colLabels=col_labels, 
                          loc='center', 
                          cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5) # Kéo giãn chiều cao các hàng ra 1.5 lần
    
    # [ĐIỂM SỬA 4]: Ghi rõ tiêu đề bảng là Dữ liệu thô
    axes[0].set_title(f'TRƯỚC PCA (Dữ liệu thô - Trích xuất {display_rows} dòng đầu)', fontsize=13, fontweight='bold')

    # --- BÊN PHẢI: BIỂU ĐỒ SAU KHI PCA ---
    total_variance = np.sum(eigenvalues)
    pc1_variance = (eigenvalues[0] / total_variance) * 100
    pc2_variance = (eigenvalues[1] / total_variance) * 100

    axes[1].scatter(projection[:, 0], projection[:, 1], 
                    c='#1f77b4', edgecolors='black', s=60, alpha=0.8)
    
    axes[1].set_title('SAU PCA (Không gian mới PC1 vs PC2)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel(f'Thành phần chính 1 (PC1) - {pc1_variance:.1f}% phương sai', fontsize=12)
    axes[1].set_ylabel(f'Thành phần chính 2 (PC2) - {pc2_variance:.1f}% phương sai', fontsize=12)
    
    axes[1].axhline(0, color='gray', linestyle='--', linewidth=1)
    axes[1].axvline(0, color='gray', linestyle='--', linewidth=1)
    axes[1].grid(True, linestyle=':', alpha=0.7)
    
    plt.suptitle('SO SÁNH BẢNG DỮ LIỆU GỐC VÀ BIỂU ĐỒ SAU PCA', fontsize=16, fontweight='heavy', color='navy')
    plt.tight_layout()
    plt.show()

# =====================================================================
# HÀM NẠP DỮ LIỆU TỪ FILE CSV
# =====================================================================
def load_csv_data(filepath):
    """
    Hàm nạp dữ liệu thuần số học.
    Toàn bộ file CSV được coi là ma trận đặc trưng (X).
    """
    try:
        # Đọc file CSV
        df = pd.read_csv(filepath, header=None)
    except FileNotFoundError:
        raise FileNotFoundError(f"LỖI: Không tìm thấy file '{filepath}'. Hãy kiểm tra lại đường dẫn!")

    # Vì cậu đã xóa cột nhãn, TOÀN BỘ dữ liệu trong bảng đều là đặc trưng (Features)
    X_raw = df.values 

    # Ép kiểu ma trận X về float để an toàn tuyệt đối cho Đại số tuyến tính
    try:
        X_raw = X_raw.astype(float)
    except ValueError:
        raise ValueError("LỖI CẤU TRÚC: File CSV của cậu vẫn còn sót chữ (string) ở đâu đó. Hãy mở file bằng Excel và kiểm tra lại toàn bộ các ô!")

    return X_raw
# LUỒNG CHẠY CHÍNH
# =====================================================================
# if __name__ == "__main__":
#     try:
#         # Tạo 50 mẫu dữ liệu
#         np.random.seed(42)
#         x = np.linspace(0, 100, 50)
#         y = 2 * x + np.random.normal(0, 5, 50) # y phụ thuộc tuyến tính vào x
#         z = np.random.normal(0, 10, 50)        # z là nhiễu ngẫu nhiên

#         raw_matrix = np.column_stack((x, y, z))
        
#         # --- THỰC THI PHẦN 1 (KHÁNH) ---
#         print("Đang chạy Phần 1: Tiền xử lý...")
#         scaled_data, valid_mask = data_standardization(raw_matrix)
#         cov_matrix = build_covariance_matrix(scaled_data)
        
#         # --- THỰC THI PHẦN 2 (TUẤN) ---
#         print("Đang chạy Phần 2: Tìm Trị riêng & Vector riêng...")
#         sorted_eigenvalues, sorted_eigenvectors = calculate_eigen_components(cov_matrix)
        
#         # --- THỰC THI PHẦN 3 (CỦA BẠN) ---
#         print("Đang chạy Phần 3: Chiếu dữ liệu và vẽ biểu đồ/bảng...")
#         num_components = 2 
#         W = sorted_eigenvectors[:, :num_components] 
#         projection = scaled_data @ W 
        
#         # Gọi hàm: Truyền scaled_data vào để nó vẽ thành bảng số liệu bên trái
#         plot_pca_results(scaled_data, projection, sorted_eigenvalues)
        
#         print("\n[THÀNH CÔNG] Thuật toán PCA đã hoàn tất!")

#     except Exception as e:
#         print(f"\n[LỖI HỆ THỐNG]: {e}")