import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- IMPORT MODULE CỦA CÁC THÀNH VIÊN ---
from pcanew import data_standardization, build_covariance_matrix
from eigenvectors_eigenvalues import calculate_eigen_components
from draw import plot_pca_results, load_csv_data



def main():
    # 1. ĐỔI TÊN FILE CSV CỦA CẬU VÀO ĐÂY
    csv_file_path = "Iris.csv" 
    
    print(f"[1] Đang nạp dữ liệu thô từ '{csv_file_path}'...")
    raw_data = load_csv_data(csv_file_path)
    print(f"    -> Đã nạp thành công: {raw_data.shape[0]} mẫu (hàng), {raw_data.shape[1]} đặc trưng (chiều).")
    
    print("[2] Đang chuẩn hóa và xử lý nhiễu (Winsorization + Z-Score)...")
    # Cậu có thể tinh chỉnh tukey_multiplier tùy thuộc độ nhiễu của data thực tế
    standardized_matrix = data_standardization(raw_data, tukey_multiplier=1.5)
    
    print("[3] Đang tính toán ma trận hiệp phương sai...")
    cov_matrix = build_covariance_matrix(standardized_matrix)
    
    print("[4] Đang phân rã Trị riêng và Vector riêng...")
    eigenvalues, eigenvectors = calculate_eigen_components(cov_matrix)
    
    print("[5] Đang chạy Phần 3: Chiếu dữ liệu và vẽ biểu đồ/bảng...")
    # Truyền raw_data cho bảng bên trái, standardized_matrix cho đồ thị bên phải
    plot_pca_results(raw_data, standardized_matrix, eigenvectors, eigenvalues, num_components=2)
        
    print("\n[THÀNH CÔNG] Thuật toán PCA đã hoàn tất trên dữ liệu thực!")

if __name__ == "__main__":
    main()