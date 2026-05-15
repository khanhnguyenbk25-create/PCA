import numpy as np

def calculate_eigen_components(cov_matrix):
    
    # 1. Tính trị riêng (eigenvalues) và vector riêng (eigenvectors)
    # Hàm này giải phương trình đặc trưng để tìm các trục của dữ liệu
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # 2. Sắp xếp trị riêng theo thứ tự giảm dần
    # PCA bắt buộc phải chọn các trục có phương sai (trị riêng) lớn nhất trước
    sorted_indices = np.argsort(eigenvalues)[::-1]
    
    # 3. Cập nhật lại mảng trị riêng và ma trận vector riêng theo thứ tự đã xếp
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    return sorted_eigenvalues, sorted_eigenvectors