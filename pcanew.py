import numpy as np

def data_standardization(raw_data_matrix, *, tukey_multiplier=1.5):
    
    # LAYER 1: TUKEY'S FENCES
    q75_vector = np.nanpercentile(raw_data_matrix, 75, axis=0)
    q25_vector = np.nanpercentile(raw_data_matrix, 25, axis=0)
    iqr_vector = q75_vector - q25_vector
    
    lower_fence = q25_vector - (tukey_multiplier * iqr_vector)
    upper_fence = q75_vector + (tukey_multiplier * iqr_vector)

    # LAYER 2: WINSORIZATION
    # Directly overwriting the source reference
    winsorized_matrix = np.clip(raw_data_matrix, lower_fence, upper_fence)

    # LAYER 3: Z-SCORE STANDARDIZATION
    mean_vector = np.nanmean(winsorized_matrix, axis=0)
    std_vector = np.nanstd(winsorized_matrix, axis=0)
    
    std_vector = np.where(std_vector < 1e-8, 1.0, std_vector)
    standardized_matrix = (winsorized_matrix - mean_vector) / std_vector

    # LAYER 4: ZERO IMPUTATION
    return np.nan_to_num(standardized_matrix, nan=0.0)


def build_covariance_matrix(standardized_matrix):
    
    N = standardized_matrix.shape[0]
    # S = (X^T @ X) / (N - 1)
    covariance_matrix = (standardized_matrix.T @ standardized_matrix) / (N - 1)
    return covariance_matrix