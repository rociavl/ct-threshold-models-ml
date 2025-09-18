import slicer
import numpy as np
import vtk
from vtk.util import numpy_support
import cv2
from skimage import exposure, filters, morphology
from skimage.exposure import rescale_intensity
import pywt
import pywt.data
import os
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scipy import ndimage
import sklearn
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from skimage.morphology import remove_small_objects
import scipy.ndimage as ndi
from skimage.segmentation import watershed
from skimage import segmentation, measure, feature, draw
from skimage.filters import sobel
from scipy.ndimage import distance_transform_edt
from skimage.restoration import denoise_nl_means
from scipy.ndimage import watershed_ift, gaussian_filter
from skimage.feature import peak_local_max
from skimage.segmentation import active_contour
from skimage.draw import ellipse
from skimage import img_as_float
from skimage.filters import gaussian
from skimage.feature import canny
from skimage.transform import rescale
from skimage.filters import gaussian, laplace
from skimage.transform import rescale
from skimage.measure import regionprops, label
from skimage.morphology import remove_small_objects
import pandas as pd
from skimage.measure import regionprops
from skimage.filters import frangi
from scipy.ndimage import median_filter
import vtk.util.numpy_support as ns
from skimage.morphology import disk
from skimage.filters import median
from skimage import morphology, measure, segmentation, filters, feature
from skimage import morphology as skmorph
from skimage.color import rgb2gray
from scipy import ndimage as ndi
from skimage.filters import gaussian
from skimage.restoration import denoise_wavelet
from skimage.exposure import adjust_gamma
from scipy.ndimage import distance_transform_edt, label
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import logging
from skimage import restoration
from enhance_ctp import (gamma_correction, sharpen_high_pass, log_transform_slices, 
                        wavelet_denoise, wavelet_nlm_denoise, morphological_operation, 
                        apply_clahe, morph_operations)
import joblib
from sklearn.preprocessing import StandardScaler
import time




def shannon_entropy(image):
    """Calculate Shannon entropy of an image."""
    import numpy as np
    # Convert to probabilities by calculating histogram
    hist, _ = np.histogram(image, bins=256, density=True)
    # Remove zeros to avoid log(0)
    hist = hist[hist > 0]
    # Calculate entropy
    return -np.sum(hist * np.log2(hist))

def extract_advanced_features(volume_array, hist=None, bin_centers=None):
    import numpy as np
    from scipy import stats
    
    features = {}
    features['min'] = np.min(volume_array)
    features['max'] = np.max(volume_array)
    features['mean'] = np.mean(volume_array)
    features['median'] = np.median(volume_array)
    features['std'] = np.std(volume_array)
    features['p25'] = np.percentile(volume_array, 25)
    features['p75'] = np.percentile(volume_array, 75)
    features['p95'] = np.percentile(volume_array, 95)
    features['p99'] = np.percentile(volume_array, 99)
    
    # Compute histogram if not provided
    if hist is None or bin_centers is None:
        hist, bin_edges = np.histogram(volume_array.flatten(), bins=256)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Handle zero-peak special case for small dot segmentation
    zero_idx = np.argmin(np.abs(bin_centers))  # Index closest to zero
    zero_peak_height = hist[zero_idx]
    features['zero_peak_height'] = zero_peak_height
    features['zero_peak_ratio'] = zero_peak_height / np.sum(hist) if np.sum(hist) > 0 else 0
    
    # Add very high percentiles that might better capture small bright dots
    features['p99.5'] = np.percentile(volume_array, 99.5)
    features['p99.9'] = np.percentile(volume_array, 99.9)
    features['p99.99'] = np.percentile(volume_array, 99.99)
    
    # Calculate skewness and kurtosis for the distribution
    features['skewness'] = stats.skew(volume_array.flatten())
    features['kurtosis'] = stats.kurtosis(volume_array.flatten())
    
    # Calculate non-zero statistics (ignoring background)
    non_zero_values = volume_array[volume_array > 0]
    if len(non_zero_values) > 0:
        features['non_zero_min'] = np.min(non_zero_values)
        features['non_zero_mean'] = np.mean(non_zero_values)
        features['non_zero_median'] = np.median(non_zero_values)
        features['non_zero_std'] = np.std(non_zero_values)
        features['non_zero_count'] = len(non_zero_values)
        features['non_zero_ratio'] = len(non_zero_values) / volume_array.size
        # Calculate skewness and kurtosis for non-zero values
        if len(non_zero_values) > 3:  # Need at least 3 points for skewness calculation
            features['non_zero_skewness'] = stats.skew(non_zero_values)
            features['non_zero_kurtosis'] = stats.kurtosis(non_zero_values)
        else:
            features['non_zero_skewness'] = 0
            features['non_zero_kurtosis'] = 0
    else:
        features['non_zero_min'] = 0
        features['non_zero_mean'] = 0
        features['non_zero_median'] = 0
        features['non_zero_std'] = 0
        features['non_zero_count'] = 0
        features['non_zero_ratio'] = 0
        features['non_zero_skewness'] = 0
        features['non_zero_kurtosis'] = 0
    
    # Find peaks (ignoring the zero peak if it's dominant)
    peaks = []
    for i in range(1, len(hist)-1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
            # Add this peak only if it's not the zero peak
            if abs(bin_centers[i]) > 0.01:  # Small tolerance to avoid numerical issues
                peaks.append((bin_centers[i], hist[i]))
    
    # Sort peaks by height (descending)
    peaks.sort(key=lambda x: x[1], reverse=True)
    
    # Extract info about top non-zero peaks
    if peaks:
        features['non_zero_peak1_value'] = peaks[0][0]
        features['non_zero_peak1_height'] = peaks[0][1]
        
        if len(peaks) > 1:
            features['non_zero_peak2_value'] = peaks[1][0]
            features['non_zero_peak2_height'] = peaks[1][1]
            features['non_zero_peak_distance'] = abs(features['non_zero_peak1_value'] - features['non_zero_peak2_value'])
        else:
            features['non_zero_peak2_value'] = features['non_zero_peak1_value']
            features['non_zero_peak2_height'] = 0
            features['non_zero_peak_distance'] = 0
    else:
        # No non-zero peaks found
        features['non_zero_peak1_value'] = features['mean']
        features['non_zero_peak1_height'] = 0
        features['non_zero_peak2_value'] = features['mean']
        features['non_zero_peak2_height'] = 0
        features['non_zero_peak_distance'] = 0
    
    # Add specialized dot detection features
    # Contrast ratios that might help identify dots
    features['contrast_ratio'] = features['max'] / features['mean'] if features['mean'] > 0 else 0
    features['p99_mean_ratio'] = features['p99'] / features['mean'] if features['mean'] > 0 else 0
    
    # Entropy
    features['entropy'] = shannon_entropy(volume_array)
    
    # Additional engineered features for model prediction
    features['range'] = features['max'] - features['min']
    features['iqr'] = features['p75'] - features['p25']
    features['iqr_to_std_ratio'] = features['iqr'] / (features['std'] + 1e-5)
    features['contrast_per_iqr'] = features['contrast_ratio'] / (features['iqr'] + 1e-5)
    features['range_to_iqr'] = features['range'] / (features['iqr'] + 1e-5)
    features['skewness_squared'] = features['skewness'] ** 2
    features['kurtosis_log'] = np.log1p(features['kurtosis'] - np.min(features['kurtosis']))
    
    return features

def predict_threshold_for_volume(volume_array, model_path):
    """Predict threshold for a given volume array using the trained model, ensuring it's within min/max range."""
    # Extract features
    features = extract_advanced_features(volume_array)
    
    # Get min and max from features
    vol_min = features['min']
    vol_max = features['max']
    
    # Load the model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model_data = joblib.load(model_path)
    model = model_data['model']
    feature_names = model_data.get('feature_names', [])
    
    # Convert features to DataFrame with correct feature order
    import pandas as pd
    feature_df = pd.DataFrame([features])
    
    # Ensure we have all expected features
    for feat in feature_names:
        if feat not in feature_df.columns:
            feature_df[feat] = 0  # Add missing features with default value
    
    # Reorder columns to match training data
    feature_df = feature_df[feature_names]
    
    # Predict threshold
    threshold = model.predict(feature_df)[0]
    
    # Ensure threshold is within volume's min/max range
    if threshold < vol_min or threshold > vol_max:
        print(f"Predicted threshold {threshold} outside volume range [{vol_min}, {vol_max}]. Using 99.97th percentile instead.")
        threshold = np.percentile(volume_array, 99.97)
    
    return threshold

def process_original_ctp(enhanced_volumes, volume_array, threshold_tracker, model_path=None):
    """Process the original CTP volume with basic enhancement techniques"""
    print("Applying Original CTP Processing approach...")
    
    # Save original volume
    enhanced_volumes['OG_volume_array'] = volume_array
    print(f"OG_volume_array shape: {enhanced_volumes['OG_volume_array'].shape}")
    
    # Predict threshold for original volume
    if model_path:
        threshold = predict_threshold_for_volume(enhanced_volumes['OG_volume_array'], model_path)
    else:
        threshold = 1742  # Fallback to fixed threshold if no model
    enhanced_volumes['DESCARGAR_OG_volume_array_1136'] = np.uint8(enhanced_volumes['OG_volume_array'] > threshold)
    threshold_tracker['OG_volume_array'] = threshold
    
    # Gaussian filter on original volume
    enhanced_volumes['OG_gaussian_volume_og'] = gaussian(enhanced_volumes['OG_volume_array'], sigma=0.3)
    # Predict threshold for gaussian filtered volume
    if model_path:
        threshold = predict_threshold_for_volume(enhanced_volumes['OG_gaussian_volume_og'], model_path)
    else:
        threshold = 1742  # Fallback to fixed threshold if no model
    enhanced_volumes['DESCARGAR_OG_gaussian_volume_og_1136'] = np.uint8(enhanced_volumes['OG_gaussian_volume_og'] > threshold)
    threshold_tracker['OG_gaussian_volume_og'] = threshold
    
    # Gamma correction on gaussian filtered volume
    enhanced_volumes['OG_gamma_volume_og'] = gamma_correction(enhanced_volumes['OG_gaussian_volume_og'], gamma=3)
    # Predict threshold for gamma corrected volume
    if model_path:
        threshold = predict_threshold_for_volume(enhanced_volumes['OG_gamma_volume_og'], model_path)
    else:
        threshold = 40  # Fallback to fixed threshold if no model
    enhanced_volumes['DESCARGAR_OG_gamma_volume_og_40'] = np.uint8(enhanced_volumes['OG_gamma_volume_og'] > threshold)
    threshold_tracker['OG_gamma_volume_og'] = threshold
    
    # Sharpen gamma corrected volume
    enhanced_volumes['OG_sharpened'] = sharpen_high_pass(enhanced_volumes['OG_gamma_volume_og'], strenght=0.8)
    # Predict threshold for sharpened volume
    if model_path:
        threshold = predict_threshold_for_volume(enhanced_volumes['OG_sharpened'], model_path)
    else:
        threshold = 74  # Fallback to fixed threshold if no model
    enhanced_volumes['DESCARGAR_OG_sharpened_74'] = np.uint8(enhanced_volumes['OG_sharpened'] > threshold)
    threshold_tracker['OG_sharpened'] = threshold
    
    return enhanced_volumes, threshold_tracker

def process_roi_gamma_mask(enhanced_volumes, final_roi, volume_array, threshold_tracker, model_path=None):
    """Process the volume using ROI and gamma mask"""
    print("Applying ROI with Gamma Mask approach...")
    
    # Apply ROI mask to gamma corrected volume
    if 'OG_gamma_volume_og' not in enhanced_volumes:
        # First apply gaussian filter
        gaussian_volume = gaussian(volume_array, sigma=0.3)
        # Then apply gamma correction
        gamma_volume = gamma_correction(gaussian_volume, gamma=3)
        enhanced_volumes['OG_gamma_volume_og'] = gamma_volume
    
    # Combine ROI mask with gamma corrected volume
    enhanced_volumes['PRUEBA_roi_plus_gamma_mask'] = (final_roi > 0) * enhanced_volumes['OG_gamma_volume_og']
    # Predict threshold for ROI plus gamma mask
    if model_path:
        threshold = predict_threshold_for_volume(enhanced_volumes['PRUEBA_roi_plus_gamma_mask'], model_path)
    else:
        threshold = 43  # Fallback to fixed threshold if no model
    enhanced_volumes['DESCARGAR_PRUEBA_roi_plus_gamma_mask_40'] = np.uint8(enhanced_volumes['PRUEBA_roi_plus_gamma_mask'] > threshold)
    threshold_tracker['PRUEBA_roi_plus_gamma_mask'] = threshold
    
    # Apply CLAHE to ROI plus gamma mask
    enhanced_volumes['PRUEBA_roi_plus_gamma_mask_clahe'] = apply_clahe(enhanced_volumes['PRUEBA_roi_plus_gamma_mask'])
    # Predict threshold for CLAHE result
    if model_path:
        threshold = predict_threshold_for_volume(enhanced_volumes['PRUEBA_roi_plus_gamma_mask_clahe'], model_path)
    else:
        threshold = 57  # Fallback to fixed threshold if no model
    enhanced_volumes['DESCARGAR_PRUEBA_THRESHOLD_CLAHE_57'] = np.uint8(enhanced_volumes['PRUEBA_roi_plus_gamma_mask_clahe'] > threshold)
    threshold_tracker['PRUEBA_roi_plus_gamma_mask_clahe'] = threshold
    
    # Apply wavelet non-local means denoising
    enhanced_volumes['PRUEBA_WAVELET_NL'] = wavelet_nlm_denoise(enhanced_volumes['PRUEBA_roi_plus_gamma_mask'], wavelet='db1')
    # Predict threshold for wavelet NL denoised volume
    if model_path:
        threshold = predict_threshold_for_volume(enhanced_volumes['PRUEBA_WAVELET_NL'], model_path)
    else:
        threshold = 40.4550  # Fallback to fixed threshold if no model
    enhanced_volumes['DESCARGAR_PRUEBA_WAVELET_NL_40.4550'] = np.uint8(enhanced_volumes['PRUEBA_WAVELET_NL'] > threshold)
    threshold_tracker['PRUEBA_WAVELET_NL'] = threshold
    
    return enhanced_volumes, threshold_tracker

def process_roi_only(enhanced_volumes, roi_volume, final_roi, threshold_tracker, model_path=None):
    """Process using only the ROI volume"""
    print("Applying ROI Only approach...")
    
    # Save ROI volume
    enhanced_volumes['roi_volume'] = roi_volume
    
    # Apply wavelet denoising
    enhanced_volumes['wavelet_only_roi'] = wavelet_denoise(enhanced_volumes['roi_volume'], wavelet='db1')
    # Predict threshold for wavelet denoised volume
    if model_path:
        threshold = predict_threshold_for_volume(enhanced_volumes['wavelet_only_roi'], model_path)
    else:
        threshold = 1000  # Fallback to fixed threshold if no model
    enhanced_volumes['DESCARGAR_WAVELET_ROI_1000'] = np.uint8(enhanced_volumes['wavelet_only_roi'] > threshold)
    threshold_tracker['wavelet_only_roi'] = threshold
    
    # Apply gamma correction to wavelet denoised volume
    enhanced_volumes['gamma_only_roi'] = gamma_correction(enhanced_volumes['wavelet_only_roi'], gamma=0.8)
    # Predict threshold for gamma corrected volume
    if model_path:
        threshold = predict_threshold_for_volume(enhanced_volumes['gamma_only_roi'], model_path)
    else:
        threshold = 160  # Fallback to fixed threshold if no model
    enhanced_volumes['DESCARGAR_GAMMA_ONLY_ROI_160'] = np.uint8(enhanced_volumes['gamma_only_roi'] > threshold)
    threshold_tracker['gamma_only_roi'] = threshold
    

    enhanced_volumes['DESCARGAR_roi_volume_features'] = roi_volume
    # Predict threshold for ROI volume
    if model_path:
        threshold = predict_threshold_for_volume(enhanced_volumes['roi_volume'], model_path)
    else:
        threshold = 980  # Fallback to fixed threshold if no model
    enhanced_volumes['DESCARGAR_Threshold_roi_volume_980'] = np.uint8(enhanced_volumes['roi_volume'] > threshold)
    threshold_tracker['roi_volume'] = threshold
    
    return enhanced_volumes, threshold_tracker

def process_roi_plus_gamma_after(enhanced_volumes, final_roi, threshold_tracker, model_path=None):
    """Process using ROI plus gamma correction after"""
    print("Applying ROI plus Gamma after approach...")
    
    if 'PRUEBA_roi_plus_gamma_mask' not in enhanced_volumes:
        print("Warning: PRUEBA_roi_plus_gamma_mask not found. Skipping this approach.")
        return enhanced_volumes, threshold_tracker
    
    # Apply gaussian filter to ROI plus gamma mask
    enhanced_volumes['2_gaussian_volume_roi'] = gaussian(enhanced_volumes['PRUEBA_roi_plus_gamma_mask'], sigma=0.3)
    # Predict threshold for gaussian volume
    if model_path:
        threshold = predict_threshold_for_volume(enhanced_volumes['2_gaussian_volume_roi'], model_path)
    else:
        threshold = 0.19  # Fallback to fixed threshold if no model
    enhanced_volumes['DESCARGAR_2_gaussian_volume_roi_0.19'] = np.uint8(enhanced_volumes['2_gaussian_volume_roi'] > threshold)
    threshold_tracker['2_gaussian_volume_roi'] = threshold
    
    # Apply gamma correction
    enhanced_volumes['2_gamma_correction'] = gamma_correction(enhanced_volumes['2_gaussian_volume_roi'], gamma=0.8)
    # Predict threshold for gamma corrected volume
    if model_path:
        threshold = predict_threshold_for_volume(enhanced_volumes['2_gamma_correction'], model_path)
    else:
        threshold = 75  # Fallback to fixed threshold if no model
    enhanced_volumes['DESCARGAR_2_gamma_correction_75'] = np.uint8(enhanced_volumes['2_gamma_correction'] > threshold)
    threshold_tracker['2_gamma_correction'] = threshold
    
    # Apply top-hat transformation
    kernel_2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    tophat_2 = cv2.morphologyEx(enhanced_volumes['2_gaussian_volume_roi'], cv2.MORPH_TOPHAT, kernel_2)
    enhanced_volumes['2_tophat'] = cv2.addWeighted(enhanced_volumes['2_gaussian_volume_roi'], 1, tophat_2, 2, 0)
    # Predict threshold for top-hat result
    if model_path:
        threshold = predict_threshold_for_volume(enhanced_volumes['2_tophat'], model_path)
    else:
        threshold = 0.17  # Fallback to fixed threshold if no model
    enhanced_volumes['DESCARGAR_2_tophat_0.17'] = np.uint8(enhanced_volumes['2_tophat'] > threshold)
    threshold_tracker['2_tophat'] = threshold
    
    # Sharpen gamma corrected volume
    enhanced_volumes['2_sharpened'] = sharpen_high_pass(enhanced_volumes['2_gamma_correction'], strenght=0.8)
    # Predict threshold for sharpened volume
    if model_path:
        threshold = predict_threshold_for_volume(enhanced_volumes['2_sharpened'], model_path)
    else:
        threshold = 75  # Fallback to fixed threshold if no model
    enhanced_volumes['DESCARGAR_2_sharpened_75'] = np.uint8(enhanced_volumes['2_sharpened'] > threshold)
    threshold_tracker['2_sharpened'] = threshold
    
    # Apply LOG transform
    enhanced_volumes['2_LOG'] = log_transform_slices(enhanced_volumes['2_tophat'], c=3)
    # Predict threshold for LOG transform
    if model_path:
        threshold = predict_threshold_for_volume(enhanced_volumes['2_LOG'], model_path)
    else:
        threshold = 75  # Fallback to fixed threshold if no model
    enhanced_volumes['DESCARGAR_2_LOG_75'] = np.uint8(enhanced_volumes['2_LOG'] > threshold)
    threshold_tracker['2_LOG'] = threshold
    
    # Apply wavelet denoising
    enhanced_volumes['2_wavelet_roi'] = wavelet_denoise(enhanced_volumes['2_LOG'], wavelet='db4')
    # Predict threshold for wavelet result
    if model_path:
        threshold = predict_threshold_for_volume(enhanced_volumes['2_wavelet_roi'], model_path)
    else:
        threshold = 74  # Fallback to fixed threshold if no model
    enhanced_volumes['DESCARGAR_2_wavelet_roi_74'] = np.uint8(enhanced_volumes['2_wavelet_roi'] > threshold)
    threshold_tracker['2_wavelet_roi'] = threshold
    
    # Apply erosion
    enhanced_volumes['2_erode'] = morphological_operation(enhanced_volumes['2_sharpened'], operation='erode', kernel_size=1)
    # Predict threshold for eroded volume
    if model_path:
        threshold = predict_threshold_for_volume(enhanced_volumes['2_erode'], model_path)
    else:
        threshold = 74  # Fallback to fixed threshold if no model
    enhanced_volumes['DESCARGAR_2_74'] = np.uint8(enhanced_volumes['2_erode'] > threshold)
    threshold_tracker['2_erode'] = threshold
    
    # Apply gaussian filter
    enhanced_volumes['2_gaussian_2'] = gaussian(enhanced_volumes['2_erode'], sigma=0.2)
    # Predict threshold for gaussian filtered volume
    if model_path:
        threshold = predict_threshold_for_volume(enhanced_volumes['2_gaussian_2'], model_path)
    else:
        threshold = 0.35  # Fallback to fixed threshold if no model
    enhanced_volumes['DESCARGAR_2_gaussian_2_0.35'] = np.uint8(enhanced_volumes['2_gaussian_2'] > threshold)
    threshold_tracker['2_gaussian_2'] = threshold
    
    # Sharpen gaussian filtered volume
    enhanced_volumes['2_sharpening_2_trial'] = sharpen_high_pass(enhanced_volumes['2_gaussian_2'], strenght=0.8)
    # Predict threshold for sharpened volume
    if model_path:
        threshold = predict_threshold_for_volume(enhanced_volumes['2_sharpening_2_trial'], model_path)
    else:
        threshold = 0.35  # Fallback to fixed threshold if no model
    enhanced_volumes['DESCARGAR_2_sharpening_2_trial_0.35'] = np.uint8(enhanced_volumes['2_sharpening_2_trial'] > threshold)
    threshold_tracker['2_sharpening_2_trial'] = threshold
    
    return enhanced_volumes, threshold_tracker

def process_wavelet_roi(enhanced_volumes, roi_volume, threshold_tracker, model_path=None):
    """Process using wavelet denoising on ROI volume"""
    print("Applying Wavelet ROI approach...")
    
    # Apply non-local means denoising with wavelet
    enhanced_volumes['NUEVO_NLMEANS'] = wavelet_nlm_denoise(roi_volume)
    # Predict threshold for NL means denoised volume
    if model_path:
        threshold = predict_threshold_for_volume(enhanced_volumes['NUEVO_NLMEANS'], model_path)
    else:
        threshold = 1215  # Fallback to fixed threshold if no model
    enhanced_volumes['DESCARGAR_NUEVO_NLMEANS_1215'] = np.uint8(enhanced_volumes['NUEVO_NLMEANS'] > threshold)
    threshold_tracker['NUEVO_NLMEANS'] = threshold
    return enhanced_volumes, threshold_tracker

def process_original_idea(enhanced_volumes, threshold_tracker, model_path=None):
    """Process using the original idea approach"""
    print("Applying Original Idea approach...")
    
    if 'PRUEBA_roi_plus_gamma_mask' not in enhanced_volumes:
        print("Warning: PRUEBA_roi_plus_gamma_mask not found. Skipping this approach.")
        return enhanced_volumes, threshold_tracker
    
    # Apply gaussian filter
    enhanced_volumes['ORGINAL_IDEA_gaussian'] = gaussian(enhanced_volumes['PRUEBA_roi_plus_gamma_mask'], sigma=0.3)
    # Predict threshold for gaussian filtered volume
    if model_path:
        threshold = predict_threshold_for_volume(enhanced_volumes['ORGINAL_IDEA_gaussian'], model_path)
    else:
        threshold = 0.148  # Fallback to fixed threshold if no model
    enhanced_volumes['DESCARGAR_ORGINAL_IDEA_gaussian_0.148'] = np.uint8(enhanced_volumes['ORGINAL_IDEA_gaussian'] > threshold)
    threshold_tracker['ORGINAL_IDEA_gaussian'] = threshold
    
    # Apply gamma correction
    enhanced_volumes['ORGINAL_IDEA_gamma_correction'] = gamma_correction(enhanced_volumes['ORGINAL_IDEA_gaussian'], gamma=2)
    # Predict threshold for gamma corrected volume
    if model_path:
        threshold = predict_threshold_for_volume(enhanced_volumes['ORGINAL_IDEA_gamma_correction'], model_path)
    else:
        threshold = 10  # Fallback to fixed threshold if no model
    enhanced_volumes['DESCARGAR_ORGINAL_IDEA_gamma_correction_10'] = np.uint8(enhanced_volumes['ORGINAL_IDEA_gamma_correction'] > threshold)
    threshold_tracker['ORGINAL_IDEA_gamma_correction'] = threshold
    
    # Sharpen gamma corrected volume
    enhanced_volumes['ORGINAL_IDEA_sharpened'] = sharpen_high_pass(enhanced_volumes['ORGINAL_IDEA_gamma_correction'], strenght=0.8)
    # Predict threshold for sharpened volume
    if model_path:
        threshold = predict_threshold_for_volume(enhanced_volumes['ORGINAL_IDEA_sharpened'], model_path)
    else:
        threshold = 10  # Fallback to fixed threshold if no model
    enhanced_volumes['DESCARGAR_ORGINAL_IDEA_sharpened_10'] = np.uint8(enhanced_volumes['ORGINAL_IDEA_sharpened'] > threshold)
    threshold_tracker['ORGINAL_IDEA_sharpened'] = threshold
    
    # Apply wavelet denoising
    enhanced_volumes['ORIGINAL_IDEA_wavelet'] = wavelet_denoise(enhanced_volumes['ORGINAL_IDEA_sharpened'])
    # Predict threshold for wavelet result
    if model_path:
        threshold = predict_threshold_for_volume(enhanced_volumes['ORIGINAL_IDEA_wavelet'], model_path)
    else:
        threshold = 10  # Fallback to fixed threshold if no model
    enhanced_volumes['DESCARGAR_ORIGINAL_IDEA_wavelet_10'] = np.uint8(enhanced_volumes['ORIGINAL_IDEA_wavelet'] > threshold)
    threshold_tracker['ORIGINAL_IDEA_wavelet'] = threshold
    
    # Apply gaussian filter
    enhanced_volumes['ORGINAL_IDEA_gaussian_2'] = gaussian(enhanced_volumes['ORGINAL_IDEA_sharpened'], sigma=0.4)
    # Predict threshold for gaussian filtered volume
    if model_path:
        threshold = predict_threshold_for_volume(enhanced_volumes['ORGINAL_IDEA_gaussian_2'], model_path)
    else:
        threshold = 0.06  # Fallback to fixed threshold if no model
    enhanced_volumes['DESCARGAR_ORGINAL_IDEA_gaussian_2_0.06'] = np.uint8(enhanced_volumes['ORGINAL_IDEA_gaussian_2'] > threshold)
    threshold_tracker['ORGINAL_IDEA_gaussian_2'] = threshold
    
    # Apply gamma correction
    enhanced_volumes['ORIGINAL_IDEA_GAMMA_2'] = gamma_correction(enhanced_volumes['ORGINAL_IDEA_gaussian_2'], gamma=1.4)
    # Predict threshold for gamma corrected volume
    if model_path:
        threshold = predict_threshold_for_volume(enhanced_volumes['ORIGINAL_IDEA_GAMMA_2'], model_path)
    else:
        threshold = 8  # Fallback to fixed threshold if no model
    enhanced_volumes['DESCARGAR_ORIGINAL_IDEA_GAMMA_2_8'] = np.uint8(enhanced_volumes['ORIGINAL_IDEA_GAMMA_2'] > threshold)
    threshold_tracker['ORIGINAL_IDEA_GAMMA_2'] = threshold
    
    # Apply top-hat transformation
    kernel_size_og = (1, 1)
    kernel_og = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size_og)
    tophat_og = cv2.morphologyEx(enhanced_volumes['ORGINAL_IDEA_gaussian_2'], cv2.MORPH_TOPHAT, kernel_og)
    enhanced_volumes['OG_tophat_1'] = cv2.addWeighted(enhanced_volumes['ORGINAL_IDEA_gaussian_2'], 1, tophat_og, 2, 0)
    # Predict threshold for top-hat result
    if model_path:
        threshold = predict_threshold_for_volume(enhanced_volumes['OG_tophat_1'], model_path)
    else:
        threshold = 0.05  # Fallback to fixed threshold if no model
    enhanced_volumes['DESCARGAR_OG_tophat_1_0.05'] = np.uint8(enhanced_volumes['OG_tophat_1'] > threshold)
    threshold_tracker['OG_tophat_1'] = threshold
    
    return enhanced_volumes, threshold_tracker

def process_first_try(enhanced_volumes, roi_volume, threshold_tracker, model_path=None):
    """Process using the first try approach"""
    print("Applying First Try approach...")
    
    # Apply gaussian filter
    enhanced_volumes['FT_gaussian'] = gaussian(roi_volume, sigma=0.3)
    # Predict threshold for gaussian filtered volume
    if model_path:
        threshold = predict_threshold_for_volume(enhanced_volumes['FT_gaussian'], model_path)
    else:
        threshold = 1209  # Fallback to fixed threshold if no model
    enhanced_volumes['DESCARGAR_FT_gaussian_1209'] = np.uint8(enhanced_volumes['FT_gaussian'] > threshold)
    threshold_tracker['FT_gaussian'] = threshold
    
    # Apply top-hat transformation
    kernel_size = (1, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    tophat_ft = cv2.morphologyEx(roi_volume, cv2.MORPH_TOPHAT, kernel)
    enhanced_volumes['FT_tophat_1'] = cv2.addWeighted(roi_volume, 1, tophat_ft, 2, 0)
    # Subtract gaussian from top-hat
    enhanced_volumes['FT_RESTA_TOPHAT_GAUSSIAN'] = enhanced_volumes['FT_tophat_1'] - gaussian(roi_volume, sigma=0.8)
    # Predict threshold for subtracted volume
    if model_path:
        threshold = predict_threshold_for_volume(enhanced_volumes['FT_RESTA_TOPHAT_GAUSSIAN'], model_path)
    else:
        threshold = 419  # Fallback to fixed threshold if no model
    enhanced_volumes['DESCARGAR_FT_RESTA_TOPHAT_GAUSSIAN_419'] = np.uint8(enhanced_volumes['FT_RESTA_TOPHAT_GAUSSIAN'] > threshold)
    threshold_tracker['FT_RESTA_TOPHAT_GAUSSIAN'] = threshold
    
    # Apply gamma correction
    enhanced_volumes['FT_gamma_correction'] = gamma_correction(enhanced_volumes['FT_gaussian'], gamma=5)
    # Predict threshold for gamma corrected volume
    if model_path:
        threshold = predict_threshold_for_volume(enhanced_volumes['FT_gamma_correction'], model_path)
    else:
        threshold = 20  # Fallback to fixed threshold if no model
    enhanced_volumes['DESCARGAR_FT_gamma_correction_20'] = np.uint8(enhanced_volumes['FT_gamma_correction'] > threshold)
    threshold_tracker['FT_gamma_correction'] = threshold
    
    # Sharpen gamma corrected volume
    enhanced_volumes['FT_sharpened'] = sharpen_high_pass(enhanced_volumes['FT_gamma_correction'], strenght=0.4)
    # Predict threshold for sharpened volume
    if model_path:
        threshold = predict_threshold_for_volume(enhanced_volumes['FT_sharpened'], model_path)
    else:
        threshold = 25  # Fallback to fixed threshold if no model
    enhanced_volumes['DESCARGAR_FT_sharpened_25'] = np.uint8(enhanced_volumes['FT_sharpened'] > threshold)
    threshold_tracker['FT_sharpened'] = threshold
    
    # Apply gaussian filter
    enhanced_volumes['FT_gaussian_2'] = gaussian(enhanced_volumes['FT_sharpened'], sigma=0.4)
    # Predict threshold for gaussian filtered volume
    if model_path:
        threshold = predict_threshold_for_volume(enhanced_volumes['FT_gaussian_2'], model_path)
    else:
        threshold = 0.054  # Fallback to fixed threshold if no model
    enhanced_volumes['DESCARGAR_FT_gaussian_2_0.054'] = np.uint8(enhanced_volumes['FT_gaussian_2'] > threshold)
    threshold_tracker['FT_gaussian_2'] = threshold
    
    # Apply gamma correction
    enhanced_volumes['FT_gamma_2'] = gamma_correction(enhanced_volumes['FT_gaussian_2'], gamma=1.2)
    # Predict threshold for gamma corrected volume
    if model_path:
        threshold = predict_threshold_for_volume(enhanced_volumes['FT_gamma_2'], model_path)
    else:
        threshold = 10  # Fallback to fixed threshold if no model
    enhanced_volumes['DESCARGAR_FT_GAMMA_2_10'] = np.uint8(enhanced_volumes['FT_gamma_2'] > threshold)
    threshold_tracker['FT_gamma_2'] = threshold

    # Apply opening operation
    enhanced_volumes['FT_opening'] = morph_operations(enhanced_volumes['FT_gamma_2'], iterations=2, kernel_shape='cross')
    # Predict threshold for opened volume
    if model_path:
        threshold = predict_threshold_for_volume(enhanced_volumes['FT_opening'], model_path)
    else:
        threshold = 10  # Fallback to fixed threshold if no model
    enhanced_volumes['DESCARGAR_FT_OPENING_10'] = np.uint8(enhanced_volumes['FT_opening'] > threshold)
    threshold_tracker['FT_opening'] = threshold
    
    # Apply closing operation
    enhanced_volumes['FT_closing'] = morph_operations(enhanced_volumes['FT_opening'], operation='close')
    # Predict threshold for closed volume
    if model_path:
        threshold = predict_threshold_for_volume(enhanced_volumes['FT_closing'], model_path)
    else:
        threshold = 17  # Fallback to fixed threshold if no model
    enhanced_volumes['DESCARGAR_FT_CLOSING_17'] = np.uint8(enhanced_volumes['FT_closing'] > threshold)
    threshold_tracker['FT_closing'] = threshold

    # Apply erosion
    enhanced_volumes['FT_erode_2'] = morphological_operation(enhanced_volumes['FT_closing'], operation='erode', kernel_size=1)
    # Predict threshold for eroded volume
    if model_path:
        threshold = predict_threshold_for_volume(enhanced_volumes['FT_erode_2'], model_path)
    else:
        threshold = 15  # Fallback to fixed threshold if no model
    enhanced_volumes['DESCARGAR_FT_ERODE_2_11'] = np.uint8(enhanced_volumes['FT_erode_2'] > threshold)
    threshold_tracker['FT_erode_2'] = threshold
    
    # Apply top-hat transformation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    tophat = cv2.morphologyEx(enhanced_volumes['FT_gaussian_2'], cv2.MORPH_TOPHAT, kernel)
    enhanced_volumes['FT_tophat'] = cv2.addWeighted(enhanced_volumes['FT_gaussian_2'], 1, tophat, 2, 0)
    # Predict threshold for top-hat result
    if model_path:
        threshold = predict_threshold_for_volume(enhanced_volumes['FT_tophat'], model_path)
    else:
        threshold = 0.061  # Fallback to fixed threshold if no model
    enhanced_volumes['DESCARGAR_FT_TOPHAT_0.061'] = np.uint8(enhanced_volumes['FT_tophat'] > threshold)
    threshold_tracker['FT_tophat'] = threshold
    
    # Apply gaussian filter
    enhanced_volumes['FT_gaussian_3'] = gaussian(enhanced_volumes['FT_tophat'], sigma=0.1)
    # Predict threshold for gaussian filtered volume
    if model_path:
        threshold = predict_threshold_for_volume(enhanced_volumes['FT_gaussian_3'], model_path)
    else:
        threshold = 0.059  # Fallback to fixed threshold if no model
    enhanced_volumes['DESCARGAR_FT_gaussian_3_0.059'] = np.uint8(enhanced_volumes['FT_gaussian_3'] > threshold)
    threshold_tracker['FT_gaussian_3'] = threshold
    return enhanced_volumes, threshold_tracker


class CTPEnhancer:
    """
    CTP (CT Perfusion) Enhancer class that provides various image enhancement techniques
    for medical imaging, specifically designed for SEEG (stereotactic EEG) applications.
    """
    
    def __init__(self):
        """Initialize the CTPEnhancer."""
        pass
    
    def enhance_ctp(self, inputVolume, inputROI=None, methods=None, outputDir=None, 
                   collect_histograms=True, train_model=False, model_path=None, 
                   descargar_only=True):
        """
        Enhance CT perfusion images using different image processing approaches.
        
        Parameters:
        -----------
        inputVolume : vtkMRMLScalarVolumeNode
            Input CT perfusion volume
        inputROI : vtkMRMLScalarVolumeNode, optional
            Region of interest mask (brain mask)
        methods : str or list, optional
            Methods to apply, can be 'all' or a list of method names
        outputDir : str, optional
            Directory to save output volumes
        collect_histograms : bool, optional
            Whether to collect histogram data
        train_model : bool, optional
            Whether to train a threshold prediction model
        model_path : str, optional
            Path to trained model for threshold prediction
        descargar_only : bool, optional
            If True, only processes and saves DESCARGAR_ volumes and roi_volume_features
            
        Returns:
        --------
        dict
            Dictionary of enhanced volume nodes
        """
        # Initialize threshold tracker
        threshold_tracker = {}
        
        # Default to 'all' methods
        if methods is None:
            methods = 'all'
        
        # Convert input volume to numpy array
        volume_array = slicer.util.arrayFromVolume(inputVolume)
        if volume_array is None or volume_array.size == 0:
            print("Input volume data is empty or invalid.")
            return None

        # Process ROI if provided
        if inputROI is not None:
            roi_array = slicer.util.arrayFromVolume(inputROI)
            roi_array = np.uint8(roi_array > 0)  # Ensure binary mask (0 or 1)
            print(f"Shape of input volume: {volume_array.shape}")
            print(f"Shape of ROI mask: {roi_array.shape}")
            
            # Process ROI
            print("Filling inside the ROI...")
            filled_roi = ndimage.binary_fill_holes(roi_array)
            print("Applying morphological closing...")
            struct_elem = morphology.ball(10)
            closed_roi = morphology.binary_closing(filled_roi, struct_elem)
            
            if closed_roi.shape != volume_array.shape:
                print("🔄 Shapes don't match. Using spacing/origin-aware resampling...")
                print(f"Volume shape: {volume_array.shape}, ROI shape: {closed_roi.shape}")
                
                try:
                    # Verify inputs exist
                    if not inputROI or not inputVolume:
                        raise ValueError("Input ROI or Volume node is invalid")
                    
                    # Create temporary node for closed ROI (same as your working script)
                    temp_roi_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "temp_roi_for_resampling")
                    temp_roi_node.SetOrigin(inputROI.GetOrigin())
                    temp_roi_node.SetSpacing(inputROI.GetSpacing())
                    ijkToRasMatrix = vtk.vtkMatrix4x4()
                    inputROI.GetIJKToRASMatrix(ijkToRasMatrix)
                    temp_roi_node.SetIJKToRASMatrix(ijkToRasMatrix)
                    slicer.util.updateVolumeFromArray(temp_roi_node, closed_roi)
                    
                    # Create output node (same as your working script)
                    resampled_roi_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "resampled_roi")
                    
                    # Get reference volume IJK to RAS matrix (from your working script)
                    reference_ijk_to_ras_matrix = vtk.vtkMatrix4x4()
                    inputVolume.GetIJKToRASMatrix(reference_ijk_to_ras_matrix)
                    
                    # Resampling parameters (exactly like your working script)
                    parameters = {
                        "inputVolume": temp_roi_node,
                        "referenceVolume": inputVolume,
                        "outputVolume": resampled_roi_node,
                        "interpolationMode": "NearestNeighbor"
                    }
                    
                    # Run resampling (exactly like your working script)
                    result = slicer.cli.runSync(slicer.modules.resamplescalarvectordwivolume, None, parameters)
                    
                    # CRITICAL: Set the matrix (this was missing!)
                    resampled_roi_node.SetIJKToRASMatrix(reference_ijk_to_ras_matrix)
                    
                    # Get resampled array
                    final_roi = slicer.util.arrayFromVolume(resampled_roi_node)
                    if final_roi is None:
                        raise RuntimeError("Failed to get resampled array")
                    
                    final_roi = (final_roi > 0.5).astype(np.uint8)
                    
                    # Verify success
                    if final_roi.shape != volume_array.shape:
                        raise RuntimeError(f"Resampling failed - shapes still don't match: {final_roi.shape} vs {volume_array.shape}")
                    
                    print(f"Resampling successful: {final_roi.shape}")
                    
                    # Clean up temporary nodes
                    slicer.mrmlScene.RemoveNode(temp_roi_node)
                    slicer.mrmlScene.RemoveNode(resampled_roi_node)
                    
                except Exception as e:
                    print(f"Resampling failed: {e}")
                    # Clean up nodes if they exist
                    try:
                        if 'temp_roi_node' in locals():
                            slicer.mrmlScene.RemoveNode(temp_roi_node)
                        if 'resampled_roi_node' in locals():
                            slicer.mrmlScene.RemoveNode(resampled_roi_node)
                    except:
                        pass
                    
                    # Fallback to original ROI (will cause broadcast error, but at least we know why)
                    final_roi = closed_roi
                    print("Using original ROI - expect broadcast error")
                    
            else:
                final_roi = closed_roi
                print("No resizing needed: ROI already has the same shape as volume.")
        else:
            print("No ROI provided. Proceeding without ROI mask.")
            final_roi = np.ones_like(volume_array)
        
        # Apply the ROI mask to the volume
        print(f'Volume shape: {volume_array.shape}, ROI shape: {final_roi.shape}')
        print(f'Volume dtype: {volume_array.dtype}, ROI dtype: {final_roi.dtype}')
        roi_volume = np.multiply(volume_array, final_roi)
        final_roi = final_roi.astype(np.uint8)

        enhanced_volumes = {}
        
        if methods == 'all' or 'original' in methods:
            # ================ APPROACH 1: ORIGINAL CTP PROCESSING ================
            enhanced_volumes, threshold_tracker = process_original_ctp(
                enhanced_volumes, volume_array, threshold_tracker, model_path)
        
        if methods == 'all' or 'roi_gamma' in methods:
            # ================ APPROACH 2: ROI WITH GAMMA MASK ================
            enhanced_volumes, threshold_tracker = process_roi_gamma_mask(
                enhanced_volumes, final_roi, volume_array, threshold_tracker, model_path)
        
        if methods == 'all' or 'roi_only' in methods:
            # ================ APPROACH 3: ROI ONLY PROCESSING ================
            enhanced_volumes, threshold_tracker = process_roi_only(
                enhanced_volumes, roi_volume, final_roi, threshold_tracker, model_path)
            
        if methods == 'all' or 'roi_plus_gamma' in methods:
            # ================ APPROACH 4: ROI PLUS GAMMA AFTER ================
            enhanced_volumes, threshold_tracker = process_roi_plus_gamma_after(
                enhanced_volumes, final_roi, threshold_tracker, model_path)
            
        if methods == 'all' or 'wavelet_roi' in methods:
            # ================ APPROACH 5: WAVELET ON ROI ================
            enhanced_volumes, threshold_tracker = process_wavelet_roi(
                enhanced_volumes, roi_volume, threshold_tracker, model_path)
            
        if methods == 'all' or 'original_idea' in methods:
            # ================ APPROACH 6: ORIGINAL IDEA ================
            enhanced_volumes, threshold_tracker = process_original_idea(
                enhanced_volumes, threshold_tracker, model_path)
            
        if methods == 'all' or 'first_try' in methods:
            # ================ APPROACH 7: FIRST TRY ================
            enhanced_volumes, threshold_tracker = process_first_try(
                enhanced_volumes, roi_volume, threshold_tracker, model_path)
        
        # Save thresholds to a file
        if outputDir is None:
            outputDir = slicer.app.temporaryPath()  
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)
            
        # Save threshold values to a text file
        threshold_file = os.path.join(outputDir, f"thresholds_{inputVolume.GetName()}.txt")
        with open(threshold_file, 'w') as f:
            f.write(f"Thresholds for {inputVolume.GetName()}\n")
            f.write("=" * 50 + "\n\n")
            
            for method, threshold in threshold_tracker.items():
                f.write(f"{method}: {threshold}\n")
        
        print(f"Saved thresholds to: {threshold_file}")

        # Filter volumes if descargar_only is True
        if descargar_only:
            enhanced_volumes = {k: v for k, v in enhanced_volumes.items() if k.startswith('DESCARGAR_')}
            print("Processing only DESCARGAR_ volumes")

        # Process each enhanced volume
        enhancedVolumeNodes = {}
        for method_name, enhanced_image in enhanced_volumes.items():
            enhancedVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
            enhancedVolumeNode.SetName(f"Enhanced_{method_name}_{inputVolume.GetName()}")
            enhancedVolumeNode.SetOrigin(inputVolume.GetOrigin())
            enhancedVolumeNode.SetSpacing(inputVolume.GetSpacing())
            ijkToRasMatrix = vtk.vtkMatrix4x4()
            inputVolume.GetIJKToRASMatrix(ijkToRasMatrix)  
            enhancedVolumeNode.SetIJKToRASMatrix(ijkToRasMatrix) 
            slicer.util.updateVolumeFromArray(enhancedVolumeNode, enhanced_image)
            enhancedVolumeNodes[method_name] = enhancedVolumeNode
            
            output_file = os.path.join(outputDir, f"Filtered_{method_name}_{inputVolume.GetName()}.nrrd")
            slicer.util.saveNode(enhancedVolumeNode, output_file)
            print(f"Saved {method_name} enhancement as: {output_file}")

        # Save files
        # Save files
        for method_name, volume_node in enhancedVolumeNodes.items():
            if outputDir:
                output_file = os.path.join(outputDir, f"Filtered_{method_name}_{inputVolume.GetName()}.nrrd")
                slicer.util.saveNode(volume_node, output_file)
                print(f"Saved {method_name} enhancement as: {output_file}")
            
        return enhancedVolumeNodes