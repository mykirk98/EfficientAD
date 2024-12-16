import cv2
import numpy as np
import os
from PIL import Image

def save_anomaly_map(fp:str, anom_arr:list) -> bool:
    """
    Save anomaly map to disk
    
    fp : file path to save anomaly map
    
    based on MVTecAD datset
    """
    return cv2.imwrite(fp, anom_arr)

def save_original_and_anom_map(src_fp:str, dst_fp:str, anom_arr:list) -> bool:
    """
    Save original image and anomaly map side by side
    
    fp : file path to save original image and anomaly map
    anom_arr : anomaly map
    
    based on MVTecAD datset
    """
    
    original_image = Image.open(src_fp)
    original_image = np.array(original_image)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    
    anom_arr = cv2.cvtColor(src=anom_arr, code=cv2.COLOR_GRAY2BGR)
    anom_arr = cv2.resize(anom_arr, (original_image.shape[1], original_image.shape[0]))
    
    orig_anom = np.concatenate([original_image, anom_arr], axis=1)
    stacked_fp = dst_fp.replace('.png', '_stacked.png')
    
    return cv2.imwrite(stacked_fp, orig_anom)