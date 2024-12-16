import cv2
import numpy as np
import os
from PIL import Image

def save_anomaly_map(fp:str, anom_arr:list) -> None:
    """
    Save anomaly map to disk
    
    fp : file path to save anomaly map
    """
    cv2.imwrite(fp, anom_arr)

def save_original_and_anom_map(fp:str, anom_arr:list) -> None:
    """
    Save original image and anomaly map side by side
    
    
    """
    
    original_image = Image.open(fp)
    original_image = np.array(original_image)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    
    anom_arr = cv2.resize(anom_arr, (original_image.shape[1], original_image.shape[0]))
    
    orig_anom = np.concatenate([original_image, anom_arr], axis=1)
    
    cv2.imwrite(fp, orig_anom)