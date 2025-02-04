import cv2
import numpy as np
import os
from PIL import Image

def save_anomaly_map(fp:str, anom_arr:list) -> bool:
    """
    Save anomaly map
    Args:    
        fp (str): file path to save anomaly map
        anom_arr : anomaly map
    """
    return cv2.imwrite(filename=fp, img=anom_arr)

def save_original_and_anom_map(src_fp:str, dst_fp:str, anom_arr:list) -> bool:
    """
    Save original image and anomaly map side by side
    
    Args:
        src_fp : file path to save original image
        dst_fp : file path to save anomaly map
        anom_arr : anomaly map
    
    based on MVTecAD datset
    """
    
    original_image = Image.open(fp=src_fp)
    original_image = np.array(original_image)
    original_image = cv2.cvtColor(src=original_image, code=cv2.COLOR_RGB2BGR)
    
    anom_arr = cv2.cvtColor(src=anom_arr, code=cv2.COLOR_GRAY2BGR)
    anom_arr = cv2.resize(anom_arr, (original_image.shape[1], original_image.shape[0]))
    
    stacked_arr = np.concatenate([original_image, anom_arr], axis=1)
    stacked_fp = dst_fp.replace('.png', '_stacked.png')
    
    return cv2.imwrite(filename=stacked_fp, img=stacked_arr)

def save_original_and_anom_map_and_mask(src_fp:str, dst_fp:str, anom_arr:list) -> bool:
    """
    Save original image, anomaly map and mask side by side
    
    src_fp : file path to save original image
    dst_fp : file path to save anomaly map
    anom_arr : anomaly map
    """
    
    original_image = Image.open(fp=src_fp)
    original_image = np.array(original_image)
    original_image = cv2.cvtColor(src=original_image, code=cv2.COLOR_RGB2BGR)
    
    gt_path =src_fp.replace('test', 'ground_truth').split('.')[0] + '_mask.png'
    gt_image = Image.open(fp=gt_path)
    gt_image = np.array(gt_image)
    gt_image = cv2.cvtColor(src=gt_image, code=cv2.COLOR_GRAY2BGR)
    
    anom_arr = cv2.cvtColor(src=anom_arr, code=cv2.COLOR_GRAY2BGR)
    anom_arr = cv2.resize(anom_arr, (original_image.shape[1], original_image.shape[0]))
    
    stacked_arr = np.concatenate([original_image, gt_image, anom_arr], axis=1)
    stacked_fp = dst_fp.replace('.png', '_stacked.png')
    
    return cv2.imwrite(stacked_fp, stacked_arr)