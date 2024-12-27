from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import os
from sklearn.metrics import roc_auc_score
import numpy as np
import cv2
from src.data import basic_transform
from torch import nn
from parameters import *

@torch.no_grad()    #TODO: image: torch.Tensor는 언제 torch.Tensor로 변환되었는지 확인하기
def predict(image: torch.Tensor, teacher: nn.Sequential, student: nn.Sequential, autoencoder: nn.Sequential,
            teacher_mean: torch.Tensor, teacher_std: torch.Tensor,
            quantiles: None|tuple=None):
    """
    Args:
        imge (torch.Tensor): input image
        teacher (nn.Sequential): teacher model
        student (nn.Sequential): student model
        autoencoder (nn.Sequential): autoencoder model
        
    Returns:
        map_combined (torch.Tensor): combined map
        map_st (torch.Tensor): student map
        map_ae (torch.Tensor): autoencoder map
    """
    
    if quantiles is not None:
        q_st_start, q_st_end, q_ae_start, q_ae_end = quantiles
    else:
        q_st_start, q_st_end, q_ae_start, q_ae_end = None, None, None, None
    
    # Normalize teacher output
    teacher_output = teacher(image)
    # teacher_output = (teacher_output - teacher_mean) / teacher_std
    teacher_output -= teacher_mean
    teacher_output /= teacher_std
    
    # student and autoencoder output
    student_output = student(image)
    autoencoder_output = autoencoder(image)
    
    map_st = torch.mean((teacher_output - student_output[:, :output_channels])**2, dim=1, keepdim=True)
    map_ae = torch.mean((autoencoder_output - student_output[:, output_channels:])**2, dim=1, keepdim=True)
    
    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
    map_combined = 0.5 * map_st + 0.5 * map_ae
    
    return map_combined, map_st, map_ae

def test(test_set: DataLoader, teacher, student, autoencoder, teacher_mean, teacher_std,
            quantiles: None|tuple=None,
            test_output_dir: str=None, desc='Running inference'):
    """ #TODO: quantile들이 어떻게 사용되는지 확인하기
    Args:
        test_set (DataLoader): test data loader
        teacher (nn.Sequential): teacher model
        student (nn.Sequential): student model
        autoencoder (nn.Sequential): autoencoder model
        teacher_mean (torch.Tensor): mean of teacher output
        teacher_std (torch.Tensor): standard deviation of teacher output
        quantiles (None|tuple): quantiles for normalization, include q_st_start, q_st_end, q_ae_start, q_ae_end
        test_output_dir (str): directory to save test output
        desc (str): description for tqdm
    
    Returns:
        auc (float): area under the curve
    """
    y_true = []
    y_score = []
    
    for image, target, path in tqdm(test_set, desc=desc):
        orig_width, orig_height = image.width, image.height
        image = basic_transform(image)    #TODO: Normalization은 언제 진행되었는지 확인하기 -> 26번째줄 default_transform 함수에서 변환됨
        image = image[None]
        image = image.cuda() if torch.cuda.is_available() else image
        
        map_combined, map_st, map_ae = predict(image=image, teacher=teacher, student=student,
                                                autoencoder=autoencoder, teacher_mean=teacher_mean, teacher_std=teacher_std,
                                                # q_st_start=q_st_start, q_st_end=q_st_end, q_ae_start=q_ae_start, q_ae_end=q_ae_end)
                                                quantiles=quantiles)
        
        map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
        map_combined = torch.nn.functional.interpolate(map_combined, (orig_height, orig_width), mode='bilinear')
        map_combined = map_combined[0, 0].cpu().numpy()

        defect_class = os.path.basename(os.path.dirname(path))
        if test_output_dir is not None:
            # img_nm = os.path.split(path)[1].split('.')[0]
            if not os.path.exists(os.path.join(test_output_dir, defect_class)):
                os.makedirs(os.path.join(test_output_dir, defect_class))
            
            map_combined = (map_combined) / (np.max(map_combined) - np.min(map_combined)) * 255
            # original_image_path = path
            # file = os.path.join(test_output_dir, defect_class, img_nm + '.png')
            
            # if defect_class == 'good':
            #     save_original_and_anom_map(src_fp=original_image_path, dst_fp=file, anom_arr=map_combined)
            # else:
            #     save_original_and_anom_map_and_mask(src_fp=original_image_path, dst_fp=file, anom_arr=map_combined)
            # cv2.imwrite(filename=file, img=map_combined)    #FIXME: activate when test mode
        
        y_true_image = 0 if defect_class == 'good' else 1
        y_score_image = np.max(map_combined)
        y_true.append(y_true_image)
        y_score.append(y_score_image)
        
    auc = roc_auc_score(y_true=y_true, y_score=y_score) * 100
    return auc