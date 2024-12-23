#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import torch, cv2
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os, random, itertools
from common import ImageFolderWithoutTarget, ImageFolderWithPath, InfiniteDataloader
from sklearn.metrics import roc_auc_score
from util.hardware import gpu_check, load_json
from util.parser_ import get_argparse
# from util.save_MVTec_AD import save_original_and_anom_map, save_original_and_anom_map_and_mask
from util.figure import loss_figure
from neuralNetwork.pdn import *
from neuralNetwork.autoEncoder import AutoEncoder
from src.teacher import teacher_normalization

# constants
seed = 42
on_gpu = torch.cuda.is_available()
out_channels = 384
image_size = 256

# data loading
default_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_ae = transforms.RandomChoice([
    transforms.ColorJitter(brightness=0.2),
    transforms.ColorJitter(contrast=0.2),
    transforms.ColorJitter(saturation=0.2)
])

def train_transform(image):
    return default_transform(image), default_transform(transform_ae(image))

def test(test_set: DataLoader, teacher, student, autoencoder, teacher_mean, teacher_std,
            quantiles: None|tuple=None,
            test_output_dir: str=None, desc='Running inference'):
    """
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
        image = default_transform(image)
        image = image[None]
        image = image.cuda() if on_gpu else image
        
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
    
    map_st = torch.mean((teacher_output - student_output[:, :out_channels])**2, dim=1, keepdim=True)
    map_ae = torch.mean((autoencoder_output - student_output[:, out_channels:])**2, dim=1, keepdim=True)
    
    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
    map_combined = 0.5 * map_st + 0.5 * map_ae
    
    return map_combined, map_st, map_ae

@torch.no_grad()        #FIXME: arguments들이 너무 많음
def map_normalization(validation_loader: DataLoader, teacher: nn.Sequential, student: nn.Sequential, autoencoder: nn.Sequential,   #TODO: teacher, student, autoencoder 결합하여 하나의 변수 인자로 받기
                                    teacher_mean: torch.Tensor, teacher_std: torch.Tensor, desc='Map normalization'):       #TODO: teacher_mean, teacher_std 결합하여 하나의 변수 인자로 받기
    """     #TODO: quantile들 확인하기
    Normalize the map using quantiles
    
    Args:
        validation_loader (DataLoader): validation data loader
        teacher (nn.Sequential): teacher model
        student (nn.Sequential): student model
        autoencoder (nn.Sequential): autoencoder model
        teacher_mean (torch.Tensor): mean of teacher output
        teacher_std (torch.Tensor): standard deviation of teacher output
        desc (str): description for tqdm
    
    Returns:
        q_st_start (torch.Tensor): start quantile of student map
        q_st_end (torch.Tensor): end quantile of student map
        q_ae_start (torch.Tensor): start quantile of autoencoder map
        q_ae_end (torch.Tensor): end quantile of autoencoder map
    """
    maps_st = []
    maps_ae = []
    # ignore augmented ae image
    for image, _ in tqdm(validation_loader, desc=desc):
        image = image.cuda() if on_gpu else image
        map_combined, map_st, map_ae = predict(image=image, teacher=teacher, student=student,
                                        autoencoder=autoencoder, teacher_mean=teacher_mean, teacher_std=teacher_std)
        maps_st.append(map_st)
        maps_ae.append(map_ae)
        
    maps_st = torch.cat(maps_st)
    maps_ae = torch.cat(maps_ae)
    
    # q : quantile을 뜻함
    q_st_start = torch.quantile(maps_st, q=0.9)
    q_st_end = torch.quantile(maps_st, q=0.995)
    q_ae_start = torch.quantile(maps_ae, q=0.9)
    q_ae_end = torch.quantile(maps_ae, q=0.995)
    
    return q_st_start, q_st_end, q_ae_start, q_ae_end

if __name__ == '__main__':
    gpu_check()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    config_file = load_json(fp='/home/msis/Work/anomalyDetector/EfficientAD/parameters.json')
    args = get_argparse()
    dataset_path = config_file['dataset'][args.dataset]['path']
    pretrain_penalty = True if args.imagenet_train_path != 'none' else False

    # create output dir
    train_output_dir = os.path.join(args.output_dir, 'trainings', args.dataset, args.category)
    test_output_dir = os.path.join(args.output_dir, 'anomaly_maps', args.dataset, args.category, 'test')
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    # load data
    full_train_set = ImageFolderWithoutTarget(os.path.join(dataset_path, args.category, 'train'),
                                            transform=transforms.Lambda(train_transform))
    test_set = ImageFolderWithPath(os.path.join(dataset_path, args.category, 'test'))
    
    if args.dataset == 'MVTec_AD':
        # mvtec dataset paper recommend 10% validation set
        train_size = int(0.9 * len(full_train_set))
        validation_size = len(full_train_set) - train_size
        rng = torch.Generator().manual_seed(seed)       #NOTE: rng: random number generator
        train_set, validation_set = torch.utils.data.random_split(full_train_set, [train_size,validation_size], rng)
    elif args.dataset == 'MVTec_LOCO_AD':
        train_set = full_train_set
        validation_set = ImageFolderWithoutTarget(os.path.join(dataset_path, args.category, 'validation'),
                                                transform=transforms.Lambda(train_transform))
    else:
        raise Exception('Unknown config.dataset')


    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    train_loader_infinite = InfiniteDataloader(train_loader)
    validation_loader = DataLoader(validation_set, batch_size=1)

    if pretrain_penalty:
        # load pretraining data for penalty
        penalty_transform = transforms.Compose([
                                    transforms.Resize((2 * image_size, 2 * image_size)),
                                    transforms.RandomGrayscale(0.3),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])
        penalty_set = ImageFolderWithoutTarget(args.imagenet_train_path, transform=penalty_transform)
        penalty_loader = DataLoader(penalty_set, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
        penalty_loader_infinite = InfiniteDataloader(penalty_loader)
    else:
        penalty_loader_infinite = itertools.repeat(None)

    # create models
    teacher, student = create_model(model_size=args.model_size, out_channels=out_channels)
    
    state_dict = torch.load(args.weights, map_location='cpu', weights_only=True)
    teacher.load_state_dict(state_dict)
    autoencoder = AutoEncoder(out_channels)

    # teacher frozen
    teacher.eval()
    student.train()
    autoencoder.train()

    if on_gpu == True:
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()

    teacher_mean, teacher_std = teacher_normalization(teacher=teacher, train_loader=train_loader)

    optimizer = torch.optim.Adam(itertools.chain(student.parameters(), autoencoder.parameters()), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.95 * args.train_steps), gamma=0.1)
    
    losses = []
    auces = []
    tqdm_obj = tqdm(range(args.train_steps))
    for iteration, (image_st, image_ae), image_penalty in zip(
            tqdm_obj, train_loader_infinite, penalty_loader_infinite):
        if on_gpu == True:
            image_st = image_st.cuda()
            image_ae = image_ae.cuda()
            if image_penalty is not None:
                image_penalty = image_penalty.cuda()
        
        with torch.no_grad():
            teacher_output_st = teacher(image_st)
            teacher_output_st = (teacher_output_st - teacher_mean) / teacher_std
        
        student_output_st = student(image_st)[:, :out_channels]
        distance_st = (teacher_output_st - student_output_st) ** 2
        d_hard = torch.quantile(distance_st, q=0.999)
        loss_hard = torch.mean(distance_st[distance_st >= d_hard])

        if image_penalty is not None:
            student_output_penalty = student(image_penalty)[:, :out_channels]
            loss_penalty = torch.mean(student_output_penalty**2)
            loss_st = loss_hard + loss_penalty
        else:
            loss_st = loss_hard

        ae_output = autoencoder(image_ae)
        with torch.no_grad():
            teacher_output_ae = teacher(image_ae)
            teacher_output_ae = (teacher_output_ae - teacher_mean) / teacher_std
        
        student_output_ae = student(image_ae)[:, out_channels:]
        distance_ae = (teacher_output_ae - ae_output)**2
        distance_stae = (ae_output - student_output_ae)**2
        loss_ae = torch.mean(distance_ae)
        loss_stae = torch.mean(distance_stae)
        loss_total = loss_st + loss_ae + loss_stae
        losses.append(loss_total.item())

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        if iteration % 10 == 0:
            tqdm_obj.set_description("Current loss: {:.4f}  ".format(loss_total.item()))

        # if iteration % 1000 == 0:
        #     torch.save(teacher, os.path.join(train_output_dir, 'teacher_tmp.pth'))
        #     torch.save(student, os.path.join(train_output_dir, 'student_tmp.pth'))
        #     torch.save(autoencoder, os.path.join(train_output_dir, 'autoencoder_tmp.pth'))

        # if iteration % 10000 == 0 and iteration > 0:
        #     # run intermediate evaluation
        #     teacher.eval()
        #     student.eval()
        #     autoencoder.eval()

        #     q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
        #                                                     validation_loader=validation_loader, teacher=teacher,
        #                                                     student=student, autoencoder=autoencoder,
        #                                                     teacher_mean=teacher_mean, teacher_std=teacher_std,
        #                                                     desc='Intermediate map normalization')
        #     auc = test(test_set=test_set, teacher=teacher, student=student,
        #                 autoencoder=autoencoder, teacher_mean=teacher_mean,
        #                 teacher_std=teacher_std, q_st_start=q_st_start,
        #                 q_st_end=q_st_end, q_ae_start=q_ae_start, q_ae_end=q_ae_end,
        #                 test_output_dir=None, desc='Intermediate inference')
        #     auces.append(auc)
        #     print('Intermediate image auc: {:.4f}'.format(auc))

        #     # teacher frozen
        #     teacher.eval()
        #     student.train()
        #     autoencoder.train()
    
    teacher.eval()
    student.eval()
    autoencoder.eval()

    #TODO: 손실함수 그래프 그리기
    # loss_figure(array=losses, dst_fp=train_output_dir)
    
    torch.save(teacher, os.path.join(train_output_dir, 'teacher_final.pth'))
    torch.save(student, os.path.join(train_output_dir, 'student_final.pth'))
    torch.save(autoencoder, os.path.join(train_output_dir, 'autoencoder_final.pth'))

    quantiles = map_normalization(
        validation_loader=validation_loader, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, desc='Final map normalization')
    auc = test(
        test_set=test_set, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean, teacher_std=teacher_std,
        quantiles=quantiles,
        test_output_dir=test_output_dir, desc='Final inference')
    print('Final image auc: {:.4f}'.format(auc))