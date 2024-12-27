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
from src.compute import test, predict

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
])      #NOTE: Algorithm 1. line 23~28  #FIXME: 논문과 구현이 살짝 다름, line23이 구현안됨

def train_transform(image):
    return default_transform(image), default_transform(transform_ae(image))

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
    
    teacher_pretrained_weight = torch.load(args.weights, map_location='cpu', weights_only=True)
    teacher.load_state_dict(teacher_pretrained_weight)
    autoencoder = AutoEncoder(out_channels)

    # teacher frozen
    teacher.eval()
    student.train()
    autoencoder.train()

    if on_gpu == True:
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()

    teacher_mean, teacher_std = teacher_normalization(teacher=teacher, train_loader=train_loader)   # NOTE: Algorithm 1. line 3~9

    optimizer = torch.optim.Adam(itertools.chain(student.parameters(), autoencoder.parameters()), lr=1e-4, weight_decay=1e-5)   #NOTE: Algorithm 1. line 11
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.95 * args.train_steps), gamma=0.1)   # 66500 / 70000 = 0.95
    
    # losses = []
    # auces = []
    tqdm_obj = tqdm(range(args.train_steps))
    for iteration, (image_st, image_ae), image_penalty in zip(
            tqdm_obj, train_loader_infinite, penalty_loader_infinite):  #NOTE: Algorithm 1. line 12
        if on_gpu == True:
            image_st = image_st.cuda()
            image_ae = image_ae.cuda()
            if image_penalty is not None:
                image_penalty = image_penalty.cuda()
        
        with torch.no_grad():
            teacher_output_st = teacher(image_st)   # (1, 384, 56, 56) teacher output for student   #NOTE: Algorithm 1. line 14
            teacher_output_st = (teacher_output_st - teacher_mean) / teacher_std    # (1, 384, 56, 56) Normalized   #NOTE: Algorithm 1. line 15
        
        student_output_st = student(image_st)[:, :out_channels, :, :]   # (1, 384, 56, 56)  student output for teacher  #NOTE: Algorithm 1. line 17
        distance_st = (teacher_output_st - student_output_st) ** 2  # (1, 384, 56, 56)  #NOTE: Algorithm 1. line 18
        d_hard = torch.quantile(distance_st, q=0.999)   #NOTE: Algorithm 1. line 19
        loss_hard = torch.mean(distance_st[distance_st >= d_hard])  #NOTE: Algorithm 1. line 20

        if image_penalty is not None:   #NOTE: Algorithm 1. line 21~22
            student_output_penalty = student(image_penalty)[:, :out_channels]
            loss_penalty = torch.mean(student_output_penalty**2)
            loss_st = loss_hard + loss_penalty
        else:
            loss_st = loss_hard

        ae_output = autoencoder(image_ae)   #CONTINUE:  #NOTE: Algorithm 1. line 29
        with torch.no_grad():
            teacher_output_ae = teacher(image_ae)   #NOTE: Algorithm 1. line 30
            teacher_output_ae = (teacher_output_ae - teacher_mean) / teacher_std    #NOTE: Algorithm 1. line 31
        
        student_output_ae = student(image_ae)[:, out_channels:, :, :] #NOTE: Algorithm 1. line 32, 33
        distance_ae = (teacher_output_ae - ae_output)**2    #NOTE: Algorithm 1. line 34
        distance_stae = (ae_output - student_output_ae)**2  #NOTE: Algorithm 1. line 35
        loss_ae = torch.mean(distance_ae)   #NOTE: Algorithm 1. line 36
        loss_stae = torch.mean(distance_stae)   #NOTE: Algorithm 1. line 37
        loss_total = loss_st + loss_ae + loss_stae  #NOTE: Algorithm 1. line 38
        # losses.append(loss_total.item())


        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        if iteration % 10 == 0:
            tqdm_obj.set_description("Current loss: {:.4f}  ".format(loss_total.item()))

        if iteration % 1000 == 0:
            torch.save(teacher, os.path.join(train_output_dir, 'teacher_tmp.pth'))
            torch.save(student, os.path.join(train_output_dir, 'student_tmp.pth'))
            torch.save(autoencoder, os.path.join(train_output_dir, 'autoencoder_tmp.pth'))

        if iteration % 10000 == 0 and iteration > 0:
            # run intermediate evaluation
            teacher.eval()
            student.eval()
            autoencoder.eval()

            q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
                                                            validation_loader=validation_loader, teacher=teacher,
                                                            student=student, autoencoder=autoencoder,
                                                            teacher_mean=teacher_mean, teacher_std=teacher_std,
                                                            desc='Intermediate map normalization')      #NOTE: Algorithm 1. line 46~51
            auc = test(test_set=test_set, teacher=teacher, student=student,
                        autoencoder=autoencoder, teacher_mean=teacher_mean,
                        teacher_std=teacher_std, q_st_start=q_st_start,
                        q_st_end=q_st_end, q_ae_start=q_ae_start, q_ae_end=q_ae_end,
                        test_output_dir=None, desc='Intermediate inference')    #NOTE:L Algorithm 1. line 52~55
            # auces.append(auc)
            print('Intermediate image auc: {:.4f}'.format(auc))

            # teacher frozen
            teacher.eval()
            student.train()
            autoencoder.train()
    
    teacher.eval()
    student.eval()
    autoencoder.eval()

    #TODO: 손실함수 그래프 그리기
    # loss_figure(array=losses, dst_fp=train_output_dir)
    
    torch.save(teacher, os.path.join(train_output_dir, 'teacher_final.pth'))
    torch.save(student, os.path.join(train_output_dir, 'student_final.pth'))
    torch.save(autoencoder, os.path.join(train_output_dir, 'autoencoder_final.pth'))



    # Inference Procedure
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