import os
import sys
import time
import json
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from math import log10

from opts import parse_opts
from mean import get_mean, get_std
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor, DriverFocusCrop, DriverCenterCrop)
from temporal_transforms import LoopPadding, TemporalRandomCrop, TemporalCenterCrop, UniformIntervalCrop
from utils import set_random_seed
from dataset import get_training_set, get_validation_set
from levellstm import Flstm
# from tensorboardX import SummaryWriter

def prediction_test(model_dir,val_loader):
    # prediction of the test data
    label_map = {0:"end_action",1:"lchange",2:"lturn",3:"rchange",4:"rturn"}

    M = 5 #  5种行为类别+dummy类
    model = torch.load(model_dir)
    model.eval()
    TP_num = np.zeros(M)
    N_num = np.zeros(M)
    P_num = np.zeros(M)
    cfu_matrix = np.zeros((M,M))
    time_before_m = 0.0
    for i, data in enumerate(val_loader):  # b,sl,...
        test_data = data[0]
        targets = data[1]
        vid = data[2]
        output = model(test_data)
        output = output.squeeze(0)
        pred = nn.functional.softmax(output,dim=1)
        value,pred_label = torch.max(pred,dim=1)
        value = value.cpu().detach()
        pred_label = pred_label.cpu().numpy()
        #阈值
        #if pred_label!=0:
        #    if value<0.8:
        #        pred_label[-1] = 0
        
        check_label = targets.numpy()
        '''
        padding_check_data = np.zeros([check_data.shape[0],check_data.shape[0],check_data.shape[1],check_data.shape[2]])
        padding_check_data[-1,:,:,:] = check_data
        for j in range(0,check_data.shape[0] - 1):
            padding_check_data[j,:,:,:] = check_data
            for k in range(j + 1,check_data.shape[0]):
                padding_check_data[j,k,:,:] = padding_check_data[j,j,:,:]
        # if i == 0:
        #    print(padding_check_data[:, :, 0])
        check_data = utils.prepare_data(check_data)
        check_label = test_data[1][i]
        pred_label = model.forward(check_data)
        #if (pred_label[1][-1]!=check_label[-1]):
        #    print(str(test_data[2][i])+"  pred:"+label_map[str(pred_label[1][-1])]+"  label:"+label_map[str(check_label[-1][0])])
            #print("  pred:"+label_map[str(pred_label[1][-1])]+"  label:"+label_map[str(check_label[-1])])
        '''
        if check_label[-1] != 0:
            N_num[check_label[-1]] += 1
            if pred_label[-1] == check_label[-1]:
                TP_num[check_label[-1]] += 1
    
        cfu_matrix[check_label[-1],pred_label[-1]] += 1

        if pred_label[-1] != 0:
            P_num[pred_label[-1]] += 1
        
        #if pred_label[-1] != check_label[-1]:
        #    print('{}:  pred:{}  label:{}'.format(vid,label_map[pred_label[-1]],label_map[check_label[-1]]))
    time_before_m = 0    
    Pr = 0
    Re = 0
    for i in range(1,5):
        Pr += TP_num[i] / P_num[i]
        Re += TP_num[i] / N_num[i]
    Pr /= 4
    Re /= 4
    for i in range(M):
        cfu_matrix[i] /= np.sum(cfu_matrix[i])
    return Pr,Re,time_before_m,cfu_matrix

def prediction_total(epoch,model_dir):

    total_fold_num = 1              #在此修改测试的fold数
    pr = np.zeros(total_fold_num)
    re = np.zeros(total_fold_num)
    t = np.zeros(total_fold_num)
    cfu_matrix = np.zeros((total_fold_num,5,5))   #混淆矩阵
    for fold_num in range(1, total_fold_num + 1):
        val_loader = loader_list[fold_num-1]
        model_dir_epoch = model_dir +'fold'+str(fold_num-1)+'/flstm-save_'+str(epoch) + '.pkl'
        #print('fold_num:',fold_num)
        #val_loader = loader_list[1]
        #model_dir_epoch = model_dir +'fold1/flstm-save_'+str(epoch) + '.pkl'
        pr[fold_num - 1], re[fold_num - 1], t[fold_num - 1],cfu_matrix[fold_num - 1] = prediction_test(model_dir_epoch,val_loader)

    Pr = np.mean(pr)
    Re = np.mean(re)
    F1 = 2 * Pr * Re / (Pr + Re)
    time_before_m = np.mean(t)
    cfu_matrix = np.mean(cfu_matrix,0)
    print('Pr:',pr,'Re:',re)
    print(' Pr: ' + str("%.3f" % Pr) + ' Re: ' + str("%.3f" % Re) + ' F1: '
        + str("%.3f" % F1) + ' Time before maneuver: ' + str("%.2f" % time_before_m))
    return F1,cfu_matrix 


if __name__ == '__main__':
    opt = parse_opts()
    set_random_seed(opt.rand_seed)
    if opt.root_path != '':
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = 'flowlstm'
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    #opt.sample_size = (144,96)
    in_sample_size = (112,112)
    out_sample_size = (144,96)
    loader_list = []

    if not opt.no_train:
        assert opt.train_crop in ['random', 'corner', 'center', 'driver focus']
    if opt.train_crop == 'random':
        crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
    elif opt.train_crop == 'corner':
        crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
    elif opt.train_crop == 'center':
        crop_method = MultiScaleCornerCrop(
            opt.scales, opt.sample_size, crop_positions=['c'])
    elif opt.train_crop == 'driver focus':
        crop_method = DriverFocusCrop(opt.scales, opt.sample_size)

    torch.manual_seed(opt.manual_seed)
    #model = Flstm(feature_dim=6+32,tag_to_ix=tag_to_ix, hidden_dim=32,batch_size=opt.batch_size,outnet_name='resnet18')
    for i in range(5):
        opt.n_fold = i
        val_spatial_transform_invideo = Compose([
            #crop_method,    #进行裁剪
            #MultiScaleRandomCrop(opt.scales, opt.sample_size),
            Scale(in_sample_size),        
            ToTensor(opt.norm_value),
        ])
        val_spatial_transform_outvideo = Compose([
            Scale(out_sample_size),        
            ToTensor(opt.norm_value),
        ])
        val_temporal_transform = UniformIntervalCrop(opt.sample_duration, opt.interval)
        #val_temporal_transform = FixInterval(opt.sample_duration, opt.interval)
        val_target_transform = None
        validation_data = get_validation_set(opt, val_spatial_transform_invideo,
                    val_spatial_transform_outvideo, val_temporal_transform, val_target_transform)
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=1,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True)
        loader_list.append(val_loader)

    max_res = 0
    best_cfu_matrix = None
    max_epoch = -1
    for e in range(opt.test_start_epoch, opt.test_end_epoch+1, opt.checkpoint):
        print('loading epoch: ' + str(e))
        temp_res,cfu_matrix = prediction_total(e, opt.resume_path)
        if temp_res > max_res:
            max_res = temp_res
            best_cfu_matrix = cfu_matrix
            max_epoch = e
    print('best_results: ' + str(max_res) + ' epoch: ' + str(max_epoch))
    print(best_cfu_matrix)

