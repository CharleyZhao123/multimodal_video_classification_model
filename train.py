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

#import torchvision.transforms.functional as F

from opts import parse_opts
from mean import get_mean, get_std
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor, DriverFocusCrop, DriverCenterCrop)
from temporal_transforms import LoopPadding, TemporalRandomCrop, TemporalCenterCrop, UniformIntervalCrop, UniformPadSample,RandomIntervalCrop
from utils import set_random_seed
from dataset import get_training_set, get_validation_set
from levellstm import Flstm                                       #车内外特征attention
#from tensorboardX import SummaryWriter

if __name__ == '__main__':
    opt = parse_opts()
    set_random_seed(opt.rand_seed)
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path, 'fold'+str(opt.n_fold))
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = 'flowlstm'
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    #resize的大小
    in_sample_size = (112,112)
    out_sample_size = (144,96)

    #print(opt)
    #with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
    #    json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)

### convlstm ##########################################################################################################
    hidden_dim = [64,64,64]   # [out, in, state]
    feature_dim = [32,64,64]
    nclass = 5
    model = Flstm(feature_dim,nclass, hidden_dim,num_dim=4,outnet_name='resnet18',innet_name='mobilefacenet').cuda()
    #writer = SummaryWriter()
    #model = nn.DataParallel(model, device_ids=None)
    #parameters = model.parameters()
    #print(model)
    #损失函数
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.NLLLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()
    #数据标准化      用于车内图像
    if opt.no_mean_norm and not opt.std_norm:   
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)
    #裁剪方式,但其实也没用到
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
    #定义三种转换
    #缩放并进行图像的tensor转化
    train_spatial_transform_invideo = Compose([
        #crop_method,    #进行随机裁剪
        #MultiScaleRandomCrop(opt.scales, opt.sample_size),
        Scale(in_sample_size),
        ToTensor(opt.norm_value),
    ])
    train_spatial_transform_outvideo = Compose([
        #crop_method,    #进行随机裁剪
        Scale(out_sample_size),        
        ToTensor(opt.norm_value),
    ])
    #train_temporal_transform =  UniformIntervalCrop(opt.sample_duration, opt.interval)  
    train_temporal_transform =  RandomIntervalCrop(opt.sample_duration, opt.interval)  #sample_duration控制采样几帧, opt.interval实际上没用
    train_target_transform = Compose([
        Scale(opt.sample_size),
        ToTensor(opt.norm_value)#, norm_method
    ])
    #随机水平翻转
    train_horizontal_flip = RandomHorizontalFlip()
    training_data = get_training_set(opt, train_spatial_transform_invideo,train_spatial_transform_outvideo, train_horizontal_flip,
                                     train_temporal_transform, train_target_transform)
    train_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_threads,
        pin_memory=True)
    #train_logger = Logger(
    #    os.path.join(opt.result_path, 'convlstm-train.log'),
    #    ['epoch', 'loss', 'lr'])
    #train_batch_logger = Logger(
    #    os.path.join(opt.result_path, 'convlstm-train_batch.log'),
    #    ['epoch', 'batch', 'iter', 'loss', 'lr'])

    optimizer = optim.Adam(model.parameters(),lr=opt.learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=opt.lr_step, gamma=0.1)
    if not opt.no_val:
        val_spatial_transform = Compose([
            Scale(opt.sample_size),
            ToTensor(opt.norm_value)#, norm_method
        ])
        val_temporal_transform = UniformIntervalCrop(opt.sample_duration, opt.interval)
        val_target_transform = val_spatial_transform
        validation_data = get_validation_set(
            opt, val_spatial_transform, val_temporal_transform, val_target_transform)
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=1,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True)
        #val_logger = Logger(
        #    os.path.join(opt.result_path, 'convlstm-val.log'), ['epoch', 'loss', 'ssim', 'psnr'])

    if opt.train_from_scratch:
        print('loading checkpoint {}'.format(opt.resume_path))
        model = torch.load(opt.resume_path)
        opt.begin_epoch = 641
        #model.load_state_dict(checkpoint)
#===============================================================================================
    print('run')
    global best_loss
    best_loss = torch.tensor(float('inf'))
    
    for epoch in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            print('train at epoch {}'.format(epoch))

            model.train()
            
            total_loss = 0
            begin = time.time()
            for i, data in enumerate(train_loader):  # b,sl,...
                train_data = data[0]
                targets = data[1]
                if not opt.no_cuda:
                    targets = targets.cuda(non_blocking=True)
                output = model(train_data)
                output = output.squeeze(0)
                loss = criterion(output,targets)
                total_loss+=loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            
            if not os.path.exists(opt.result_path):
                os.makedirs(opt.result_path)
            if epoch % opt.checkpoint == 0:
                save_file_path = os.path.join(opt.result_path,'flstm-save_{}.pkl'.format(epoch))
                torch.save(model, save_file_path)
                #states = {
                #    'epoch': epoch + 1,
                #    'arch': opt.arch,
                #    'state_dict': model.state_dict(),
                #    'optimizer': optimizer.state_dict(),
                #}
        print('total_loss:  ',total_loss.item())
        print('epoch_time:  ',time.time()-begin)
        # if not opt.no_val:
        #     model.eval()
        #     end_time = time.time()

        #     for i, (inputs, targets) in enumerate(val_loader):
        #         data_time.update(time.time() - end_time)

        #         if not opt.no_cuda:
        #             targets = targets.cuda(non_blocking=True)
        #         inputs = Variable(inputs, volatile=True)
        #         targets = Variable(targets, volatile=True)
        #         outputs = model(inputs)
        #         loss = criterion(outputs, targets)

        #         batch_time.update(time.time() - end_time)
        #         end_time = time.time()

        #         print('Epoch: [{0}][{1}/{2}]\t'
        #               'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #               'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #               'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #               'SSIM {ssim_loss.val:.4f}({ssim_loss.avg:.4f})\t'
        #               'PSNR {psnr_loss.val:.4f}({psnr_loss.avg:.4f})'.format(
        #                   epoch,
        #                   i + 1,
        #                   len(val_loader),
        #                   batch_time=batch_time,
        #                   data_time=data_time,
        #                   loss=losses,
        #                   ssim_loss=ssim_losses,
        #                   psnr_loss=psnr_losses))

        #     val_logger.log({'epoch': epoch, 'loss': losses.avg, 'ssim': ssim_losses.avg, 'psnr': psnr_losses.avg})

        #     is_best = losses.avg < best_loss
        #     best_loss = min(losses.avg, best_loss)
        #     print('\n The best prec is %.4f' % best_loss)
        #     if is_best:
        #         states = {
        #             'epoch': epoch + 1,
        #              'arch': opt.arch,
        #             'state_dict': model.state_dict(),
        #             'optimizer': optimizer.state_dict(),
        #           }
        #         save_file_path = os.path.join(opt.result_path,
        #                             'convlstm-save_best.pth')
        #         torch.save(states, save_file_path)

        # print('total_loss:  ',total_loss.item())
        # print('epoch_time:  ',time.time()-begin)
        # writer.add_histogram('zz/total_loss', total_loss, epoch)
        # writer.add_scalar('data/total_loss', total_loss, epoch)
        # writer.add_text('zz/text', 'epoch: ' + str(epoch) + ' loss: ' + str(total_loss), epoch)
        




