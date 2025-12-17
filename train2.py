r""" AFANet training (validation) code """
import os
# 建议显卡设置在命令行指定，或者保持这里的设置
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import argparse
import pdb
from datetime import datetime, timedelta
import torch.optim as optim
import torch.nn as nn
import torch
import random
import numpy as np
import matplotlib.pyplot as plt # [新增] 引入画图库

from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
from model.AFANet import afanet
import clip

# [新增] 设置 matplotlib 后端，防止在无界面的服务器上报错
plt.switch_backend('agg') 

def setup_seed(seed):
    print(f'seed: {seed}')
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'

# [新增] 绘制 Loss 曲线的函数
def plot_loss_curve(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'loss_curve.png'))
    plt.close()

def train(epoch, model, dataloader, optimizer, training, stage):
    r""" Train AFANet """  
    model.module.train_mode() if training else model.module.eval()
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):
        
        # 1. forward propagation
        batch = utils.to_cuda(batch)
        
        logit_mask_q, logit_mask_s, losses = model(
            query_img=batch['query_img'],              
            support_img=batch['support_imgs'].squeeze(1),   
            support_cam=batch['support_cams'].squeeze(1),  
            query_cam=batch['query_cam'], stage=stage,     
            query_mask=batch['query_mask'],
            support_mask=batch['support_masks'].squeeze(1),
            class_id = batch['class_id'])
        pred_mask_q = logit_mask_q.argmax(dim=1)  

        # 2. Compute loss & update model parameters               
        loss = losses.mean()
        if training:
            optimizer.zero_grad()  
            loss.backward()
            optimizer.step()
        
        # 3. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask_q, batch)  
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone()) 
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

    # Write evaluation results
    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou


if __name__ == '__main__':
    
    # Arguments parsing
    parser = argparse.ArgumentParser(description='AFANet: Adaptive Frequency-Aware Network')
    parser.add_argument('--datapath', type=str, default='/home/xhd/XD/datasets/AFANet_datasets/')
    parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal', 'coco', 'fss'])
    parser.add_argument('--logpath', type=str, default='')   
    parser.add_argument('--bsz', type=int, default=16)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--niter', type=int, default=50) 
    parser.add_argument('--nworker', type=int, default=32)
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--stage', type=int, default=2) 
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50', 'resnet101'])

    parser.add_argument('--traincampath', type=str, default='/home/xhd/XD/datasets/AFANet_datasets/CAM_VOC_Train/')
    parser.add_argument('--valcampath', type=str, default='/home/xhd/XD/datasets/AFANet_datasets/CAM_VOC_Val/')
    parser.add_argument('--seed', type=int, default=6776)
    
    # [新增] 断点续训参数
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint file to resume from')

    args = parser.parse_args()
    
    print(f'seed: {args.seed}')
    setup_seed(args.seed)
    
    Logger.initialize(args, training=True)
    
    # 稍微放宽这个限制，防止单卡测试报错
    if torch.cuda.device_count() > 1:
        assert args.bsz % torch.cuda.device_count() == 0

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())

    # Model initialization
    clip_model, _ = clip.load('RN50', device= device, jit=False)
    model = afanet(args.backbone, False, args.benchmark, clip_model)
    Logger.log_params(model)

    model = nn.DataParallel(model)
    model.to(device)

    # Helper classes (for training) initialization
    optimizer = optim.Adam([{"params": model.parameters(), "lr": args.lr}])
    
    Evaluator.initialize()  

    # Dataset initialization
    FSSDataset.initialize(img_size=400, datapath=args.datapath, use_original_imgsize=False)
    
    dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn',
                                                 cam_train_path=args.traincampath, cam_val_path=args.valcampath)
    dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val',
                                                 cam_train_path=args.traincampath, cam_val_path=args.valcampath)

    # Train AFANet
    best_val_miou = float('-inf')
    best_epoch = 0
    start_epoch = 0 # [新增] 默认从0开始

    # [新增] 用于记录 Loss 以便绘图
    train_loss_history = []
    val_loss_history = []

    # [新增] 断点续训加载逻辑
    if args.resume:
        if os.path.isfile(args.resume):
            Logger.info(f"==> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            best_val_miou = checkpoint['best_val_miou']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            
            # 尝试恢复 loss history (如果之前保存了的话，为了简单这里不从文件读列表，从空开始画)
            # 如果你想完美恢复画图，需要把 history 也存在 checkpoint 里
            if 'train_loss_history' in checkpoint:
                train_loss_history = checkpoint['train_loss_history']
                val_loss_history = checkpoint['val_loss_history']
                
            Logger.info(f"==> Loaded checkpoint (epoch {checkpoint['epoch']})")
        else:
            Logger.info(f"==> No checkpoint found at '{args.resume}'")

    print(f'Start training from epoch: {start_epoch}, Total epochs: {args.niter}')
    
    linux_os_start_time = datetime.now() 
    print(f'Start time: {linux_os_start_time}')

    for epoch in range(start_epoch, args.niter):  
        
        epoch_start_time = datetime.now() # [新增] 记录本轮开始时间

        # Training
        trn_loss, trn_miou, trn_fb_iou = train(epoch, model, dataloader_trn, optimizer, training=True, stage=args.stage)
        
        # Validation
        with torch.no_grad():
            val_loss, val_miou, val_fb_iou = train(epoch, model, dataloader_val, optimizer, training=False, stage=args.stage)

        # [新增] 记录 Loss
        train_loss_history.append(trn_loss.item())
        val_loss_history.append(val_loss.item())
        
        # [新增] 实时绘制 Loss 曲线并保存
        plot_loss_curve(train_loss_history, val_loss_history, Logger.logpath)

        # Save the best model
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            best_epoch = epoch
            Logger.save_model_miou(model, epoch, val_miou)
        
        # [新增] 每10个epoch保存一个 Checkpoint (用于断点续训)
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(Logger.logpath, f'checkpoint_ep{epoch}.pth')
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_miou': best_val_miou,
                'train_loss_history': train_loss_history,
                'val_loss_history': val_loss_history
            }
            torch.save(state, checkpoint_path)
            print(f'Checkpint saved: {checkpoint_path}')

        Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
        Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
        Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
        Logger.tbd_writer.flush()

        # [新增] 时间预测功能
        epoch_end_time = datetime.now()
        epoch_duration = epoch_end_time - epoch_start_time
        remaining_epochs = args.niter - (epoch + 1)
        estimated_remaining_time = epoch_duration * remaining_epochs
        estimated_finish_time = datetime.now() + estimated_remaining_time
        
        print(f'--------------------------------------------------')
        print(f'Epoch {epoch} finished.')
        print(f'Duration: {epoch_duration}')
        print(f'Estimated Remaining Time: {estimated_remaining_time}')
        print(f'Predicted Finish Time: {estimated_finish_time.strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'--------------------------------------------------\n')

    print(f"epoch:{best_epoch} best_val_miou: {best_val_miou}")

    Logger.tbd_writer.close()
    Logger.info('==================== Finished Training ====================')

    # Finish time statistics
    linux_os_end_time = datetime.now()
    total_time = linux_os_end_time - linux_os_start_time

    print(f'Start time: {linux_os_start_time}')
    print(f'end_time: {linux_os_end_time}')
    print(f'total_time:{total_time}')