r""" AFANet training (validation) code """
import os
import argparse
import pdb
from datetime import datetime, timedelta
import torch.optim as optim
import torch.nn as nn
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import re 

from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
from model.AFANet import afanet
import clip

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

def plot_loss_curve(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 5))
    # 确保长度一致，防止画图报错
    min_len = min(len(train_losses), len(val_losses))
    if min_len > 0:
        plt.plot(range(min_len), train_losses[:min_len], label='Train Loss', color='blue')
        plt.plot(range(min_len), val_losses[:min_len], label='Val Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'loss_curve.png'))
    plt.close()

# 从 log.txt 恢复 Loss 历史的函数
def recover_history_from_log(log_file_path):
    train_losses = []
    val_losses = []
    
    if not os.path.exists(log_file_path):
        print(f"Warning: Log file not found at {log_file_path}")
        return [], []

    print(f"Parsing log file to recover history: {log_file_path}")
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
        
    # 正则表达式匹配 log.txt 中的结果行
    train_pattern = re.compile(r'\*\*\* Training \[@Epoch (\d+)\] Avg L: ([\d\.]+)')
    val_pattern = re.compile(r'\*\*\* Validation \[@Epoch (\d+)\] Avg L: ([\d\.]+)')
    
    train_dict = {}
    val_dict = {}

    for line in lines:
        t_match = train_pattern.search(line)
        v_match = val_pattern.search(line)
        
        if t_match:
            epoch = int(t_match.group(1))
            loss = float(t_match.group(2))
            train_dict[epoch] = loss
        
        if v_match:
            epoch = int(v_match.group(1))
            loss = float(v_match.group(2))
            val_dict[epoch] = loss

    max_epoch = max(max(train_dict.keys(), default=-1), max(val_dict.keys(), default=-1))
    
    for i in range(max_epoch + 1):
        if i in train_dict:
            train_losses.append(train_dict[i])
        if i in val_dict:
            val_losses.append(val_dict[i])
            
    print(f"Recovered {len(train_losses)} epochs from log file.")
    return train_losses, val_losses

def train(epoch, model, dataloader, optimizer, training, stage):
    r""" Train AFANet """  
    model.module.train_mode() if training else model.module.eval()
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):
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

        loss = losses.mean()
        if training:
            optimizer.zero_grad()  
            loss.backward()
            # === 梯度裁剪 ===
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
        
        area_inter, area_union = Evaluator.classify_prediction(pred_mask_q, batch)  
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone()) 
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='AFANet: Adaptive Frequency-Aware Network')
    parser.add_argument('--datapath', type=str, default='/home/xhd/XD/datasets/AFANet_datasets/')
    parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal', 'coco', 'fss'])
    parser.add_argument('--logpath', type=str, default='')   
    parser.add_argument('--bsz', type=int, default=16)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--lr_prompt', type=float, default=2e-3, help='learning rate for prompt learner')
    parser.add_argument('--niter', type=int, default=50) 
    parser.add_argument('--nworker', type=int, default=16)
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--stage', type=int, default=2) 
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50', 'resnet101'])

    parser.add_argument('--traincampath', type=str, default='/home/xhd/XD/datasets/AFANet_datasets/CAM_VOC_Train/')
    parser.add_argument('--valcampath', type=str, default='/home/xhd/XD/datasets/AFANet_datasets/CAM_VOC_Val/')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint file to resume from')

    args = parser.parse_args()
    
    print(f'seed: {args.seed}')
    setup_seed(args.seed)
    
    Logger.initialize(args, training=True)
    
    if torch.cuda.device_count() > 1:
        assert args.bsz % torch.cuda.device_count() == 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())

    # Model initialization
    clip_model, _ = clip.load('RN50', device= device, jit=False)

    # === 强制转为 FP32 ===
    clip_model.float() 

    model = afanet(args.backbone, False, args.benchmark, clip_model)
    Logger.log_params(model)

    model = nn.DataParallel(model)
    model.to(device)

    # ================= 优化器配置 =================
    # CoOp 策略：Prompt 参数用大 LR，其他参数用正常 LR，Backbone/CLIP 冻结
    
    param_groups = []
    prompt_params = []
    decoder_params = []
    
    # 调试 CoOp 参数是否正确被识别
    found_ctx = False

    for name, param in model.named_parameters():
        if "prompt_learner.ctx" in name:
            # 1. Prompt 向量 (核心)
            param.requires_grad = True
            prompt_params.append(param)
            found_ctx = True
        elif "prompt_learner" in name or "clip_model" in name:
            # 2. CLIP 内部参数 (冻结)
            param.requires_grad = False
        elif "backbone" in name:
            # 3. Backbone (通常冻结)
            param.requires_grad = False
        else:
            # 4. AFANet Decoder / Linear (正常训练)
            param.requires_grad = True
            decoder_params.append(param)
            
    if not found_ctx:
        print("WARNING: CoOp Context Vector (ctx) NOT found! Please check model structure.")
    else:
        print("DEBUG: CoOp Context Vector detected and set to trainable.")
    
    num_prompt_elements = sum(p.numel() for p in prompt_params)
    num_decoder_elements = sum(p.numel() for p in decoder_params)

    Logger.info(f"Num of Prompt Tensors: {len(prompt_params)}")
    Logger.info(f"Num of Prompt Params (Elements): {num_prompt_elements}")

    Logger.info(f"Num of Decoder Tensors: {len(decoder_params)}")
    Logger.info(f"Num of Decoder Params (Elements): {num_decoder_elements}")

    optimizer = optim.Adam([
        {"params": prompt_params, "lr": args.lr_prompt},    # CoOp 推荐 LR
        {"params": decoder_params, "lr": args.lr} 
    ])
    # ====================================================

    Evaluator.initialize()  
    FSSDataset.initialize(img_size=400, datapath=args.datapath, use_original_imgsize=False)
    
    dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn',
                                                 cam_train_path=args.traincampath, cam_val_path=args.valcampath)
    dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val',
                                                 cam_train_path=args.traincampath, cam_val_path=args.valcampath)

    best_val_miou = float('-inf')
    best_epoch = 0
    start_epoch = 0 
    train_loss_history = []
    val_loss_history = []
    best_model_state_cache = None # 初始化缓存

    # ================= 断点续训逻辑 (鲁棒版) =================
    if args.resume:
        if os.path.isfile(args.resume):
            Logger.info(f"==> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            
            # 1. 恢复基础训练状态
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_miou = checkpoint.get('best_val_miou', float('-inf'))
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            
            # 2. 恢复 Loss History
            if 'train_loss_history' in checkpoint and len(checkpoint['train_loss_history']) > 0:
                Logger.info("==> Recovered loss history from checkpoint.")
                train_loss_history = checkpoint['train_loss_history']
                val_loss_history = checkpoint['val_loss_history']
            else:
                # 兼容旧逻辑：尝试从 log.txt 解析
                Logger.info("==> No history in checkpoint. Trying to recover from log.txt...")
                log_file = os.path.join(Logger.logpath, 'log.txt')
                recovered_train, recovered_val = recover_history_from_log(log_file)
                if len(recovered_train) > 0:
                    train_loss_history = recovered_train
                    val_loss_history = recovered_val
                else:
                    Logger.info("==> Could not recover history. Starting fresh plotting.")

            # 裁剪 history 防止重复
            if len(train_loss_history) > start_epoch:
                train_loss_history = train_loss_history[:start_epoch]
                val_loss_history = val_loss_history[:start_epoch]

            # 3. 恢复 Best Model Cache (关键步骤)
            if 'best_model_state_cache' in checkpoint:
                best_model_state_cache = checkpoint['best_model_state_cache']
                Logger.info(f"==> Recovered best model cache (mIoU: {best_val_miou:.4f}) from RAM snapshot.")
            else:
                # 尝试从磁盘加载现有的 best_model.pt 补救
                best_model_disk_path = os.path.join(Logger.logpath, 'best_model.pt')
                if os.path.exists(best_model_disk_path):
                    Logger.info(f"==> Loading existing best model from disk: {best_model_disk_path}")
                    try:
                        disk_checkpoint = torch.load(best_model_disk_path, map_location='cpu')
                        if isinstance(disk_checkpoint, dict) and 'state_dict' in disk_checkpoint:
                            best_model_state_cache = disk_checkpoint['state_dict']
                        elif isinstance(disk_checkpoint, dict):
                            best_model_state_cache = disk_checkpoint
                        else:
                            best_model_state_cache = None
                        Logger.info("==> Successfully initialized best model cache from disk.")
                    except:
                        best_model_state_cache = None
                        Logger.info("==> Failed to load disk model. Cache is None.")
                else:
                    best_model_state_cache = None
                    Logger.info("==> No cache found. Will start caching from next best.")

            Logger.info(f"==> Loaded checkpoint (resume from epoch {start_epoch})")
        else:
            Logger.info(f"==> No checkpoint found at '{args.resume}'")
    # ==========================================================

    print(f'Start training from epoch: {start_epoch}, Total epochs: {args.niter}')
    linux_os_start_time = datetime.now() 

    for epoch in range(start_epoch, args.niter):  
        epoch_start_time = datetime.now() 
        
        # === Temperature Annealing (温度退火) ===
        if isinstance(model, nn.DataParallel):
            current_temp = model.module.adjust_temperature(epoch, args.niter)
        else:
            current_temp = model.adjust_temperature(epoch, args.niter)
        
        Logger.info(f"Epoch [{epoch}/{args.niter}] - Updated Sparse Block Temperature to: {current_temp:.4f}")
        # ========================================

        trn_loss, trn_miou, trn_fb_iou = train(epoch, model, dataloader_trn, optimizer, training=True, stage=args.stage)
        
        with torch.no_grad():
            val_loss, val_miou, val_fb_iou = train(epoch, model, dataloader_val, optimizer, training=False, stage=args.stage)

        train_loss_history.append(trn_loss.item())
        val_loss_history.append(val_loss.item())
        
        plot_loss_curve(train_loss_history, val_loss_history, Logger.logpath)

        # 始终保存最佳模型 (覆盖式)
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            best_epoch = epoch
            
            Logger.save_model_miou(model, epoch, val_miou)
            
            # [关键] 更新内存中的最佳参数缓存
            current_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            best_model_state_cache = {k: v.cpu().clone() for k, v in current_state.items()}
            
            Logger.info(f"  ==> New Best mIoU: {best_val_miou:.4f} (at Epoch {epoch}) - Cached in memory.")
        
        # === 周期性保存 (Checkpoint & Snapshot) ===
        if (epoch + 1) % 10 == 0:
            # A. 保存常规 Checkpoint
            checkpoint_path = os.path.join(Logger.logpath, f'checkpoint_ep{epoch}.pth')
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(), 
                'optimizer': optimizer.state_dict(),
                'best_val_miou': best_val_miou,
                'train_loss_history': train_loss_history, 
                'val_loss_history': val_loss_history,
                'best_model_state_cache': best_model_state_cache 
            }
            torch.save(state, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')

            # B. 保存累计最优快照
            if best_model_state_cache is not None:
                snapshot_name = f'best_model_snapshot_0_{epoch}.pt'
                snapshot_path = os.path.join(Logger.logpath, snapshot_name)
                
                torch.save(best_model_state_cache, snapshot_path)
                Logger.info(f"==> [Snapshot] Saved best model of epochs 0-{epoch} to {snapshot_name} (mIoU: {best_val_miou:.4f})")
            else:
                Logger.info(f"==> [Snapshot] Skipped (No best model found yet).")

        Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
        Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
        Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
        Logger.tbd_writer.flush()

        epoch_end_time = datetime.now()
        epoch_duration = epoch_end_time - epoch_start_time
        remaining_epochs = args.niter - (epoch + 1)
        estimated_remaining_time = epoch_duration * remaining_epochs
        estimated_finish_time = datetime.now() + estimated_remaining_time
        
        print(f'--------------------------------------------------')
        print(f'Epoch {epoch} finished. Duration: {epoch_duration}')
        print(f'Predicted Finish Time: {estimated_finish_time.strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'--------------------------------------------------\n')

    print(f"epoch:{best_epoch} best_val_miou: {best_val_miou}")
    Logger.tbd_writer.close()
    Logger.info('==================== Finished Training ====================')