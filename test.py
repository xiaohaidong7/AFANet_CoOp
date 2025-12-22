r""" AFANet testing code  """
import argparse
import os
import torch.nn.functional as F
import torch.nn as nn
import torch
from model.AFANet import afanet
from common.logger import Logger, AverageMeter
from common.vis import Visualizer
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
import clip

def test(model, dataloader, nshot, stage):
    r""" Test AFANet """

    # Freeze randomness during testing for reproducibility
    utils.fix_randseed(6776)
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):

        # 1. Hypercorrelation Squeeze Networks forward pass
        batch = utils.to_cuda(batch)

        pred_mask = model.module.predict_mask_nshot(batch, nshot=nshot, class_id=batch['class_id'],
                                                    stage=stage)  # AFANet
        
        assert pred_mask.size() == batch['query_mask'].size()

        # 2. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)

        # Visualize predictions
        if Visualizer.visualize:
            Visualizer.visualize_prediction_batch(batch['support_imgs'], batch['support_masks'],
                                                  batch['query_img'], batch['query_mask'],
                                                  pred_mask,
                                                  batch['class_id'], idx,
                                                  area_inter[1].float() / area_union[1].float(),
                                                  batch['query_name'])

    # Write evaluation results
    average_meter.write_result('Test', 0)
    miou, fb_iou = average_meter.compute_iou()
    return miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='AFANet Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='/home/xhd/XD/datasets/AFANet_datasets/')
    parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal', 'coco', 'fss'])
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--bsz', type=int, default=1)  # must be 1
    parser.add_argument('--nworker', type=int, default=16)
    parser.add_argument('--load', type=str, required=True, help='Path to the model to load')
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50', 'resnet101'])
    parser.add_argument('--visualize', action='store_true') 
    parser.add_argument('--use_original_imgsize', action='store_true') 
    parser.add_argument('--stage', type=int, default=2) 

    parser.add_argument('--traincampath', type=str, default='/home/xhd/XD/datasets/AFANet_datasets/CAM_VOC_Train/')
    parser.add_argument('--valcampath', type=str, default='/home/xhd/XD/datasets/AFANet_datasets/CAM_VOC_Val/')
    parser.add_argument('--vispath', type=str, default='./vis/')

    args = parser.parse_args()
    Logger.initialize(args, training=False)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model initialization
    clip_model, _ = clip.load('RN50', device=device, jit=False)
    
    # [修改 1] 强制转为 FP32，与 Train 保持一致
    clip_model.float() 

    model = afanet(args.backbone, args.use_original_imgsize, args.benchmark, clip_model)
    model.eval()

    Logger.log_params(model)
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())

    model = nn.DataParallel(model)
    model.to(device)

    # [修改 2] 鲁棒的模型加载逻辑
    if args.load == '':
        raise Exception('Pretrained model not specified.')
    
    Logger.info(f'Loading model from {args.load}...')
    try:
        # map_location 避免 GPU 数量不一致时的报错
        checkpoint = torch.load(args.load, map_location=device)
        
        # 检查是否是 Checkpoint 字典（包含 epoch, optimizer 等）
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            Logger.info("Loaded state_dict from checkpoint dictionary.")
        else:
            # 假设是纯粹的 state_dict (如 best_model.pt)
            model.load_state_dict(checkpoint)
            Logger.info("Loaded raw state_dict.")
            
    except Exception as e:
        Logger.info(f"Error loading model: {e}")
        raise e

    # [修改 3] 强制设置推理时的 Temperature
    # 训练结束时 temperature 为 5.0 (或者你设定的 end_temp)。
    # 测试时应使用这个值，以获得最清晰的 Mask。
    # 注意：如果 TSL 模块不在 module 下（单卡），代码会自动处理，但 DataParallel 下通常在 module 里
    inference_temp = 5.0
    if hasattr(model.module, 'tsl'):
         model.module.tsl.update_all_temperatures(inference_temp)
    elif hasattr(model, 'tsl'): # 兼容单卡情况
         model.tsl.update_all_temperatures(inference_temp)
    
    Logger.info(f"Set inference temperature to: {inference_temp}")

    # Helper classes initialization
    Evaluator.initialize()
    Visualizer.initialize(args.visualize, args.vispath)

    FSSDataset.initialize(img_size=400, datapath=args.datapath, use_original_imgsize=args.use_original_imgsize)
    dataloader_test = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val', args.nshot,
                                                  cam_train_path=args.traincampath, cam_val_path=args.valcampath)

    # Test AFANet
    with torch.no_grad():
        test_miou, test_fb_iou = test(model, dataloader_test, args.nshot, args.stage)

    Logger.info('Fold %d mIoU: %5.2f \t FB-IoU: %5.2f' % (args.fold, test_miou.item(), test_fb_iou.item()))
    Logger.info('==================== Finished Testing ====================')