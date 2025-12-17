import os
import torch
import clip
from PIL import Image
from pytorch_grad_cam import GradCAM
import cv2
import argparse
from data.dataset import FSSDataset

# 定义 VOC 类别
PASCAL_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor']

def get_cam_from_alldata(clip_model, preprocess, d=None, datapath=None, campath=None):
    # 自动创建不存在的文件夹
    os.makedirs(campath, exist_ok=True)

    dataset_all = d.dataset.img_metadata
    L = len(dataset_all)
    
    # 【优化1】提前计算 Prompt，不再在循环里重复算
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in PASCAL_CLASSES]).to(device)

    # 【优化2】GradCAM 初始化移出循环！(这是解决卡死和变慢的核心)
    target_layers = [clip_model.visual.layer4[-1]]
    cam = GradCAM(model=clip_model, target_layers=target_layers, use_cuda=True)
    
    print(f"--> 正在处理目录: {campath} (共 {L} 张)")

    for ll in range(L):
        # 打印进度条 (每100张显示一次)
        if ll % 100 == 0:
            print(f"Progress: [{ll}/{L}] ({(ll/L)*100:.1f}%)")

        filename = dataset_all[ll][0]
        # 【修复路径】使用 os.path.join 自动处理斜杠
        img_path = os.path.join(datapath, filename + '.jpg')
        
        if not os.path.exists(img_path):
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            img_input = preprocess(img).unsqueeze(0).to(device)
            class_name_id = dataset_all[ll][1]

            # 刷新一下 text features 状态
            clip_model.get_text_features(text_inputs)

            # 生成 CAM
            grayscale_cam = cam(input_tensor=img_input, target_category=class_name_id)
            grayscale_cam = grayscale_cam[0, :]
            
            # 缩放并保存
            grayscale_cam = cv2.resize(grayscale_cam, (50, 50))
            grayscale_cam = torch.from_numpy(grayscale_cam)
            
            # 【修复保存名】确保文件保存在文件夹内部
            save_name = f"{filename}--{class_name_id}.pt"
            save_path = os.path.join(campath, save_name)
            
            torch.save(grayscale_cam, save_path)
            # 这里我删除了 print，所以不会刷屏
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    print(f"Done! 目录 {campath} 完成。\n")


if __name__ == '__main__':
    # 【修复】使用 0 号显卡，避免报错
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    parser = argparse.ArgumentParser(description='IMR')
    # 【修复】默认路径改为你电脑上的真实路径
    parser.add_argument('--imgpath', type=str, default='/home/xhd/XD/datasets/AFANet_datasets/VOC2012/JPEGImages')
    parser.add_argument('--traincampath', type=str, default='/home/xhd/XD/datasets/AFANet_datasets/CAM_VOC_Train')
    parser.add_argument('--valcampath', type=str, default='/home/xhd/XD/datasets/AFANet_datasets/CAM_VOC_Val')
    parser.add_argument('--bsz', type=int, default=32)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)
    
    # 加载 CLIP
    model_clip, preprocess = clip.load('RN50', device, jit=False)
    
    # 初始化数据集
    FSSDataset.initialize(img_size=400, datapath='/home/xhd/XD/datasets/AFANet_datasets/', use_original_imgsize=False)

    # 处理 Train
    print("====== 开始处理 Train Set ======")
    for i in range(4):
        print(f"加载 Train Fold {i}...")
        dataloader = FSSDataset.build_dataloader('pascal', args.bsz, 1, i, 'train', 1)
        get_cam_from_alldata(model_clip, preprocess, d=dataloader, datapath=args.imgpath, campath=args.traincampath)

    # 处理 Val
    print("====== 开始处理 Val Set ======")
    for i in range(4):
        print(f"加载 Val Fold {i}...")
        dataloader = FSSDataset.build_dataloader('pascal', args.bsz, 1, i, 'val', 1)
        get_cam_from_alldata(model_clip, preprocess, d=dataloader, datapath=args.imgpath, campath=args.valcampath)

    print('All Done!')