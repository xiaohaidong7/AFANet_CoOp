#文本--->热力图处理---------------------------------------------------------------------------------------
python generate_cam_voc.py --traincampath /home/xhd/XD/datasets/AFANet_datasets/CAM_VOC_Train
                           --valcampath /home/xhd/XD/datasets/AFANet_datasets/CAM_VOC_Val
#文本--->热力图处理---------------------------------------------------------------------------------------


#训练脚本-------------------------------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python train.py \
  --backbone vgg16 \
  --fold 3 \
  --benchmark pascal \
  --bsz 8 \
  --lr 4e-4 \
  --lr_prompt 0.002 \
  --niter 50 \
  --stage 2 \
  --traincampath /home/xhd/XD/datasets/AFANet_datasets/CAM_VOC_Train/ \
  --valcampath /home/xhd/XD/datasets/AFANet_datasets/CAM_VOC_Val/ \
  --datapath /home/xhd/XD/datasets/AFANet_datasets/
#训练脚本-------------------------------------------------------------------------------------------------



#验证脚本-------------------------------------------------------------------------------------------------
python test.py \
  --backbone vgg16 \
  --fold 2 \
  --benchmark pascal \
  --nshot 5 \
  --load logs/TRAIN/VOC/pascal_vgg16_fold2_1231/best_model.pt \
  --datapath /home/xhd/XD/datasets/AFANet_datasets/ \
  --traincampath /home/xhd/XD/datasets/AFANet_datasets/CAM_VOC_Train/ \
  --valcampath /home/xhd/XD/datasets/AFANet_datasets/CAM_VOC_Val/
  # --vispath ./vis_results/
#验证脚本-------------------------------------------------------------------------------------------------

#测试demo脚本-------------------------------------------------------------------------------------------------
python train.py \
  --backbone vgg16 \
  --fold 0 \
  --benchmark pascal \
  --bsz 4 \
  --lr 4e-4 \
  --lr_prompt 0.002 \
  --niter 50 \
  --stage 2 \
  --logpath "test" \
  --traincampath /home/xhd/XD/datasets/AFANet_datasets/CAM_VOC_Train/ \
  --valcampath /home/xhd/XD/datasets/AFANet_datasets/CAM_VOC_Val/ \
  --datapath /home/xhd/XD/datasets/AFANet_datasets/
#测试demo脚本-------------------------------------------------------------------------------------------------