r""" Logging during training/testing """
import datetime
import logging
import os

from tensorboardX import SummaryWriter
import torch


class AverageMeter:
    r""" Stores loss, evaluation results """
    def __init__(self, dataset):
        self.benchmark = dataset.benchmark
        self.class_ids_interest = dataset.class_ids
        self.class_ids_interest = torch.tensor(self.class_ids_interest).cuda()

        if self.benchmark == 'pascal':
            self.nclass = 20
        elif self.benchmark == 'coco':
            self.nclass = 80
        elif self.benchmark == 'fss':
            self.nclass = 1000

        self.intersection_buf = torch.zeros([2, self.nclass]).float().cuda()
        self.union_buf = torch.zeros([2, self.nclass]).float().cuda()
        self.ones = torch.ones_like(self.union_buf)
        self.loss_buf = []

    def update(self, inter_b, union_b, class_id, loss):
        self.intersection_buf.index_add_(1, class_id, inter_b.float())  
        self.union_buf.index_add_(1, class_id, union_b.float())
        if loss is None:
            loss = torch.tensor(0.0)
        self.loss_buf.append(loss)

    def compute_iou(self):
        iou = self.intersection_buf.float() / \
              torch.max(torch.stack([self.union_buf, self.ones]), dim=0)[0]   #
        iou = iou.index_select(1, self.class_ids_interest)
        miou = iou[1].mean() * 100   

        fb_iou = (self.intersection_buf.index_select(1, self.class_ids_interest).sum(dim=1) /
                  self.union_buf.index_select(1, self.class_ids_interest).sum(dim=1)).mean() * 100

        return miou, fb_iou

    def write_result(self, split, epoch):
        iou, fb_iou = self.compute_iou()

        loss_buf = torch.stack(self.loss_buf)
        msg = '\n*** %s ' % split
        msg += '[@Epoch %02d] ' % epoch
        msg += 'Avg L: %6.5f   ' % loss_buf.mean()
        msg += 'mIoU: %5.2f    ' % iou
        msg += 'FB-IoU: %5.2f    ' % fb_iou

        msg += '***\n'
        Logger.info(msg)

    def write_process(self, batch_idx, datalen, epoch, write_batch_idx=20):
        if batch_idx % write_batch_idx == 0:
            msg = '[Epoch: %02d] ' % epoch if epoch != -1 else ''
            msg += '[Batch: %04d/%04d] ' % (batch_idx+1, datalen)
            iou, fb_iou = self.compute_iou()
            if epoch != -1:
                loss_buf = torch.stack(self.loss_buf)
                msg += 'L: %6.5f   ' % loss_buf[-1]
                msg += 'Avg L: %6.5f   ' % loss_buf.mean()
            msg += 'mIoU: %5.2f   |   ' % iou
            msg += 'FB-IoU: %5.2f' % fb_iou
            Logger.info(msg)


class Logger:
    r""" Writes evaluation results of training/testing """
    @classmethod
    def initialize(cls, args, training):
        # [关键修改] 必须先定义日期，否则训练模式下无法读取到 date_str
        date_str = datetime.datetime.now().__format__('%m%d')

        # 1. 确定父目录和子文件夹名称
        if training:
            # =========== [训练模式] ===========
            # 根据数据集区分父目录: ./logs/TRAIN/VOC 或 ./logs/TRAIN/COCO
            parent_dir = './logs/TRAIN/VOC' if args.benchmark == 'pascal' else './logs/TRAIN/COCO'
            
            # 如果命令行未指定 --logpath，默认按照格式命名
            if args.logpath == '':
                # 格式: pascal_resnet50_fold1_1227
                folder_name = f'{args.benchmark}_{args.backbone}_fold{args.fold}_{date_str}'
            else:
                folder_name = args.logpath
                
        else:
            # =========== [测试/验证模式] ===========
            # 根据数据集区分父目录: ./logs/TEST/VOC 或 ./logs/TEST/COCO
            parent_dir = './logs/TEST/VOC' if args.benchmark == 'pascal' else './logs/TEST/COCO'
            
            # 格式: 1shot_pascal_resnet50_fold1_1227
            folder_name = f'{args.nshot}shot_{args.benchmark}_{args.backbone}_fold{args.fold}_{date_str}'

        # 2. 拼接生成完整的路径
        # 例如: ./logs/TEST/VOC/1shot_pascal_resnet50_fold1_1227
        cls.logpath = os.path.join(parent_dir, folder_name)
        
        cls.benchmark = args.benchmark
        
        # 3. 递归创建文件夹
        os.makedirs(cls.logpath, exist_ok=True)

        # 4. 配置日志文件 (log.txt)
        logging.basicConfig(filemode='w',
                            filename=os.path.join(cls.logpath, 'log.txt'),
                            level=logging.INFO,
                            format='%(message)s',
                            datefmt='%m-%d %H:%M:%S')

        # Console log config
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        # Tensorboard writer
        cls.tbd_writer = SummaryWriter(os.path.join(cls.logpath, 'tbd/runs'))

        # Log arguments
        logging.info('\n:=========== Few-shot Seg. with AFANet ===========')
        for arg_key in args.__dict__:
            logging.info('| %20s: %-24s' % (arg_key, str(args.__dict__[arg_key])))
        logging.info(':================================================\n')

    @classmethod
    def info(cls, msg):
        r""" Writes log message to log.txt """
        logging.info(msg)

    @classmethod
    def save_model_miou(cls, model, epoch, val_miou):
        torch.save(model.state_dict(), os.path.join(cls.logpath, 'best_model.pt'))
        cls.info('Model saved @%d w/ val. mIoU: %5.2f.\n' % (epoch, val_miou))

    @classmethod
    def log_params(cls, model):
        backbone_param = 0
        learner_param = 0
        for k in model.state_dict().keys():
            n_param = model.state_dict()[k].numel()
            if k.split('.')[0] in 'backbone':
                if k.split('.')[1] in ['classifier', 'fc']:  # as fc layers are not used in AFANet
                    continue
                backbone_param += n_param
            else:
                learner_param += n_param
        Logger.info('Backbone # param.: %d' % backbone_param)
        Logger.info('Learnable # param.: %d' % learner_param)
        Logger.info('Total # param.: %d' % (backbone_param + learner_param))