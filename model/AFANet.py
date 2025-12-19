import pdb
from functools import reduce
from operator import add
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from torchvision.models import vgg
from torchvision.ops import deform_conv2d
import clip

# 请确保这些模块在他的文件中存在，或者根据你的目录结构调整引用
from .base.feature import extract_feat_vgg, extract_feat_res
from .base.correlation import Correlation
from .learner import HPNLearner
from model.afa_module import NeighborConnectionDecoder, BasicConv2d # Octave 被 DCN 替代，不需要了
from generate_cam_voc import PASCAL_CLASSES
from generate_cam_coco import COCO_CLASSES

# ==========================================================================================
# 1. [新增] 支持外部 Mask 注入的可变形卷积 DCNv2
# ==========================================================================================
class PlugAndPlayDCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super(PlugAndPlayDCN, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        
        # 核心卷积权重
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        
        # 预测 offset 和 mask (眼睛)
        self.p_conv = nn.Conv2d(in_channels, 3 * kernel_size * kernel_size, 
                                kernel_size=kernel_size, padding=padding, stride=stride)
        
        # 初始化：确保初始状态接近普通卷积
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.p_conv.weight)
        nn.init.zeros_(self.p_conv.bias)

    def forward(self, x, external_mask=None):
        # 1. 预测 Offset 和 内部 Mask
        out = self.p_conv(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat([o1, o2], dim=1)
        
        # DCN 自身学到的 Mask (0~1)
        mask = torch.sigmoid(mask)
        
        # 2. [关键] 注入 CAM 外部掩码
        if external_mask is not None:
            # 确保 external_mask 和 x 的尺寸一致 (插值对齐)
            if external_mask.shape[-2:] != x.shape[-2:]:
                external_mask = F.interpolate(external_mask, size=x.shape[-2:], 
                                              mode='bilinear', align_corners=True)
            
            # 广播机制：如果 CAM 说是背景(0)，则强制 DCN 也不关注该区域
            mask = mask * external_mask 
        
        return deform_conv2d(x, offset, self.weight, self.bias, 
                             stride=self.stride, padding=self.padding, 
                             mask=mask)

# ==========================================================================================
# 2. [新增] CoOp Prompt Learner (修复了 device 和硬编码问题)
# ==========================================================================================
class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, n_ctx=16):
        super().__init__()
        n_cls = len(classnames)
        ctx_dim = clip_model.ln_final.weight.shape[0]
        dtype = clip_model.dtype
        device = clip_model.token_embedding.weight.device 

        # 1. 初始化 Context
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors) 

        # 2. 处理 Class Prompts
        prompts = [name.replace("_", " ") for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]) 
        # 移动到正确设备
        tokenized_prompts = tokenized_prompts.to(device)
        
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :]) 
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.clip_model = clip_model 
        
        # 保存 tokenized 数据以便 forward 使用，避免重复 tokenize
        self.class_tokenized = tokenized_prompts

    def forward(self):
        ctx = self.ctx 
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1) 

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = torch.cat(
            [prefix, ctx, suffix], dim=1
        ) 

        # CLIP Transformer Encoding
        x = prompts + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)

        # 文本投影
        # 使用 init 中保存的 token，确保设备正确
        tokenized = self.class_tokenized.to(self.token_prefix.device)
        
        # 从 prompts 中取出对应类别的特征
        text_features = x[torch.arange(self.n_cls), tokenized.argmax(dim=-1)] @ self.clip_model.text_projection

        return text_features

# ==========================================================================================
# 3. AFANet 主模型
# ==========================================================================================
class afanet(nn.Module):

    def __init__(self, backbone, use_original_imgsize, benchmark, clip_model):
        super(afanet, self).__init__()

        # 1. Backbone Initialization
        self.backbone_type = backbone
        self.use_original_imgsize = use_original_imgsize
        
        if backbone == 'vgg16':
            self.backbone = vgg.vgg16(pretrained=False)
            # 请确认路径是否正确
            ckpt = torch.load('/home/xhd/XD/seg/AFANet_CoOp/pretrained/vgg16-397923af.pth', weights_only=True)
            self.backbone.load_state_dict(ckpt)
            self.feat_ids = [17, 19, 21, 24, 26, 28, 30]
            self.extract_feats = extract_feat_vgg
            nbottlenecks = [2, 2, 3, 3, 3, 1]
        elif backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=False)
            ckpt = torch.load('/home/xhd/XD/seg/AFANet/pretrained/resnet50-19c8e357.pth', map_location='cpu')
            self.backbone.load_state_dict(ckpt)
            self.feat_ids = list(range(4, 17))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 6, 3]
            self.conv1024_512 = nn.Conv2d(1024, 512, kernel_size=1)
        elif backbone == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=True)
            self.feat_ids = list(range(4, 34))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 23, 3]

        # 2. PromptLearner Initialization
        if benchmark == 'pascal':
            class_names = PASCAL_CLASSES
        elif benchmark == 'coco':
            class_names = COCO_CLASSES
        else:
            raise Exception('Unavailable benchmark')

        self.prompt_learner = PromptLearner(class_names, clip_model, n_ctx=16)
        
        # Freeze CLIP parameters
        for p in self.prompt_learner.clip_model.parameters():
            p.requires_grad = False

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]

        self.backbone.eval()
        self.hpn_learner = HPNLearner(list(reversed(nbottlenecks[-3:])))
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        # 3. [替换] 使用 PlugAndPlayDCN 替换原有的 Octave
        is_vgg_backbone = (backbone == 'vgg16')
        
        if is_vgg_backbone: 
             # VGG 通常只处理一种尺度或特定层
             self.vgg_fam = PlugAndPlayDCN(512, 64)
        else:
             # ResNet 多尺度处理
             self.fam_low = PlugAndPlayDCN(512, 64)
             self.fam_mid = PlugAndPlayDCN(1024, 64)
             self.fam_hig = PlugAndPlayDCN(2048, 64)

        self.ncd = NeighborConnectionDecoder()

        self.state = nn.Parameter(torch.zeros([1, 128, 50, 50]))
        self.convz0 = nn.Conv2d(769, 512, kernel_size=1, padding=0)

        self.ncd_convz0 = nn.Conv2d(385, 512, kernel_size=1, padding=0)

        self.convz1 = nn.Conv2d(256, 64, kernel_size=3, padding=1, groups=8, dilation=1)
        self.convz2 = nn.Conv2d(256, 64, kernel_size=3, padding=1, groups=8, dilation=1)

        self.convr0 = nn.Conv2d(769, 512, kernel_size=1, padding=0)

        self.ncd_convr0 = nn.Conv2d(385, 512, kernel_size=1, padding=0)

        self.convr1 = nn.Conv2d(256, 64, kernel_size=3, padding=1, groups=8, dilation=1)
        self.convr2 = nn.Conv2d(256, 64, kernel_size=3, padding=1, groups=8, dilation=1)

        self.ncd_convh0 = nn.Conv2d(385, 512, kernel_size=1, padding=0)

        self.convh0 = nn.Conv2d(769, 512, kernel_size=1, padding=0)
        self.convh1 = nn.Conv2d(256, 64, kernel_size=3, padding=1, groups=8, dilation=1)
        self.convh2 = nn.Conv2d(256, 64, kernel_size=3, padding=1, groups=8, dilation=1)

        outch1, outch2, outch3 = 16, 64, 128
        self.decoder1 = nn.Sequential(nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU())

        self.decoder2 = nn.Sequential(nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(outch2, 2, (3, 3), padding=(1, 1), bias=True))

        self.res = nn.Sequential(nn.Conv2d(3, 10, kernel_size=1),
                                 nn.GELU(),
                                 nn.Conv2d(10, 2, kernel_size=1))

        self.bn = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)

        self.linear_1024_625 = nn.Linear(1024, 625)

    def forward(self, query_img, support_img, support_cam, query_cam,
                query_mask=None, support_mask=None, stage=2, w='same', class_id=None):
        
        # 1. Feature Extraction
        with torch.no_grad():
            query_feats = self.extract_feats(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats = self.extract_feats(support_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)

        # 2. CoOp Text Features Generation
        text_features_raw = self.prompt_learner() 
        clip_text_features = self.linear_1024_625(text_features_raw.float())
        clip_text_features = clip_text_features[class_id].unsqueeze(1) 
        batch, _, _ = clip_text_features.size()
        clip_text_features = clip_text_features.view(batch, 1, 25, 25)

        # 3. Prepare Features for FAM (DCN)
        if len(query_feats) == 7:  # VGG Backbone
            isvgg = True
            q_mid_feat = F.interpolate(query_feats[3] + query_feats[4] + query_feats[5],
                                        (50, 50), mode='bilinear', align_corners=True)
            s_mid_feat = F.interpolate(support_feats[3] + support_feats[4] + support_feats[5],
                                        (50, 50), mode='bilinear', align_corners=True)

            support_low = support_feats[1]
            support_mid = support_feats[5]
            support_hig = support_feats[6]

            query_low = query_feats[1]
            query_mid = query_feats[5]
            query_hig = query_feats[6]

        else:
            isvgg = False  # ResNet-50 Backbone
            support_low = support_feats[3]
            support_mid = support_feats[9]
            support_hig = support_feats[12]

            query_low = query_feats[3]
            query_mid = query_feats[9]
            query_hig = query_feats[12]

        # =====================================================================
        # [修复] 准备外部 Mask (CAM)，必须在传入 DCN 之前定义
        # =====================================================================
        # 确保维度为 [B, 1, H, W]
        s_cam_mask = support_cam.unsqueeze(1) if support_cam.dim() == 3 else support_cam
        q_cam_mask = query_cam.unsqueeze(1) if query_cam.dim() == 3 else query_cam
        
        # 4. Feature Adaptation Module (FAM) using DCN
        if isvgg == True: # VGG
            support_fam_low = self.vgg_fam(support_low, external_mask=s_cam_mask)
            support_fam_mid = self.vgg_fam(support_mid, external_mask=s_cam_mask)
            support_fam_hig = self.vgg_fam(support_hig, external_mask=s_cam_mask)

            query_fam_low = self.vgg_fam(query_low, external_mask=q_cam_mask)
            query_fam_mid = self.vgg_fam(query_mid, external_mask=q_cam_mask)
            query_fam_hig = self.vgg_fam(query_hig, external_mask=q_cam_mask)

        else: # ResNet
            support_fam_low = self.fam_low(support_low, external_mask=s_cam_mask)
            support_fam_mid = self.fam_mid(support_mid, external_mask=s_cam_mask)
            support_fam_hig = self.fam_hig(support_hig, external_mask=s_cam_mask)

            query_fam_low = self.fam_low(query_low, external_mask=q_cam_mask)
            query_fam_mid = self.fam_mid(query_mid, external_mask=q_cam_mask)
            query_fam_hig = self.fam_hig(query_hig, external_mask=q_cam_mask)
        
        # 5. Neighbor Connection Decoder (NCD)
        support_ncd_feats = self.ncd(support_fam_low, support_fam_mid, support_fam_hig)
        query_ncd_feats = self.ncd(query_fam_low, query_fam_mid, query_fam_hig)

        support_ncd_feats = self.bn(support_ncd_feats)
        support_ncd_feats = self.relu(support_ncd_feats)

        query_ncd_feats = self.bn(query_ncd_feats)
        query_ncd_feats = self.relu(query_ncd_feats)

        with torch.no_grad():
            s_ncd_reshape = F.interpolate(support_ncd_feats, scale_factor=0.5, mode='bilinear')
            s_multimodal = torch.mul(s_ncd_reshape, clip_text_features)
            s_multimodal = F.interpolate(s_multimodal, scale_factor=2, mode='bilinear')

            q_ncd_reshape = F.interpolate(query_ncd_feats, scale_factor=0.5, mode='bilinear')
            q_multimodal = torch.mul(q_ncd_reshape, clip_text_features)
            q_multimodal = F.interpolate(q_multimodal, scale_factor=2, mode='bilinear')

        query_feats_masked = self.mask_feature(query_feats, query_cam.clone())
        support_feats_masked = self.mask_feature(support_feats, support_cam.clone())

        corr_query = Correlation.multilayer_correlation(query_feats, support_feats_masked, self.stack_ids)
        corr_support = Correlation.multilayer_correlation(support_feats, query_feats_masked, self.stack_ids)

        # 这里的 cam 之前已经被 unsqueeze 处理过了吗？
        # 原代码有这行，保留以防万一，但注意上面已经定义了 s_cam_mask
        # 这里对 query_cam, support_cam 重新赋值用于后面的 concat
        query_cam = query_cam.unsqueeze(1) if query_cam.dim() == 3 else query_cam
        support_cam = support_cam.unsqueeze(1) if support_cam.dim() == 3 else support_cam

        bsz = query_img.shape[0]

        with torch.no_grad():
            mfa_state_query = self.state.expand(bsz, -1, -1, -1)
            mfa_state_support = self.state.expand(bsz, -1, -1, -1)

        losses = 0
        
        # 6. Iterative Multi-Resolution (IMR) Loop
        for ss in range(stage): 

            # mfa query
            after4d_query = self.hpn_learner.forward_conv4d(corr_query)

            imr_x_query = torch.cat([query_cam, after4d_query, q_multimodal, mfa_state_query], dim=1)

            imr_x_query_z = self.ncd_convz0(imr_x_query)
            imr_z_query1 = self.convz1(imr_x_query_z[:, :256])
            imr_z_query2 = self.convz2(imr_x_query_z[:, 256:])

            imr_z_query = torch.sigmoid(torch.cat([imr_z_query1, imr_z_query2], dim=1))

            imr_x_query_r = self.ncd_convr0(imr_x_query)
            imr_r_query1 = self.convr1(imr_x_query_r[:, :256])
            imr_r_query2 = self.convr2(imr_x_query_r[:, 256:])
            imr_r_query = torch.sigmoid(torch.cat([imr_r_query1, imr_r_query2], dim=1))

            imr_x_query_h = self.ncd_convh0(
                torch.cat([query_cam, after4d_query, q_multimodal, imr_r_query * mfa_state_query], dim=1))
            imr_h_query1 = self.convh1(imr_x_query_h[:, :256])
            imr_h_query2 = self.convh2(imr_x_query_h[:, 256:])
            imr_h_query = torch.cat([imr_h_query1, imr_h_query2], dim=1)

            state_new_query = torch.tanh(imr_h_query)
            mfa_state_query = (1 - imr_z_query) * mfa_state_query + imr_z_query * state_new_query

            # MFA support
            after4d_support = self.hpn_learner.forward_conv4d(corr_support)
            imr_x_support = torch.cat([support_cam, after4d_support, s_multimodal, mfa_state_support], dim=1)

            imr_x_support_z = self.ncd_convz0(imr_x_support)
            imr_z_support1 = self.convz1(imr_x_support_z[:, :256])
            imr_z_support2 = self.convz2(imr_x_support_z[:, 256:])
            imr_z_support = torch.sigmoid(torch.cat([imr_z_support1, imr_z_support2], dim=1))

            imr_x_support_r = self.ncd_convr0(imr_x_support)
            imr_r_support1 = self.convr1(imr_x_support_r[:, :256])
            imr_r_support2 = self.convr2(imr_x_support_r[:, 256:])
            imr_r_support = torch.sigmoid(torch.cat([imr_r_support1, imr_r_support2], dim=1))

            imr_x_support_h = self.ncd_convh0(
                torch.cat([support_cam, after4d_support, s_multimodal, imr_r_support * mfa_state_support], dim=1))
            imr_h_support1 = self.convh1(imr_x_support_h[:, :256])
            imr_h_support2 = self.convh2(imr_x_support_h[:, 256:])
            imr_h_support = torch.cat([imr_h_support1, imr_h_support2], dim=1)

            state_new_support = torch.tanh(imr_h_support)
            mfa_state_support = (1 - imr_z_support) * mfa_state_support + imr_z_support * state_new_support

            # decoder
            hypercorr_decoded_s = self.decoder1(mfa_state_support + after4d_support)

            upsample_size = (hypercorr_decoded_s.size(-1) * 2,) * 2  # (100, 100)
            hypercorr_decoded_s = F.interpolate(hypercorr_decoded_s, upsample_size, mode='bilinear', align_corners=True)
            logit_mask_support = self.decoder2(hypercorr_decoded_s)  # mask s  ([8, 2, 100, 100])

            hypercorr_decoded_q = self.decoder1(mfa_state_query + after4d_query)

            upsample_size = (hypercorr_decoded_q.size(-1) * 2,) * 2
            hypercorr_decoded_q = F.interpolate(hypercorr_decoded_q, upsample_size, mode='bilinear', align_corners=True)
            logit_mask_query = self.decoder2(hypercorr_decoded_q)

            logit_mask_support = self.res(
                torch.cat(
                    [logit_mask_support, F.interpolate(support_cam, (100, 100), mode='bilinear', align_corners=True)],
                    dim=1))
            logit_mask_query = self.res(
                torch.cat([logit_mask_query, F.interpolate(query_cam, (100, 100), mode='bilinear', align_corners=True)],
                          dim=1))

            # loss computation
            if query_mask is not None:  # for training
                if not self.use_original_imgsize:
                    logit_mask_query_temp = F.interpolate(logit_mask_query, support_img.size()[2:], mode='bilinear',
                                                          align_corners=True)
                    logit_mask_support_temp = F.interpolate(logit_mask_support, support_img.size()[2:], mode='bilinear',
                                                            align_corners=True)
                loss_q_stage = self.compute_objective(logit_mask_query_temp, query_mask)
                loss_s_stage = self.compute_objective(logit_mask_support_temp, support_mask)
                losses = losses + loss_q_stage + loss_s_stage

            if ss != stage - 1:
                support_cam = logit_mask_support.softmax(dim=1)[:, 1]
                query_cam = logit_mask_query.softmax(dim=1)[:, 1]
                query_feats_masked = self.mask_feature(query_feats, query_cam)
                support_feats_masked = self.mask_feature(support_feats, support_cam)
                corr_query = Correlation.multilayer_correlation(query_feats, support_feats_masked, self.stack_ids)
                corr_support = Correlation.multilayer_correlation(support_feats, query_feats_masked, self.stack_ids)

                query_cam = F.interpolate(query_cam.unsqueeze(1), (50, 50), mode='bilinear', align_corners=True)
                support_cam = F.interpolate(support_cam.unsqueeze(1), (50, 50), mode='bilinear', align_corners=True)

        if query_mask is not None:
            return logit_mask_query_temp, logit_mask_support_temp, losses
        else:
            # test
            if not self.use_original_imgsize:
                logit_mask_query = F.interpolate(
                    logit_mask_query, support_img.size()[2:], mode='bilinear', align_corners=True)
                logit_mask_support = F.interpolate(
                    logit_mask_support, support_img.size()[2:], mode='bilinear', align_corners=True)
            return logit_mask_query, logit_mask_support

    def mask_feature(self, features, support_mask):
        for idx, feature in enumerate(features):
            mask = F.interpolate(
                support_mask.unsqueeze(1).float(), feature.size()[2:], mode='bilinear', align_corners=True)
            features[idx] = features[idx] * mask
        return features

    def predict_mask_nshot(self, batch, nshot, class_id, stage):
        # Perform multiple prediction given (nshot) number of different support sets
        logit_mask_agg = 0
        for s_idx in range(nshot):

            logit_mask, logit_mask_s = self(query_img=batch['query_img'],
                                            support_img=batch['support_imgs'][:, s_idx],
                                            support_cam=batch['support_cams'][:, s_idx],
                                            query_cam=batch['query_cam'],
                                            class_id=batch['class_id'], stage=stage)
            if self.use_original_imgsize:
                org_qry_imsize = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
                logit_mask = F.interpolate(logit_mask, org_qry_imsize, mode='bilinear', align_corners=True)

            logit_mask_agg += logit_mask.argmax(dim=1).clone()
            if nshot == 1:
                return logit_mask_agg

        # Average & quantize predictions given threshold (=0.5)
        bsz = logit_mask_agg.size(0)
        max_vote = logit_mask_agg.view(bsz, -1).max(dim=1)[0]
        max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
        max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
        pred_mask = logit_mask_agg.float() / max_vote
        pred_mask[pred_mask < 0.5] = 0
        pred_mask[pred_mask >= 0.5] = 1

        return pred_mask

    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).long()
        return self.cross_entropy_loss(logit_mask, gt_mask)

    def train_mode(self):
        self.train()
        self.backbone.eval()  # to prevent BN from learning data statistics with exponential averaging