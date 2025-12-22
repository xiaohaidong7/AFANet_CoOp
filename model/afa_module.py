import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import clip
from torchvision.ops import deform_conv2d

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, 
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes) 
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates]#format with class
            texts = clip.tokenize(texts).to(device) # tokenize
            class_embeddings = model.encode_text(texts) # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights.t()

# 受CAM影响的可变形卷积========================================================================================================================
class PlugAndPlayDCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super(PlugAndPlayDCN, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        
        # 1. 核心卷积权重
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        
        # 2. 预测 offset 和 mask
        self.p_conv = nn.Conv2d(in_channels, 3 * kernel_size * kernel_size, 
                                kernel_size=kernel_size, padding=padding, stride=stride)

        # [新增] 定义 offset 的最大偏移限制 (通常设为 kernel_size 左右)
        self.max_offset_limit = float(kernel_size)
        
        # 初始化
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.p_conv.weight)
        nn.init.zeros_(self.p_conv.bias)

    def forward(self, x, external_mask=None):
        # x: [B, C, H, W]
        # external_mask: [B, 1, H, W] (比如 CAM)
        
        # 1. 预测 Offset 和 内部 Mask
        out = self.p_conv(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat([o1, o2], dim=1)
        
        # [优化1] 限制 Offset 范围，增加稳定性
        # 将偏移限制在 [-k, k] 之间，防止训练初期梯度爆炸导致的采样点飞离
        offset = torch.tanh(offset) * self.max_offset_limit

        # DCN 自身学到的 Mask (0~1)
        mask = torch.sigmoid(mask)
        
        # === [关键修改] 注入 CAM 掩码 ===
        if external_mask is not None:
            # 确保 external_mask 和 x 的尺寸一致 (需要插值)
            if external_mask.shape[-2:] != x.shape[-2:]:
                external_mask = F.interpolate(external_mask, size=x.shape[-2:], 
                                              mode='bilinear', align_corners=True)
            
            # 广播机制：[B, 1, H, W] * [B, 9, H, W] -> [B, 9, H, W]
            # 逻辑：如果 CAM 说是背景(0)，那么无论 DCN 想看哪里，最终权重都被置为 0
            # mask = mask * external_mask 
            mask = mask + 0.5 * external_mask  # 软注入，保留部分 DCN 自身信息
        # ==============================
        
        return deform_conv2d(x, offset, self.weight, self.bias, 
                             stride=self.stride, padding=self.padding, 
                             mask=mask)
# ========================================================================================================================

# 稀疏融合========================================================================================================================


class LearnableResidualSparseBlock(nn.Module):
    def __init__(self, in_channels, init_ratio=0.5, temp=5.0):
        """
        Args:
            in_channels: 输入通道
            init_ratio: 期望初始保留的比例 (0.0 ~ 1.0)。
                        例如 0.6 表示初始状态下，当输入特征为平均值时，Mask输出约为 0.6。
            temp: 温度系数，控制 Mask 的陡峭程度。
        """
        super(LearnableResidualSparseBlock, self).__init__()
        
        # 1. 显著性特征提取
        self.salience_conv = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1) # BN 保证 scores 分布在 0 附近，方差为 1
        )
        
        self.temp = temp

        # 2. 自动计算阈值的初始值
        # 数学推导: Sigmoid( (0 - threshold) * temp ) = init_ratio
        # => threshold = - (ln(ratio / (1-ratio))) / temp
        # 限制 ratio 防止 log(0)
        eps = 1e-6
        init_ratio = max(eps, min(1 - eps, init_ratio))
        init_logit = math.log(init_ratio / (1 - init_ratio))
        init_thresh = -init_logit / temp
        
        self.threshold = nn.Parameter(torch.tensor(init_thresh))
        
        # 3. 增强权重
        self.enhance_weight = nn.Parameter(torch.tensor(1.0))
        
        # 4. 用于可视化/调试的缓存变量
        self.last_mask = None 

    def forward(self, x):
        # x: [B, C, H, W]
        
        # [B, 1, H, W]
        scores_logit = self.salience_conv(x)
        
        # 生成 Mask
        # (Score - Threshold) * Temp
        diff = (scores_logit - self.threshold) * self.temp
        mask = torch.sigmoid(diff)
        
        # 缓存 Mask 用于可视化分析 (Detach 防止影响梯度)
        if not self.training:
            self.last_mask = mask.detach()
        
        # 残差增强
        out = x + (x * mask * self.enhance_weight)
        
        return out

    def update_temperature(self, new_temp):
        """在训练过程中调用此函数可实现退火策略"""
        self.temp = new_temp

class TriScaleLearnableFusion(nn.Module):
    def __init__(self, channels=64, out_channels=128):
        super(TriScaleLearnableFusion, self).__init__()
        
        # 1. 稀疏增强 (清洗)
        self.enhance_s = LearnableResidualSparseBlock(channels, init_ratio=0.7) # Small/Deep
        self.enhance_m = LearnableResidualSparseBlock(channels, init_ratio=0.6) # Mid
        self.enhance_l = LearnableResidualSparseBlock(channels, init_ratio=0.5) # Large/Shallow
        
        # 2. 交互对齐层 (新增)
        self.conv_align_s2m = BasicConv2d(channels, channels, 3, padding=1)
        self.conv_align_m2l = BasicConv2d(channels, channels, 3, padding=1)
        
        # 3. 融合层
        self.fusion_conv = nn.Sequential(
            BasicConv2d(channels * 3, out_channels, 3, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1) 
        )

    def forward(self, x1, x2, x3):
        # x1: Small (Deep), x2: Mid, x3: Large (Shallow)
        
        # Step 1: 稀疏清洗
        x_s = self.enhance_s(x1)
        x_m = self.enhance_m(x2)
        x_l = self.enhance_l(x3)
        
        # Step 2: Top-Down 语义引导 (避免硬编码尺寸)
        # Deep (Small) 指导 Mid
        target_size_m = x_m.shape[2:] 
        x_s_up = F.interpolate(x_s, size=target_size_m, mode='bilinear', align_corners=True)
        x_m_gated = x_m * self.conv_align_s2m(x_s_up) # 乘法交互
        
        # Mid 指导 Large
        target_size_l = x_l.shape[2:]
        x_m_up = F.interpolate(x_m_gated, size=target_size_l, mode='bilinear', align_corners=True)
        x_l_gated = x_l * self.conv_align_m2l(x_m_up) # 乘法交互
        
        # Step 3: 最终融合
        x_s_final = F.interpolate(x_s, size=target_size_l, mode='bilinear', align_corners=True)
        x_m_final = x_m_up 
        x_l_final = x_l_gated
        
        cat = torch.cat([x_s_final, x_m_final, x_l_final], dim=1)
        return self.fusion_conv(cat)

    # [新增] 统一更新内部三个稀疏块的温度    
    def update_all_temperatures(self, new_temp):
        self.enhance_s.update_temperature(new_temp)
        self.enhance_m.update_temperature(new_temp)
        self.enhance_l.update_temperature(new_temp)
    # ========================================================================================================================