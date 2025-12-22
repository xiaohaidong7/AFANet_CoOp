import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d
import math

class PlugAndPlayDCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super(PlugAndPlayDCN, self).__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        
        # 1. 核心卷积层：最终用于处理采样后的特征
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        
        # 2. Offset & Mask Predictor
        # 通道数计算：每个点 2个offset + 1个mask，共 3 * K^2
        # 我们用一个 3x3 卷积来预测这些空间信息
        self.p_conv = nn.Conv2d(in_channels, 3 * kernel_size * kernel_size, 
                                kernel_size=kernel_size, padding=padding, stride=stride)
        
        # 初始化设置
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.p_conv.weight)
        nn.init.zeros_(self.p_conv.bias)

    def forward(self, x):
        # 预测 offset 和 mask
        out = self.p_conv(x)
        
        # 分离 offset 和 mask
        # 假设前 2*K*K 是 offset，后 K*K 是 mask
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat([o1, o2], dim=1)
        
        # Mask 需要通过 sigmoid 限制在 0~1 之间
        mask = torch.sigmoid(mask)
        
        # 使用 torchvision 的高效实现
        return deform_conv2d(x, offset, self.weight, self.bias, 
                             stride=self.stride, padding=self.padding, 
                             mask=mask)

class LearnableResidualSparseBlock(nn.Module):
    def __init__(self, in_channels, init_ratio=0.5, temp=5.0):
        """
        可学习比例的残差稀疏增强模块
        Args:
            in_channels: 输入通道数
            init_ratio: 初始的保留比例 (0~1)，初始化阈值用
            temp: 温度系数，控制 Mask 的软硬程度。
                  值越大，Mask 越接近 0/1 (二值化)；值越小，Mask 越平滑。
        """
        super(LearnableResidualSparseBlock, self).__init__()
        
        # 1. 显著性特征提取 (生成 Score Map)
        self.salience_conv = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1) # 加 BN 归一化，让 scores 分布更稳定
        )
        
        # 2. 可学习的阈值参数 (Learnable Threshold)
        # 初始化为 0.0 (对应 Sigmoid 后的 0.5)
        # 这里的 threshold 是在 Logit 层面操作的
        self.threshold = nn.Parameter(torch.tensor(0.0))
        
        # 3. 温度系数
        self.temp = temp

        # 4. 可学习的增强强度 (Scale)
        # 允许网络学习 mask 部分叠加的权重，初始设为 1.0
        self.enhance_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # x: [B, C, H, W]
        
        # --- 1. 计算显著性分数 (Raw Scores) ---
        # scores_logit: [B, 1, H, W]
        scores_logit = self.salience_conv(x)
        
        # --- 2. 生成动态 Mask (Differentiable) ---
        # 核心公式: Mask = Sigmoid( (Score - Threshold) * Temperature )
        # 如果 Score > Threshold，结果趋向 1；反之趋向 0。
        # 这是一个可微分的过程，梯度可以传回 self.threshold
        diff = scores_logit - self.threshold
        mask = torch.sigmoid(diff * self.temp)
        
        # --- 3. 残差增强 (Residual Enhancement) ---
        # 原始特征 + (原始特征 * 掩码 * 强度)
        # 这样，mask 为 0 的地方保持原样，mask 为 1 的地方特征值被放大
        out = x + (x * mask * self.enhance_weight)
        
        return out

class TriScaleLearnableFusion(nn.Module):
    def __init__(self, channels=64, out_channels=128):
        super(TriScaleLearnableFusion, self).__init__()
        
        # 为每个尺度初始化不同的比例
        # 小图语义强，可能只需要聚焦核心，初始阈值设高一点
        # 大图背景多，但也可能包含微小细节，让网络自己去学
        self.enhance_s = LearnableResidualSparseBlock(channels, init_ratio=0.6)
        self.enhance_m = LearnableResidualSparseBlock(channels, init_ratio=0.6)
        self.enhance_l = LearnableResidualSparseBlock(channels, init_ratio=0.6)
        
        # 融合层
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2, x3):
        """
        x1: [B, 64, 13, 13]
        x2: [B, 64, 25, 25]
        x3: [B, 64, 50, 50]
        """
        
        # 1. 独立的分支内残差增强
        x1_enh = self.enhance_s(x1)
        x2_enh = self.enhance_m(x2)
        x3_enh = self.enhance_l(x3)
        
        # 2. 统一尺度到 50x50
        target_size = x3.size()[2:] 
        
        x1_up = F.interpolate(x1_enh, size=target_size, mode='bilinear', align_corners=True)
        x2_up = F.interpolate(x2_enh, size=target_size, mode='bilinear', align_corners=True)
        x3_up = x3_enh 
        
        # 3. 拼接
        cat = torch.cat([x1_up, x2_up, x3_up], dim=1) # [B, 192, 50, 50]
        
        # 4. 融合压缩
        out = self.fusion_conv(cat) # [B, 128, 50, 50]
        
        return out

# --- 测试与验证 ---
if __name__ == "__main__":
    model = TriScaleLearnableFusion()
    
    # 打印可学习参数，确认 threshold 存在
    print("检查可学习参数:")
    for name, param in model.named_parameters():
        if "threshold" in name:
            print(f"{name}: {param.item()}")
            
    x1 = torch.randn(2, 64, 13, 13)
    x2 = torch.randn(2, 64, 25, 25)
    x3 = torch.randn(2, 64, 50, 50)
    
    out = model(x1, x2, x3)
    print(f"\n输出尺寸: {out.shape}")
# if __name__ == "__main__":
#     # 测试 PlugAndPlayDCN 模块
#     model = PlugAndPlayDCN(in_channels=64, out_channels=64)
#     input_tensor = torch.randn(1, 64, 224, 224)  # 示例输入
#     output = model(input_tensor)
#     print(output.shape)  # 应该输出 torch.Size([1, 64, 224, 224])