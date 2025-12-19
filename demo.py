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
if __name__ == "__main__":
    # 测试 PlugAndPlayDCN 模块
    model = PlugAndPlayDCN(in_channels=64, out_channels=64)
    input_tensor = torch.randn(1, 64, 224, 224)  # 示例输入
    output = model(input_tensor)
    print(output.shape)  # 应该输出 torch.Size([1, 64, 224, 224])