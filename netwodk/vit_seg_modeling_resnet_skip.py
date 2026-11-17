import math

from os.path import join as pjoin
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义Mish
class Mish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

mish =Mish()

def np2th(weights, conv=False): #通过函数将numpy数组转化为pytorch张量
    """Possibly convert HWIO to OIHW."""
    if conv: #当 conv=True 时，函数会将权重从 HWIO 格式转换为 OIHW 格式
        weights = weights.transpose([3, 2, 0, 1]) #[高度, 宽度, 输入通道数, 输出通道数] 变为 [输出通道数, 输入通道数, 高度, 宽度]
    return torch.from_numpy(weights) #torch.from_numpy() 将NumPy数组转换为PyTorch张量


class StdConv2d(nn.Conv2d): #继承nn.Conv2d类，并添加forward方法

    def forward(self, x):
        w = self.weight # 获取权重，从父类继承的卷积核权重，形状为 [out_channels, in_channels, kernel_height, kernel_width] 对应 （cin, cout，卷积核高，卷积核宽）
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False) ## 计算权重的均值和方差，dim=[1, 2, 3]: 沿着输入通道、高度和宽度维度计算（即对每个输出通道的所有权重计算），keepdim=True: 保持维度，结果形状为 [out_channels, 1, 1, 1]
        w = (w - m) / torch.sqrt(v + 1e-5) # 标准化权重： (权重 - 均值) / 标准差    （上述的 v ,m 形状都为 [out_channels, 1, 1, 1]）
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups) # 使用标准化的权重进行卷积操作


def conv3x3(cin, cout, stride=1, groups=1, bias=False): # 创建一个3x3的卷积层，输入通道数为cin，输出通道数为cout，步长为stride，组数为groups，是否使用偏置为bias
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups) #图像输出大小 保持不变


def conv1x1(cin, cout, stride=1, bias=False): # 创建一个1x1的卷积层，输入通道数为cin，输出通道数为cout，步长为stride，是否使用偏置为bias
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module): # 预激活(Pre-activation)机制的Bottleneck残差块
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):# 初始化函数，输入参数包括：输入通道数cin，输出通道数cout，中间通道数cmid，步长stride
        super().__init__()
        cout = cout or cin # 如果输出通道数为None，则将其设置为输入通道数
        cmid = cmid or cout//4 # 如果中间通道数为None，则将其设置为输出通道数的1/4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6) # 创建一个GroupNorm层，分组数为32，输入通道数为cmid，输出通道数为cmid，eps为1e-6
        self.conv1 = conv1x1(cin, cmid, bias=False) #大小不变
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6) # 创建一个GroupNorm层，分组数为32，输入通道数为cmid，输出通道数为cmid，eps为1e-6
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!! 原始代码中这一步为 conv1 !!! 图像输出大小 保持不变
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)#图像输出大小 保持不变
        self.relu = nn.ReLU(inplace=True)
        self.mish = mish

        if (stride != 1 or cin != cout): #在输入和输出的 空间维度（stride） 或 通道数（cin ≠ cout） 不匹配时，通过一个投影（Projection）操作使二者能够相加。
            # Projection also with pre-activation according to paper. （stride != 1：主分支的卷积会降低特征图分辨率，残差连接也需要同步下采样 ；cin != cout：输入和输出的通道数不同，需要用 1x1 卷积调整通道数。）
            self.downsample = conv1x1(cin, cout, stride, bias=False) # 不仅 完成了 下采样操作，还将 输入的通道数调整到输出通道数，
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'): # hasattr(object, attribute) 是 Python 的内置函数，用于检查对象 object 是否包含指定的属性 attribute。
            residual = self.downsample(x) # 对于数据中 ，若 stride != 1 or cin != cout 则会进行 1*1 卷积，并进行 GroupNorm 操作，完成调整
            residual = self.gn_proj(residual)


        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x))) # 1*1 卷积，并进行 GroupNorm 操作，完成激活
        #print("conv1 output shape:", y.shape)
        y = self.relu(self.gn2(self.conv2(y))) # 3*3 卷积，并进行 GroupNorm 操作，完成激活
        #print("conv2 output shape:", y.shape)
        y = self.gn3(self.conv3(y)) # 1*1 卷积，并进行 GroupNorm 操作
        #print("conv3 output shape:", y.shape)

        y = self.mish(residual + y) # 残差连接
        return y

    def load_from(self, weights, n_block, n_unit):
        conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True) #pjoin 是 os.path.join 的别名，用于拼接路径，n_block 和 n_unit 分别表示当前块和单元的编号，np2th 是将 numpy 数组转换为 torch 张量的函数
        conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)

        gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])

        gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])

        gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])

        self.conv1.weight.copy_(conv1_weight) #self.conv1.weight 是第一个卷积层的可学习权重参数， copy_ 是PyTorch张量的原地(in-place)复制方法，下划线表示原地操作，直接修改目标张量而不创建新对象
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            proj_conv_weight = np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True)
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")])
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))

class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor): #block_units: 一个列表，指定每个残差块组(block)中包含的基本单元(unit)数量,[3, 4, 6]表示第一个block有3个unit，第二个有4个，第三个有6个
        super().__init__()
        width = int(64 * width_factor) # width_factor是一个宽度缩放因子，用于灵活调整模型的容量（参数量） ,  64是ResNet系列模型中第一个卷积层输出通道数的经典默认值(在原始ResNet论文（如ResNet-50/101）中：输入图像经过7x7卷积（stride=2）后，输出通道数固定为64。)
        self.width = width # 当width_factor=1时：width=64;当width_factor=2时：width=128; 参数文件中为 1

        self.root = nn.Sequential(OrderedDict([ #根网络部分(root)部分--卷积，归一化，激活函数，# 池化
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)), # 卷积层，输出的结果的 高宽 为原来的1/2，并且通道数发生变化
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))
        #主体网络部分(body)   nn.Sequential 是PyTorch中用于按顺序组合多个模块的容器，而 OrderedDict 是Python中保持元素插入顺序的字典。
        self.body = nn.Sequential(OrderedDict([ #  残差网络部分(body)部分--三个残差块
            ('block1/', nn.Sequential(OrderedDict(
                [('unit1/', PreActBottleneck(cin=width, cout=width * 4, cmid=width))] + #图像大小不变，变化的是通道数，变为原来的4倍
                [(f'unit{i:d}/', PreActBottleneck(cin=width * 4, cout=width * 4, cmid=width)) for i in
                 range(2, block_units[0] + 1)],  #从unit2到unitN（N由block_units[0]决定）
            ))),
            ('block2/', nn.Sequential(OrderedDict(
                [('unit1/', PreActBottleneck(cin=width * 4, cout=width * 8, cmid=width * 2, stride=2))] + #stride 为 2 会使得 尺寸减半
                [(f'unit{i:d}/', PreActBottleneck(cin=width * 8, cout=width * 8, cmid=width * 2)) for i in
                 range(2, block_units[1] + 1)],
            ))),
            ('block3/', nn.Sequential(OrderedDict(
                [('unit1/', PreActBottleneck(cin=width * 8, cout=width * 16, cmid=width * 4, stride=2))] +
                [(f'unit{i:d}/', PreActBottleneck(cin=width * 16, cout=width * 16, cmid=width * 4)) for i in
                 range(2, block_units[2] + 1)],
            ))),
        ]))


        self.reduce_channels = nn.Conv2d(1792, 1024, kernel_size=1)  # 1792是concat后的通道数，1024是目标通道数






    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        #print("ResNetV2部分")
        #print('input size: ',x.shape)
        x = self.root(x) # 特征尺寸减半 ，通道数 变化
        #print("root部分的输出尺寸为",x.shape)
        features.append(x) # 将第一个特征图添加到features列表中    特征图的尺寸为 （b, c, in_size, in_size）
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x) # 池化操作，将特征图尺寸减半
        #print("池化部分的输出尺寸为",x.shape)
        for i in range(len(self.body)-1): #中间层的特征用于构建特征金字塔或多尺度表示,但最后一个block的输出通常作为最终特征直接返回，所以不进行操作，则循环处理block1和block2
            #print("第",i,"个block的输入尺寸为",x.shape)
            x = self.body[i](x) #将输入x通过第i个block进行前向传播
            right_size = int(in_size / 4 / (i+1)) #in_size / 4：根网络（stride2卷积）和池化层（stride2）共下采样4倍     ；  /(i+1)：随着block深度增加，预期尺寸逐步减小
            # ——第一个block（i=0）：in_size/4，第二个block（i=1）：in_size/8，注意 其中 block2 中 有 stride 的 更改设置为 2
            if x.size()[2] != right_size: # 如果特征图尺寸不符合预期，则需要进行填充操作
                pad = right_size - x.size()[2] # 计算需要填充的尺寸，计算尺寸差异
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size) #验证差异是否合理。 assert（断言）是 Python 的一种调试工具，用于在代码中检查某个条件是否为 True。如果条件为 False，则程序会抛出 AssertionError 并终止执行（除非被捕获）。
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device) ## 创建一个全零的特征图，尺寸为[batch_size, channels, right_size, right_size]
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:] ## 填充原始特征图，使得其尺寸符合预期，将原始特征图x的值复制到新特征图的左上角区域
            else:
                feat = x
            #print("第",i,"个block的输出尺寸为",feat.shape)
            features.append(feat)# 将填充后的特征图添加到features列表中

            if i == 0:
                one = x
                #print("第1个block的输出尺寸为", one.shape)
                one = nn.AdaptiveAvgPool2d((28, 28))(one)  # 自适应平均池化操作，将特征图尺寸调整为28x28
                #print("第1个block池化后的输出尺寸为", one.shape)
            elif i == 1:
                two = x
                #print("第2个block的输出尺寸为", x.shape)
                concat_feat2 = torch.cat([one, two], dim=1)  # 拼接后的通道数: 256 + 512 = 768
                two_up = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(concat_feat2)  # 28x28 -> 14x14
                #print("第2个block池化后的输出尺寸为", two_up.shape)
            else:
                three = x
                #print("第3个block的输出尺寸为", x.shape)
        # print("第3个block的输入尺寸为", x.shape)
        x = self.body[-1](x) # 最后一个block的输出作为最终特征返回
        # three = x
        # concat_feat3 = torch.cat([two_up, three], dim=1)  # 拼接后的通道数: 768+1024 = 1792
        # # 添加1x1卷积层调整通道数
        # concat_feat3 = self.reduce_channels(concat_feat3)  # 输出尺寸: [batch, 1024, 14, 14]
        # #print("第3个block的输出尺寸为", concat_feat3.shape)
        # x = concat_feat3
        return x, features[::-1] # 返回最终特征和倒序的features列表


import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# 假设 StdConv2d, PreActBottleneck, gn (group norm辅助函数) 等已定义
# ... (如果需要，我可以提供这些辅助类的代码，但假设你环境里已有) ...

# ==========================================
# 1. 新增的特征门控增强模块
# ==========================================
# class SEBlock(nn.Module):
#     """SE Attention"""
#
#     def __init__(self, channels, reduction=16):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channels, channels // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channels // reduction, channels, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)
#
#
# class SpatialGate(nn.Module):
#     """浅层特征 -> 深层 Gate 信号"""
#
#     def __init__(self, in_channels, out_channels, scale_factor):
#         super().__init__()
#         if scale_factor > 1:
#             self.pool = nn.AvgPool2d(kernel_size=scale_factor, stride=scale_factor)
#         else:
#             self.pool = nn.Identity()
#
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         x = self.pool(x)
#         return self.conv(x)
#
#
# class MultiScaleEncoderGate(nn.Module):
#     def __init__(self, c1, c2, c3):
#         super().__init__()
#         # c1: Block1 Out, c2: Block2 Out, c3: Block3 Out
#
#         # Y1 (H/4) -> H/16 (Scale=4)
#         self.gate_branch1 = SpatialGate(c1, c3, scale_factor=4)
#         # Y2 (H/8) -> H/16 (Scale=2)
#         self.gate_branch2 = SpatialGate(c2, c3, scale_factor=2)
#
#         self.fuse_conv = nn.Sequential(
#             nn.Conv2d(c3, c3, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(c3),
#             nn.ReLU(inplace=True)
#         )
#         self.se = SEBlock(c3, reduction=16)
#         self.beta = nn.Parameter(torch.tensor(0.5))
#
#     def forward(self, y1, y2, y3):
#         # 1. 生成 Gate
#         g1 = self.gate_branch1(y1)
#         g2 = self.gate_branch2(y2)
#
#         # 2. 增强 Y3
#         y3_enhanced = y3 * (1.0 + g1 + g2)
#
#         # 3. 融合提纯
#         out = self.fuse_conv(y3_enhanced)
#         out = self.se(out)
#
#         # 4. 残差
#         return y3 + self.beta * out
#
#
# # ==========================================
# # 2. 修改后的 ResNetV2
# # ==========================================
# class ResNetV2(nn.Module):
#     """Implementation of Pre-activation (v2) ResNet mode."""
#
#     def __init__(self, block_units, width_factor):
#         super().__init__()
#         width = int(64 * width_factor)
#         self.width = width
#
#         # Root
#         self.root = nn.Sequential(OrderedDict([
#             ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
#             ('gn', nn.GroupNorm(32, width, eps=1e-6)),
#             ('relu', nn.ReLU(inplace=True)),
#         ]))
#
#         # Body
#         self.body = nn.Sequential(OrderedDict([
#             ('block1/', nn.Sequential(OrderedDict(
#                 [('unit1/', PreActBottleneck(cin=width, cout=width * 4, cmid=width))] +
#                 [(f'unit{i:d}/', PreActBottleneck(cin=width * 4, cout=width * 4, cmid=width)) for i in
#                  range(2, block_units[0] + 1)],
#             ))),
#             ('block2/', nn.Sequential(OrderedDict(
#                 [('unit1/', PreActBottleneck(cin=width * 4, cout=width * 8, cmid=width * 2, stride=2))] +
#                 [(f'unit{i:d}/', PreActBottleneck(cin=width * 8, cout=width * 8, cmid=width * 2)) for i in
#                  range(2, block_units[1] + 1)],
#             ))),
#             ('block3/', nn.Sequential(OrderedDict(
#                 [('unit1/', PreActBottleneck(cin=width * 8, cout=width * 16, cmid=width * 4, stride=2))] +
#                 [(f'unit{i:d}/', PreActBottleneck(cin=width * 16, cout=width * 16, cmid=width * 4)) for i in
#                  range(2, block_units[2] + 1)],
#             ))),
#         ]))
#
#         # 【插入点 1】: 实例化门控模块
#         # 根据 ResNet 结构:
#         # Block1 输出通道 = width * 4
#         # Block2 输出通道 = width * 8
#         # Block3 输出通道 = width * 16
#         self.encoder_gate = MultiScaleEncoderGate(
#             c1=width * 4,
#             c2=width * 8,
#             c3=width * 16
#         )
#
#     def forward(self, x):
#         features = []
#         b, c, in_size, _ = x.size()
#
#         # 1. Root
#         x = self.root(x)
#         features.append(x)  # features[0]: Root feature
#         x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
#
#         # 2. Body Loop (处理 Block 1 和 Block 2)
#         # 用一个临时列表保存每个 Block 的输出，供 Gate 使用
#         block_outputs = []
#
#         for i in range(len(self.body) - 1):  # i=0 (Block1), i=1 (Block2)
#             x = self.body[i](x)
#
#             # 这里的 padding 逻辑保持原样
#             right_size = int(in_size / 4 / (i + 1))
#             if x.size()[2] != right_size:
#                 pad = right_size - x.size()[2]
#                 assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
#                 feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
#                 feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
#             else:
#                 feat = x
#
#             # 将处理后的特征加入 features 列表供 Decoder 使用
#             features.append(feat)  # features[1]: Block1, features[2]: Block2
#
#             # 同时保存到 block_outputs 供 Gate 使用
#             block_outputs.append(feat)
#
#         # 3. Block 3 处理
#         x = self.body[-1](x)  # x 现在是 Block3 的输出
#
#         # 提取各个尺度的特征用于增强
#         # y1: Block1 输出 (features[1] 或 block_outputs[0])
#         # y2: Block2 输出 (features[2] 或 block_outputs[1])
#         # y3: Block3 输出 (x)
#         y1 = block_outputs[0]
#         y2 = block_outputs[1]
#         y3 = x
#
#         # 【插入点 2】: 调用门控模块增强 Y3
#         y3_enhanced = self.encoder_gate(y1, y2, y3)
#
#         # 4. 返回结果
#         # 返回增强后的 y3 作为最终输出
#         # features列表通常包含 [Root, Block1, Block2]，Decoder需要倒序使用
#         # 这里你可以选择是否把 y3 也加进 features，或者保持原样。
#         # 原始 TransUNet 返回 x, features[::-1]
#         # x 传给 ViT, features 传给 Decoder
#
#         return y3_enhanced, features[::-1]