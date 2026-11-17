# This file is licensed under Apache-2.0
# Copyright (c) NXAI GmbH and its affiliates 2024
# Benedikt Alkin, Maximilian Beck, Korbinian Pöppel
import math
import warnings
from enum import Enum

import einops
import torch
import torch.nn.functional as F
from torch import nn

from vision_lstm_util import interpolate_sincos, to_ntuple, VitPatchEmbed, VitPosEmbed2d, DropPath, SequenceConv2d
#interpolate_sincos：通过插值完成数据的缩放
#to_ntuple:将x转换为n元组
#VitPatchEmbed:ViT的patch嵌入层
#VitPosEmbed2d:ViT的position嵌入层
#DropPath:Dropout路径
#SequenceConv2d:二维序列卷积层

class SequenceTraversal(Enum):#设定遍历方向
    ROWWISE_FROM_TOP_LEFT = "rowwise_from_top_left"# 从左上角开始，按行遍历（从左到右，从上到下）
    ROWWISE_FROM_BOT_RIGHT = "rowwise_from_bot_right"# 从右下角开始，按行遍历（从右到左，从下到上）

#bias_linspace_init_:初始化偏置向量
def bias_linspace_init_(param: torch.Tensor, start: float = 3.4, end: float = 6.0) -> torch.Tensor:#start: 线性空间的起始值；end: 线性空间的结束值；param为 要初始化的1维张量（通常是偏置）
    """Linearly spaced bias init across dimensions."""
    assert param.dim() == 1, f"param must be 1-dimensional (typically a bias), got {param.dim()}" #断言确保参数是一维的。param为 要初始化的1维张量（通常是偏置）
    n_dims = param.shape[0]
    init_vals = torch.linspace(start, end, n_dims)#根据偏置向量的长度生成等间隔的数值序列。start和end分别是起始值和结束值
    with torch.no_grad():
        param.copy_(init_vals)#在不计算梯度的上下文中，将生成的序列值复制到参数中。
    return param
'''
# 初始化一个长度为5的偏置向量
bias = torch.zeros(5)
bias_linspace_init_(bias, start=0.0, end=1.0)
print(bias)  # 输出: tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])

# 使用默认参数初始化
bias2 = torch.zeros(3)
bias_linspace_init_(bias2)
print(bias2)  # 输出: tensor([3.4000, 4.7000, 6.0000])
'''

def small_init_(param: torch.Tensor, dim: int) -> torch.Tensor:#param为 要初始化的1维张量（通常是偏置）；dim为参数的维度
    '''
    用于Transformer中的线性层（包括自注意力层和前馈网络层）的权重初始化；这种方法旨在避免梯度爆炸或消失，使得模型在训练初期保持稳定。
    自注意力机制中的Q、K、V投影矩阵
    前馈神经网络(FFN)的权重矩阵
    输出投影层权重
    '''
    """说明这个初始化方法源自《Transformers without Tears》论文，
    Fills the input Tensor with values according to the method described in Transformers without Tears: Improving
    the Normalization of Self-Attention - Nguyen, T. & Salazar, J. (2019), using a normal distribution.
    Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py.
    """
    std = math.sqrt(2 / (5 * dim))#标准的 He 初始化中，公式是 sqrt(2 / fan_in)，其中 fan_in 指的是该层权重所连接的前一层神经元的数量（即输入特征的数量）。
    # 在这里，5 被硬编码为 fan_in 的值。这意味着编写这段代码的人假设，这个权重矩阵的输入特征数是 5。
    torch.nn.init.normal_(param, mean=0.0, std=std)#使用均值为0，计算出的标准差的正态分布来初始化参数
    return param


def wang_init_(param: torch.Tensor, dim: int, num_blocks: int):
    """ Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py. """
    '''
    该初始化方法考虑了网络深度（通过num_blocks参数，例如Transformer的层数）。这种初始化方法可能会随着网络深度的增加而减小权重的标准差，从而适应更深层的网络结构。
    主要针对：
    整个Transformer块/层的权重参数
    深度网络中的多层结构
    '''
    std = 2 / num_blocks / math.sqrt(dim)# 为多块结构（如Transformer块）设计的专用初始化
    torch.nn.init.normal_(param, mean=0.0, std=std)#用均值为0，计算出的标准差的正态分布来初始化参数
    return param
#使用
"""
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, layer_id, total_layers):
        super().__init__()

        # 自注意力部分 - 使用small_init_
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        small_init_(self.self_attention.wq.weight, dim=d_model)
        small_init_(self.self_attention.wk.weight, dim=d_model)
        small_init_(self.self_attention.wv.weight, dim=d_model)
        small_init_(self.self_attention.wo.weight, dim=d_model)

        # 前馈网络部分 - 使用small_init_
        self.ffn = FeedForward(d_model, d_ff)
        small_init_(self.ffn.w1.weight, dim=d_model)
        small_init_(self.ffn.w2.weight, dim=d_model)

        # 或者对整个层使用wang_init_
        # wang_init_(self.self_attention.wq.weight, dim=d_model, num_blocks=total_layers)
        # wang_init_(self.ffn.w1.weight, dim=d_model, num_blocks=total_layers)

def initialize_transformer_model(model):
   初始化整个Transformer模型
    total_layers = len(model.layers)

    for layer_id, layer in enumerate(model.layers):
        # 方法1：混合使用两种初始化
        small_init_(layer.self_attn.q_proj.weight, dim=model.d_model)
        small_init_(layer.self_attn.k_proj.weight, dim=model.d_model)
        small_init_(layer.self_attn.v_proj.weight, dim=model.d_model)
        small_init_(layer.self_attn.out_proj.weight, dim=model.d_model)

        # 方法2：使用wang_init_考虑网络深度
        wang_init_(layer.ffn.linear1.weight, dim=model.d_model, num_blocks=total_layers)
        wang_init_(layer.ffn.linear2.weight, dim=model.d_model, num_blocks=total_layers)
"""
#并行稳定化 mLSTM 单元详细解析
def parallel_stabilized_simple(
        queries: torch.Tensor,      # (B, NH, S, DH) - 查询张量
        keys: torch.Tensor,         # (B, NH, S, DH) - 键张量
        values: torch.Tensor,       # (B, NH, S, DH) - 值张量
        igate_preact: torch.Tensor, # (B, NH, S, 1) - 输入门预激活
        fgate_preact: torch.Tensor, # (B, NH, S, 1) - 遗忘门预激活
        lower_triangular_matrix: torch.Tensor = None,  # (S,S) - 因果掩码
        stabilize_rowwise: bool = True,  # 行稳定化策略
        eps: float = 1e-6,         # 数值稳定性常数
) -> torch.Tensor:
    """
    B: Batch size
    NH: Number of Heads (多头)
    S: Sequence Length (序列长度)
    DH: Dimension of Head (特征维度)（每个头的维度）

    This is the mLSTM cell in parallel form.
    This version is stabilized. We control the range of exp() arguments by
    ensuring that they are always smaller than 0.0 by subtracting the maximum.

    Args:
        :param queries: (torch.Tensor) (B, NH, S, DH)
        :param keys: (torch.Tensor) (B, NH, S, DH)
        :param values: (torch.Tensor) (B, NH, S, DH)
        :param igate_preact: (torch.Tensor) (B, NH, S, 1)
        :param fgate_preact: (torch.Tensor) (B, NH, S, 1)
        :param lower_triangular_matrix: (torch.Tensor) (S,S). Defaults to None.
        :param stabilize_rowwise: (bool) Wether to stabilize the combination matrix C rowwise (take maximum per row).
            Alternative: Subtract the maximum over all rows. Defaults to True.
        :param eps: (float) small constant to avoid division by 0. Defaults to 1e-6.

    Returns:
        torch.Tensor: (B, NH, S, DH), h_tilde_state
    """
    # 获取输入张量的形状信息
    B, NH, S, DH = queries.shape#获取输入张量 queries 的形状，并将其存储在变量 B, NH, S, DH（批次，头数，序列长度，特征维度） 中。
    _dtype, _device = queries.dtype, queries.device#获取输入张量 queries 的数据类型和设备信息，并将其存储在变量 _dtype 和 _device 中。
    '''
    第一步：构建遗忘门衰减矩阵 (Log-Space)，即每个位置的遗忘门累积效应
    '''
    # forget gate matrix#
    #mLSTM 的核心在于记忆的衰减。在并行模式下，我们需要计算从时刻 j 到时刻 i 的累积遗忘值
    log_fgates = torch.nn.functional.logsigmoid(fgate_preact)  # (B, NH, S, 1)，log sigmoid 将遗忘门预激活转换为对数空间，使用log-sigmoid确保数值稳定。
    #创建下三角布尔矩阵，用于实现因果注意力（只能看到当前及之前的时间步）
    if lower_triangular_matrix is None or S < lower_triangular_matrix.size(-1):
        #lower_triangular_matrix is None：没有提供预定义的掩码，需要新建一个掩码
        #S < lower_triangular_matrix.size(-1)：提供的掩码尺寸小于当前序列长度，需要重新创建合适尺寸的掩码
        ltr = torch.tril(torch.ones((S, S), dtype=torch.bool, device=_device))#torch.tril(torch.ones((S, S), dtype=torch.bool))构建下三角矩阵（下面为True,上三角为False）
    else:#重用提供的掩码（性能优化）
        ltr = lower_triangular_matrix
    assert ltr.dtype == torch.bool, f"lower_triangular_matrix must be of dtype bool, got {ltr.dtype}"#强制要求你传入的 lower_triangular_matrix 必须是 bool 类型张量，如果不是，就直接报错终止程序。

    log_fgates_cumsum = torch.cat(#在序列开始处添加零，然后计算累积和。这表示从序列开始到每个位置的遗忘门累积效应。
        [
            torch.zeros((B, NH, 1, 1), dtype=_dtype, device=_device),
            torch.cumsum(log_fgates, dim=-2),
        ],
        dim=-2,
    )  # (B, NH, S+1, 1)，S+1,是因为加入了时刻0这一行
    #torch.zeros((B, NH, 1, 1)): 在序列开始处添加一个零向量
    #torch.cumsum(log_fgates, dim=-2): 沿序列维度计算累积和
    #torch.cat(..., dim=-2): 在序列维度上拼接

    # for each batch/head this is a matrix of shape (S+1, S+1) containing the cumsum of the log forget gate values
    # in the second dimension (colum dimension). Each row has the same is a copy of the first row.
    # First entry of each row is zero.
    rep_log_fgates_cumsum = log_fgates_cumsum.repeat(1, 1, 1, S + 1)  # (B, NH, S+1, S+1)，沿最后一个维度重复 S+1 次。也就是构成一个横轴去，纵轴刻度都为时间t的矩阵
    # Now in each row cut off / subtract the forgetgate values of the later timesteps
    # where col j > row i
    _log_fg_matrix = rep_log_fgates_cumsum - rep_log_fgates_cumsum.transpose(-2, -1)  # (B, NH, S+1, S+1)
    # Causal masking & selection of the correct submatrix, such that forgetgate at timestep t is not applied
    # to the input at timestep t
    log_fg_matrix = torch.where(ltr, _log_fg_matrix[:, :, 1:, 1:], -float("inf"))  # (B, NH, S, S)
    # #数学原理：构建一个矩阵，其中元素 (i,j) 表示从时间步 j 到 i 的遗忘门累积乘积的对数：
    '''举例：
    log_fgates_cumsum = [[0.0], [-0.3], [-0.5], [-0.6]]  # (4,1)
    rep_log_fgates_cumsum = [[0.0, 0.0, 0.0, 0.0],
                         [-0.3, -0.3, -0.3, -0.3],
                         [-0.5, -0.5, -0.5, -0.5],
                         [-0.6, -0.6, -0.6, -0.6]]  # (4,4)
    _log_fg_matrix = [[ 0.0,  0.3,  0.5,  0.6],   # 行0 - 列[0,1,2,3]
                      [-0.3,  0.0,  0.2,  0.3],   # 行1 - 列[0,1,2,3]  
                      [-0.5, -0.2,  0.0,  0.1],   # 行2 - 列[0,1,2,3]
                      [-0.6, -0.3, -0.1,  0.0]]   # 行3 - 列[0,1,2,3]
    子矩阵 = [[ 0.0,  0.2,  0.3],
         [-0.2,  0.0,  0.1],
         [-0.3, -0.1,  0.0]]

    下三角掩码 = [[True, False, False],
            [True,  True, False],
            [True,  True,  True]]

    log_fg_matrix = [[ 0.0, -inf, -inf],
                 [-0.2,  0.0, -inf],
                 [-0.3, -0.1,  0.0]]
    '''
    '''
    第二步：构建门控衰减矩阵 D
    '''
    # gate decay matrix D (combination of forget gate and input gate)
    log_D_matrix = log_fg_matrix + igate_preact.transpose(-2, -1)  # (B, NH, S, S)# D = Decay * Input_Gate# 在对数域：log(D) = log(Decay) + log(Input_Gate)
    #其中通过 transpose(-2, -1) 将输入门预激活转换为 (1, S) 的矩阵,而(s,s)-(1,s) 会令(1,s)通过广播形式变为(s,s)参与运算
    # D matrix stabilization
    if stabilize_rowwise:
        max_log_D, _ = torch.max(log_D_matrix, dim=-1, keepdim=True)  # (B, NH, S, 1)
    else:
        max_log_D = torch.max(log_D_matrix.view(B, NH, -1), dim=-1, keepdim=True)[0].unsqueeze(-1)
        # (B, NH, 1, 1)
    log_D_matrix_stabilized = log_D_matrix - max_log_D  # (B, NH, S, S)
    D_matrix = torch.exp(log_D_matrix_stabilized)  # (B, NH, S, S) #稳定化版本，得到稳定的门激活矩阵

    keys_scaled = keys / math.sqrt(DH) #Key 的缩放，除以 DH 的平方根

    # combination matrix C
    qk_matrix = queries @ keys_scaled.transpose(-2, -1)  # (B, NH, S, S)# 计算查询和键的内积，即查询和键的乘积矩阵
    C_matrix = qk_matrix * D_matrix  # (B, NH, S, S)
    normalizer = torch.maximum(C_matrix.sum(dim=-1, keepdim=True).abs(), torch.exp(-max_log_D))  # (B, NH, S, 1)
    # (B, NH, S, S)
    C_matrix_normalized = C_matrix / (normalizer + eps)#稳定化的C（细胞状态）
    #eps 是为了防止除以 0 的情况，防止出现 NaN 或 Inf 值
    # retrieved values
    h_tilde_state = C_matrix_normalized @ values  # (B, NH, S, DH)

    return h_tilde_state


class LinearHeadwiseExpand(nn.Module):
    """
    这是一个多头分块扩展线性层，专将输入张量通过多头分块的方式进行维度扩展，每个头使用独立的线性变换。
    参数效率: 总参数从 512×512=262,144 减少到 8×64×64=32,768；  计算并行: 每个头可以并行计算；  表示能力: 每个头学习不同的特征变换
    This is a structured projection layer that projects the input to a higher dimension.
    It only allows integer up-projection factors, i.e. the output dimension is a multiple of the input dimension.
    """

    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        assert dim % num_heads == 0 # 检查输入维度是否可以被头数整除，否则抛出异常
        self.dim = dim
        self.num_heads = num_heads

        dim_per_head = dim // num_heads # 每个头部的维度
        self.weight = nn.Parameter(torch.empty(num_heads, dim_per_head, dim_per_head))#每个头有独立的变换矩阵，即每个头有独立的权重矩阵,未初始化，值随机
        if bias:#如果有偏置，则创建偏置参数
            self.bias = nn.Parameter(torch.empty(dim))
        else:#没有偏置，则不创建偏置参数
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):#权重: 使用缩放正态分布初始化，标准差基于输入维度
        nn.init.normal_(self.weight.data, mean=0.0, std=math.sqrt(2 / 5 / self.weight.shape[-1]))
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #分离出头维度
        x = einops.rearrange(x, "... (nh d) -> ... nh d", nh=self.num_heads)#将输入张量的最后一个维度（特征维度）分解为两个维度：头数和每个头的维度。输入: (2, 5, 512)， 将512分解为8×64，输出: (2, 5, 8, 64)
        #对每个头应用独立的线性变换
        x = einops.einsum(#使用einops.einsum进行张量乘法，即矩阵乘法；将分头后的x与各个头对应的weight进行相乘
            x,#将分头后的x与各个头对应的weight进行相乘
            self.weight,
            "... nh d, nh out_d d -> ... nh out_d",#输入x的维度为... nh d，weight的维度为nh out_d d，其中以d为接口进行矩阵乘法，输出x的维度为... nh out_d
        )
        #重新组合头维度
        x = einops.rearrange(x, "... nh out_d -> ... (nh out_d)")#进行重排，将结果最后的nh,out_d维度合并为最后一个维度
        #添加偏置
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self):#用于返回模块的额外表示信息，当打印模块时，这些信息会显示出来。
        return (
            f"dim={self.dim}, "
            f"num_heads={self.num_heads}, "
            f"bias={self.bias is not None}, "
        )


class CausalConv1d(nn.Module):#因果卷积——应用于一维数据
    """
    Implements causal depthwise convolution of a time series tensor.        实现时间序列张量的因果深度可分离卷积。
    Input:  Tensor of shape (B,T,F), i.e. (batch, time, feature)            输入: 形状为 (B,T,F) 的张量，即 (批次, 时间, 特征)
    Output: Tensor of shape (B,T,F)                                         输出: 形状为 (B,T,F) 的张量

    Args:
        feature_dim: number of features in the input tensor                 输入张量的特征数
        kernel_size: size of the kernel for the depthwise convolution       深度可分离卷积的核大小
        causal_conv_bias: whether to use bias in the depthwise convolution  是否在深度可分离卷积中使用偏置
        channel_mixing: whether to use channel mixing (i.e. groups=1) or not (i.e. groups=feature_dim)是否使用通道混合 (即 groups=1) 或不使用 (即 groups=feature_dim)
                        If True, it mixes the convolved features across channels.如果为 True，则在通道间混合卷积后的特征。
                        If False, all the features are convolved independently.  如果为 False，所有特征独立卷积。
    """

    def __init__(self, dim, kernel_size=4, bias=True):
        super().__init__()
        self.dim = dim#输入张量的特征数
        self.kernel_size = kernel_size#卷积核大小
        self.bias = bias#是否使用偏置
        # padding of this size assures temporal causality.
        self.pad = kernel_size - 1#卷积的填充大小，保证了时间上的因果性
        self.conv = nn.Conv1d(
            in_channels=dim,#输入通道数
            out_channels=dim,#输出通道数
            kernel_size=kernel_size,#卷积核大小
            padding=self.pad,#填充大小
            groups=dim,#分组卷积，即每个通道独立卷积  #实现深度可分离卷积，每个输入通道有独立的滤波器
            bias=bias,#偏置
        )
        self.reset_parameters()

    def reset_parameters(self):#使用PyTorch默认的卷积层参数初始化方法。
        self.conv.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # conv requires dim first
        x = einops.rearrange(x, "b l d -> b d l")#将输入张量转换为 (B, D, L) 的形状，其中 B 是批次大小，D 是特征维度，L 是序列长度
        # causal conv1d
        x = self.conv(x)#卷积操作，应用深度可分离卷积。由于填充了 kernel_size-1，输出序列长度会变为 L + (kernel_size-1)。
        x = x[:, :, :-self.pad]#裁剪掉填充的部分，即最后self.pad个时间步
        # back to dim last
        x = einops.rearrange(x, "b d l -> b l d")#将输出张量转换为 (B, L, D) 的形状，其中 B 是批次大小，L 是序列长度，D 是特征维度
        return x


class LayerNorm(nn.Module):#对PyTorch的LayerNorm的封装，但增加了一些可选功能，比如可选的偏置和残差权重。
    # 完成了对于残差模式，标准模式的权重初始化，以及归一化层的定义
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False.
    对于权重weights；
     如果用于残差连接 (residual_weight=True)，初始化为 0 (以便在 weight_proxy 中成为 1.0)。
     如果用于标准层 (residual_weight=False)，初始化为 1 (标准行为)。
     对于偏置bias:
     如果存在 (bias=True)，初始化为 0。
     """

    def __init__(
            self,
            ndim: int = -1,# 归一化的维度大小，默认为-1，即自动推断，-1可能是一个占位符，意在提醒用户必须传入正确的维度
            weight: bool = True,# 是否使用可学习的缩放参数（gamma）
            bias: bool = False,#是否使用可学习的偏置参数（beta）
            eps: float = 1e-5,#防止除零的小常数
            residual_weight: bool = True,# 特殊的权重初始化策略
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(ndim)) if weight else None#weight=True，创建全零的缩放参数
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None#bias=True，创建全零的偏置参数
        self.eps = eps
        self.residual_weight = residual_weight
        self.ndim = ndim
        self.reset_parameters()

    @property
    def weight_proxy(self) -> torch.Tensor:#-> torch.Tensor:说明结果输出为一个张量
        if self.weight is None:#如果weight为None，返回None
            return None
        if self.residual_weight:#残差权重：1 + 可学习参数——如果residual_weight为True，返回1.0 + weight（确定使用特殊权重初始化）
            return 1.0 + self.weight
        else:#否则，返回weight，标准权重
            return self.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(#F.layer_norm是PyTorch中的一个函数，用于应用层归一化，即对输入张量进行归一化处理
            x,
            normalized_shape=(self.ndim,),## 归一化的形状，对输入张量的最后self.ndim个维度进行归一化
            weight=self.weight_proxy,#可选的缩放权重，用于调整归一化后的值。必须与normalized_shape形状相同。如果为None，则不使用缩放。
            bias=self.bias,
            eps=self.eps,
        )#ormalized_shape必须与输入张量的最后若干维度的形状匹配。例如，如果输入张量的形状为(B, T, D)，并且self.ndim = D，那么归一化将在每个特征向量（最后一个维度）上进行，即对每个特征向量计算均值和方差，然后进行归一化。

    def reset_parameters(self):
        if self.weight_proxy is not None:#只有当模块配置为使用可学习权重时（weight=True），才进行权重初始化。
            if self.residual_weight:
                nn.init.zeros_(self.weight)# 残差模式：初始化为0
            else:
                nn.init.ones_(self.weight)# 标准模式：初始化为1
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class MultiHeadLayerNorm(LayerNorm):#继承上述的LayerNorm层归一化
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, "Input must be 4D tensor (B, NH, S, DH)"#需要输入是四维的张量，即(B, NH, S, DH)
        B, NH, S, DH = x.shape

        gn_in_1 = x.transpose(1, 2)  # (B, S, NH, DH)维度变换
        gn_in_2 = gn_in_1.reshape(B * S, NH * DH)  # (B * S, NH * DH)维度变换，变为2维的张量
        out = F.group_norm(#F.group_norm是PyTorch中的一个函数，用于应用组归一化，即对输入张量进行归一化处理
            gn_in_2,
            num_groups=NH,#NH，即头数，分为NH组进行组别归一化
            weight=self.weight_proxy,#残差权重
            bias=self.bias,
            eps=self.eps,
        )  # .to(x.dtype)
        # (B * S), (NH * DH) -> (B, S, NH, DH) -> (B, NH, S, DH)
        out = out.view(B, S, NH, DH).transpose(1, 2)
        return out


class MatrixLSTMCell(nn.Module): #并行的mlstm块，得到最后的h_tilde_state
    def __init__(self, dim, num_heads, norm_bias=True):
        super().__init__()
        self.dim = dim#输入特征的维度
        self.num_heads = num_heads#头数

        self.igate = nn.Linear(3 * dim, num_heads)#将拼接后的q,k,v(共3*dim维)映射到num_heads维，线性层
        self.fgate = nn.Linear(3 * dim, num_heads)#将拼接后的q,k,v(共3*dim维)映射到num_heads维，线性层
        self.outnorm = MultiHeadLayerNorm(ndim=dim, weight=True, bias=norm_bias)#多头LayerNorm，用于输出归一化
        self.causal_mask_cache = {}#一个缓存字典，键是 (序列长度 S, 设备名称)，值是对应的因果掩码。作用：避免重复创建相同的掩码（尤其是序列长度和设备不变时），节省计算资源。
        self.reset_parameters()#初始化网络参数
        # (在你原有的 init 代码末尾加上这一行)
        self.gamma_leap = nn.Parameter(torch.zeros(1))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,leap_mask=None) -> torch.Tensor:
        B, S, _ = q.shape  # (B, S, H)#输入: q, k, v三个张量，形状都是(Batch, Sequence, Hidden)
        #获取batch大小B和序列长度S
        if_gate_input = torch.cat([q, k, v], dim=-1)#将q,k,v在最后一个维度拼接，用于输入门和遗忘门的计算
        #将q,k,v进行重塑为多头模式，每个头的维度DH = dim / num_heads
        q = q.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)
        k = k.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)
        v = v.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)
        # 调整维度顺序，将头维度放在序列维度之前
        q = q.transpose(1, 2)  # (B, NH, S, DH)
        k = k.transpose(1, 2)  # (B, NH, S, DH)
        v = v.transpose(1, 2)  # (B, NH, S, DH)

        # compute input and forget gate pre-activations
        igate_preact = self.igate(if_gate_input)  # (B, S, NH)，得到输入门的预激活值，通过线性层，将[B,S,3*H]线性变换为[B, S, NH]
        igate_preact = igate_preact.transpose(-1, -2).unsqueeze(-1)  # (B, NH, S, 1)调整维度为(B, NH, S, 1)，与多头格式匹配
        fgate_preact = self.fgate(if_gate_input)  # (B, S, NH)，得到遗忘门的预激活值
        fgate_preact = fgate_preact.transpose(-1, -2).unsqueeze(-1)  # (B, NH, S, 1)#调整维度为(B, NH, S, 1)，与多头格式匹配
        # ==========================================================
        # 【新增：Leap 掩码劫持门控】
        # ==========================================================
        if leap_mask is not None:
            # 调整 mask 形状对齐，并利用 PyTorch Broadcast 自动广播到 head_dim
            # 假设 leap_mask 输入时是 [B, S, 1]
            m = leap_mask.transpose(1, 2).unsqueeze(1)  # 变成 [B, 1, 1, S] 或视你的头维度调整

            # 零初始化安全锁
            m_safe = 1.0 + self.gamma_leap * (m - 1.0)

            # 物理阻断与瞬移
            igate_preact = igate_preact + torch.log(m_safe + 1e-6)
            fgate_preact = fgate_preact * m_safe + 20.0 * (1.0 - m_safe)
        # ==========================================================
        # cache causal mask to avoid memory allocation in every iteration
        '''
        这段代码检查缓存中是否已有当前序列长度 S 对应设备的掩码。如果没有，它会创建一个 S×S 的下三角布尔矩阵（只允许当前位置及之前的位置被关注），并将其存储到缓存中，以提高效率。
核心目的
这段代码的目的是：高效地管理和复用用于序列计算的下三角掩码（Causal Mask），避免在每一次前向传播中都重新创建和分配内存给这个掩码。
在处理序列数据时（如语言模型），我们通常需要确保当前时间步的输出只依赖于当前和过去的时间步，不能“偷看”未来的信息。这就是因果掩码的作用。
关键概念解释
序列长度 
S: 当前批次中序列的长度。如果模型正在处理一个长度为 50 的句子，则 
S=50。
因果掩码 (Causal Mask): 一个 
S×S 的矩阵，通常由 torch.tril（Lower Triangular，下三角）创建。在注意力机制中，这个掩码会被用来将矩阵中对应于未来位置的注意力分数设置为 
−∞（或在布尔掩码中标记为 False），从而阻止信息流向未来。
缓存 (self.causal_mask_cache): 这是一个 Python 字典，用于存储已经创建好的掩码。它以 (序列长度, 设备信息) 作为键。
工作流程与示例
假设 self.dim = 512，self.num_heads = 8，并且我们正在处理 GPU 上的数据。
##    场景一：第一次处理长度 
S=10 的序列（不在缓存中）
检查缓存: if 10 in self.causal_mask_cache: → False (缓存是空的或没有 
S=10 的)。
创建掩码: 代码执行 else 分支。
torch.ones(10, 10, dtype=torch.bool, device=q.device) 创建一个 
10×10 的全 True张量。
torch.tril(...) 将其转换为下三角形式，得到 causal_mask。
\text{T} & \text{F} & \text{F} & \dots & \text{F} \\ \text{T} & \text{T} & \text{F} & \dots & \text{F} \\ \text{T} & \text{T} & \text{T} & \dots & \text{F} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ \text{T} & \text{T} & \text{T} & \dots & \text{T} \end{pmatrix}$$
缓存: 这个新创建的掩码被存入缓存：
self.causal_mask_cache[(10, 'cuda:0')] = causal_mask (假设 Q 在 cuda:0 上)。
##   场景二：第二次处理长度 
S=10 的序列（在缓存中）
检查缓存: if 10 in self.causal_mask_cache: → True。
设备检查: 缓存键还包括设备信息 str(q.device)。如果当前 Q 仍然在同一个设备上（例如 cuda:0），则匹配成功。
复用: 代码执行 if 分支，直接从缓存中取出之前创建好的 
10×10 掩码：
causal_mask = self.causal_mask_cache[(10, 'cuda:0')]。
没有进行昂贵的 torch.ones 和 torch.tril 操作。
##   场景三：处理不同长度 
S=20 的序列（不在缓存中）
检查缓存: if 20 in self.causal_mask_cache: → False。
创建新掩码: 模型会创建一个新的 
20×20 的下三角掩码，并将其缓存起来。
为什么需要缓存设备信息 str(q.device)？
PyTorch 要求所有参与计算的张量必须在同一个设备上（CPU 或特定的 GPU）。如果模型在 
S=10 时在 cuda:0 上运行，缓存了一个在 cuda:0 上的掩码；但在下一次 
S=10 时，如果 Q 张量被移动到了 cuda:1，那么它不能使用 cuda:0 上的掩码。通过在缓存键中包含设备信息，可以确保只使用与当前输入张量在同一设备上的缓存掩码。
总结
缓存因果掩码是优化深度学习模型性能的常见手段，特别是对于 RNN/Transformer 等序列模型，它可以显著减少在每个时间步重复分配和初始化大型布尔张量所带来的计算开销和内存碎片。
        '''
        if S in self.causal_mask_cache:#S 是序列长度(sequence length)，掩码的设备（device=q.device）与输入数据 q 保持一致，避免因设备不同导致的计算错误（例如 GPU 张量不能直接和 CPU 掩码运算）。
            causal_mask = self.causal_mask_cache[(S, str(q.device))]
        else:
            causal_mask = torch.tril(torch.ones(S, S, dtype=torch.bool, device=q.device))#下三角(包括对角线)为True，上三角为False
            self.causal_mask_cache[(S, str(q.device))] = causal_mask

        h_state = parallel_stabilized_simple(#输出隐藏状态，形状为(B, NH, S, DH)
            queries=q,
            keys=k,
            values=v,
            igate_preact=igate_preact,
            fgate_preact=fgate_preact,
            lower_triangular_matrix=causal_mask,
        )  # (B, NH, S, DH)

        h_state_norm = self.outnorm(h_state)  # (B, NH, S, DH) 进行多头归一化处理
        h_state_norm = h_state_norm.transpose(1, 2).reshape(B, S, -1)  # (B, NH, S, DH) -> (B, S, NH, DH) -> (B, S, H)

        return h_state_norm

    def reset_parameters(self):#参数初始化方法，用于设置网络层的初始权重和偏置
        self.outnorm.reset_parameters()#多头LayerNorm的参数初始化,为 LayerNorm 层内部的可学习参数设定它们在训练开始时的第一个状态（即初始化）
        # forget gate initialization
        torch.nn.init.zeros_(self.fgate.weight)#初始化遗忘门的权重为0
        bias_linspace_init_(self.fgate.bias, start=3.0, end=6.0)#初始化遗忘门的偏置，使用线性插值初始化方法，从3.0到6.0
        # input gate initialization
        torch.nn.init.zeros_(self.igate.weight)#初始化输入门的权重为0
        torch.nn.init.normal_(self.igate.bias, mean=0.0, std=0.1)#初始化输入门的偏置


class ViLLayer(nn.Module):
    def __init__(
            self,
            dim,                    # 输入特征维度
            direction,              # 序列遍历方向
            expansion=2,            # 内部维度扩展因子
            qkv_block_size=4,       # QKV分块大小
            proj_bias=True,         # 投影层是否使用偏置
            norm_bias=True,         # 归一化层是否使用偏置
            conv_bias=True,         # 卷积层是否使用偏置
            conv_kernel_size=4,     # 卷积核大小
            conv_kind="2d",         # 卷积类型
            init_weights="original", # 权重初始化方法
            seqlens=None,           # 序列长度信息
            num_blocks=None,        # 块数（用于初始化）
    ):
        super().__init__()
        assert dim % qkv_block_size == 0#检查输入特征维度是否可以被qkv_block_size整除
        self.dim = dim
        self.direction = direction
        self.expansion = expansion
        self.qkv_block_size = qkv_block_size
        self.proj_bias = proj_bias
        self.conv_bias = conv_bias
        self.conv_kernel_size = conv_kernel_size
        self.conv_kind = conv_kind
        self.init_weights = init_weights
        self.num_blocks = num_blocks
        #MLSTM是先进行升维，再经过mLSTM块的，再降维
        inner_dim = expansion * dim #将维度扩展（升维），内部扩展后的维度
        num_heads = inner_dim // qkv_block_size#QKV的头数，较小的 qkv_block_size → 更多的头数 → 更细粒度的特征学习
        self.proj_up = nn.Linear(
            in_features=dim,
            out_features=2 * inner_dim,
            bias=proj_bias,
        )#上投影层： 将输入维度 dim 投影到 2 * inner_dim，为后续分叉做准备。
        self.q_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )#LinearHeadwiseExpand 是一个自定义的线性变换层，专门设计用于按头(head-wise)进行特征扩展，较于liner再分头，该方法可以大大减少参数使用
        # 在 LinearHeadwiseExpand 内部：假设输入为[B, S, 256]
        # 1. 重塑输入x_reshaped = x_mlstm.view(B, S, num_heads, head_dim)  # [B, S, 64, 4]
        # 2. 对每个头应用线性变换 (每个头有 4×4 的权重矩阵)，头分别进行 [4] -> [4] 的线性变换
        # 3. 合并所有头： # [B, S, 64, 4]
        # 4. 重塑回原始形状# [B, S, 256]
        self.k_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )
        self.v_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )#QKV投影层： 使用头部分离的线性变换，为矩阵LSTM准备查询、键、值。

        if conv_kind == "causal1d":## 因果一维卷积
            self.conv = CausalConv1d(
                dim=inner_dim,
                kernel_size=conv_kernel_size,
                bias=conv_bias,
            )
        elif conv_kind == "2d":## 2D卷积
            assert conv_kernel_size % 2 == 1, \
                f"same output shape as input shape is required -> even kernel sizes not supported"  ## 确保卷积核为奇数，当卷积核大小为奇数时，可以对称地填充两侧：
            self.conv = SequenceConv2d(
                in_channels=inner_dim,
                out_channels=inner_dim,
                kernel_size=conv_kernel_size,
                padding=conv_kernel_size // 2,# 保持序列长度不变
                groups=inner_dim, # 深度可分离卷积
                bias=conv_bias,
                seqlens=seqlens,
            )
        else:#既不是一维也不是二维数据，不予处理
            raise NotImplementedError
        self.mlstm_cell = MatrixLSTMCell(
            dim=inner_dim,
            num_heads=qkv_block_size,
            norm_bias=norm_bias,
        )#设置并行化的msltm模块，参数为输入维度，QKV头数，是否使用归一化偏置
        #创建了一个从 x_mlstm 到最终输出的跳跃路径，绕过了矩阵LSTM的复杂计算。就是mlstm框架最右边那个跳跃连接
        self.learnable_skip = nn.Parameter(torch.ones(inner_dim))# 可学习的跳跃连接，对跳跃连接中各个进行数据进行加权调整
        #此处为初始化都为1，即每个维度上的权重都为1
        self.proj_down = nn.Linear(
            in_features=inner_dim,
            out_features=dim,
            bias=proj_bias,
        )#下投影层： 将内部维度 inner_dim 投影到 dim，为最终输出做准备。降维
        self.reset_parameters()#初始化参数

    def forward(self, x: torch.Tensor,leap_mask=None) -> torch.Tensor:
        B, S, _ = x.shape#获取输入的批次大小B，序列长度S，以及其他特征维度_

        # alternate direction in successive layers
        if self.direction == SequenceTraversal.ROWWISE_FROM_TOP_LEFT:
            pass## 保持原顺序
        elif self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x = x.flip(dims=[1])#反向翻转序列
        else:
            raise NotImplementedError

        # up-projection
        x_inner = self.proj_up(x)#上投影层，将输入x_inner的维度从[B, S, dim]变为[B, S, 2 * inner_dim]
        x_mlstm, z = torch.chunk(x_inner, chunks=2, dim=-1)#将 x_inner 沿着最后一个维度(dim=-1)平均分成2块：
        '''
        假设 inner_dim = 256, 则 x_inner: [B, S, 512]
        x_mlstm: [B, S, 256]  # 前一半特征,x_mlstm 分支 - 序列建模通路
        z: [B, S, 256]        # 后一半特征,z 分支 - 门控信号通路
        '''
        # mlstm branch
        x_mlstm_conv = self.conv(x_mlstm)#卷积层，对x_mlstm_conv进行卷积操作
        x_mlstm_conv_act = F.silu(x_mlstm_conv)# SiLU激活
        q = self.q_proj(x_mlstm_conv_act)
        k = self.k_proj(x_mlstm_conv_act)#再XLSTM对与mlstm的设定就是Q,K进行SILU激活后获得
        v = self.v_proj(x_mlstm)#按照XLSTM论文对于mlstm块的设定，V的获得，不需要进行SILU激活
        h_tilde_state = self.mlstm_cell(q=q, k=k, v=v)#将Q,K,V输入mlstm模块中，得到稳定的隐藏状态输出
        h_tilde_state_skip = h_tilde_state + (self.learnable_skip * x_mlstm_conv_act)#通过跳跃连接，完成mlstm最左侧的LSkip部分

        # output / z branch
        h_state = h_tilde_state_skip * F.silu(z)#将mlstm的输出与门控信号通路的输出相乘

        # down-projection
        x = self.proj_down(h_state)#下投影层，将内部维度 inner_dim 投影到 dim，为最终输出做准备。

        # reverse alternating flip
        #结束时的方向恢复
        if self.direction == SequenceTraversal.ROWWISE_FROM_TOP_LEFT:
            pass
        elif self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x = x.flip(dims=[1])
        else:
            raise NotImplementedError

        return x

    def reset_parameters(self):#初始化模型参数,small_init_ 和 wang_init_见  line 46 和 line 64
        # init inproj
        small_init_(self.proj_up.weight, dim=self.dim)#初始化上投影层的权重，dim=self.dim,前馈网络初始化方法
        if self.proj_up.bias is not None:
            nn.init.zeros_(self.proj_up.bias)
        # init outproj (original mLSTM uses num_blocks=1)
        if self.init_weights == "original":#num_blocks=1: 对应原始 mLSTM 论文的设置，即只有一块mlstm模块
            wang_init_(self.proj_down.weight, dim=self.dim, num_blocks=1)
        elif self.init_weights == "original-fixed":#num_blocks=self.num_blocks，即有多个mlstm模块进行堆叠
            wang_init_(self.proj_down.weight, dim=self.dim, num_blocks=self.num_blocks)
        else:
            raise NotImplementedError
        if self.proj_down.bias is not None:#如果有偏置，初始化为0
            nn.init.zeros_(self.proj_down.bias)
        # 3. 初始化可学习的跳跃连接 (learnable_skip)
        nn.init.ones_(self.learnable_skip)

        # 4. 初始化 QKV 投影层
        def _init_qkv_proj(qkv_proj: LinearHeadwiseExpand):
            # use the embedding dim instead of the inner embedding dim
            small_init_(qkv_proj.weight, dim=self.dim)
            if qkv_proj.bias is not None:
                nn.init.zeros_(qkv_proj.bias)

        _init_qkv_proj(self.q_proj)#初始化方法一样，但初始化结果可不一样
        _init_qkv_proj(self.k_proj)
        _init_qkv_proj(self.v_proj)

        self.mlstm_cell.reset_parameters()


class ViLBlock(nn.Module):#封装 ViLLayer,
    # 并添加层归一化，(Layer Normalization): 稳定训练过程。
    # 残差连接 (Residual Connection): 允许信息和梯度在深层网络中有效流动，避免梯度消失。
    # 路径丢弃 (DropPath / Stochastic Depth): 一种强大的正则化技术，防止模型过拟合。
    def __init__(
            self,
            dim, #输入特征维度
            direction,#序列遍历方向
            drop_path=0.0,#dropout概率
            conv_kind="2d",#卷积类型
            conv_kernel_size=3,#卷积核大小
            proj_bias=True,#投影层是否使用偏置
            norm_bias=True,#归一化层是否使用偏置
            seqlens=None,#序列长度信息
            num_blocks=None,#块数（用于初始化）
            init_weights="original",#权重初始化方法
    ):
        super().__init__()
        self.dim = dim
        self.direction = direction
        self.drop_path = drop_path
        self.conv_kind = conv_kind
        self.conv_kernel_size = conv_kernel_size
        self.init_weights = init_weights

        self.drop_path = DropPath(drop_prob=drop_path)
        self.norm = LayerNorm(ndim=dim, weight=True, bias=norm_bias)
        self.layer = ViLLayer(
            dim=dim,
            direction=direction,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel_size,
            seqlens=seqlens,
            norm_bias=norm_bias,
            proj_bias=proj_bias,
            num_blocks=num_blocks,
            init_weights=init_weights,
        )

        self.reset_parameters()

    ## 这是一个辅助函数，定义了块的主计算路径（_forward_path），即：
    def _forward_path(self, x):
        x = self.norm(x)
        x = self.layer(x)
        return x
    ## 这是块的正式前向传播方法，即：
    def forward(self, x: torch.Tensor,leap_mask=None) -> torch.Tensor:#在训练期间，整个块的主计算路径（_forward_path）有 drop_path 的概率被“丢弃”（即跳过），此时块的行为就是一个恒等函数。这是一种正则化方法，可以看作是作用于整个层的 Dropout。
        #应用 DropPath 和残差连接,drop_path=0.0：始终使用该块,    drop_path=0.1：10%的概率跳过该块
        x = self.drop_path(x, self._forward_path)#在训练时随机跳过整个block，实现模型正则化
        return x

    def reset_parameters(self):#初始化模型参数
        self.layer.reset_parameters()
        self.norm.reset_parameters()


class ViLBlockPair(nn.Module):#双向处理单元，包含两个方向相反的 ViLBlock
    def __init__(
            self,
            dim,
            drop_path=0.0,
            conv_kind="2d",
            conv_kernel_size=3,
            proj_bias=True,
            norm_bias=True,
            seqlens=None,
            num_blocks=None,
            init_weights="original",
    ):
        super().__init__()
        self.rowwise_from_top_left = ViLBlock(
            dim=dim,
            direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT,
            drop_path=drop_path,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel_size,
            proj_bias=proj_bias,
            norm_bias=norm_bias,
            seqlens=seqlens,
            num_blocks=num_blocks,
            init_weights=init_weights,
        )
        self.rowwise_from_bot_right = ViLBlock(
            dim=dim,
            direction=SequenceTraversal.ROWWISE_FROM_BOT_RIGHT,
            drop_path=drop_path,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel_size,
            proj_bias=proj_bias,
            norm_bias=norm_bias,
            seqlens=seqlens,
            num_blocks=num_blocks,
            init_weights=init_weights,
        )

    def forward(self, x,leap_mask=None):#  包含两个方向相反的 ViLBlock
        x = self.rowwise_from_top_left(x)
        x = self.rowwise_from_bot_right(x)
        return x


class VisionLSTM2(nn.Module):
    def __init__(
            self,
            dim=192,# 模型的特征维度/隐藏维度
            input_shape=(3, 224, 224),#输入图像的形状
            patch_size=16,#图像patch的大小,num_patches = (224 // 16) × (224 // 16) = 14 × 14 = 196
            depth=12,#模型的深度，ViLBlockPair 的数量。注意，一个 ViLBlockPair 内部包含两个 ViLBlock（一个前向，一个后向），所以实际的 ViLLayer 数量是 depth * 2。
            output_shape=(1000,),#模型的输出维度,在分类模式下，是 (num_classes,)，例如 (1000,) 代表 ImageNet 的1000个类别。在特征模式下，通常为 None，输出形状由 pooling 策略决定。
            mode="classifier",#模型的模式，可以是"classifier"或"features"
            #"classifier": 模型末尾带有一个分类头，用于图像分类任务。    "features": 模型不带分类头，用于提取图像特征，可作为其他任务的骨干网络。
            pooling="bilateral_flatten",#在模型主干之后，如何将序列化的 patch 特征聚合成一个单一的特征向量。
            #使用双向拼接，是因为，第一个patch经过正向LSTM处理；最后一个patch经过反向LSTM处理；两者都包含了丰富的上下文信息。该聚合类似与VIT中的CLS token（全局特征信息聚合）这一做法
            #"bilateral_flatten": 双向拼接。将序列的第一个和最后一个 patch 的特征向量拼接起来。这是一种利用双向处理信息的有效方式。——图像分类
            # "bilateral_avg": 双向平均。将第一个和最后一个 patch 的特征向量取平均。——图像分类
            #"to_image": 将序列特征重新组合成图像形状的特征图 [B, D, H, W]；——图像分割       None: 不进行池化，直接输出序列特征 [B, S, D]。——特征提取 (用于序列处理下游任务)
            drop_path_rate=0.0,#随机深度（DropPath）的最大丢弃率。
            drop_path_decay=False,#否让 drop_path_rate 随着网络深度的增加而线性增长。这是一个常见的正则化策略，浅层网络丢弃率低，深层网络丢弃率高。
            stride=None,  # 步长（用于patch嵌入）
            legacy_norm=False,  # 是否使用旧版归一化
            conv_kind="2d",  # 卷积类型
            conv_kernel_size=3,  # 卷积核大小
            proj_bias=True,  # 投影层是否使用偏置
            norm_bias=True,  # 归一化层是否使用偏置
            init_weights="original",  # 权重初始化方法
    ):
        if depth == 24 and dim < 1024:
            warnings.warn(
                "A single VisionLSTM2 block consists of two subblocks (one for each traversal direction). "
                "ViL-T, ViL-S and ViL-B therefore use depth=12 instead of depth=24, are you sure you want to use "
                "depth=24?"
            )#作者指出，他们的一个 ViLBlockPair 对应 ViT 的一个 Transformer Block。所以，如果用户想构建类似 ViT-Large 的模型，depth 应该是 12，而不是 24。这个警告是为了防止用户因误解而配置出一个异常深的网络。
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        ndim = len(self.input_shape) - 1#空间维度数 (2 for 2D images)
        self.patch_size = to_ntuple(patch_size, n=ndim)#patch大小,to_ntuple将vision_lstm_utils.to_ntuple()函数转换为元组
        self.dim = dim
        self.depth = depth
        self.stride = stride
        self.mode = mode
        self.pooling = pooling
        self.drop_path_rate = drop_path_rate
        self.drop_path_decay = drop_path_decay
        self.conv_kind = conv_kind
        self.conv_kernel_size = conv_kernel_size
        self.proj_bias = proj_bias
        self.norm_bias = norm_bias
        self.init_weights = init_weights

        # initialize patch_embed
        self.patch_embed = VitPatchEmbed(
            dim=dim,
            stride=stride,
            num_channels=self.input_shape[0],
            resolution=self.input_shape[1:],
            patch_size=self.patch_size,
        )#初始化patch_embed,完成数据分块操作，输出结果为[batch_size,len_H,len_W,dim]

        # pos embed
        self.pos_embed = VitPosEmbed2d(seqlens=self.patch_embed.seqlens, dim=dim)#初始化pos_embed,完成位置嵌入操作，输出结果为[batch_size,len_H,len_W,dim]

        # calculate stochastic depth per block
        if drop_path_decay and drop_path_rate > 0.:#如果 drop_path_decay 为 True
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]#创建一个从 0 到 drop_path_rate 线性增长的丢弃率列表，每个 ViLBlockPair 都会得到一个递增的丢弃率。
        else:#
            dpr = [drop_path_rate] * depth#否则，所有块都使用相同的 drop_path_rate。

        # merge two blocks into a blockpair to keep depth equal to the depth of transformers
        # useful to keep layer-wise lr decay implementations consistent with transformers
        self.blocks = nn.ModuleList(#模型的核心主干，由 depth 个 ViLBlockPair 堆叠而成。
            [
                ViLBlockPair(
                    dim=dim,
                    drop_path=dpr[i],
                    conv_kind=conv_kind,
                    seqlens=self.patch_embed.seqlens,
                    proj_bias=proj_bias,
                    norm_bias=norm_bias,
                    num_blocks=depth * 2,
                    init_weights=init_weights,
                )
                for i in range(depth)
            ],
        )
        if pooling == "bilateral_flatten" and mode == "classifier":#如果是pool方法为双向拼接   以及  模式为分类模式
            head_dim = dim * 2
        else:
            head_dim = dim
        self.norm = LayerNorm(dim, bias=norm_bias, eps=1e-6)#设置归一化层，在所有 ViLBlockPair 之后的主归一化层。
        # LEGACY: not needed but was used during training
        if legacy_norm:#旧版归一化（兼容性）
            self.legacy_norm = nn.LayerNorm(head_dim)
        else:
            self.legacy_norm = nn.Identity()

        # head
        if mode == "features":
            # 特征提取模式
            if self.output_shape is not None:
                pass
                #warnings.warn(f"passed mode=features -> output_shape is ignored ({self.output_shape})")#特征模式，忽略所给出的output_shape
            self.head = None
            if self.pooling is None:
                self.output_shape = (self.patch_embed.num_patches, dim)  # 所有patch特征
            elif self.pooling == "to_image":
                self.output_shape = (dim, *self.patch_embed.seqlens)  # 恢复图像形状
            else:#其他情况，会警告这是一个不被支持的池化方式
                warnings.warn(f"passed invalid pooling -> pooling is ignored ({self.pooling})")
                self.pooling = None

        elif mode == "classifier":
            # 分类模式
            #len(self.output_shape) == 1      必须是一维元组，即为(1000,)，不能为（x,y）形式
            assert self.output_shape is not None and len(self.output_shape) == 1, \
                f"define number of classes via output_shape=(num_classes,) (e.g. output_shape=(1000,) for ImageNet-1K"
            #在分类器模式（mode="classifier"）下。它要求用户必须通过 output_shape 指定一个一维的元组（即分类的类别数）。例如，对于ImageNet-1K，输出应该是1000类，所以 output_shape=(1000,)。
            self.head = nn.Linear(head_dim, self.output_shape[0])#分类头构建，为匹配ImageNet-1K，输出是1000类，进行了线性层进行放缩
            # MAE风格的权重初始化（"Masked Autoencoders Are Scalable Vision Learners" (2021)）
            nn.init.trunc_normal_(self.head.weight, std=2e-5)#Truncated Normal 初始化，截断正态分布
            nn.init.zeros_(self.head.bias)
        else:
            raise NotImplementedError

    def load_state_dict(self, state_dict, strict=True):
        # interpolate pos_embed for different resolution (e.g. for fine-tuning on higher-resolution)
        old_pos_embed = state_dict["pos_embed.embed"]
        if old_pos_embed.shape != self.pos_embed.embed.shape:
            state_dict["pos_embed.embed"] = interpolate_sincos(embed=old_pos_embed, seqlens=self.pos_embed.seqlens)
        # remove head and adapt layernorm for feature extraction
        if self.mode == "features":
            state_dict.pop("head.weight", None)
            state_dict.pop("head.bias", None)
            # legacy_norm uses head dim (is doubled for bilateral_concat) -> not usable for feature extraction
            cur_sd = self.state_dict()
            state_dict["legacy_norm.weight"] = cur_sd["legacy_norm.weight"]
            state_dict["legacy_norm.bias"] = cur_sd["legacy_norm.bias"]
        return super().load_state_dict(state_dict=state_dict, strict=strict)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed.embed"}

    def forward(self, x):
        # embed patches
        x = self.patch_embed(x)#分块
        # add pos_embed
        x = self.pos_embed(x)#位置嵌入
        #此时输出结果为[batch_size,len_H,len_W,dim]
        # flatten to 1d
        x = einops.rearrange(x, "b ... d -> b (...) d")#将输入的张量展平为一维，即[batch_size,len_H,len_W,dim]变为[batch_size,len_H*len_W,dim]

        # apply blocks
        for block in self.blocks:#主干网络进行循环，完成12次的 ViLBlockPair
            x = block(x)
        x = self.norm(x)#对结果进行归一化操作
        # pool——池化策略
        if self.pooling is None:
            x = self.legacy_norm(x)
        elif self.pooling == "to_image":
            x = self.legacy_norm(x)
            seqlen_h, seqlen_w = self.patch_embed.seqlens
            x = einops.rearrange(
                x,
                "b (seqlen_h seqlen_w) dim -> b dim seqlen_h seqlen_w",
                seqlen_h=seqlen_h,
                seqlen_w=seqlen_w,
            )
        elif self.pooling == "bilateral_avg":
            # norm after pooling
            x = (x[:, 0] + x[:, -1]) / 2#取出序列的第一个 patch 特征 x[:, 0] (大小为 [B, D]) 和最后一个 x[:, -1] (大小为 [B, D])。得到[B, D]的输出
            x = self.legacy_norm(x)
        elif self.pooling == "bilateral_flatten":
            # norm after pooling
            x = torch.concat([x[:, 0], x[:, -1]], dim=1)#取出序列的第一个 patch 特征 x[:, 0] (大小为 [B, D]) 和最后一个 x[:, -1] (大小为 [B, D])拼接，得到[B, 2D]的输出
            x = self.legacy_norm(x)
        else:
            raise NotImplementedError(f"pooling '{self.pooling}' is not implemented")
        #如果感觉池化会损失一些细节信息，就不应该池化。这正好对应了 VisionLSTM2 中提供的 "to_image" 和 None 两种“非池化”策略。
        # head
        if self.head is not None:#如果需要分类，则需要再经过一个线形层
            x = self.head(x)#如果模型处于分类模式，将池化后的特征向量 x 送入线性头，得到最终的分类 logits

        return x
