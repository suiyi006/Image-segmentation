# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from . import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import ResNetV2
logger = logging.getLogger(__name__)
from .add1 import AIFIGate_v12
# 在文件开头导入雷达
from .add_im2 import SemanticBottleneckRadar,SplitConcatFuseGate_Light
ATTENTION_Q = "MultiHeadDotProductAttention_1/query/"
ATTENTION_K = "MultiHeadDotProductAttention_1/key/"
ATTENTION_V = "MultiHeadDotProductAttention_1/value/"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out/"
FC_0 = "MlpBlock_3/Dense_0/"
FC_1 = "MlpBlock_3/Dense_1/"
ATTENTION_NORM = "LayerNorm_0/"
MLP_NORM = "LayerNorm_2/"


def np2th(weights, conv=False):
    #通过函数将numpy数组转化为pytorch张量
    """Possibly convert HWIO to OIHW."""
    if conv:
        #transpose函数 返回一个新的张量，数据不变，仅维度顺序改变
        weights = weights.transpose([3, 2, 0, 1])
        #这一步是将权重从 HWIO[高度, 宽度, 输入通道, 输出通道]  转换为 OIHW[输出通道, 输入通道, 高度, 宽度]
    return torch.from_numpy(weights)
    #from_numpy()函数用于将numpy数组转换为PyTorch张量

def swish(x):# swish激活函数
    #Swish 是一种在深度学习中使用的非线性激活函数，通常用于神经网络中的隐藏层，以引入非线性特性，帮助网络学习复杂的模式。
    return x * torch.sigmoid(x)

#通过使用 ACT2FN 字典，可以根据字符串键（如 'gelu'、'relu' 或 'swish'）来获取相应的激活函数。
ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"] # 注意力头数量，从vit_seg_configs中获取,看之前的 import的文件
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads) # 注意力头大小
        #总隐藏层大小（hidden_size）被均匀分配到每个注意力头。多头注意力的核心思想是将输入拆分为多个头并行计算，最后拼接结果
        self.all_head_size = self.num_attention_heads * self.attention_head_size # 计算所有注意力头加起来的总维度大小，这是一个完整性检查，确保维度分配正确。
        #
        #初始化一个线性层（Linear），其输入维度为config.hidden_size，输出维度为所有注意力头的总维度大小self.all_head_size
        #从而生成 Q K V 三个张量
        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        #生成一个线性层，用于在注意力机制之后对输出进行变换，将拼接后的多头输出映射回 原始维度
        self.out = Linear(config.hidden_size, config.hidden_size)

        #生成两个 dropout 层，用于防止过拟合
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        #作用于 注意力权重矩阵（softmax 后的概率），防止注意力机制过于依赖少数关键位置。
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])
        #作用于 最终输出（self.out 之后），防止网络过拟合输出特征。

        #生成一个softmax层，用于计算注意力概率
        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        #对输入张量x进行形状变换和转置操作，以便于后续的计算。

        #首先获取输入张量x的形状（即各维度的大小），然后去掉最后一个维度的大小，接着在其后添加两个新的维度大小：self.num_attention_heads和self.attention_head_size
        #将输入张量的维度变换为(batch_size, height, width, num_attention_heads, attention_head_size)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        #将输入张量的维度变换为new_x_shape
        #view()函数是 PyTorch 中的一个张量方法（torch.Tensor.view()），功能类似于 NumPy 的 reshape。它 不改变数据本身，仅改变张量的形状（shape）。
        x = x.view(*new_x_shape)
        #对变换后的张量进行转置操作，使得最后两个维度交换位置，即(batch_size, num_attention_heads, attention_head_size, height, width)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
    #生成Q K V --> 多头注意力机制 --> 多头合并 --> 线性层 --> 输出
    #生成Q K V 三个张量
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        #将Q K V 三个张量变换为(batch_size, num_attention_heads, attention_head_size, height, width)
    #多头注意力机制
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        #计算注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    #计算注意力概率
        attention_probs = self.softmax(attention_scores)
        #如果 self.vis=True，保存 attention_probs 用于可视化（如热力图）
        weights = attention_probs if self.vis else None
        #对注意力概率随机失活，防止过拟合
        attention_probs = self.attn_dropout(attention_probs)

        #将注意力概率 attention_probs 与 value_layer 相乘，得到加权后的上下文表示。
        context_layer = torch.matmul(attention_probs, value_layer)
        #将维度交换位置，为之后的 多头合并做准备
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    #定义 多头合并 后的 数据形状
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        #通过 view() 函数将 context_layer 变换为 new_context_layer_shape 的大小
        context_layer = context_layer.view(*new_context_layer_shape)
    #一个线性层，将多头合并后的数据映射回原始维度
        attention_output = self.out(context_layer)
        #对输出进行随机失活
        attention_output = self.proj_dropout(attention_output)
        #返回注意力输出和注意力权重
        return attention_output, weights

#Transformer 中的 MLP 层 即 后面的Feed forward 块
class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        #第一个全连接层，将输入从hidden_size维度映射到mlp_dim维度
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        #第二个全连接层，将mlp_dim维度映射回hidden_size维度
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        #使用GELU(Gaussian Error Linear Unit)激活函数，这是Transformer中常用的激活函数
        self.act_fn = ACT2FN["gelu"]
        #用于防止过拟合，在训练时随机丢弃部分神经元
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()


    def _init_weights(self):#初始化权重
        #使用Xavier均匀初始化，这种初始化方法考虑了输入和输出的维度，有助于保持梯度的稳定
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        #使用标准差为 1e-6 的正态分布初始化了它们的偏置
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
    #全连接层--gelu激活层--dropout层--全连接层--dropout层
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        # hybrid标记是否使用 ResNet 混合架构（True 表示使用 ResNet 提取特征后再分块），适用于高分辨率图像（计算量更小）
        # hybrid 为 False 时 直接对原图分块，输入 Transformer。
        self.hybrid = None
        # 模型配置对象，包含 patches、transformer、resnet 等相关参数。
        self.config = config
        # 确保 img_size 是 (height, width) 形式（如 (224, 224)）
        img_size = _pair(img_size)
        #参数设置 ，hybrid 模式下的 补丁大小 和 分块数量 以及 直接分块下的 补丁大小 和 分块数量
        #判断 config 设定中是否制定了 补丁大小，有 就使用 ResNet 混合架构，  否则 就直接分块
        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"] # grid_size 用于对  ResNet 预处理后的特征图 进行 进一步分块，grid_size 决定如何划分这些块。
            #以下除以 16 是因为 ResNet 提取的特征图是 原图的 16 倍下采样的特征图
            #再 除以grid_size 是为了 将特征图划分成 grid_size[0] 份， 计算每个 patch 在特征图上的大小。
            #grid_size 决定 到底划分 多少块 ，Hybrid  VIT 先 决定 要划分为 多少块
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
            #需要ResNet 预处理
            self.hybrid = True
        else:
            #没有说明 grid_size 要划分为 多少块 ， 就要去 直接分块，将设定的 大小 参数进行获取
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            #直接分块
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
            # 添加双向LSTM层
            # self.lstm = nn.LSTM(
            #     input_size=in_channels,
            #     hidden_size=in_channels//2,  # 双向LSTM会将两个方向的输出拼接，所以hidden_size设为一半
            #     num_layers=2,
            #     bidirectional=True,
            #     batch_first=True
            # )
        # 初始化补丁嵌入层，将输入图像转换为补丁嵌入
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        # 初始化位置嵌入层，将位置信息嵌入到补丁嵌入中
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        # 初始化 dropout 层，用于防止过拟合
        self.dropout = Dropout(config.transformer["dropout_rate"])
        #self.add =DWDualBranchGate(channels=768)

    def forward(self, x):
        # 如果使用 ResNet 混合架构，则首先使用 ResNet 提取特征，然后再进行分块
        if self.hybrid:
            x, features = self.hybrid_model(x)

            # '''增加 LSTM 处理 '''
            # # 在ResNet输出后添加2D LSTM处理
            # B, C, H, W = x.shape
            # # 将空间维度展平为序列 (B, C, H*W) -> (B, H*W, C)
            # x = x.view(B, C, -1).permute(0, 2, 1)
            # #x, _ = self.lstm(x)  # (B, H*W, C)
            # # 恢复空间维度 (B, H, W, C) -> (B, C, H, W)
            # x = x.permute(0, 2, 1).view(B, C, H, W)
            # #print('LSTM处理完成',x.shape)


        #否则 不需要 ResNet 预处理
        else:
            features = None
        #print(f"RseNet 输出结果大小 {x.shape}")
        #print(f"进行 Embedding")
        # 将输入图像转换为补丁嵌入


        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        #print(x.size())
        # x = self.add(x)
        x = x.flatten(2) #指定从 第2个维度开始展平（维度索引从0开始），而保留前两个维度不变。
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        # 将位置嵌入添加到补丁嵌入中,得到 Embeddings
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        #print(f"Embeddings 输出结果大小 {embeddings.shape}")
        # 返回嵌入（embeddings 作为 Transformer 的输入）和 特征（features 用作 辅助任务（如分割）的特征融合 （跳跃连接））
        return embeddings, features


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        #设定隐藏层大小
        self.hidden_size = config.hidden_size
        #设定 注意力机制 前的 归一化层
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        #设定 前向传播 后的 归一化层
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        #设定 前向传播 后的 全连接层，注意力机制 后的 Feed forward 部分
        self.ffn = Mlp(config)
        #设定 注意力机制 层
        self.attn = Attention(config, vis)

    def forward(self, x):#前向传播
        #注意力模块
        #归一化--注意力机制--残差连接
        h = x
        x = self.attention_norm(x)
        #attn该函数  结果是  返回注意力输出和注意力权重
        x, weights = self.attn(x)
        x = x + h
        #Feed forward
        #归一化--全连接层--残差连接
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        #print('Transformer模块结束后得到的数据大小',x.shape)
        return x, weights

    def load_from(self, weights, n_block): #load_from 方法是用来从预训练权重中加载参数的函数，它接受两个参数：weights和n_block，分别表示预训练权重和块的索引。
        ROOT = f"Transformer/encoderblock_{n_block}/"

        #注意力机制部分的参数（权重 和 偏置）；  前馈神经网络(FFN)部分的参数；   两个归一化层(LayerNorm)的参数

        with torch.no_grad():#torch.no_grad() 是 PyTorch 中的一个上下文管理器（context manager），用于禁用梯度计算，通常在模型推理（inference）或参数更新时使用，以减少内存消耗并加速计算。
        #在加载权重时，我们不希望这些操作影响模型的梯度计算：
            ##通过函数       np2th          将numpy数组转化为pytorch张量，     见 36 line
            #pjoin 是 os.path.join 的一个变体或别名，用于跨平台安全地拼接文件路径。它的作用是将多个路径片段组合成一个完整的路径，并自动处理不同操作系统（Windows、Linux、macOS）的路径分隔符问题。
            #weights[path]——作用：从预训练权重字典中提取对应路径的参数 ； 输入：上一步生成的路径字符串 ； 输出：通常是NumPy数组形式的权重值
            #ROOT 是 路径的根目录，ATTENTION_Q 是 注意力机制 的 查询权重 的 名称 (前几行有定义)，kernel 是权重的名称,表示这是权重矩阵（而非偏置），t() 是转置操作，将矩阵的行和列互换。
            #.view(self.hidden_size, self.hidden_size) 表示将权重矩阵的维度变换为 (hidden_size, hidden_size)
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)
            #self.attn.query → 注意力机制的查询（Query）线性层  追朔 其对应的 Linear 函数 有对应的 weight 定义
            #copy_() → PyTorch 的原地复制方法 ，直接赋值（如 self.attn.query.weight = query_weight）会破坏 PyTorch 的计算图。copy_() 能保留张量的自动微分属性。
            # query_weight → 从预训练文件中加载并处理后的权重张量
            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):#实现多层 Transformer 的特征编码,作为 Vit 的一部分
    def __init__(self, config, vis):
    #config: 包含模型配置的对象，如 hidden_size、num_layers 等; vis: 布尔值，控制是否返回注意力权重用于可视化
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList() #nn.ModuleList(): 存储多个 Transformer 层 (Block)
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6) #LayerNorm: 最终的层归一化，稳定训练过程
        self.n_layers = config.transformer["num_layers"]
        for _ in range(config.transformer["num_layers"]): #transformer["num_layers"],见 vit_seg_config.py 中 设定
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer)) #copy.deepcopy: 确保每个 Block 是独立的（避免参数共享）

    def forward(self, hidden_states):  #hidden_states 来自 238行 对于 Attention 函数的引用





        #print('一共有', self.n_layers, '层 Transformer')
        attn_weights = []
        for i in range(self.n_layers):  # 使用range遍历，而不是for layer_block in self.layer
            #print(f'第{i + 1}层')  # 修改print语句
            hidden_states, weights = self.layer[i](hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        #print('Transformer 输出结果大小', encoded.shape)
        return encoded, attn_weights

from .Vil import VilEncoderWithLoad
class Transformer(nn.Module): #整合 整个 ViT 的前向流程
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size) # 初始化嵌入层，返回的是 (embeddings, features) 元组 ，在forward进行区分
        #self.encoder = Encoder(config, vis) # 初始化编码器层，多个 Transformer 层
        self.encoder = VilEncoderWithLoad(
            config=config,
            vis=vis,
            seqlens=(14, 14),
            conv_kind="causal1d",  # 对齐预训练 conv1d
            conv_kernel_size=4,  # 与预训练保持一致更好
            norm_bias=True,
            proj_bias=True,
        )
        # 【新增 1：初始化语义雷达】
        # 注意 num_classes 根据你的实际数据集类别数设置 (Synapse 是 9 类)
        self.radar = SemanticBottleneckRadar(dim=config.hidden_size, num_classes=config['n_classes'])
        # 【新增 2：外部空间抛光的残差参数】(初始化为0)
        self.gamma_bg_polish = nn.Parameter(torch.zeros(1))
        self.gamma_fg_polish = nn.Parameter(torch.zeros(1))
    def forward(self, input_ids):
        # embedding_output, features = self.embeddings(input_ids)
        # encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        # return encoded, attn_weights, embedding_output,features #返回 最后归一化结果 ，注意力权重列表 和 辅助任务（如分割）的特征

        # # 1. 图像分块与扁平化
        # embedding_output, features = self.embeddings(input_ids)  # [B, 196, 768]
        # # ==========================================================
        # # 2. 雷达扫描全图，生成 Leap Mask 和 外部抛光概率图
        # # ==========================================================
        # leap_mask, P_bg, P_fg, aux_logits = self.radar(embedding_output)
        # # 3. 内部支路：送入带有 Leap 掩码保护的 ViL (只处理序列，空间防错乱)
        # encoded, attn_weights = self.encoder(embedding_output, leap_mask=leap_mask)
        # return encoded, attn_weights, embedding_output,features,leap_mask, P_bg, P_fg, aux_logits

        # 1. 提取基础特征
        embedding_output, features = self.embeddings(input_ids)

        # 2. 雷达输出解耦概率
        leap_mask, P_bg, aux_logits = self.radar(embedding_output)

        # 3. 内部支路：前景驱动的 ViL 门控
        encoded, attn_weights = self.encoder(embedding_output, leap_mask=leap_mask)

        # ==========================================================
        # 4. 外部支路：背景驱动的空间边缘锐化 (按照你的新思路)
        # ==========================================================
        # 仅对背景噪声进行乘法物理阻断
        bg_suppressed = encoded * P_bg

        # 负残差融合：从原特征中“抠除”背景杂质
        # 依然保留 gamma_bg 零初始化以保证前期稳定性
        x_final = encoded - self.gamma_bg_polish * bg_suppressed

        return encoded, attn_weights, embedding_output, features, leap_mask, P_bg, x_final, aux_logits
class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels, #输入通道数
            out_channels, #输出通道数
            kernel_size, #卷积核大小
            padding=0, #掩码大小
            stride=1, #步长
            use_batchnorm=True, #是否使用批量归一化
    ):
        conv = nn.Conv2d(    #卷积层
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm), #重要操作：当使用批量归一化时，禁用卷积层的偏置项，因为BN已经包含可学习的偏移参数
        )                             #有BN时：BN(Conv(x) + b) 中的 b 会被BN的减均值操作消除
        relu = nn.ReLU(inplace=True) #激活函数

        bn = nn.BatchNorm2d(out_channels) #批量归一化层

        super(Conv2dReLU, self).__init__(conv, bn, relu) #前向传播时自动按此顺序执行：conv, bn, relu


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, use_gate=False, gate_type=None):
        super().__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        if gate_type == 'v12' and skip_channels > 0:
            self.fuse = AIFIGate_v12(in_channels, skip_channels, out_channels)
            self.conv1 = None
        elif gate_type == 'weak' and skip_channels > 0:
            # 调用你之前 add_im2 里的轻量级模块
            self.fuse = SplitConcatFuseGate_Light(in_channels, skip_channels, out_channels)
            self.conv1 = None
        else:
            self.fuse = None
            # 处理没有 skip 时的通道匹配
            actual_in = in_channels + skip_channels if skip_channels > 0 else in_channels
            self.conv1 = Conv2dReLU(actual_in, out_channels, kernel_size=3, padding=1)

        self.conv2 = Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1)


    def forward(self, x, skip=None):
        x = self.up(x)  # x 通道为 64

        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

            if self.fuse is not None:
                x = self.fuse(x, skip)
            else:
                x = torch.cat([x, skip], dim=1)  # 64 + 16 = 80
                x = self.conv1(x)
        else:
            # 补丁：如果 skip 丢失，且 conv1 期望合并通道，必须进行补齐或修正
            if self.conv1 is not None:
                if not hasattr(self, 'fuse') or self.fuse is None:
                    # 如果 conv1 已经定义为接收 80 通道，但 skip 没来
                    if x.size(1) < self.conv1[0].in_channels:
                        # 方案一：补零（最快但不一定最好）
                        pad_size = self.conv1[0].in_channels - x.size(1)
                        dummy_skip = torch.zeros(x.size(0), pad_size, x.size(2), x.size(3)).to(x.device)
                        x = torch.cat([x, dummy_skip], dim=1)
                x = self.conv1(x)

        x = self.conv2(x)
        return x

class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = copy.deepcopy(config)
        # 获取基础通道
        head_channels = config.hidden_size
        self.decoder_channels = config.decoder_channels  # (256, 128, 64, 16)

        # 自动获取 skip_channels，如果没有定义则设为默认
        self.skip_channels = config.skip_channels if hasattr(config, 'skip_channels') else [512, 256, 64, 16]

        in_channels = [head_channels] + list(self.decoder_channels[:-1])
        out_channels = self.decoder_channels

        self.blocks = nn.ModuleList()
        for i in range(len(self.decoder_channels)):
            # 混合策略配置
            if i == 0:
                gate_type = 'weak'  # 0层：轻量级保底
            elif i in [1, 2]:
                gate_type = 'v12'  # 中间层：高精度交互
            else:
                gate_type = None  # 顶层：直接上采样

            self.blocks.append(DecoderBlock(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                skip_channels=self.skip_channels[i],
                gate_type=gate_type
            ))

    def forward(self, hidden_states, features=None):
        B, L, D = hidden_states.shape
        H = W = int(math.sqrt(L))
        x = hidden_states.transpose(1, 2).reshape(B, D, H, W)

        for i, block in enumerate(self.blocks):
            skip = features[i] if (features is not None and i < len(features)) else None
            x = block(x, skip=skip)
        return x
class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):# upsampling：上采样比例（默认为1，即不上采样）
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2) # 保持空间尺寸不变，改变通道数——将高维特征(in_channels)转换为类别数(out_channels)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity() #当upsampling > 1时：使用双线性插值上采样，将低分辨率预测上采样至原始图像尺寸 ，   否则使用恒等映射（即不上采样）
        super().__init__(conv2d, upsampling) #定义执行顺序：先卷积 → 后上采样



class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes #ImageNet-21k 是 ImageNet-1k（1000 类）的扩展版，实际包含约 21843 个类别
        self.zero_head = zero_head #Flas：分类头使用默认初始化。————True：在训练时，分类头的权重会被初始化为零（或接近零的小随机值）。
        self.classifier = config.classifier # 设定 为 seg （config文件中的设定）
        self.transformer = Transformer(config, img_size, vis) # Transformer 编码器
        self.decoder = DecoderCup(config) # 解码器
        self.segmentation_head = SegmentationHead( # 先卷积 → 后上采样  ———— 生成最终的分类结果
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config
        # 【新增 1：初始化语义雷达】
        # 注意 num_classes 根据你的实际数据集类别数设置 (Synapse 是 9 类)
        self.radar = SemanticBottleneckRadar(dim=config.hidden_size, num_classes=config['n_classes'])
        # 【新增 2：外部空间抛光的残差参数】(初始化为0)
        self.gamma_bg_polish = nn.Parameter(torch.zeros(1))
        self.gamma_fg_polish = nn.Parameter(torch.zeros(1))
        #self.add =  StackedSPPF_DualGate(channels=512)
        self.embeddings = Embeddings(config, img_size=img_size)  # 初始化嵌入层，返回的是 (embeddings, features) 元组 ，在forward进行区分
        # self.encoder = Encoder(config, vis) # 初始化编码器层，多个 Transformer 层
        self.encoder = VilEncoderWithLoad(
            config=config,
            vis=vis,
            seqlens=(14, 14),
            conv_kind="causal1d",  # 对齐预训练 conv1d
            conv_kernel_size=4,  # 与预训练保持一致更好
            norm_bias=True,
            proj_bias=True,
        )
    # def forward(self, x):
    #     if x.size()[1] == 1: #（B,C,H,W） ———— 如果输入的 x 只有一个通道，即灰度图像，则将其扩展为三通道
    #         x = x.repeat(1,3,1,1) # (B, 3, H, W),变为3通道
    #     #Transformer - > Decoder -> SegmentationHead 流程
    #         # 1. 图像分块与扁平化
    #         encoded, attn_weights, embedding_output, features, leap_mask, P_bg, P_fg, aux_logits = self.transformer(x)
    #
    #         # ==========================================================
    #         # 4. 外部支路：空间特征抛光与锐化 (Tri-Gate Re-calibration)
    #         # ==========================================================
    #         # 抑制背景残留噪声
    #         bg_suppressed = encoded * P_bg
    #         # 增强/锐化前景边缘
    #         fg_enhanced = encoded * P_fg
    #
    #         # 残差加成，确保 Epoch 0 时等价于 baseline
    #         x_final = encoded - self.gamma_bg_polish * bg_suppressed + self.gamma_fg_polish * fg_enhanced
    #
    #         # 5. 送入解码器
    #         x_decoded = self.decoder(x_final, features)
    #         logits = self.segmentation_head(x_decoded)
    #
    #         # 【关键】：必须同时返回 aux_logits 给外面算 Loss！
    #         return logits, aux_logits

    # def forward(self, x):
    #     # 1. 通道预处理
    #     if x.size()[1] == 1:
    #         x = x.repeat(1, 3, 1, 1)
    #
    #         # 2. 内部 Transformer 处理 (内含语义雷达与 Leap 门控劫持)
    #     # 修正了拼写错误 trasnformer -> transformer
    #     # 按照 Transformer.forward 的输出解包
    #     encoded, attn_weights, embedding_output, features, leap_mask, P_bg, aux_logits = self.transformer(x)
    #
    #     # ==========================================================
    #     # 3. 外部支路：基于背景概率的空间特征抛光
    #     # ==========================================================
    #     # 非对称设计：ViL 内部已通过前景引导吸收了特征，
    #     # 外部只需利用背景概率 P_bg 剔除边缘溢出的杂质
    #     bg_noise = encoded * P_bg  # 提取背景区域的冗余响应
    #
    #     # 负残差融合：x_final = x_vil - gamma * bg_noise
    #     x_final = encoded - self.gamma_bg_polish * bg_noise
    #
    #     # 4. 解码与输出
    #     x_decoded = self.decoder(x_final, features)
    #     logits = self.segmentation_head(x_decoded)
    #
    #     # 返回主结果和辅助 Logits (用于训练脚本计算辅助 Loss)
    #     return logits, aux_logits

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

            # ==========================================================
        # 完美接住 Transformer 传出的 8 个变量
        # ==========================================================
        encoded, attn_weights, embedding_output, features, leap_mask, P_bg, x_final, aux_logits = self.transformer(x)

        # 直接提取我们需要继续往下传的 x_final 和 features
        x_decoded = self.decoder(x_final, features)
        logits = self.segmentation_head(x_decoded)

        # 同样返回主结果和辅助 Logits
        return logits, aux_logits

    def load_vil_from_nxai(self, ckpt_path="../load/vil2_base16_e400_in1k.th", map_location="cpu", strict: bool = False, verbose: bool = True):
        """
        只加载 Vision-LSTM (ViLBlockPair) 的预训练权重到 transformer.encoder
        - 不加载 patch_embed/pos_embed（你的 Embeddings 仍用原本 position_embeddings）
        - 不加载 ResNet50（hybrid_model）相关
        """
        assert hasattr(self, "transformer") and hasattr(self.transformer, "encoder"), "transformer.encoder not found"
        assert hasattr(self.transformer.encoder, "load_from_nxai"), \
            "transformer.encoder must be VilEncoderWithLoad (missing load_from_nxai method)"
        return self.transformer.encoder.load_from_nxai(
            checkpoint_path=ckpt_path,
            map_location=map_location,
            strict=strict,
            verbose=verbose,
        )
    def load_from(self, weights):
        """从 ViT npz 预训练权重加载：
        - patch_embeddings
        - position_embeddings
        - ResNet50 (hybrid 部分)
        不再加载 Transformer encoder（因为已经换成 VilEncoderWithLoad）。
        """
        with torch.no_grad():
            res_weight = weights

            # 1) Embeddings: patch_embeddings
            self.transformer.embeddings.patch_embeddings.weight.copy_(
                np2th(weights["embedding/kernel"], conv=True)
            )
            self.transformer.embeddings.patch_embeddings.bias.copy_(
                np2th(weights["embedding/bias"])
            )

            # 2) 位置编码：Transformer/posembed_input/pos_embedding
            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings

            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1] - 1 == posemb_new.size()[1]:
                # 去掉 cls token
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info(
                    "load_pretrained: resized variant: %s to %s"
                    % (posemb.size(), posemb_new.size())
                )
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # 3) 不再加载 Transformer.encoder 的权重（原来的 Block + encoder_norm）
            #    —— 因为现在 encoder 是 VilEncoderWithLoad，会从 NX-AI 的 ckpt 加载

            # 4) 如果使用 ResNet50 hybrid，则继续加载 ResNet 权重（不受 ViL 影响）
            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(
                    np2th(res_weight["conv_root/kernel"], conv=True)
                )
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)
    # def load_from(self, weights): # 从预训练权重文件加载参数到当前模型，支持不同尺寸的位置编码插值调整
    #     with torch.no_grad(): # 禁用梯度计算，以免在加载权重时产生不必要的计算
    #         #
    #         res_weight = weights #获取权重
    #         #Transformer 前 的 Embedding 部分的权重pretrained（VIT --  Embedding  ->  Encoder(多个Transformer; Layernor )）
    #         self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
    #         self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
    #         # Encoder 归一化 部分的权重pretrained
    #         self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
    #         self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))
    #         # Vit 块划分————位置编码，从预训练权重文件中加载位置编码(位置嵌入)的操作
    #         posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
    #         #获取当前模型初始化好的位置编码层
    #         posemb_new = self.transformer.embeddings.position_embeddings
    #         if posemb.size() == posemb_new.size(): # 通过条件语句 对 位置编码进行调整，以适应当前模型的需求，如果获取的以及初始化的大小一致，则以获得的为主
    #             self.transformer.embeddings.position_embeddings.copy_(posemb)
    #         elif posemb.size()[1]-1 == posemb_new.size()[1]:#仅多一个token(通常是class token)，则去掉多余的token  常在第一个，从而[:,1:]
    #             posemb = posemb[:, 1:]
    #             self.transformer.embeddings.position_embeddings.copy_(posemb)
    #         else:
    #             logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size())) ## 日志记录尺寸变化
    #             ntok_new = posemb_new.size(1) ## 获取新的token数量
    #             if self.classifier == "seg":
    #                 _, posemb_grid = posemb[:, :1], posemb[0, 1:] ## 分割class token和图像token
    #             gs_old = int(np.sqrt(len(posemb_grid))) ## 计算原始和新的网格尺寸
    #             gs_new = int(np.sqrt(ntok_new))
    #             print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
    #             posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
    #             zoom = (gs_new / gs_old, gs_new / gs_old, 1)
    #             posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
    #             posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
    #             posemb = posemb_grid
    #             self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))
    #
    #         # Encoder whole
    #         for bname, block in self.transformer.encoder.named_children():
    #             for uname, unit in block.named_children():
    #                 unit.load_from(weights, n_block=uname)
    #
    #         if self.transformer.embeddings.hybrid:
    #             self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
    #             gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
    #             gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
    #             self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
    #             self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)
    #
    #             for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
    #                 for uname, unit in block.named_children():
    #                     unit.load_from(res_weight, n_block=bname, n_unit=uname)

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}


