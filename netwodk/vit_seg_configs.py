import ml_collections

def get_b16_config():
    """Returns the ViT-B/16 configuration."""

    #图像分块配置
    config = ml_collections.ConfigDict()

    #主干参数配置
    config.patches = ml_collections.ConfigDict({'size': (16, 16)}) #像被分割成大小为 16×16 的 Patches
    config.hidden_size = 768 # 隐藏层维度

    #Transformer配置
    config.transformer = ml_collections.ConfigDict() # config.transformer 是一个嵌套的 ConfigDict，用于存储 Transformer 模型的具体参数。
    config.transformer.mlp_dim = 3072 # Transformer 中多层感知机（MLP）的维度为 3072。
    config.transformer.num_heads = 12 # 多头注意力机制，注意力头的数量为 12
    config.transformer.num_layers = 10 # Transformer有 12 层
    config.transformer.attention_dropout_rate = 0.0 # 注意力dropout率为 0.0
    config.transformer.dropout_rate = 0.1 # Transformer 中的 dropout 率为 0.1，用于防止过拟合。
    #对于Transformer的每个层（包括自注意力层和前馈神经网络层），都有10%的神经元会被随机丢弃。

    #分类与输出配置
    config.classifier = 'seg' # 分类器类型为seg
    config.representation_size = None # 表示输出的特征维度，默认为None，即不进行特征提取
    config.resnet_pretrained_path = None # 预训练的ResNet模型的路径，默认为None，即不使用预训练模型
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_16.npz'
    # 预训练模型的路径，默认为None，即不使用预训练模型
    #某些 ViT 变体或下游任务（如分割）的代码中，可能用 patch_size 作为快捷参数，避免重复定义二维尺寸。此时 patch_size=16 等价于 patches.size=(16, 16)。
    config.patch_size = 16

    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = 'softmax'
    return config


def get_testing():
    # 返回一个用于测试的最小配置
    """Returns a minimal configuration for testing."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_r50_b16_config():
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = get_b16_config()
    config.patches.grid = (16, 16) #添加新的属性grid，用于指定图像分割成补丁的网格大小或补丁的大小
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9) # 定义ResNet50的层数，分别为3、4、9层
    #即，3，4，9层的残差快构建的ResNet50模型，每一个残差块包含 包含 3 个卷积层
    config.resnet.width_factor = 1 #各阶段通道数保持原设计的  1（设置的参数）/1（原设计的值）

    config.classifier = 'seg'
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 2 #分割输出的类别数，输出的类别数为2，（如二分类：背景+目标）
    config.n_skip = 3 # 跳跃连接（Skip Connection）的数量（通常从编码器的中间层跳过连接到解码器）
    config.activation = 'softmax'

    return config


def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.patches.size = (32, 32)
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_32.npz'
    return config


def get_l16_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 16
    config.transformer.num_layers = 24
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.representation_size = None

    # custom
    config.classifier = 'seg'
    config.resnet_pretrained_path = None
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-L_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = 'softmax'
    return config


def get_r50_l16_config():
    """Returns the Resnet50 + ViT-L/16 configuration. customized """
    config = get_l16_config()
    config.patches.grid = (16, 16)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 39)
    config.resnet.width_factor = 1

    config.classifier = 'seg'
    config.resnet_pretrained_path = '../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 2
    config.activation = 'softmax'
    return config


def get_l32_config():
    """Returns the ViT-L/32 configuration."""
    config = get_l16_config()
    config.patches.size = (32, 32)
    return config


def get_h14_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (14, 14)})
    config.hidden_size = 1280
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 5120
    config.transformer.num_heads = 16
    config.transformer.num_layers = 32
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None

    return config
