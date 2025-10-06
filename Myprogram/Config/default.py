from yacs.config import CfgNode as CN  # 从yacs库导入配置节点类，负责层级化配置管理

_C = CN()  # 创建根节点配置对象，后续所有子配置都会挂载在这里

# Device setup -------------------------------------------------
_C.SYSTEM = CN()  # 定义系统级配置子节点，记录硬件信息
_C.SYSTEM.NUM_GPUS = 1  # 指定默认使用的GPU数量
_C.SYSTEM.DEVICE = 'cuda'  # 默认运行设备为CUDA，如果没有GPU可改为'cpu'

# Dataset ------------------------------------------------------
_C.DATASET = CN()  # 数据集相关配置子节点
_C.DATASET.ROOT = './datasets'  # 数据集根目录，所有路径都基于此处
_C.DATASET.TRAIN = 'train.json'  # 训练集episode标注文件名
_C.DATASET.VAL = 'val.json'  # 验证集episode标注文件名
_C.DATASET.TEST = 'test.json'  # 测试集episode标注文件名
_C.DATASET.IMAGE_DIR = 'images'  # 存放原始图像的相对子目录
_C.DATASET.MASK_DIR = 'masks'  # 存放掩码图片的相对子目录
_C.DATASET.NUM_CLASSES = 1  # 任务输出的类别数，这里保持1表示单个实例任务
_C.DATASET.NUM_SHOTS = 1  # 每个episode支持集的图像张数（one-shot设置）
_C.DATASET.NUM_QUERIES = 1  # 每个episode查询集的图像张数
_C.DATASET.EPISODES_PER_EPOCH = 1000  # 训练阶段每个epoch随机采样的episode数量
_C.DATASET.VAL_EPISODES = 200  # 验证/测试阶段评估的episode数量
_C.DATASET.IMG_SIZE = 256  # 图像与掩码统一缩放后的边长
_C.DATASET.MEAN = [0.485, 0.456, 0.406]  # 图像归一化时使用的均值（与ImageNet一致）
_C.DATASET.STD = [0.229, 0.224, 0.225]  # 图像归一化时使用的标准差
_C.DATASET.COLOR_FORMAT = 'RGB'  # 指定读取图像的颜色通道顺序
_C.DATASET.USE_JSON = True  # 是否读取JSON episode标注；为False时按文件夹结构自动生成episode
_C.DATASET.IMAGE_EXTS = ['.jpg', '.png', '.jpeg']  # 遍历文件夹时认为是图像的文件后缀
_C.DATASET.MASK_EXTS = ['.png']  # 遍历文件夹时认为是掩码的文件后缀
_C.DATASET.USE_ALL_QUERIES = None  # None表示自动决定；True/False强制控制是否使用全部query样本
_C.DATASET.USE_ALL_QUERIES_EVAL = True  # 验证/测试阶段是否自动使用全部query样本
_C.DATASET.MIN_MASK_AREA = 0.01  # 掩码最小有效面积占比（相对于输入分辨率）
_C.DATASET.MASK_ERASE_RATIO = 0.0  # 训练阶段随机遮挡掩码的比例
_C.DATASET.TRAIN_JSON = 'train.json'
_C.DATASET.VAL_JSON = 'val.json'
_C.DATASET.TEST_JSON = 'test.json'
_C.DATASET.AUTO_BUILD = True  # 是否训练前自动按种子生成train/val json
_C.DATASET.TRAIN_SPLIT_RATIO = None  # 可选：按照比例切分子类，None表示不使用自动切分
_C.DATASET.SPLIT_SEED = 42  # 自动切分子类时使用的随机种子，保证可复现
_C.DATASET.TRAIN_SUBCLASSES = []  # 可选：手工指定属于训练集的子类名列表
_C.DATASET.VAL_SUBCLASSES = []  # 可选：手工指定属于验证集的子类名列表

# Model --------------------------------------------------------
_C.MODEL = CN()  # 模型结构相关配置子节点
_C.MODEL.NAME = 'CapeFormerSegmentation'  # 模型名称，便于日志记录
_C.MODEL.BACKBONE = 'resnet50'  # 骨干网络结构名称，与CapeFormer保持一致
_C.MODEL.PRETRAINED = ''  # 预训练权重路径，留空表示不加载外部权重
_C.MODEL.IMG_SIZE = 256  # 模型内部假定的输入图像尺寸
_C.MODEL.D_MODEL = 256  # Transformer隐藏维度大小
_C.MODEL.NHEAD = 8  # 多头注意力的头数
_C.MODEL.NUM_ENCODER_LAYERS = 3  # Transformer编码器层数
_C.MODEL.NUM_DECODER_LAYERS = 3  # Transformer解码器层数
_C.MODEL.DIM_FEEDFORWARD = 2048  # 前馈网络的隐藏层维度
_C.MODEL.DROPOUT = 0.1  # Transformer模块中的dropout比率
_C.MODEL.MASK_DIM = 256  # 预测掩码特征的维度
_C.MODEL.JOINT_ENCODER_LAYERS = 3  # Query-Support联合精炼编码层数，0表示关闭
_C.MODEL.USE_MULTISCALE_FUSION = False  # 是否启用多尺度特征融合
_C.MODEL.MULTISCALE_REFERENCE = 'res5'  # 多尺度融合对齐的参考层
_C.MODEL.USE_INFO_FUSION = False  # 是否启用InformationFusion模块（通常在第二阶段微调时使用）

# Training -----------------------------------------------------
_C.TRAIN = CN()  # 训练阶段超参数子节点
_C.TRAIN.BATCH_SIZE = 1  # episodic batch size (episodes per iteration)  # 每次迭代处理的episode数量
_C.TRAIN.NUM_WORKERS = 4  # DataLoader并行加载的工作线程数
_C.TRAIN.LR = 1e-4  # Adam/SGD等优化器的学习率
_C.TRAIN.WEIGHT_DECAY = 1e-4  # 权重衰减项，控制正则化强度
_C.TRAIN.EPOCHS = 100  # 计划训练的总epoch数
_C.TRAIN.LOG_INTERVAL = 10  # 日志打印的迭代间隔
_C.TRAIN.CHECKPOINT_DIR = './Checkpoints'  # 模型权重保存目录
_C.TRAIN.GRAD_CLIP = 0.0  # 梯度裁剪阈值，0表示不裁剪
_C.TRAIN.LR_STEP = [60, 90]  # MultiStepLR里程碑
_C.TRAIN.LR_FACTOR = 0.1  # MultiStepLR衰减因子
_C.TRAIN.AUTO_RESUME = False  # 是否自动从checkpoint恢复
_C.TRAIN.PRINT_FREQ = 10  # 与LOG_INTERVAL保持一致，兼容DSLPT风格
_C.TRAIN.EVAL_EPOCH_FREQ = 1  # 每多少个epoch执行一次验证

# Evaluation ---------------------------------------------------
_C.EVAL = CN()  # 评估阶段配置子节点
_C.EVAL.THRESHOLD = 0.5  # 将预测掩码转换为二值时使用的阈值
_C.EVAL.METRIC = 'IoU'  # 评估指标名称，这里使用交并比

# Misc ---------------------------------------------------------
_C.SEED = 1  # 全局随机种子，确保训练过程可复现
# Debug --------------------------------------------------------
_C.DEBUG = CN()
_C.DEBUG.ENABLE = False
_C.DEBUG.PRINT_TENSOR_SHAPES = False
_C.DEBUG.PRINT_STATS = False
# Augmentation / Meta-learning helpers -------------------------
_C.AUG = CN()  # 数据增强与元学习相关的辅助配置
_C.AUG.FLIP_PROB = 0.5  # 水平翻转的概率
_C.AUG.COLOR_JITTER = (0.1, 0.1, 0.1, 0.05)  # brightness, contrast, saturation, hue  # 颜色抖动幅度配置
_C.AUG.MASK_DROPOUT = 0.0  # 按部件随机丢弃支持掩码的概率，鼓励模型依赖其余提示


# 返回配置默认拷贝
def get_cfg_defaults():  # 提供外部调用的接口，返回一份默认配置副本
    # clone一份配置，避免全局状态被外部修改
    return _C.clone()  # 返回克隆出的配置对象，调用方可以安全修改
