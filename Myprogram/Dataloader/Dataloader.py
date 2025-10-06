import json  # 导入JSON工具，用于读取episode标注
import os  # 提供路径拼接与文件存在性检查
import random  # 用于随机抽取episode和样本
from typing import Any, Dict, List, Optional, Set, Tuple  # 类型注解，描述列表、字典和元组
from collections import defaultdict  # 以子类为单位聚合episodes
from pathlib import Path  # 从路径推断子类名称

import numpy as np  # 数值计算库，主要用于随机选择和种子设置
from PIL import Image  # 处理图像读写与格式转换

import torch  # 深度学习张量库
from torch.utils.data import Dataset  # PyTorch数据集基类
from torchvision import transforms  # 常用图像预处理工具

from utils import mask_area  # 掩码面积计算工具
from Dataloader.augmentations import EpisodeAugmentor
try:
    from tools.auto_split_json import generate_splits  # 根据随机种子自动拆分train/val
except ImportError:
    from Myprogram.tools.auto_split_json import generate_splits


class FewShotSegmentationDataset(Dataset):
    """Episodic dataset for part segmentation.

    Annotation JSON格式示例（建议带上子类信息，方便按文件夹划分）::
        [
            {
                "episode_id": "animal_001",
                "subclass": "subclass_A",  # 可选，未提供时将按路径首级目录推断
                "support": [
                    {
                        "image": "subclass_A/images/support_0001.jpg",
                        "masks": [
                            "subclass_A/masks/support_0001_part_head.png",
                            "subclass_A/masks/support_0001_part_wing.png"
                        ],
                        "part_names": ["wing", "tail"]  # 可选
                    }
                ],
                "query": [
                    {
                        "image": "subclass_A/images/query_0001.jpg",
                        "masks": [
                            "subclass_A/masks/query_0001_part_head.png",
                            "subclass_A/masks/query_0001_part_wing.png"
                        ]
                    }
                ]
            }
        ]

    当 ``cfg.DATASET.USE_JSON = False`` 时，会按照以下目录结构自动构建 episode::

        Data/
          subclass_A/
            images/
              support_0001.jpg
              query_0001.jpg
            masks/
              support_0001/part_head.png
              support_0001/part_wing.png
              query_0001/part_head.png
              query_0001/part_wing.png
    """

    # 初始化few-shot分割数据集并建立episode列表
    def __init__(
        self,
        cfg,
        root: str,
        ann_file: str,
        split: str = 'train',
        is_train: bool = True,
        episodes: int = 1000,
        num_shots: int = 1,
        num_queries: int = 1,
        img_size: int = 256,
        mean: List[float] | Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: List[float] | Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.cfg = cfg  # 保存配置引用，便于访问数据及模型参数
        self.root = os.path.abspath(root)  # 将数据根目录转换为绝对路径，便于后续拼接
        self.split = split  # 记录当前数据划分(train/val/test)，驱动子类拆分逻辑
        self.is_train = is_train  # 标记当前数据集是否用于训练
        self.num_shots = num_shots  # 支持集样本数（shots）
        self.num_queries = num_queries  # 查询集样本数
        self.use_json = bool(getattr(cfg.DATASET, 'USE_JSON', True))  # 标记是否从JSON读取episode
        self.image_exts = {ext.lower() for ext in getattr(cfg.DATASET, 'IMAGE_EXTS', ['.jpg', '.png', '.jpeg'])}  # 允许的图像后缀
        self.mask_exts = {ext.lower() for ext in getattr(cfg.DATASET, 'MASK_EXTS', ['.png'])}  # 允许的掩码后缀
        use_all_query_cfg = getattr(cfg.DATASET, 'USE_ALL_QUERIES', None)  # 读取是否使用全部query的开关
        if use_all_query_cfg is None:  # None表示自动决定
            self.use_all_queries = not self.use_json  # 使用文件夹模式时默认全部query，其余保持原逻辑
        else:
            self.use_all_queries = bool(use_all_query_cfg)  # 显式配置优先

        min_area_ratio = float(getattr(cfg.DATASET, 'MIN_MASK_AREA', 0.0))
        self.min_mask_pixels = max(0, int(round(min_area_ratio * img_size * img_size)))
        self.mask_erase_ratio = float(getattr(cfg.DATASET, 'MASK_ERASE_RATIO', 0.0))

        self.image_dir = cfg.DATASET.IMAGE_DIR  # 图像目录相对路径
        self.mask_dir = cfg.DATASET.MASK_DIR  # 掩码目录相对路径

        self.episodes: List[dict] = []
        if self.use_json:
            ann_path = self._maybe_build_annotation(cfg, ann_file)
            raw_annotations = self._load_json_annotations(ann_path)
            self._parse_json_annotations(raw_annotations)
        else:  # 走文件夹自动构建episode的路径
            self.episodes = self._build_episodes_from_folder()  # 根据目录结构生成episodes


        # 构建子类分组：根据元信息或路径前缀
        self.subclass_to_episodes = defaultdict(list)  # 子类名称 -> episode列表
        for epi in self.episodes:  # 遍历所有episode
            subclass = epi.get('subclass')  # 先尝试读取显式子类字段
            if subclass is None and epi['support']:  # 若未提供则根据路径推断
                support_path = Path(epi['support'][0]['image'])  # 取第一张support图像路径并转为Path
                subclass = self._infer_subclass_from_path(support_path)  # 根据路径推断子类名称
            if subclass is None:  # 若依然为空则回退到default标签
                subclass = 'default'
            epi['subclass'] = subclass  # 回写子类信息，后续流程统一使用
            self.subclass_to_episodes[subclass].append(epi)  # 将episode加入对应子类

        self.all_subclass_names = sorted(self.subclass_to_episodes.keys())  # 记录所有可用子类
        self.subclass_partitions = self._build_subclass_partitions()  # 基于配置生成train/val子类划分

        if self.split == 'train':  # 按当前数据集划分选择子类集合
            active_names = self.subclass_partitions.get('train', [])
        elif self.split == 'val':
            active_names = self.subclass_partitions.get('val', [])
        else:
            active_names = self.subclass_partitions.get('test', [])

        self.active_subclass_names = list(active_names)  # 存储当前数据集应使用的子类列表
        if not self.active_subclass_names:  # 若划分为空则直接报错提示配置问题
            raise ValueError(f'No subclasses assigned to split {self.split}.')
        self.active_subclass_to_episodes = {name: self.subclass_to_episodes[name] for name in self.active_subclass_names}  # 构建子类到episode映射

        # 评估阶段按照固定顺序遍历当前划分内的episode
        self.eval_episodes = []  # 保存验证/测试时要遍历的episode
        for name in self.active_subclass_names:  # 只遍历当前split包含的子类
            self.eval_episodes.extend(self.subclass_to_episodes[name])

        if self.is_train:
            # 训练阶段要求覆盖被划分到该split的全部子类，因此直接展平所有episode
            self.train_episode_pool = []  # 顺序收集当前划分内的所有episode
            for name in self.active_subclass_names:
                self.train_episode_pool.extend(self.subclass_to_episodes[name])
            if not self.train_episode_pool:
                raise ValueError(f'No training episodes found for subclasses {self.active_subclass_names}.')
        else:
            self.train_episode_pool = None

        if not self.episodes:  # 没有有效episode时直接报错
            raise ValueError('No valid episodes found in annotation file.')

        self.to_tensor = transforms.Compose([  # 定义图像预处理流水线
            transforms.Resize((img_size, img_size), interpolation=Image.BILINEAR),  # 缩放图像到固定尺寸
            transforms.ToTensor(),  # 转成张量并归一化到[0,1]
            transforms.Normalize(mean=mean, std=std),  # 按ImageNet均值方差标准化
        ])
        self.mask_to_tensor = transforms.Compose([  # 定义掩码预处理
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),  # 使用最近邻缩放保持二值属性
            transforms.PILToTensor(),  # 转成张量但不归一化
        ])

        # 元学习相关的增强配置，用于提升support->query泛化
        aug_cfg = getattr(cfg, 'AUG', None)
        if aug_cfg is not None:
            self.flip_prob = float(getattr(aug_cfg, 'FLIP_PROB', 0.0))
            jitter_cfg = getattr(aug_cfg, 'COLOR_JITTER', None)
            if jitter_cfg:
                if isinstance(jitter_cfg, (list, tuple)) and len(jitter_cfg) >= 4:
                    brightness, contrast, saturation, hue = jitter_cfg[:4]
                else:  # 允许只给一个幅度
                    brightness = contrast = saturation = jitter_cfg
                    hue = 0.0
                hue = max(min(hue, 0.5), -0.5)  # 确保取值合法
                self.color_jitter = transforms.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue,
                )
            else:
                self.color_jitter = None
            self.mask_dropout_prob = float(getattr(aug_cfg, 'MASK_DROPOUT', 0.0))
        else:
            self.flip_prob = 0.0
            self.color_jitter = None
            self.mask_dropout_prob = 0.0

        self.augmentor = EpisodeAugmentor(
            is_train=is_train,
            color_jitter=self.color_jitter,
            flip_prob=self.flip_prob,
            mask_dropout_prob=self.mask_dropout_prob,
            min_mask_pixels=self.min_mask_pixels,
            mask_erase_ratio=self.mask_erase_ratio,
        )

        if not is_train:  # 验证/测试阶段固定随机种子
            random.seed(1)
            np.random.seed(0)

        # 统计总共需要处理的支持/查询图像数量，用于日志前缀
        self.total_samples = sum(len(ep['support']) + len(ep['query']) for ep in self.episodes)
        self._processed_samples = 0

    # ------------------------------------------------------------------
    # JSON / 文件构建辅助函数
    # ------------------------------------------------------------------

    def _maybe_build_annotation(self, cfg, ann_file: str) -> str:
        """Ensure the requested annotation JSON exists. When AUTO_BUILD 为真时缺失会自动生成."""
        ann_path = os.path.join(self.root, ann_file)
        if os.path.exists(ann_path):
            return ann_path

        if not getattr(cfg.DATASET, 'AUTO_BUILD', False):
            raise FileNotFoundError(f'Annotation {ann_path} not found. Set DATASET.AUTO_BUILD=True to auto-generate.')

        root_path = Path(self.root)
        ratio = cfg.DATASET.TRAIN_SPLIT_RATIO if cfg.DATASET.TRAIN_SPLIT_RATIO is not None else 0.8
        ratio = max(0.0, min(1.0, float(ratio)))
        train_json = root_path / cfg.DATASET.TRAIN_JSON
        val_json = root_path / cfg.DATASET.VAL_JSON
        test_json = root_path / cfg.DATASET.TEST_JSON
        print(
            f"[INFO] Missing annotation {ann_file}; auto-building train/val JSON at {root_path}"
            f" (ratio={ratio}, seed={cfg.DATASET.SPLIT_SEED})."
        )
        generate_splits(root_path, train_json, val_json, ratio, cfg.DATASET.SPLIT_SEED, test_json)
        return os.path.join(self.root, ann_file)

    def _load_json_annotations(self, ann_path: str) -> List[dict]:
        with open(ann_path, 'r') as f:
            return json.load(f)

    def _parse_json_annotations(self, raw: List[dict]) -> None:
        for idx, episode in enumerate(raw):  # 遍历每个episode条目
            support_set = episode.get('support', [])
            query_set = episode.get('query', [])
            if not support_set or not query_set:
                continue

            resolved_support = [self._resolve_entry(item) for item in support_set]
            resolved_query = [self._resolve_entry(item) for item in query_set]

            self.episodes.append(
                {
                    'episode_id': episode.get('episode_id', f'episode_{idx}'),
                    'subclass': episode.get('subclass'),
                    'support': resolved_support,
                    'query': resolved_query,
                }
            )

    # 解析路径，支持绝对路径或相对路径
    def _resolve_path(self, path: str, default_dir: str) -> str:
        if os.path.isabs(path):  # 已是绝对路径直接返回
            return path
        candidates = [  # 构造两个候选：直接拼接root或拼接默认子目录
            os.path.join(self.root, path),
            os.path.join(self.root, default_dir, path),
        ]
        for candidate in candidates:  # 依次尝试两个候选路径
            if os.path.exists(candidate):  # 找到存在的路径立即返回
                return candidate
        return os.path.join(self.root, path)  # 都不存在则返回root+原始路径，错误稍后抛出

    def _resolve_entry(self, item: dict) -> dict:
        resolved = {
            'image': self._resolve_path(item['image'], self.image_dir),  # 解析图像路径
        }
        mask_mode = item.get('mask_mode', 'multi_channel')
        resolved['mask_mode'] = mask_mode
        if 'masks' in item and item['masks']:  # 若提供多部件掩码列表
            resolved['masks'] = [self._resolve_path(p, self.mask_dir) for p in item['masks']]
        elif 'mask' in item:  # 兼容旧格式，只含单个mask
            resolved['masks'] = [self._resolve_path(item['mask'], self.mask_dir)]
        else:
            raise ValueError(f"Item {item} must contain 'masks' or 'mask'")  # 缺少掩码字段时报错

        if mask_mode != 'multi_channel':
            raise ValueError("Only multi-channel PNG masks are supported. Please regenerate annotation.")

        if 'part_names' in item:  # 如果标注里有部件名称则一并保留
            resolved['part_names'] = item['part_names']
        return resolved

    # 依据图像路径推断所属子类名称
    def _infer_subclass_from_path(self, path: Path) -> str:
        parts = path.parts  # 拆分路径层级
        for marker in ('images', 'masks'):  # 优先找到images或masks目录
            if marker in parts:
                idx = parts.index(marker)
                if idx > 0:
                    return parts[idx - 1]  # 取其上一级目录作为子类名
        return parts[-2] if len(parts) >= 2 else 'default'  # 兜底逻辑

    # 从超类/子类文件夹结构自动构造episodes
    def _build_episodes_from_folder(self) -> list:
        root_dir = Path(self.root)
        if not root_dir.is_dir():
            raise FileNotFoundError(f'Dataset root {root_dir} not found.')

        episodes = []  # 临时保存构造出的episodes
        for super_dir in sorted(root_dir.iterdir()):
            if not super_dir.is_dir():
                continue
            for subclass_dir in sorted(super_dir.iterdir()):  # 遍历每个子类文件夹
                if not subclass_dir.is_dir():
                    continue
                subclass_name = subclass_dir.name
                images_dir = subclass_dir / 'images'
                masks_dir = subclass_dir / 'maks'
                if not masks_dir.exists():
                    masks_dir = subclass_dir / 'masks'
                if not images_dir.is_dir() or not masks_dir.is_dir():
                    continue

                image_entries = []
                for img_path in sorted(images_dir.iterdir()):  # 遍历images目录
                    if not img_path.is_file():
                        continue
                    if img_path.suffix.lower() not in self.image_exts:
                        continue
                    mask_info = self._collect_masks_for_image(masks_dir, img_path.stem)
                    if mask_info is None:
                        continue
                    entry = {
                        'image': str(img_path),
                        'masks': mask_info['paths'],
                        'mask_mode': mask_info['mode'],
                    }
                    if mask_info.get('part_names'):
                        entry['part_names'] = mask_info['part_names']
                    image_entries.append(entry)

                if len(image_entries) <= 1:
                    continue

                for idx, support_entry in enumerate(image_entries):
                    query_entries = [image_entries[j] for j in range(len(image_entries)) if j != idx]
                    episodes.append(
                        {
                            'episode_id': f'{super_dir.name}_{subclass_name}_{Path(support_entry["image"]).stem}',
                            'subclass': subclass_name,
                            'support': [support_entry],
                            'query': query_entries,
                        }
                    )

        return episodes

    def _collect_masks_for_image(self, masks_dir: Path, image_stem: str):
        for ext in self.mask_exts:
            candidate = masks_dir / f'{image_stem}{ext}'
            if candidate.is_file():
                return {'paths': [str(candidate)], 'mode': 'multi_channel'}

        # 不再支持多文件模式，若未找到多通道掩码则跳过该样本
        print(f"[WARN] skip sample '{image_stem}' because multi-channel mask not found in {masks_dir}")
        return None

    def _load_multi_channel_mask(self, mask_path: str) -> List[Image.Image]:
        arr = np.array(Image.open(mask_path))
        channels = []
        if arr.ndim == 2:  # 单通道标签图，数值表示部件ID
            max_label = int(arr.max())
            for label in range(max_label + 1):
                channel = (arr == label).astype(np.uint8) * 255
                channels.append(Image.fromarray(channel))
        elif arr.ndim == 3:  # 多通道掩码
            for idx in range(arr.shape[2]):
                channel = arr[..., idx]
                if channel.dtype != np.uint8:
                    channel = (channel > 0).astype(np.uint8) * 255
                else:
                    channel = (channel > 0).astype(np.uint8) * 255
                channels.append(Image.fromarray(channel))
        else:
            raise ValueError(f'Unsupported mask format for {mask_path}, array shape {arr.shape}')

        return channels

    # 基于配置构建train/val/test子类划分
    def _build_subclass_partitions(self) -> dict:
        partitions = {}  # 用字典保存各split对应的子类列表

        def _filter_existing(names):  # 过滤不存在的子类并保持原有顺序
            filtered = []
            for name in names:
                if name in self.subclass_to_episodes and name not in filtered:
                    filtered.append(name)
            return filtered

        train_manual = list(getattr(self.cfg.DATASET, 'TRAIN_SUBCLASSES', []))  # 读取手工指定的训练子类
        if train_manual:  # 若配置了显式列表则优先采用
            partitions['train'] = _filter_existing(train_manual)

        val_manual = list(getattr(self.cfg.DATASET, 'VAL_SUBCLASSES', []))  # 读取手工指定的验证子类
        if val_manual:
            partitions['val'] = _filter_existing(val_manual)

        train_ratio = getattr(self.cfg.DATASET, 'TRAIN_SPLIT_RATIO', None)  # 读取按比例划分配置
        if train_ratio is not None and self.all_subclass_names:  # 只有存在子类时才进行比例划分
            try:
                ratio_value = float(train_ratio)  # 将比例转换为浮点数
            except (TypeError, ValueError) as exc:  # 比例配置非法时给出明确报错
                raise ValueError(f'cfg.DATASET.TRAIN_SPLIT_RATIO must be numeric, got {train_ratio}') from exc
            ratio_value = min(max(ratio_value, 0.0), 1.0)  # 将比例限制在[0,1]
            seed_value = int(getattr(self.cfg.DATASET, 'SPLIT_SEED', 42))  # 拿到随机拆分使用的种子
            shuffled = self.all_subclass_names.copy()  # 拷贝一个子类列表用于洗牌
            random.Random(seed_value).shuffle(shuffled)  # 用固定种子打乱，确保可复现
            split_idx = max(1, int(round(len(shuffled) * ratio_value))) if shuffled else 0  # 按比例计算训练子类数量，至少为1
            if split_idx >= len(shuffled):  # 若全部分到训练侧则验证侧回退到全部子类
                auto_train = shuffled
                auto_val = shuffled
            else:
                auto_train = shuffled[:split_idx]  # 按比例截取训练子类
                auto_val = shuffled[split_idx:]  # 剩余子类作为验证子类
                if not auto_val:  # 兜底保障验证集不为空
                    auto_val = shuffled
            auto_train = _filter_existing(auto_train)
            auto_val = _filter_existing(auto_val)
            if auto_train:
                partitions.setdefault('train', auto_train)  # 若未手动指定则采用自动划分
            if auto_val:
                partitions.setdefault('val', auto_val)

        partitions['train'] = _filter_existing(partitions.get('train', [])) or self.all_subclass_names.copy()  # 训练划分兜底为全部子类
        partitions['val'] = _filter_existing(partitions.get('val', [])) or self.all_subclass_names.copy()  # 验证划分兜底为全部子类
        partitions['test'] = _filter_existing(partitions.get('test', [])) or self.all_subclass_names.copy()  # 测试划分默认使用全部子类

        return partitions  # 返回拆分结果

    # 返回总episode数量或每个epoch的采样量
    def __len__(self) -> int:
        if self.is_train:  # 训练阶段直接覆盖全部子类内的episode
            return len(self.train_episode_pool)
        return len(self.eval_episodes)  # 验证/测试阶段遍历所有episode

    # 构建单个episode的支持集与查询集样本
    def __getitem__(self, idx: int):
        if self.is_train:
            episode = self.train_episode_pool[idx]  # 直接按索引取出对应episode，确保覆盖完整子类
            support_entries = self._sample_entries(
                episode['support'],
                self.num_shots,
                randomize=True,
            )
            query_entries = self._sample_entries(
                episode['query'],
                self.num_queries,
                randomize=True,
                use_all=self.use_all_queries,
            )
        else:
            episode = self.eval_episodes[idx]
            support_entries = self._sample_entries(
                episode['support'],
                self.num_shots,
                self.is_train,
            )
            use_all_queries = False
            if self.use_all_queries:
                use_all_queries = True
            else:
                use_all_queries = bool(getattr(self.cfg.DATASET, 'USE_ALL_QUERIES_EVAL', False))
            query_entries = self._sample_entries(
                episode['query'],
                self.num_queries,
                self.is_train,
                use_all=use_all_queries,
            )

        support_images, support_masks = self._load_batch(support_entries, label='support')  # 加载支持集图像与掩码
        query_images, query_masks = self._load_batch(query_entries, label='query')  # 加载查询集图像与掩码

        support_image_names = [Path(entry['image']).name for entry in support_entries]
        query_image_names = [Path(entry['image']).name for entry in query_entries]

        support_part_meta = self._extract_part_names(support_entries, support_masks.shape[1])
        query_part_meta = self._extract_part_names(query_entries, query_masks.shape[1])

        (
            support_masks,
            query_masks,
            alignment_info,
        ) = self._align_support_query_masks(
            support_masks,
            query_masks,
            support_part_meta,
            query_part_meta,
            episode_id=episode['episode_id'],
            support_image_names=support_image_names,
            query_image_names=query_image_names,
        )

        num_parts = support_masks.shape[1]  # 记录当前episode部件数量，后续模型需要

        meta = {  # 组装元信息，方便调试或可视化
            'episode_id': episode['episode_id'],
            'support_paths': [entry['image'] for entry in support_entries],
            'query_paths': [entry['image'] for entry in query_entries],
            'num_parts': num_parts,
            'part_names': alignment_info['part_names'],
            'part_alignment': alignment_info,
        }

        return {  # 返回模型训练所需的张量与元数据
            'support_images': support_images,
            'support_masks': support_masks,
            'query_images': query_images,
            'query_masks': query_masks,
            'meta': meta,
        }

    # 随机或顺序抽取指定数量的样本
    def _sample_entries(self, entries, k: int, randomize: bool, *, use_all: bool = False):
        if use_all:  # 需要使用全部样本时直接返回
            return list(entries)
        if len(entries) >= k:  # 数据量充足时
            if randomize:
                indices = np.random.choice(len(entries), k, replace=False)  # 随机不放回采样
            else:
                indices = list(range(k))  # 验证阶段直接取前k个
        else:  # 数据量不足时允许重复
            indices = np.random.choice(len(entries), k, replace=True) if randomize else list(range(len(entries)))
            while len(indices) < k:  # 顺序模式下补齐索引，保证数量
                indices.append(indices[-1])
        return [entries[i] for i in indices]

    def _open_image(self, path: str) -> Image.Image:
        return Image.open(path).convert('RGB')

    def _load_mask_images(self, entry: dict):
        mask_paths = entry.get('masks') or []
        mask_mode = entry.get('mask_mode', 'multi_channel')
        if mask_mode == 'multi_channel':
            if not mask_paths:
                raise ValueError(f"Entry {entry['image']} missing multi-channel mask path")
            mask_imgs = self._load_multi_channel_mask(mask_paths[0])
        else:
            mask_imgs = [Image.open(mask_path).convert('L') for mask_path in mask_paths]
        return mask_imgs, mask_paths, mask_mode

    def _apply_augmentations(self, img: Image.Image, masks: List[Image.Image], *, is_support: bool):
        return self.augmentor.apply(img, masks, is_support=is_support)

    def _build_part_tensors(
        self,
        mask_images: List[Image.Image],
        mask_paths,
        mask_mode: str,
        prefix: str,
        *,
        is_support: bool,
    ):
        part_tensors = []
        warned = False
        for part_idx, mask_img in enumerate(mask_images):
            tensor = self._mask_tensor(mask_img)
            tensor = self.augmentor.refine_mask_tensor(tensor, is_support=is_support)
            if mask_area(tensor) == 0:
                src_mask_path = mask_paths[0] if mask_mode == 'multi_channel' else mask_paths[min(part_idx, len(mask_paths) - 1)]
                print(f"{prefix} WARN zero-area mask after downsample: {src_mask_path} (part_idx={part_idx})")
                warned = True
            part_tensors.append(tensor)
        return part_tensors, warned

    def _ensure_part_consistency(self, expected_parts, part_tensors):
        if expected_parts is None:
            return len(part_tensors)
        if len(part_tensors) != expected_parts:
            raise ValueError(
                f"Inconsistent part count within episode: expected {expected_parts}, got {len(part_tensors)}"
            )
        return expected_parts

    # 批量加载图像与掩码
    def _load_batch(self, batch_entries, *, label: str):
        assert label in ('support', 'query'), f'Unknown label {label}'
        is_support = label == 'support'
        images, masks = [], []
        expected_parts = None  # 记录当前episode的部件数量以保证一致
        for entry in batch_entries:
            self._processed_samples += 1
            prefix = f"[{self._processed_samples}/{self.total_samples}] [{label}]"
            img = self._open_image(entry['image'])
            mask_imgs, mask_paths, mask_mode = self._load_mask_images(entry)
            aug_img, aug_masks = self._apply_augmentations(img, mask_imgs, is_support=is_support)

            images.append(self.to_tensor(aug_img))

            part_tensors, warned = self._build_part_tensors(
                aug_masks,
                mask_paths,
                mask_mode,
                prefix,
                is_support=is_support,
            )
            expected_parts = self._ensure_part_consistency(expected_parts, part_tensors)

            part_stack = torch.stack(part_tensors)
            if is_support:
                part_stack = self.augmentor.apply_dropout(part_stack)

            masks.append(part_stack)

            if not warned:
                stem = Path(entry['image']).name
                print(f"{prefix} Done {stem}")

        images_tensor = torch.stack(images)
        masks_tensor = torch.stack(masks)
        return images_tensor, masks_tensor

    def _extract_part_names(self, entries: List[dict], part_count: int) -> Dict[str, Any]:
        """提取部件名称并补齐长度，返回唯一名称及显示名称映射。"""

        if part_count <= 0:
            return {'names': [], 'display_map': {}, 'provided_flags': []}

        raw_names: List[str] = []
        for entry in entries:
            names = entry.get('part_names') or []
            if names:
                raw_names = [name.strip() for name in names]
                break

        canonical_names: List[str] = []
        provided_flags: List[bool] = []
        for idx in range(part_count):
            if idx < len(raw_names) and raw_names[idx]:
                base_name = raw_names[idx]
                provided_flags.append(True)
            else:
                base_name = f'part_{idx}'
                provided_flags.append(False)
            canonical_names.append(base_name)

        unique_names: List[str] = []
        display_map: Dict[str, str] = {}
        unique_flags: List[bool] = []
        seen: Dict[str, int] = {}
        for idx, base_name in enumerate(canonical_names):
            occurrence = seen.get(base_name, 0)
            seen[base_name] = occurrence + 1
            if occurrence == 0:
                unique = base_name
            else:
                unique = f'{base_name}#{occurrence}'
            unique_names.append(unique)
            display_map[unique] = base_name
            unique_flags.append(provided_flags[idx])

        return {
            'names': unique_names,
            'display_map': display_map,
            'provided_flags': unique_flags,
        }

    def _align_support_query_masks(
        self,
        support_masks: torch.Tensor,
        query_masks: torch.Tensor,
        support_part_meta: Dict[str, Any],
        query_part_meta: Dict[str, Any],
        *,
        episode_id: str,
        support_image_names: List[str],
        query_image_names: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """对齐support/query的掩码通道，确保两侧具有一致的部件集合。"""

        support_names = support_part_meta['names']
        query_names = query_part_meta['names']
        support_flags = support_part_meta.get('provided_flags', [True] * len(support_names))
        query_flags = query_part_meta.get('provided_flags', [True] * len(query_names))

        support_count = support_masks.shape[1]
        query_count = query_masks.shape[1]
        min_count = min(support_count, query_count)

        support_keys: List[Tuple[str, Any]] = []
        query_keys: List[Tuple[str, Any]] = []

        for idx in range(min_count):
            support_has_name = idx < len(support_flags) and support_flags[idx]
            query_has_name = idx < len(query_flags) and query_flags[idx]
            if support_has_name and query_has_name:
                support_keys.append(('name', support_names[idx]))
                query_keys.append(('name', query_names[idx]))
            else:
                support_keys.append(('index', idx))
                query_keys.append(('index', idx))

        for idx in range(min_count, support_count):
            if idx < len(support_flags) and support_flags[idx]:
                support_keys.append(('name', support_names[idx]))
            else:
                support_keys.append(('index', idx))

        for idx in range(min_count, query_count):
            if idx < len(query_flags) and query_flags[idx]:
                query_keys.append(('name', query_names[idx]))
            else:
                query_keys.append(('index', idx))

        union_keys: List[Tuple[str, Any]] = []
        seen_keys: Set[Tuple[str, Any]] = set()

        def _register_key(key: Tuple[str, Any]):
            if key not in seen_keys:
                union_keys.append(key)
                seen_keys.add(key)

        for key in support_keys:
            _register_key(key)
        for key in query_keys:
            _register_key(key)

        if not union_keys:
            raise ValueError(
                f"Episode {episode_id} has no valid part channels in support/query masks."
            )

        support_lookup = {name: idx for idx, name in enumerate(support_names)}
        query_lookup = {name: idx for idx, name in enumerate(query_names)}

        support_display_map = support_part_meta['display_map']
        query_display_map = query_part_meta['display_map']

        def _display_for_key(key: Tuple[str, Any]) -> str:
            kind, value = key
            if kind == 'name':
                return support_display_map.get(
                    value,
                    query_display_map.get(value, value),
                )
            idx = int(value)
            candidates = []
            if idx < len(support_names):
                name = support_names[idx]
                candidates.append(support_display_map.get(name))
            if idx < len(query_names):
                name = query_names[idx]
                candidates.append(query_display_map.get(name))
            for candidate in candidates:
                if candidate:
                    return candidate
            return f'part_{idx}'

        support_channels: List[torch.Tensor] = []
        query_channels: List[torch.Tensor] = []
        support_missing: List[str] = []
        query_missing: List[str] = []
        support_reorder: List[Optional[int]] = []
        query_reorder: List[Optional[int]] = []

        for key in union_keys:
            display_name = _display_for_key(key)
            kind, value = key
            if kind == 'name' and value in support_lookup:
                idx = support_lookup[value]
                support_channels.append(support_masks[:, idx : idx + 1])
                support_reorder.append(idx)
            elif kind == 'index' and int(value) < support_count:
                idx = int(value)
                support_channels.append(support_masks[:, idx : idx + 1])
                support_reorder.append(idx)
            else:
                support_channels.append(
                    support_masks.new_zeros(
                        (support_masks.shape[0], 1, support_masks.shape[2], support_masks.shape[3])
                    )
                )
                support_missing.append(display_name)
                support_reorder.append(None)

            if kind == 'name' and value in query_lookup:
                idx = query_lookup[value]
                query_channels.append(query_masks[:, idx : idx + 1])
                query_reorder.append(idx)
            elif kind == 'index' and int(value) < query_count:
                idx = int(value)
                query_channels.append(query_masks[:, idx : idx + 1])
                query_reorder.append(idx)
            else:
                query_channels.append(
                    query_masks.new_zeros(
                        (query_masks.shape[0], 1, query_masks.shape[2], query_masks.shape[3])
                    )
                )
                query_missing.append(display_name)
                query_reorder.append(None)

        aligned_support = torch.cat(support_channels, dim=1)
        aligned_query = torch.cat(query_channels, dim=1)

        if aligned_support.shape[1] != aligned_query.shape[1]:
            raise RuntimeError(
                f"Episode {episode_id} alignment failed: support has {aligned_support.shape[1]} parts,"
                f" query has {aligned_query.shape[1]} parts."
            )

        union_display_names = [_display_for_key(key) for key in union_keys]

        if support_missing:
            print(
                f"[WARN] Episode {episode_id} support lacks parts {support_missing}; padded zero masks to align query."
                f" Support images: {support_image_names}"
            )
        if query_missing:
            print(
                f"[WARN] Episode {episode_id} query lacks parts {query_missing}; padded zero masks to align support."
                f" Query images: {query_image_names}"
            )

        info = {
            'support_parts_before': support_masks.shape[1],
            'query_parts_before': query_masks.shape[1],
            'part_names': union_display_names,
            'aligned_parts': aligned_support.shape[1],
            'support_missing_parts': support_missing,
            'query_missing_parts': query_missing,
            'support_reorder_indices': support_reorder,
            'query_reorder_indices': query_reorder,
            'alignment_keys': [
                {
                    'type': kind,
                    'value': int(value) if kind == 'index' else value,
                }
                for kind, value in union_keys
            ],
        }

        return aligned_support, aligned_query, info

    # 将掩码图转换为归一化张量
    def _mask_tensor(self, mask: Image.Image) -> torch.Tensor:
        tensor = self.mask_to_tensor(mask).float()  # 应用掩码预处理并转浮点
        if tensor.max() > 0:
            tensor = tensor / tensor.max()  # 将掩码归一化到0-1区间
        return tensor.squeeze(0)  # 去掉单通道维度


# 根据配置构建DataLoader
def build_dataloader(cfg, split: str, is_train: bool):
    if split == 'train':
        ann = cfg.DATASET.TRAIN  # 训练集标注文件名
        episodes = cfg.DATASET.EPISODES_PER_EPOCH  # 每个epoch的episode数
    elif split == 'val':
        ann = cfg.DATASET.VAL  # 验证集标注文件名
        episodes = cfg.DATASET.VAL_EPISODES  # 验证阶段使用的episode数
    elif split == 'test':
        ann = cfg.DATASET.TEST  # 测试集标注文件名
        episodes = cfg.DATASET.VAL_EPISODES  # 测试沿用验证episode数量
    else:
        raise ValueError(split)  # 未知split直接报错

    dataset = FewShotSegmentationDataset(
        cfg,
        cfg.DATASET.ROOT,
        ann,
        split=split,
        is_train=is_train,
        episodes=episodes,
        num_shots=cfg.DATASET.NUM_SHOTS,
        num_queries=cfg.DATASET.NUM_QUERIES,
        img_size=cfg.DATASET.IMG_SIZE,
        mean=cfg.DATASET.MEAN,
        std=cfg.DATASET.STD,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE if is_train else 1,
        shuffle=bool(is_train),
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=True,
    )

    return loader
