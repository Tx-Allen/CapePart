from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}
MASK_EXTS = {'.png'}


@dataclass
class SampleEntry:
    super_name: str
    subclass: str
    image_path: Path  # absolute
    rel_image: str
    mask_paths: List[str]
    mask_mode: str  # currently仅支持 multi_channel
    stem: str


def _find_mask_paths(masks_dir: Path, stem: str):
    # 仅支持单PNG多通道；若不存在则返回None
    for ext in MASK_EXTS:
        candidate = masks_dir / f'{stem}{ext}'
        if candidate.is_file():
            return [candidate], 'multi_channel'
    return None, None


def collect_samples(root: Path) -> List[SampleEntry]:
    root = root.resolve()
    samples: List[SampleEntry] = []
    for super_dir in sorted(root.iterdir()):
        if not super_dir.is_dir():
            continue
        for subclass_dir in sorted(super_dir.iterdir()):
            if not subclass_dir.is_dir():
                continue
            images_dir = subclass_dir / 'images'
            masks_dir = subclass_dir / 'maks'
            if not masks_dir.exists():
                masks_dir = subclass_dir / 'masks'
            if not images_dir.is_dir() or not masks_dir.is_dir():
                continue

            for img_path in sorted(images_dir.iterdir()):
                if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                stem = img_path.stem
                mask_paths, mode = _find_mask_paths(masks_dir, stem)
                if mask_paths is None:
                    print(f"[WARN] Skip {img_path} because multi-channel mask not found")
                    continue
                rel_image = img_path.relative_to(root).as_posix()
                rel_masks = [m.relative_to(root).as_posix() for m in mask_paths]
                samples.append(
                    SampleEntry(
                        super_name=super_dir.name,
                        subclass=subclass_dir.name,
                        image_path=img_path,
                        rel_image=rel_image,
                        mask_paths=rel_masks,
                        mask_mode=mode,
                        stem=stem,
                    )
                )
    return samples


def group_samples_by_subclass(samples: List[SampleEntry]) -> Dict[str, List[SampleEntry]]:
    grouped: Dict[str, List[SampleEntry]] = {}
    for sample in samples:
        key = f"{sample.super_name}/{sample.subclass}"
        grouped.setdefault(key, []).append(sample)
    return grouped


def samples_to_episodes(samples: List[SampleEntry]):
    episodes = []
    grouped = group_samples_by_subclass(samples)
    for key, entries in grouped.items():
        if len(entries) <= 1:
            continue
        super_name, subclass = key.split('/', 1)
        entries_sorted = sorted(entries, key=lambda e: e.stem)
        for idx, support in enumerate(entries_sorted):
            query = [e for j, e in enumerate(entries_sorted) if j != idx]
            episode = {
                'episode_id': f'{super_name}_{subclass}_{support.stem}',
                'subclass': subclass,
                'support': [{
                    'image': support.rel_image,
                    'masks': support.mask_paths,
                    'mask_mode': support.mask_mode,
                }],
                'query': [{
                    'image': q.rel_image,
                    'masks': q.mask_paths,
                    'mask_mode': q.mask_mode,
                } for q in query],
            }
            episodes.append(episode)
    return episodes
