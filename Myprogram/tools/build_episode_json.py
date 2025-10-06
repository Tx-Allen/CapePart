import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.episode_utils import collect_samples, samples_to_episodes  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description='根据目录结构生成few-shot episode JSON')
    parser.add_argument('--root', required=True, help='数据根目录，例如 /datasets/train')
    parser.add_argument('--output', required=True, help='输出 JSON 文件路径')
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise FileNotFoundError(root)

    samples = collect_samples(root)
    episodes = samples_to_episodes(samples)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(episodes, f, ensure_ascii=False, indent=2)
    print(f'Wrote {len(episodes)} episodes to {args.output}')


if __name__ == '__main__':
    main()
