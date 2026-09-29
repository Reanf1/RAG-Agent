"""读取项目唯一的 YAML 配置，不依赖当前启动目录。"""

from pathlib import Path

import yaml


def load_config() -> dict:
    """按模块位置定位项目根目录，返回配置字典。"""
    path = Path(__file__).resolve().parents[2] / "config.yaml"
    with path.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)
