"""Configuration utilities for the Transport Optimization AI package."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any, Dict

import yaml


@dataclass
class Config:
    """Simple container around configuration dictionaries."""

    raw: Dict[str, Any]

    def __getitem__(self, item: str) -> Any:
        return self.raw[item]

    def get(self, item: str, default: Any = None) -> Any:
        return self.raw.get(item, default)


def load_config(path: str | pathlib.Path) -> Config:
    """Load YAML configuration from *path* and return a :class:`Config` wrapper."""

    config_path = pathlib.Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Config(raw=data)


def default_config_path() -> pathlib.Path:
    """Return the default configuration path."""

    return pathlib.Path(__file__).resolve().parents[2] / "conf" / "config.yaml"
