"""Hydra entry point for the Self TIME augmentation sweep."""

from __future__ import annotations

import logging
import os
import random
import sys
from importlib import import_module
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from timebench.evaluation.data import DEFAULT_CONFIG_PATH, load_dataset_config
from timebench.proposal import validate_self_augmentations


MODELS = {
    "chronos2": ("timebench.model_loading.chronos2", "run_chronos2_experiment"),
    "ts_icl": ("timebench.model_loading.ts_icl", "run_tsicl_experiment"),
}


def _path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _datasets(values: list[str], config_path: Path | None) -> list[str]:
    if values == ["all_datasets"]:
        config = load_dataset_config(config_path)
        return list(config.get("datasets", {}).keys())
    if "all_datasets" in values:
        raise ValueError("all_datasets must be the only dataset selector")
    return values


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(version_base=None, config_path="../conf", config_name="self_time")
def main(config: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    model = str(config.model)
    if model not in MODELS:
        raise ValueError(f"Self TIME supports exactly {tuple(MODELS)}")
    if str(config.instance_normalization) != "zscore":
        raise ValueError("Self TIME requires instance_normalization=zscore")

    os.environ["TIME_EXPERIMENT"] = "self_augmentation"
    module_name, function_name = MODELS[model]
    run_experiment = getattr(import_module(module_name), function_name)
    config_path = _path(config.dataset_config) or DEFAULT_CONFIG_PATH.resolve()
    datasets = _datasets(list(config.datasets), config_path)
    settings = config.model_settings[model]
    terms = None if config.terms is None else list(config.terms)

    seen_names: set[str] = set()
    augmentation_sets: list[tuple[str, tuple[str, ...]]] = []
    for entry in config.augmentation_sets:
        name = str(entry.name)
        if not name or name in seen_names:
            raise ValueError(f"Augmentation-set names must be unique: {name!r}")
        seen_names.add(name)
        augmentation_sets.append(
            (name, validate_self_augmentations(list(entry.channels)))
        )

    logging.info(
        "Self TIME model=%s datasets=%d augmentation_sets=%s seed=%d dataset_config=%s",
        model,
        len(datasets),
        [name for name, _ in augmentation_sets],
        int(config.seed),
        config_path,
    )
    for augmentation_id, channels in augmentation_sets:
        logging.info(
            "Self TIME configuration=%s channels=%s",
            augmentation_id,
            list(channels),
        )
        for index, dataset in enumerate(datasets, start=1):
            _set_seed(int(config.seed))
            logging.info(
                "Self TIME dataset=%s position=%d/%d configuration=%s",
                dataset,
                index,
                len(datasets),
                augmentation_id,
            )
            run_experiment(
                dataset_name=dataset,
                terms=terms,
                output_dir=None if config.output_dir is None else str(_path(config.output_dir)),
                batch_size=int(settings.batch_size),
                context_length=int(settings.context_length),
                config_path=config_path,
                model_path=_path(settings.model_path),
                self_augmentations=channels,
                augmentation_id=augmentation_id,
                seed=int(config.seed),
                target_mode=str(config.target_mode),
                instance_normalization="zscore",
            )


if __name__ == "__main__":
    main()
