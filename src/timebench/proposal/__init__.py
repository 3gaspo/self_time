"""Self TIME proposal: causal self-augmentation covariates."""

from .self_augmentations import (
    SELF_AUGMENTATIONS,
    build_past_self_covariates,
    validate_self_augmentations,
)

__all__ = [
    "SELF_AUGMENTATIONS",
    "build_past_self_covariates",
    "validate_self_augmentations",
]
