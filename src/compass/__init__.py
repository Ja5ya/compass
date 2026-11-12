"""COMPASS core: modality-agnostic multi-objective analysis on pipeline artifacts."""

from compass.contracts import RECORD_REQUIRED_KEYS, validate_record, validate_dataset_layout
from compass.objectives import (
    ObjectivesConfig,
    TASK_COLUMN,
    apply_fa_scores,
    apply_objectives,
    compute_fp_fs_scores,
    compute_f_A,
)
from compass.paths import (
    artifacts_root,
    data_source_root,
    dataset_dir,
    resolve_artifact_paths,
    shared_dir,
)

__all__ = [
    "RECORD_REQUIRED_KEYS",
    "ObjectivesConfig",
    "TASK_COLUMN",
    "apply_fa_scores",
    "apply_objectives",
    "artifacts_root",
    "compute_f_A",
    "compute_fp_fs_scores",
    "data_source_root",
    "dataset_dir",
    "resolve_artifact_paths",
    "shared_dir",
    "validate_dataset_layout",
    "validate_record",
]
