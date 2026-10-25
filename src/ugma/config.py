from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(slots=True)
class DataConfig:
    kind: str
    seq_len: int
    pred_len: int
    train_ratio: float
    val_ratio: float
    batch_size: int
    num_workers: int = 0
    path: str | None = None
    reference_path: str | None = None
    timestamp_column: str = "date"
    length: int = 1800
    channels: int = 8
    layers: int = 3
    network_complexity: float = 0.5
    shift_start: float = 0.8
    shift_severity: float = 0.7


@dataclass(slots=True)
class BackboneConfig:
    name: str
    hidden_dim: int
    epochs: int
    learning_rate: float
    weight_decay: float
    early_stopping_patience: int | None = None


@dataclass(slots=True)
class UGMAConfig:
    d_model: int
    encoder_epochs: int
    risk_pretrain_epochs: int
    enhancement_epochs: int
    learning_rate: float
    weight_decay: float
    bank_size: int
    top_k: int
    temporal_gap: int
    contrastive_temperature: float
    retrieval_temperature: float
    risk_loss_weight: float
    risk_rank_weight: float
    shift_augmentation_probability: float
    shift_augmentation_severity: float
    fallback_similarity_threshold: float
    topology_strength: float
    use_hcb: bool
    use_dfm: bool
    use_risk_gate: bool
    training_protocol: str = "enhanced_pretrain"
    risk_monitor_mode: str = "shift_augmented"
    normalization_mode: str = "none"
    gate_mode: str = "confidence_scaled"
    positive_metric: str = "cosine"
    topology_mode: str = "multiplex"
    dfm_mode: str = "multiplex_graph_attention"
    retrieval_encoder: str = "predictive_contrastive"
    retrieval_scope: str = "all"
    gate_bias: float = -5.0
    safety_risk_quantile: float = 1.0
    safety_temperature: float = 0.05
    safety_monitor: str = "risk"
    early_stopping_patience: int | None = None


@dataclass(slots=True)
class ExperimentConfig:
    seed: int
    device: str
    output_dir: str
    data: DataConfig
    backbone: BackboneConfig
    ugma: UGMAConfig


def parse_config(raw: dict) -> ExperimentConfig:
    return ExperimentConfig(
        seed=int(raw["seed"]),
        device=str(raw["device"]),
        output_dir=str(raw["output_dir"]),
        data=DataConfig(**raw["data"]),
        backbone=BackboneConfig(**raw["backbone"]),
        ugma=UGMAConfig(**raw["ugma"]),
    )


def load_config(path: str | Path) -> ExperimentConfig:
    return parse_config(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
