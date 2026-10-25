import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from ugma.data import WindowDataset
from ugma.models.backbones import DLinear
from ugma.models.ugma import UGMAModel
from ugma.training import _future_nearest_positive, train_backbone, train_enhancement


class ConstantForecast(nn.Module):
    def __init__(self, pred_len: int, channels: int) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(0.0))
        self.pred_len = pred_len
        self.channels = channels

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.value.expand(len(history), self.pred_len, self.channels)


def test_backbone_early_stopping_uses_patience_after_best_epoch() -> None:
    values = np.zeros((80, 2), dtype=np.float32)
    dataset = WindowDataset(values, seq_len=12, pred_len=6, target_start=20, target_end=70)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    model = ConstantForecast(pred_len=6, channels=2)
    records = train_backbone(
        model,
        loader,
        loader,
        epochs=10,
        learning_rate=0.0,
        weight_decay=0.0,
        device=torch.device("cpu"),
        early_stopping_patience=2,
    )
    assert len(records) == 3


def test_manuscript_joint_stage_updates_risk_but_not_frozen_encoder() -> None:
    torch.manual_seed(2026)
    values = np.random.default_rng(2026).normal(size=(220, 2)).astype(np.float32)
    train = WindowDataset(values, seq_len=12, pred_len=6, target_start=20, target_end=140)
    validation = WindowDataset(values, seq_len=12, pred_len=6, target_start=140, target_end=210)
    train_loader = DataLoader(train, batch_size=16, shuffle=False)
    validation_loader = DataLoader(validation, batch_size=16, shuffle=False)
    model = UGMAModel(
        DLinear(12, 6, 2),
        12,
        6,
        2,
        d_model=8,
        top_k=2,
        retrieval_temperature=0.1,
        topology=None,
        topology_strength=0.0,
        temporal_gap=1,
        fallback_similarity_threshold=0.0,
        dfm_mode="channel_independent",
        risk_monitor_mode="manuscript_mlp",
    )
    model.bank.set_entries(
        torch.randn(4, 8),
        torch.randn(4, 6),
        torch.tensor([0, 1, 0, 1]),
        torch.tensor([0, 0, 200, 200]),
    )
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    risk_before = [parameter.detach().clone() for parameter in model.risk_monitor.parameters()]
    encoder_before = [parameter.detach().clone() for parameter in model.encoder.parameters()]

    records = train_enhancement(
        model,
        train_loader,
        validation_loader,
        epochs=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        risk_weight=1.0,
        risk_rank_weight=0.0,
        shift_probability=0.0,
        shift_severity=0.0,
        device=torch.device("cpu"),
    )

    assert len(records) == 1
    assert any(
        not torch.equal(before, after)
        for before, after in zip(risk_before, model.risk_monitor.parameters())
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(encoder_before, model.encoder.parameters())
    )


def test_future_positive_metric_matches_manuscript_euclidean_rule() -> None:
    futures = torch.tensor([[1.0, 0.0], [10.0, 0.0], [1.0, 1.0]])
    starts = torch.tensor([0, 100, 200])
    cosine = _future_nearest_positive(futures, starts, temporal_gap=50, metric="cosine")
    euclidean = _future_nearest_positive(futures, starts, temporal_gap=50, metric="euclidean")
    assert cosine[0].item() == 1
    assert euclidean[0].item() == 2


def test_future_positive_selection_rejects_missing_nonoverlap_candidate() -> None:
    futures = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    starts = torch.tensor([0, 10, 20])
    with pytest.raises(ValueError, match="No non-overlapping predictive positive"):
        _future_nearest_positive(futures, starts, temporal_gap=30, metric="euclidean")
