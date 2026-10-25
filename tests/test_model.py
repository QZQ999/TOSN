from pathlib import Path

import pytest
import torch

from ugma.models.backbones import DLinear, build_backbone
from ugma.models.ugma import GraphAwareDFM, RiskMonitor, UGMAModel


def build_model() -> UGMAModel:
    channels, seq_len, pred_len = 6, 48, 24
    topology = torch.eye(channels).repeat(3, 1, 1)
    backbone = DLinear(seq_len, pred_len, channels)
    model = UGMAModel(
        backbone,
        seq_len,
        pred_len,
        channels,
        d_model=32,
        top_k=4,
        retrieval_temperature=0.1,
        topology=topology,
        topology_strength=0.3,
        temporal_gap=48,
        fallback_similarity_threshold=0.0,
    )
    keys = torch.randn(20, 32)
    futures = torch.randn(20, pred_len)
    nodes = torch.arange(20) % channels
    starts = torch.arange(20) * 48
    model.bank.set_entries(keys, futures, nodes, starts)
    return model


def test_ugma_shapes_and_frozen_backbone() -> None:
    model = build_model()
    x = torch.randn(3, 48, 6)
    output, aux = model(x)
    assert output.shape == (3, 24, 6)
    assert aux["risk"].shape == (3, 6)
    assert aux["reference"].shape == (3, 24, 6)
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())


def test_bank_records_train_timestamps() -> None:
    model = build_model()
    assert model.bank.target_starts.max().item() == 19 * 48


def test_bank_can_preserve_manuscript_history_triple() -> None:
    model = build_model()
    histories = torch.randn(20, 48)
    model.bank.set_entries(
        model.bank.keys,
        model.bank.futures,
        model.bank.nodes,
        model.bank.target_starts,
        histories,
    )
    assert torch.equal(model.bank.histories, histories)


def test_retrieval_masks_complete_window_overlap() -> None:
    model = build_model()
    queries = torch.randn(2, 6, 32)
    query_starts = torch.tensor([5 * 48, 10 * 48])
    retrieved = model.bank.retrieve(queries, query_starts)
    gap = (retrieved.candidate_target_starts - query_starts[:, None, None]).abs()
    assert model.bank.temporal_gap >= 48 + 24
    assert torch.all(gap >= model.bank.temporal_gap)


@pytest.mark.parametrize("name", ["iTransformer", "PatchTST", "TimeMixer", "TimesNet", "FEDformer"])
def test_official_tslib_backbone_forward(name: str) -> None:
    project_root = Path(__file__).resolve().parents[1]
    if not (project_root / "external" / "tslib").exists():
        pytest.skip("official Time-Series-Library checkout is not present")
    backbone = build_backbone(name, 96, 24, 7, 64, project_root)
    with torch.no_grad():
        output = backbone(torch.randn(2, 96, 7))
    assert output.shape == (2, 24, 7)


@pytest.mark.parametrize("scope", ["all", "same_sensor", "cross_sensor"])
def test_raw_retrieval_variants(scope: str) -> None:
    channels, seq_len, pred_len = 6, 48, 24
    model = UGMAModel(
        DLinear(seq_len, pred_len, channels),
        seq_len,
        pred_len,
        channels,
        d_model=32,
        top_k=2,
        retrieval_temperature=0.1,
        topology=torch.eye(channels).repeat(3, 1, 1),
        topology_strength=0.3,
        temporal_gap=1,
        fallback_similarity_threshold=0.0,
        retrieval_encoder="raw_cosine",
        retrieval_scope=scope,
    )
    keys = torch.randn(24, seq_len)
    nodes = torch.arange(24) % channels
    model.bank.set_entries(keys, torch.randn(24, pred_len), nodes, torch.arange(24) * 10)
    output, _ = model(torch.randn(2, seq_len, channels))
    assert output.shape == (2, pred_len, channels)


def test_retrieval_falls_back_when_no_legal_candidate_exists() -> None:
    model = build_model()
    model.use_dfm = False
    model.bank.retrieval_scope = "same_sensor"
    model.bank.set_entries(
        torch.randn(2, 32),
        torch.randn(2, 24),
        torch.tensor([0, 0]),
        torch.tensor([0, 48]),
    )
    output, aux = model(torch.randn(1, 48, 6), torch.tensor([0]))
    assert torch.isfinite(output).all()
    assert torch.all(aux["retrieval_confidence"][:, 1:] == 0)
    assert torch.allclose(output[:, :, 1:], aux["backbone"][:, :, 1:])


def test_shift_safety_factor_actually_scales_both_gates() -> None:
    model = build_model()
    x = torch.randn(2, 48, 6)
    _, original = model(x)
    model.set_safety_shift_threshold(-100.0)
    _, safe = model(x)
    assert torch.all(safe["safety_factor"] < 1e-6)
    assert torch.all(safe["alpha"] < original["alpha"] * 1e-5)
    assert torch.all(safe["beta"] < original["beta"] * 1e-5)


def test_dfm_modes_only_instantiate_layers_they_use() -> None:
    topology = torch.eye(6).repeat(3, 1, 1)
    independent = GraphAwareDFM(24, 6, 32, topology, "channel_independent")
    sensor = GraphAwareDFM(24, 6, 32, topology, "sensor_attention")
    multiplex = GraphAwareDFM(24, 6, 32, topology, "multiplex_graph_attention")

    assert independent.graph is None
    assert independent.sensor_attention is None
    assert sensor.graph is None
    assert sensor.sensor_attention is not None
    assert multiplex.graph is not None
    assert multiplex.sensor_attention is not None

    parameter_counts = [
        sum(parameter.numel() for parameter in module.parameters())
        for module in (independent, sensor, multiplex)
    ]
    assert parameter_counts[0] < parameter_counts[1] < parameter_counts[2]
    forecast = torch.randn(2, 24, 6)
    assert independent(forecast).shape == forecast.shape
    assert sensor(forecast).shape == forecast.shape
    assert multiplex(forecast).shape == forecast.shape


def test_manuscript_risk_monitor_excludes_shift_extension() -> None:
    disabled = RiskMonitor(48, 24, 6, 32, "disabled")
    manuscript = RiskMonitor(48, 24, 6, 32, "manuscript_mlp")
    revised = RiskMonitor(48, 24, 6, 32, "softplus_mlp")
    extension = RiskMonitor(48, 24, 6, 32, "shift_augmented")
    assert manuscript.stat_weight_raw is None
    assert manuscript.stat_bias is None
    assert revised.stat_weight_raw is None
    assert revised.stat_bias is None
    assert extension.stat_weight_raw is not None
    assert extension.stat_bias is not None
    disabled_count = sum(parameter.numel() for parameter in disabled.parameters())
    manuscript_count = sum(parameter.numel() for parameter in manuscript.parameters())
    extension_count = sum(parameter.numel() for parameter in extension.parameters())
    assert disabled_count == 0 < manuscript_count < extension_count
    history = torch.randn(3, 48, 6)
    forecast = torch.randn(3, 24, 6)
    assert torch.equal(disabled(history, forecast), torch.zeros(3, 6))
    assert torch.all(manuscript(history, forecast) >= 0)
    assert torch.all(revised(history, forecast) > 0)
    assert torch.all(extension(history, forecast) >= 0)


def test_softplus_risk_head_retains_gradient_for_negative_pre_activation() -> None:
    monitor = RiskMonitor(48, 24, 6, 32, "softplus_mlp")
    output_layer = monitor.risk_head[-2]
    assert isinstance(output_layer, torch.nn.Linear)
    with torch.no_grad():
        output_layer.weight.zero_()
        output_layer.bias.fill_(-5.0)
    history = torch.randn(3, 48, 6)
    forecast = torch.randn(3, 24, 6)
    prediction = monitor(history, forecast)
    assert torch.all(prediction > 0)
    torch.nn.functional.mse_loss(prediction, torch.ones_like(prediction)).backward()
    assert output_layer.bias.grad is not None
    assert torch.all(output_layer.bias.grad.abs() > 0)


def test_risk_control_does_not_change_dfm_or_gate_initialization() -> None:
    def initialized(mode: str) -> UGMAModel:
        torch.manual_seed(2026)
        return UGMAModel(
            DLinear(48, 24, 6),
            48,
            24,
            6,
            d_model=32,
            top_k=4,
            retrieval_temperature=0.1,
            topology=None,
            topology_strength=0.0,
            temporal_gap=48,
            fallback_similarity_threshold=0.0,
            dfm_mode="channel_independent",
            risk_monitor_mode=mode,
            normalization_mode="revin",
            gate_mode="manuscript_topk",
        )

    disabled = initialized("disabled")
    enabled = initialized("softplus_mlp")
    for module_name in ("encoder", "dfm", "local_gate", "global_gate"):
        disabled_state = getattr(disabled, module_name).state_dict()
        enabled_state = getattr(enabled, module_name).state_dict()
        assert disabled_state.keys() == enabled_state.keys()
        assert all(
            torch.equal(disabled_state[key], enabled_state[key])
            for key in disabled_state
        )


def test_revin_round_trip_and_bank_normalization() -> None:
    model = UGMAModel(
        DLinear(48, 24, 6),
        48,
        24,
        6,
        d_model=32,
        top_k=4,
        retrieval_temperature=0.1,
        topology=None,
        topology_strength=0.0,
        temporal_gap=48,
        fallback_similarity_threshold=0.0,
        dfm_mode="channel_independent",
        risk_monitor_mode="manuscript_mlp",
        normalization_mode="revin",
        gate_mode="manuscript_topk",
    )
    history = torch.randn(3, 48, 6) * 2.0 + 5.0
    forecast = torch.randn(3, 24, 6) * 2.0 + 5.0
    normalized_history, normalized_forecast, mean, scale = model.normalize_context(
        history, forecast
    )
    assert torch.allclose(normalized_history.mean(dim=1), torch.zeros(3, 6), atol=1e-6)
    assert torch.allclose(normalized_forecast * scale + mean, forecast, atol=1e-6)

    flat_history = history.transpose(1, 2).reshape(-1, 48)
    flat_future = forecast.transpose(1, 2).reshape(-1, 24)
    normalized_bank_history, normalized_bank_future = model.normalize_bank_entries(
        flat_history, flat_future
    )
    assert torch.allclose(
        normalized_bank_history.mean(dim=-1),
        torch.zeros(len(normalized_bank_history)),
        atol=1e-6,
    )
    assert normalized_bank_future.shape == flat_future.shape
    assert model.global_gate[0].in_features == 5

    extension = build_model()
    assert extension.global_gate[0].in_features == 2
