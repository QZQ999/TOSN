from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class PredictiveEncoder(nn.Module):
    """Channel-wise history encoder used by predictive contrastive retrieval."""

    def __init__(self, seq_len: int, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        keys = self.net(history.transpose(1, 2))
        return F.normalize(keys, dim=-1)


class RawHistoryEncoder(nn.Module):
    """Non-learned raw-history cosine key for the retrieval ablation."""

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return F.normalize(history.transpose(1, 2), dim=-1)


class RiskMonitor(nn.Module):
    """Predict per-channel backbone MAE for reviewer-diagnostic variants.

    ``disabled`` is the retained forecasting path after the reviewer-directed
    validation rejected risk guidance; it has no parameters and returns zeros.
    No enabled mode is a posterior variance. ``manuscript_mlp`` preserves the
    submitted Linear-ReLU-Linear-ReLU head for diagnosis. ``softplus_mlp`` is a
    smooth-positive diagnostic control, while ``shift_augmented`` adds the
    reconstructed statistics branch used only in robustness audits.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        channels: int,
        d_model: int,
        mode: str = "shift_augmented",
    ) -> None:
        super().__init__()
        if mode not in {"disabled", "manuscript_mlp", "softplus_mlp", "shift_augmented"}:
            raise ValueError(
                "risk monitor mode must be disabled, manuscript_mlp, softplus_mlp, "
                "or shift_augmented"
            )
        self.mode = mode
        if mode == "disabled":
            self.history_projection = None
            self.forecast_projection = None
            self.channel_embedding = None
            self.risk_head = None
            self.register_parameter("stat_weight_raw", None)
            self.register_parameter("stat_bias", None)
            return
        self.history_projection = nn.Linear(seq_len, d_model)
        self.forecast_projection = nn.Linear(pred_len, d_model)
        self.channel_embedding = nn.Embedding(channels, d_model)
        activation: nn.Module = nn.ReLU() if mode in {"manuscript_mlp", "softplus_mlp"} else nn.GELU()
        head = [nn.Linear(d_model * 3, d_model), activation, nn.Linear(d_model, 1)]
        if mode == "manuscript_mlp":
            head.append(nn.ReLU())
        elif mode == "softplus_mlp":
            head.append(nn.Softplus())
        self.risk_head = nn.Sequential(*head)
        if mode == "shift_augmented":
            self.stat_weight_raw = nn.Parameter(torch.full((8,), -2.0))
            self.stat_bias = nn.Parameter(torch.tensor(-1.0))
        else:
            self.register_parameter("stat_weight_raw", None)
            self.register_parameter("stat_bias", None)

    @staticmethod
    def shift_statistics(history: torch.Tensor, forecast: torch.Tensor) -> torch.Tensor:
        h_mean = history.mean(dim=1)
        h_std = history.std(dim=1, unbiased=False)
        p_mean = forecast.mean(dim=1)
        p_std = forecast.std(dim=1, unbiased=False)
        h_slope = (history[:, -1] - history[:, 0]) / max(1, history.shape[1] - 1)
        p_slope = (forecast[:, -1] - forecast[:, 0]) / max(1, forecast.shape[1] - 1)
        statistics = torch.stack(
            [
                h_mean.abs(),
                (h_std - 1.0).abs(),
                h_slope.abs(),
                p_mean.abs(),
                (p_std - 1.0).abs(),
                p_slope.abs(),
                (p_mean - h_mean).abs(),
                (forecast[:, 0] - history[:, -1]).abs(),
            ],
            dim=-1,
        )
        return torch.log1p(statistics)

    def forward(self, history: torch.Tensor, forecast: torch.Tensor) -> torch.Tensor:
        batch, _, channels = history.shape
        if self.mode == "disabled":
            return history.new_zeros((batch, channels))
        h = self.history_projection(history.transpose(1, 2))
        p = self.forecast_projection(forecast.transpose(1, 2))
        ids = torch.arange(channels, device=history.device)
        e = self.channel_embedding(ids)[None].expand(batch, -1, -1)
        learned = self.risk_head(torch.cat([h, p, e], dim=-1)).squeeze(-1)
        if self.mode in {"manuscript_mlp", "softplus_mlp"}:
            return learned
        statistics = self.shift_statistics(history, forecast)
        monotone = (statistics * F.softplus(self.stat_weight_raw)).sum(dim=-1) + self.stat_bias
        return F.softplus(learned + monotone)


class DistributionShiftMonitor(nn.Module):
    """Diagonal Mahalanobis score over train-fitted history statistics."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.register_buffer("feature_mean", torch.zeros(channels, 6))
        self.register_buffer("feature_scale", torch.ones(channels, 6))

    @staticmethod
    def features(history: torch.Tensor) -> torch.Tensor:
        mean = history.mean(dim=1)
        std = history.std(dim=1, unbiased=False)
        slope = (history[:, -1] - history[:, 0]) / max(1, history.shape[1] - 1)
        difference_std = history.diff(dim=1).std(dim=1, unbiased=False)
        centered = history - mean[:, None, :]
        lag_numerator = (centered[:, 1:] * centered[:, :-1]).mean(dim=1)
        lag_one = lag_numerator / centered.square().mean(dim=1).clamp_min(1e-6)
        half = history.shape[1] // 2
        recent_change = history[:, half:].mean(dim=1) - history[:, :half].mean(dim=1)
        return torch.stack([mean, std, slope, difference_std, lag_one, recent_change], dim=-1)

    def set_reference(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        self.feature_mean.copy_(mean)
        self.feature_scale.copy_(scale.clamp_min(1e-5))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        z = (self.features(history) - self.feature_mean) / self.feature_scale
        return z.square().mean(dim=-1).sqrt()


class MultiplexGraphMixer(nn.Module):
    def __init__(self, d_model: int, topology: torch.Tensor | None) -> None:
        super().__init__()
        if topology is None:
            topology = torch.empty(0)
        self.register_buffer("topology", topology.float())
        layers = max(1, len(topology))
        self.layer_logits = nn.Parameter(torch.zeros(layers))
        self.message = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.topology.numel() == 0:
            return features
        weights = torch.softmax(self.layer_logits, dim=0)
        adjacency = torch.einsum("l,lij->ij", weights, self.topology)
        propagated = torch.einsum("ij,bqjd->bqid", adjacency, features)
        return self.norm(features + self.message(propagated))


class GraphAwareDFM(nn.Module):
    """Temporal residual corrector followed by multiplex cross-sensor mixing."""

    def __init__(
        self,
        pred_len: int,
        channels: int,
        d_model: int,
        topology: torch.Tensor | None,
        mode: str = "multiplex_graph_attention",
    ) -> None:
        super().__init__()
        valid_modes = {"channel_independent", "sensor_attention", "multiplex_graph_attention"}
        if mode not in valid_modes:
            raise ValueError(f"Unknown DFM mode {mode!r}; expected one of {sorted(valid_modes)}")
        self.mode = mode
        self.value_embedding = nn.Linear(1, d_model)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(temporal_layer, num_layers=2)
        self.graph = (
            MultiplexGraphMixer(d_model, topology)
            if mode == "multiplex_graph_attention"
            else None
        )
        if mode != "channel_independent":
            sensor_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=d_model * 2,
                dropout=0.1,
                batch_first=True,
                norm_first=True,
            )
            self.sensor_attention = nn.TransformerEncoder(sensor_layer, num_layers=1)
        else:
            self.sensor_attention = None
        self.output = nn.Linear(d_model, 1)

    def forward(self, forecast: torch.Tensor) -> torch.Tensor:
        batch, horizon, channels = forecast.shape
        z = self.value_embedding(forecast.transpose(1, 2).unsqueeze(-1))
        z = self.temporal(z.reshape(batch * channels, horizon, -1))
        z = z.reshape(batch, channels, horizon, -1).transpose(1, 2)
        if self.graph is not None:
            z = self.graph(z)
        if self.sensor_attention is not None:
            z = self.sensor_attention(z.reshape(batch * horizon, channels, -1))
            z = z.reshape(batch, horizon, channels, -1)
        return self.output(z).squeeze(-1)


@dataclass
class RetrievalOutput:
    reference: torch.Tensor
    similarities: torch.Tensor
    candidate_nodes: torch.Tensor
    candidate_target_starts: torch.Tensor
    valid_candidates: torch.Tensor


class ConditionBank(nn.Module):
    """Train-only channel-level memory with optional multiplex topology prior."""

    def __init__(
        self,
        top_k: int,
        temperature: float,
        topology: torch.Tensor | None,
        topology_strength: float,
        temporal_gap: int,
        retrieval_scope: str = "all",
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.temperature = temperature
        self.topology_strength = topology_strength
        # Effective separation between target starts.  The caller expands the
        # declared gap to at least seq_len + pred_len so complete input/target
        # intervals cannot overlap.
        self.temporal_gap = temporal_gap
        if retrieval_scope not in {"all", "same_sensor", "cross_sensor"}:
            raise ValueError("retrieval_scope must be all, same_sensor, or cross_sensor")
        self.retrieval_scope = retrieval_scope
        self.register_buffer("keys", torch.empty(0))
        self.register_buffer("histories", torch.empty(0))
        self.register_buffer("futures", torch.empty(0))
        self.register_buffer("nodes", torch.empty(0, dtype=torch.long))
        self.register_buffer("target_starts", torch.empty(0, dtype=torch.long))
        if topology is None:
            topology = torch.empty(0)
        self.register_buffer("topology", topology.float())

    def set_entries(
        self,
        keys: torch.Tensor,
        futures: torch.Tensor,
        nodes: torch.Tensor,
        target_starts: torch.Tensor,
        histories: torch.Tensor | None = None,
    ) -> None:
        self.keys = F.normalize(keys.detach(), dim=-1)
        self.histories = (
            histories.detach() if histories is not None else torch.empty(0, device=keys.device)
        )
        self.futures = futures.detach()
        self.nodes = nodes.detach().long()
        self.target_starts = target_starts.detach().long()

    def _node_proximity(self) -> torch.Tensor | None:
        if self.topology.numel() == 0:
            return None
        proximity = self.topology.mean(dim=0)
        return proximity / proximity.amax(dim=-1, keepdim=True).clamp_min(1e-8)

    def retrieve(
        self,
        queries: torch.Tensor,
        query_target_starts: torch.Tensor | None = None,
    ) -> RetrievalOutput:
        batch, channels, d_model = queries.shape
        flat_queries = queries.reshape(batch * channels, d_model)
        cosine = flat_queries @ self.keys.T

        query_nodes = torch.arange(channels, device=queries.device).repeat(batch)
        if self.retrieval_scope == "same_sensor":
            cosine = cosine.masked_fill(query_nodes[:, None] != self.nodes[None, :], float("-inf"))
        elif self.retrieval_scope == "cross_sensor":
            cosine = cosine.masked_fill(query_nodes[:, None] == self.nodes[None, :], float("-inf"))

        if query_target_starts is not None:
            query_starts = query_target_starts.repeat_interleave(channels)
            too_close = (query_starts[:, None] - self.target_starts[None, :]).abs() < self.temporal_gap
            cosine = cosine.masked_fill(too_close, float("-inf"))

        proximity = self._node_proximity()
        if proximity is not None and self.topology_strength > 0:
            prior = proximity[query_nodes][:, self.nodes]
            cosine = cosine + self.topology_strength * prior

        k = min(self.top_k, self.keys.shape[0])
        scores, indices = torch.topk(cosine, k=k, dim=-1)
        valid = torch.isfinite(scores)
        safe_scores = torch.where(valid, scores, torch.zeros_like(scores))
        safe_scores = safe_scores - safe_scores.amax(dim=-1, keepdim=True)
        unnormalized = torch.exp(safe_scores / self.temperature) * valid
        weights = unnormalized / unnormalized.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        selected = self.futures[indices]
        reference = torch.einsum("bk,bkq->bq", weights, selected)
        reference = reference.reshape(batch, channels, -1).transpose(1, 2)
        return RetrievalOutput(
            reference=reference,
            similarities=scores.reshape(batch, channels, k),
            candidate_nodes=self.nodes[indices].reshape(batch, channels, k),
            candidate_target_starts=self.target_starts[indices].reshape(batch, channels, k),
            valid_candidates=valid.reshape(batch, channels, k),
        )


class UGMAModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        seq_len: int,
        pred_len: int,
        channels: int,
        d_model: int,
        top_k: int,
        retrieval_temperature: float,
        topology: torch.Tensor | None,
        topology_strength: float,
        temporal_gap: int,
        fallback_similarity_threshold: float,
        use_hcb: bool = True,
        use_dfm: bool = True,
        use_risk_gate: bool = True,
        dfm_mode: str = "multiplex_graph_attention",
        retrieval_encoder: str = "predictive_contrastive",
        retrieval_scope: str = "all",
        gate_bias: float = -5.0,
        safety_temperature: float = 0.05,
        risk_monitor_mode: str = "shift_augmented",
        normalization_mode: str = "none",
        gate_mode: str = "confidence_scaled",
    ) -> None:
        super().__init__()
        if normalization_mode not in {"none", "revin"}:
            raise ValueError("normalization_mode must be none or revin")
        if gate_mode not in {"confidence_scaled", "manuscript_topk"}:
            raise ValueError("gate_mode must be confidence_scaled or manuscript_topk")
        self.normalization_mode = normalization_mode
        self.gate_mode = gate_mode
        self.top_k = top_k
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        if retrieval_encoder == "predictive_contrastive":
            self.encoder = PredictiveEncoder(seq_len, d_model)
        elif retrieval_encoder == "raw_cosine":
            self.encoder = RawHistoryEncoder()
        else:
            raise ValueError("retrieval_encoder must be predictive_contrastive or raw_cosine")
        # Reviewer controls must differ only in the monitor itself.  Construct
        # it in an isolated RNG stream so enabling or removing the monitor does
        # not silently change DFM/gate initialisation through RNG consumption.
        risk_seed = (torch.initial_seed() + 10_000_019) % (2**63)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(risk_seed)
            self.risk_monitor = RiskMonitor(
                seq_len,
                pred_len,
                channels,
                d_model,
                risk_monitor_mode,
            )
        self.shift_monitor = DistributionShiftMonitor(channels)
        self.bank = ConditionBank(
            top_k,
            retrieval_temperature,
            topology,
            topology_strength,
            max(temporal_gap, seq_len + pred_len),
            retrieval_scope,
        )
        self.dfm = GraphAwareDFM(pred_len, channels, d_model, topology, dfm_mode)
        self.local_gate = nn.Linear(1, 1)
        global_gate_inputs = top_k + 1 if gate_mode == "manuscript_topk" else 2
        self.global_gate = nn.Sequential(
            nn.Linear(global_gate_inputs, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        nn.init.zeros_(self.local_gate.weight)
        nn.init.constant_(self.local_gate.bias, gate_bias)
        nn.init.zeros_(self.global_gate[-1].weight)
        nn.init.constant_(self.global_gate[-1].bias, gate_bias)
        self.use_hcb = use_hcb
        self.use_dfm = use_dfm
        self.use_risk_gate = use_risk_gate
        self.fallback_similarity_threshold = fallback_similarity_threshold
        self.register_buffer("risk_calibration_slope", torch.tensor(1.0))
        self.register_buffer("risk_calibration_intercept", torch.tensor(0.0))
        self.safety_risk_threshold = float("inf")
        self.safety_shift_threshold = float("inf")
        self.safety_temperature = safety_temperature

    def set_risk_calibration(self, slope: float, intercept: float) -> None:
        self.risk_calibration_slope.fill_(slope)
        self.risk_calibration_intercept.fill_(intercept)

    def set_safety_risk_threshold(self, threshold: float) -> None:
        self.safety_risk_threshold = float(threshold)

    def set_safety_shift_threshold(self, threshold: float) -> None:
        self.safety_shift_threshold = float(threshold)

    def normalize_context(
        self,
        history: torch.Tensor,
        forecast: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized_history, mean, scale = self.normalize_history(history)
        return normalized_history, (forecast - mean) / scale, mean, scale

    def normalize_history(
        self,
        history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.normalization_mode == "revin":
            mean = history.mean(dim=1, keepdim=True)
            scale = history.var(dim=1, keepdim=True, unbiased=False).add(1e-5).sqrt()
            return (history - mean) / scale, mean, scale
        mean = torch.zeros_like(history[:, :1])
        scale = torch.ones_like(history[:, :1])
        return history, mean, scale

    def normalize_bank_entries(
        self,
        histories: torch.Tensor,
        futures: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.normalization_mode != "revin":
            return histories, futures
        mean = histories.mean(dim=-1, keepdim=True)
        scale = histories.var(dim=-1, keepdim=True, unbiased=False).add(1e-5).sqrt()
        return (histories - mean) / scale, (futures - mean) / scale

    @staticmethod
    def realized_risk(target: torch.Tensor, aux: dict[str, torch.Tensor]) -> torch.Tensor:
        normalized_target = (target - aux["normalization_mean"]) / aux["normalization_scale"]
        return (normalized_target - aux["normalized_backbone"]).abs().mean(dim=1)

    def train(self, mode: bool = True) -> "UGMAModel":
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(
        self,
        history: torch.Tensor,
        target_starts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with torch.no_grad():
            backbone_forecast = self.backbone(history)

        normalized_history, normalized_backbone, normalization_mean, normalization_scale = (
            self.normalize_context(history, backbone_forecast)
        )

        raw_risk = self.risk_monitor(normalized_history, normalized_backbone)
        risk = (self.risk_calibration_slope * raw_risk + self.risk_calibration_intercept).clamp_min(0.0)
        shift_score = self.shift_monitor(normalized_history)
        local = self.dfm(normalized_backbone) if self.use_dfm else torch.zeros_like(normalized_backbone)

        if self.use_hcb and self.bank.keys.numel() > 0:
            retrieval = self.bank.retrieve(self.encoder(normalized_history), target_starts)
            has_valid = retrieval.valid_candidates.any(dim=-1)
            reference = torch.where(
                has_valid[:, None, :],
                retrieval.reference,
                normalized_backbone,
            )
            valid_count = retrieval.valid_candidates.sum(dim=-1).clamp_min(1)
            similarity_vector = torch.where(
                retrieval.valid_candidates,
                retrieval.similarities,
                torch.zeros_like(retrieval.similarities),
            )
            if similarity_vector.shape[-1] < self.top_k:
                similarity_vector = F.pad(
                    similarity_vector,
                    (0, self.top_k - similarity_vector.shape[-1]),
                )
            similarity = similarity_vector.sum(dim=-1) / valid_count
            retrieval_confidence = torch.sigmoid(
                (retrieval.similarities.amax(dim=-1) - self.fallback_similarity_threshold) / 0.1
            )
        else:
            reference = normalized_backbone
            similarity = torch.zeros_like(risk)
            similarity_vector = torch.zeros(*risk.shape, self.top_k, device=risk.device, dtype=risk.dtype)
            retrieval_confidence = torch.zeros_like(risk)

        if self.use_risk_gate:
            gate_risk = risk
        else:
            gate_risk = torch.zeros_like(risk)
        alpha = torch.sigmoid(self.local_gate(gate_risk.unsqueeze(-1))).squeeze(-1)
        if self.gate_mode == "manuscript_topk":
            beta_input = torch.cat([gate_risk.unsqueeze(-1), similarity_vector], dim=-1)
            beta = torch.sigmoid(self.global_gate(beta_input)).squeeze(-1)
        else:
            beta_input = torch.stack([gate_risk, similarity], dim=-1)
            beta = torch.sigmoid(self.global_gate(beta_input)).squeeze(-1) * retrieval_confidence
        if self.safety_shift_threshold < float("inf"):
            safety_factor = torch.sigmoid(
                (self.safety_shift_threshold - shift_score) / self.safety_temperature
            )
        elif self.safety_risk_threshold < float("inf"):
            safety_factor = torch.sigmoid(
                (self.safety_risk_threshold - risk) / self.safety_temperature
            )
        else:
            safety_factor = torch.ones_like(risk)
        alpha = alpha * safety_factor
        beta = beta * safety_factor

        enhanced_normalized = (
            normalized_backbone
            + alpha[:, None, :] * local
            + beta[:, None, :] * (reference - normalized_backbone)
        )
        enhanced = enhanced_normalized * normalization_scale + normalization_mean
        aux = {
            "backbone": backbone_forecast,
            "normalized_backbone": normalized_backbone,
            "normalization_mean": normalization_mean,
            "normalization_scale": normalization_scale,
            "risk": risk,
            "raw_risk": raw_risk,
            "alpha": alpha,
            "beta": beta,
            "reference": reference,
            "retrieval_confidence": retrieval_confidence,
            "retrieval_similarity": similarity,
            "safety_factor": safety_factor,
            "shift_score": shift_score,
            "local": local,
        }
        return enhanced, aux
