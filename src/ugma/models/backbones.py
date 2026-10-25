from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


class DLinear(nn.Module):
    """Channel-independent linear trend/seasonal baseline."""

    def __init__(self, seq_len: int, pred_len: int, channels: int, moving_average: int = 25) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.moving_average = moving_average
        self.seasonal = nn.Linear(seq_len, pred_len)
        self.trend = nn.Linear(seq_len, pred_len)
        nn.init.constant_(self.seasonal.weight, 1.0 / seq_len)
        nn.init.constant_(self.trend.weight, 1.0 / seq_len)

    def _moving_average(self, x: torch.Tensor) -> torch.Tensor:
        padding = (self.moving_average - 1) // 2
        xt = x.transpose(1, 2)
        front = xt[:, :, :1].repeat(1, 1, padding)
        end = xt[:, :, -1:].repeat(1, 1, padding)
        padded = torch.cat([front, xt, end], dim=-1)
        trend = torch.nn.functional.avg_pool1d(padded, self.moving_average, stride=1)
        return trend.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = self._moving_average(x)
        seasonal = x - trend
        return (self.seasonal(seasonal.transpose(1, 2)) + self.trend(trend.transpose(1, 2))).transpose(1, 2)


class TinyTransformer(nn.Module):
    """Compact multivariate transformer used only for pipeline verification."""

    def __init__(self, seq_len: int, pred_len: int, channels: int, d_model: int = 64) -> None:
        super().__init__()
        self.input = nn.Linear(channels, d_model)
        layer = nn.TransformerEncoderLayer(d_model, 4, d_model * 2, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.time_projection = nn.Linear(seq_len, pred_len)
        self.output = nn.Linear(d_model, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(self.input(x))
        hidden = self.time_projection(hidden.transpose(1, 2)).transpose(1, 2)
        return self.output(hidden)


class SpectralAdaptiveProxy(nn.Module):
    """Explicitly named proxy; it is not the unavailable MFDA-Former."""

    def __init__(self, seq_len: int, pred_len: int, channels: int, d_model: int = 64) -> None:
        super().__init__()
        bins = seq_len // 2 + 1
        self.freq_gate = nn.Parameter(torch.zeros(channels, bins))
        self.projection = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(x.transpose(1, 2), dim=-1)
        gated = spectrum * torch.sigmoid(self.freq_gate)[None]
        filtered = torch.fft.irfft(gated, n=x.shape[1], dim=-1)
        return self.projection(filtered).transpose(1, 2)


class TSLibBackbone(nn.Module):
    """Adapter around unmodified official Time-Series-Library model classes."""

    def __init__(
        self,
        name: str,
        seq_len: int,
        pred_len: int,
        channels: int,
        d_model: int,
        tslib_root: str | Path,
    ) -> None:
        super().__init__()
        root = str(Path(tslib_root).resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        module = importlib.import_module(f"models.{name}")
        cfg = SimpleNamespace(
            task_name="long_term_forecast",
            seq_len=seq_len,
            label_len=seq_len // 2,
            pred_len=pred_len,
            enc_in=channels,
            dec_in=channels,
            c_out=channels,
            d_model=d_model,
            n_heads=4,
            e_layers=2,
            d_layers=1,
            d_ff=d_model * 2,
            moving_avg=25,
            factor=1,
            dropout=0.1,
            embed="timeF",
            freq="h",
            activation="gelu",
            output_attention=False,
            distil=True,
            individual=False,
            class_strategy="projection",
            use_norm=1,
            channel_independence=1,
            decomp_method="moving_avg",
            down_sampling_layers=2,
            down_sampling_window=2,
            down_sampling_method="avg",
            use_future_temporal_feature=0,
            top_k=5,
            num_kernels=6,
            seasonal_patterns="Monthly",
            p_hidden_dims=[128, 128],
            p_hidden_layers=2,
        )
        self.model = module.Model(cfg)
        self.label_len = cfg.label_len
        self.pred_len = pred_len
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        x_mark = torch.zeros(batch, x.shape[1], 4, device=x.device, dtype=x.dtype)
        dec = torch.cat(
            [x[:, -self.label_len :], torch.zeros(batch, self.pred_len, self.channels, device=x.device, dtype=x.dtype)],
            dim=1,
        )
        dec_mark = torch.zeros(batch, dec.shape[1], 4, device=x.device, dtype=x.dtype)
        prediction = self.model(x, x_mark, dec, dec_mark)
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        return prediction[:, -self.pred_len :, :]


def build_backbone(
    name: str,
    seq_len: int,
    pred_len: int,
    channels: int,
    hidden_dim: int,
    project_root: str | Path,
) -> nn.Module:
    key = name.lower()
    if key == "dlinear":
        return DLinear(seq_len, pred_len, channels)
    if key == "tiny_transformer":
        return TinyTransformer(seq_len, pred_len, channels, hidden_dim)
    if key == "spectral_proxy":
        return SpectralAdaptiveProxy(seq_len, pred_len, channels, hidden_dim)
    official_names = {
        "itransformer": "iTransformer",
        "patchtst": "PatchTST",
        "timemixer": "TimeMixer",
        "timesnet": "TimesNet",
        "fedformer": "FEDformer",
    }
    return TSLibBackbone(
        official_names[key],
        seq_len,
        pred_len,
        channels,
        hidden_dim,
        Path(project_root) / "external" / "tslib",
    )

