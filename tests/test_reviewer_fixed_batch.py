from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import reviewer_shift_fixed_batch_20260911 as fixed


def test_padding_is_discarded_and_physical_shape_fixed():
    class Toy(torch.nn.Module):
        def forward(self, x, starts=None):
            assert len(x) == 32 and len(starts) == 32
            return x * 2, {"backbone": x, "risk": x.mean(1), "scalar": torch.tensor(1.)}
    model = fixed.fixed_physical_forward(Toy())
    for n in (1, 7, 16, 32):
        x = torch.randn(n, 96, 7)
        pred, aux = model(x, torch.arange(n))
        assert torch.equal(pred, x * 2)
        assert torch.equal(aux["backbone"], x)
        assert aux["risk"].shape == (n, 7)


def test_logical_partitions_have_identical_outputs():
    class Toy(torch.nn.Module):
        def forward(self, x, starts=None):
            return x.sin(), {"backbone": x}
    model = fixed.fixed_physical_forward(Toy())
    x = torch.randn(32, 96, 7)
    whole = model(x)[0]
    parts = torch.cat([model(x[:16])[0], model(x[16:])[0]])
    assert torch.equal(whole, parts)
