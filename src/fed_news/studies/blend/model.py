"""MODEL: two networks trained as one, their rate-change forecasts mixed by a learned weight alpha.

Used by blend/run_networks.py.

Branch R is the Study 4 version 2 network (models/neural.py): it outputs one number, the rate change. Branch C is the
CORAL network (studies/coral/networks.py): it outputs the two ordinal levels (cut 00, hold 10, hike 11), turned into a
rate change as P(cut) * size_cut + P(hike) * size_hike. The forecast is

    change = alpha * change_C + (1 - alpha) * change_R,    alpha = sigmoid(a)

and `a` is a parameter like any other: the squared error of the mixed change is backpropagated into alpha and into
both branches, so the three are tuned together. The CORAL loss is added so that branch C keeps giving usable class
probabilities. The squared error is divided by the training variance of the target so both loss terms are of order 1.
"""
import numpy as np
import torch
import torch.nn as nn

from fed_news.models.linear import move_sizes
from fed_news.models.neural import build_model
from fed_news.studies.coral.core import coral_loss, init_biases, levels
from fed_news.studies.coral.networks import build_coral_model


class BlendNet(nn.Module):
    """Regression branch + CORAL branch + the mixing weight. Each branch reads its own columns of the input."""

    def __init__(self, arch: str, reg_cols, coral_cols, hidden: int, dropout: float, seed: int, layers: int = 1,
                 heads: int = 4):
        super().__init__()
        self.reg = build_model(arch, len(reg_cols), hidden, dropout, seed, layers=layers, heads=heads)
        self.coral = build_coral_model(arch, len(coral_cols), hidden, dropout, seed, layers=layers, heads=heads)
        self.a = nn.Parameter(torch.zeros(()))  # alpha = sigmoid(a) starts at 0.5
        self.register_buffer("reg_idx", torch.as_tensor(list(reg_cols), dtype=torch.long))
        self.register_buffer("coral_idx", torch.as_tensor(list(coral_cols), dtype=torch.long))
        self.register_buffer("sizes", torch.tensor([-0.25, 0.0, 0.25]))
        self.register_buffer("mu", torch.zeros(()))
        self.register_buffer("sd", torch.ones(()))

    def prepare(self, ranks, y_change) -> None:
        """Set everything fixed by the training rows: the CORAL start biases, the mean move per class, and the
        target's mean and standard deviation (branch R forecasts the standardised change, as in Study 4)."""
        ranks, y = np.asarray(ranks, dtype=int), np.asarray(y_change, dtype=float)
        init_biases(self.coral.head, ranks)
        sd = float(y.std())
        with torch.no_grad():
            self.sizes.copy_(torch.as_tensor(move_sizes(ranks, y), dtype=torch.float32))
            self.mu.fill_(float(y.mean()))
            self.sd.fill_(sd if sd > 0 else 1.0)

    @property
    def alpha(self) -> torch.Tensor:
        """The weight on the CORAL branch, between 0 and 1."""
        return torch.sigmoid(self.a)

    def forward(self, x):
        """(mixed change, CORAL level logits, branch R change, branch C change), changes in rate units."""
        change_r = self.reg(x[..., self.reg_idx]) * self.sd + self.mu
        logits = self.coral(x[..., self.coral_idx])
        lp = torch.sigmoid(logits)
        change_c = (1.0 - lp[:, 0]) * self.sizes[0] + lp[:, 1] * self.sizes[2]
        return self.alpha * change_c + (1.0 - self.alpha) * change_r, logits, change_r, change_c


def blend_loss(model: BlendNet, x, y, lv) -> torch.Tensor:
    """Squared error of the mixed change (in units of the training variance) plus the CORAL loss of branch C."""
    change, logits, _, _ = model(x)
    return nn.functional.mse_loss(change, y) / model.sd ** 2 + coral_loss(logits, lv)


def _tensor(a, device: str) -> torch.Tensor:
    """Float32 tensor on `device`."""
    return torch.tensor(np.asarray(a), dtype=torch.float32, device=device)


def fit_blend(model: BlendNet, X, y_change, ranks, lr: float, weight_decay: float, batch_size: int, epochs: int,
              device: str, seed: int = 0, X_val=None, y_val=None, patience: int | None = None) -> tuple[float, int]:
    """Mini-batch Adam on the joint loss (same shuffling as models/neural.py). Without validation rows: exactly
    `epochs` epochs. With them: stop after `patience` epochs without a better validation squared error of the mixed
    change (rate units); returns that best value and its epoch."""
    model.prepare(ranks, y_change)
    model.to(device)
    xt, yt, lv = _tensor(X, device), _tensor(y_change, device), levels(ranks).to(device)
    free = [model.a, model.coral.head.bias]  # the mixing weight and the CORAL thresholds are not weight-decayed
    rest = [p for p in model.parameters() if all(p is not f for f in free)]
    opt = torch.optim.Adam([{"params": rest, "weight_decay": weight_decay}, {"params": free, "weight_decay": 0.0}],
                           lr=lr)
    rng = np.random.default_rng(seed)
    best, best_epoch, wait = np.inf, 1, 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.as_tensor(rng.permutation(len(yt)), device=device)
        for s in range(0, len(yt), batch_size):
            idx = order[s: s + batch_size]
            opt.zero_grad()
            loss = blend_loss(model, xt[idx], yt[idx], lv[idx])
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            loss.backward()
            opt.step()
        if X_val is None:
            continue
        v = float(np.mean((predict_blend(model, X_val, device)["change"] - np.asarray(y_val, dtype=float)) ** 2))
        if not np.isfinite(v):
            raise ValueError("non-finite validation loss")
        if v < best - 1e-12:
            best, best_epoch, wait = v, epoch, 0
        else:
            wait += 1
            if wait >= patience:
                break
    return best, best_epoch


def predict_blend(model: BlendNet, X, device: str) -> dict:
    """Forecasts for X (no gradient): mixed `change`, level probabilities `lp`, each branch's change, and alpha."""
    model.to(device).eval()
    with torch.no_grad():
        change, logits, change_r, change_c = model(_tensor(X, device))
        out = {"change": change, "lp": torch.sigmoid(logits), "change_reg": change_r, "change_coral": change_c}
        return {k: v.float().cpu().numpy().astype(float) for k, v in out.items()} | {"alpha": float(model.alpha)}
