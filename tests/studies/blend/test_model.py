import numpy as np
import pytest
import torch

from fed_news.studies.blend.model import BlendNet, fit_blend, predict_blend

REG, CORAL = [0, 1, 2], [0, 1, 3]


def _data(n=40, lookback=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, lookback, 4)).astype(np.float32)
    ranks = rng.integers(0, 3, n)
    return X, (ranks - 1) * 0.25 + rng.normal(scale=0.01, size=n), ranks


@pytest.mark.parametrize("arch", ["lstm", "gru", "tft"])
def test_the_forecast_is_the_alpha_weighted_mix_of_the_two_branches(arch):
    X, y, ranks = _data()
    model = BlendNet(arch, REG, CORAL, hidden=8, dropout=0.0, seed=0)
    model.prepare(ranks, y)
    out = predict_blend(model, X, "cpu")
    assert out["alpha"] == pytest.approx(0.5)
    assert np.allclose(out["change"], 0.5 * out["change_coral"] + 0.5 * out["change_reg"], atol=1e-6)
    sizes = model.sizes.numpy()
    assert np.allclose(out["change_coral"], (1 - out["lp"][:, 0]) * sizes[0] + out["lp"][:, 1] * sizes[2], atol=1e-6)


def test_alpha_one_gives_the_coral_branch_and_alpha_zero_the_regression_branch():
    X, y, ranks = _data()
    model = BlendNet("gru", REG, CORAL, hidden=8, dropout=0.0, seed=0)
    model.prepare(ranks, y)
    with torch.no_grad():
        model.a.fill_(50.0)
    out = predict_blend(model, X, "cpu")
    assert np.allclose(out["change"], out["change_coral"], atol=1e-6)
    with torch.no_grad():
        model.a.fill_(-50.0)
    out = predict_blend(model, X, "cpu")
    assert np.allclose(out["change"], out["change_reg"], atol=1e-6)


def test_training_moves_alpha_and_the_weights_of_both_branches():
    X, y, ranks = _data()
    model = BlendNet("lstm", REG, CORAL, hidden=8, dropout=0.0, seed=0)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    fit_blend(model, X, y, ranks, lr=1e-2, weight_decay=0.0, batch_size=16, epochs=3, device="cpu")
    after = model.state_dict()
    assert 0.0 < float(model.alpha) < 1.0 and float(model.alpha) != 0.5
    assert not torch.equal(before["reg.head.weight"], after["reg.head.weight"])
    assert not torch.equal(before["coral.head.weight.weight"], after["coral.head.weight.weight"])


def test_each_branch_reads_only_its_own_columns():
    X, y, ranks = _data()
    model = BlendNet("gru", REG, CORAL, hidden=8, dropout=0.0, seed=0)
    model.prepare(ranks, y)
    base = predict_blend(model, X, "cpu")
    X2 = X.copy()
    X2[:, :, 3] += 5.0  # column 3 belongs to the CORAL branch only
    moved = predict_blend(model, X2, "cpu")
    assert np.allclose(base["change_reg"], moved["change_reg"])
    assert not np.allclose(base["change_coral"], moved["change_coral"])


def test_validation_run_stops_early_and_reports_a_finite_squared_error():
    X, y, ranks = _data(60)
    model = BlendNet("gru", REG, CORAL, hidden=8, dropout=0.0, seed=0)
    best, epoch = fit_blend(model, X[:40], y[:40], ranks[:40], 1e-2, 0.0, 16, 50, "cpu", X_val=X[40:], y_val=y[40:],
                            patience=3)
    assert np.isfinite(best) and 1 <= epoch <= 50
