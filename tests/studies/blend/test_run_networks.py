import numpy as np
import pandas as pd

from fed_news.studies.blend import run_networks as bn
from fed_news.studies.coral import run_networks as rn


def test_each_branch_gets_its_own_tone_and_the_baseline_gets_none():
    names = lambda idx: [bn.COLS[i] for i in idx]
    _, reg, coral = bn.ARMS[bn.TONE]
    assert names(reg) == rn.SM + bn.C_TONE and names(coral) == rn.SM + bn.HD_TONE
    _, reg, coral = bn.ARMS[bn.BASE]
    assert names(reg) == rn.SM and names(coral) == rn.SM
    assert bn.ARMS[bn.PLACEBO] == ("placebo", *bn.ARMS[bn.TONE][1:])


def _seed(y_pred, lp1, alpha):
    return pd.DataFrame({"date": pd.to_datetime(["2020-01-29", "2020-03-18"]), "y_true": [1.5, 0.25], "lag": 1.5,
                         "class_true": [1, 0], "y_pred": y_pred, "y_reg": y_pred, "y_coral": y_pred, "lp1": lp1,
                         "lp2": 0.0, "alpha": alpha})


def test_ensemble_averages_the_seeds_and_classes_the_mixed_change_with_the_band():
    ens = bn.ensemble([_seed([1.5, 1.0], [1.0, 0.2], 0.4), _seed([1.6, 1.2], [1.0, 0.4], 0.6)])
    assert np.allclose(ens["y_pred"], [1.55, 1.1]) and np.allclose(ens["alpha"], 0.5)
    assert list(ens["class_pred"]) == [1, 0]  # +0.05 is inside the band (hold); -0.4 is a cut
    assert np.allclose(ens[["p_cut", "p_hold", "p_hike"]], [[0.0, 1.0, 0.0], [0.7, 0.3, 0.0]])


def test_holm_runs_over_the_primary_squared_error_tests_only():
    row = lambda arch, model, against, period, p: {"arch": arch, "model": model, "against": against,
                                                   "period": period, "subset": "scheduled", "sq_nw_p": p}
    comp = pd.DataFrame([row("lstm", bn.TONE, bn.BASE, "from2010", 0.01), row("gru", bn.TONE, bn.BASE, "from2010", 0.02),
                         row("tft", bn.TONE, bn.BASE, "from2010", 0.3), row("lstm", bn.TONE, bn.BASE, "full", 0.001),
                         row("lstm", bn.PLACEBO, bn.BASE, "from2010", 0.001)])
    adj = bn.primary_holm(comp)
    assert np.allclose(adj[:3], [0.03, 0.04, 0.3]) and adj[3:].isna().all()


def test_placebo_reading_needs_real_beating_placebo_and_placebo_not_beating_no_tone():
    rows = []
    for arch, (real_vs_pl, pl_vs_base) in {"lstm": (0.03, 0.5), "gru": (0.2, 0.5), "tft": (0.03, 0.04)}.items():
        rows += [{"arch": arch, "model": bn.TONE, "against": bn.PLACEBO, "sq_nw_p": real_vs_pl},
                 {"arch": arch, "model": bn.PLACEBO, "against": bn.BASE, "sq_nw_p": pl_vs_base}]
    comp = pd.DataFrame(rows).assign(period="from2010", subset="scheduled")
    assert bn.reading(comp) == {"lstm": True, "gru": False, "tft": False}


def test_best_network_is_the_one_with_the_lowest_mean_validation_error():
    s = lambda *v: {i: {"cv_mse": x} for i, x in enumerate(v)}
    assert bn.best_arch({"lstm": s(0.02, 0.04), "gru": s(0.01, 0.03), "tft": s(0.05, 0.001)}) == "gru"
