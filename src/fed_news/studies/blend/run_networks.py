"""ATTEMPT 15 (exploratory): the regression network and the CORAL network blended with a learned weight (CUDA).

One model per forecast (blend/model.py): branch R reads S+M + the yearly ChronoBERT tone "C" and forecasts the rate
change (Study 4 version 2); branch C reads S+M + the distilled tone "Hd" and outputs the ordinal levels (CORAL study);
alpha mixes the two rate changes and is trained together with both branches.
Three arms per network type: no tone in either branch, the real tones, and both tones shifted by half the panel
(placebo). Protocol as in the CORAL study: tuning points 2001, 2005, ..., 2025, 4 shared folds, 20 TPE trials, refits
every 8 meetings, 8 seeds. The tuning objective is the validation squared error of the mixed change.
Primary: one-sided Newey-West (8 lags) on the squared rate error, real tones against no tone, scheduled meetings from
2010, Holm over the 3 network types. "Best" network type: the lowest mean validation error at tuning (no test data).
The log loss and the tests built on it use branch C's class probabilities; branch R has none.
Reads:   the distilled-study panels (<data>/t2*.pkl, doc_stance*.pkl, coral/, distill/); for the parent comparisons
         results/distill/networks/preds_<arch>_S+M+Hd.csv and, if present,
         ../fed-news-forecasting-final/results/study4/neural/preds_<arch>_S+M+C.csv.
Writes:  results/blend/networks/: settings_*.json, preds_*.csv, metrics.csv, comparison.csv, small_sample.csv,
         parents.csv, summary.json.
Run:     python -m fed_news.studies.blend.run_networks tune
         python -m fed_news.studies.blend.run_networks forecast
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from fed_news import config
from fed_news.evaluation.metrics import metrics
from fed_news.evaluation.significance import holm
from fed_news.models.linear import classes_from_change
from fed_news.models.neural import make_sequences, standardise
from fed_news.models.walk_forward import train_mask
from fed_news.studies.blend.model import BlendNet, fit_blend, predict_blend
from fed_news.studies.coral import run_networks as rn
from fed_news.studies.coral.core import class_probs
from fed_news.studies.coral.from2010 import START
from fed_news.studies.distill import run_networks as dn
from fed_news.studies.networks.run_networks import pair_stats
from fed_news.studies.sigtests import tests as st
from fed_news.studies.simple_model.run_simple_model import log_loss

C_TONE, HD_TONE = ["c_stance", "c_stance_chg"], ["hd_stance", "hd_stance_chg"]
COLS = rn.SM + C_TONE + HD_TONE  # every network input; each branch reads its own columns of it
_idx = lambda names: [COLS.index(c) for c in names]
# arm -> (panel, branch R columns, branch C columns)
ARMS = {"blend-S+M": ("real", _idx(rn.SM), _idx(rn.SM)),
        "blend-S+M+tone": ("real", _idx(rn.SM + C_TONE), _idx(rn.SM + HD_TONE)),
        "blend-placebo": ("placebo", _idx(rn.SM + C_TONE), _idx(rn.SM + HD_TONE))}
TONE, BASE, PLACEBO = "blend-S+M+tone", "blend-S+M", "blend-placebo"
PAIRS = [(TONE, BASE), (PLACEBO, BASE), (TONE, PLACEBO)]
OUT = config.RESULTS / "blend" / "networks"
PERIODS = {"from2010": START, "full": None}
V2_PREDS = config.REPO.parent / "fed-news-forecasting-final" / "results" / "study4" / "neural"


def panels() -> dict[str, pd.DataFrame]:
    """One panel with both tones on the meetings of the earlier studies, and its placebo (both tones shifted by
    half the panel). "C" is the same panel under the name `_subsets` looks up for the unscheduled flag."""
    pan = dn.panels()
    real = pan["C"].copy()
    real[C_TONE], real[HD_TONE] = pan["C"][rn.D].to_numpy(), pan["Hd"][rn.D].to_numpy()
    return {"real": real, "placebo": rn.placebo_panel(real, C_TONE + HD_TONE), "C": real}


def _build(arch: str, arm: str, p: dict, seed: int) -> BlendNet:
    """The blend model of one arm from a settings dict."""
    _, reg, coral = ARMS[arm]
    return BlendNet(arch, reg, coral, p["hidden"], p["dropout"], seed, layers=p.get("layers", 1),
                    heads=p.get("heads", 4))


def cv_score(X, y_change, ranks, point: int, arch: str, arm: str, p: dict, device: str, seed: int = 0):
    """Mean validation squared error of the mixed change over the shared folds, and the median best epoch; infinite
    if a fold has fewer than MIN_TRAIN_SEQ training sequences for this candidate's lookback."""
    lb, vals, epochs = p["lookback"], [], []
    for train_end, va in rn.folds_for(point, 0, max(rn.LOOKBACKS)):
        tr = np.arange(lb - 1, train_end)
        if len(tr) < rn.MIN_TRAIN_SEQ:
            return np.inf, 1
        mask = np.zeros(len(X), bool)
        mask[tr] = True
        Z = standardise(X, mask)
        v, e = fit_blend(_build(arch, arm, p, seed), make_sequences(Z, tr, lb), y_change[tr], ranks[tr], p["lr"],
                         p["weight_decay"], p["batch"], rn.MAX_EPOCHS, device, seed, make_sequences(Z, va, lb),
                         y_change[va], rn.PATIENCE)
        vals.append(v)
        epochs.append(e)
    return float(np.mean(vals)), max(1, int(np.median(epochs)))


def tune_point(X, y_change, ranks, point: int, arch: str, arm: str, device: str, n_trials: int) -> dict:
    """Optuna (TPE, seed 0) at one tuning point; the best settings, their epochs and validation squared error."""
    def objective(trial):
        """Validation squared error of one candidate."""
        score, epochs = cv_score(X, y_change, ranks, point, arch, arm, rn.suggest(trial, arch, rn.LOOKBACKS), device)
        trial.set_user_attr("epochs", epochs)
        return score
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=n_trials)
    best = study.best_trial
    return {**best.params, "epochs": int(best.user_attrs["epochs"]), "cv_mse": float(best.value),
            "n_feasible_trials": int(sum(bool(np.isfinite(t.value)) for t in study.trials if t.value is not None))}


def _settings_path(out: Path, arch: str, arm: str) -> Path:
    """File of the tuned settings for one network type and arm."""
    return out / f"settings_{arch}_{arm}.json"


def _fingerprint(df: pd.DataFrame, arch: str, arm: str) -> str:
    """Fingerprint of what a tuning run reads; the arm is part of it because the arms share their columns."""
    return rn.panel_fingerprint(df, COLS, f"{arch}|{arm}")


def _load_settings(out: Path, arch: str, arm: str, df: pd.DataFrame) -> dict:
    """Saved settings keyed by row index (empty if none), refused if tuned on other dates or other values."""
    path = _settings_path(out, arch, arm)
    done = {int(k): v for k, v in json.loads(path.read_text()).items()} if path.exists() else {}
    fp = _fingerprint(df, arch, arm)
    rn._check_dates(done, df, arch, arm)
    if any(s["panel"] != fp for s in done.values()):
        raise ValueError(f"{arch} {arm}: settings were tuned on a panel with other values; delete them and retune")
    return done


def tune(pan: dict, device: str, out: Path = OUT, n_trials: int = rn.N_TRIALS, archs=rn.ARCHS) -> None:
    """Tune every arm of every network type at every tuning point (resumes from saved settings)."""
    out.mkdir(parents=True, exist_ok=True)
    for arch in archs:
        for arm, (which, _, _) in ARMS.items():
            df = pan[which]
            done = _load_settings(out, arch, arm, df)
            X, y, ranks = df[COLS].to_numpy(float), df["y_change"].to_numpy(float), df["y_class"].to_numpy(int)
            for pt in rn.tuning_points(df.index):
                if pt in done:
                    continue
                t0 = time.time()
                done[pt] = tune_point(X, y, ranks, pt, arch, arm, device, n_trials) | {
                    "date": str(df.index[pt].date()), "panel": _fingerprint(df, arch, arm)}
                rn._write_json(_settings_path(out, arch, arm), done)
                print(f"{arch} {arm} {df.index[pt].date()}: {done[pt]} ({time.time() - t0:.0f}s)", flush=True)


def run_tuned(df: pd.DataFrame, arch: str, arm: str, settings: dict, seed: int, device: str) -> pd.DataFrame:
    """Walk-forward forecasts of one blend model with the tuned settings and one seed. Each refit trains on the
    meetings before the forecast meeting only (`train_mask`)."""
    X, y_change = df[COLS].to_numpy(float), df["y_change"].to_numpy(float)
    lag, y_level, ranks, dates = df["s_rate_lag"].to_numpy(float), df["y_level"].to_numpy(float), \
        df["y_class"].to_numpy(int), df.index
    test, points = np.flatnonzero(dates >= pd.Timestamp(config.TEST_START)), sorted(settings)
    out, model, current = [], None, None
    for k, i in enumerate(test):
        pt = rn.point_for(i, points)
        if k % rn.REFIT_EVERY == 0 or pt != current:
            current, p = pt, settings[pt]
            train = train_mask(len(df), i, "T2")
            Z = standardise(X, train)
            rows = np.flatnonzero(train)
            rows = rows[rows >= p["lookback"] - 1]
            model = _build(arch, arm, p, seed)
            fit_blend(model, make_sequences(Z, rows, p["lookback"]), y_change[rows], ranks[rows], p["lr"],
                      p["weight_decay"], p["batch"], p["epochs"], device, seed)
        f = predict_blend(model, make_sequences(Z, np.array([i]), p["lookback"]), device)
        if not np.isfinite(f["change"]).all():
            raise ValueError(f"{arch} {arm}: non-finite forecast at {dates[i].date()}")
        out.append({"date": dates[i], "y_true": y_level[i], "lag": lag[i], "class_true": int(ranks[i]),
                    "y_pred": lag[i] + f["change"][0], "y_reg": lag[i] + f["change_reg"][0],
                    "y_coral": lag[i] + f["change_coral"][0], "lp1": f["lp"][0, 0], "lp2": f["lp"][0, 1],
                    "alpha": f["alpha"]})
    return pd.DataFrame(out)


def seed_forecast(df, arch: str, arm: str, settings: dict, seed: int, device: str, path: Path) -> pd.DataFrame:
    """One seed's forecasts: reused from `path` only if made with these exact settings; written atomically."""
    h = rn.settings_hash(settings)
    if path.exists():
        cached = pd.read_csv(path, parse_dates=["date"], dtype={"settings_hash": str})
        if len(cached) == int((df.index >= pd.Timestamp(config.TEST_START)).sum()) and (
                cached["settings_hash"] == h).all():
            return cached
    out = run_tuned(df, arch, arm, settings, seed, device).assign(settings_hash=h)
    tmp = path.with_name(path.name + ".tmp")
    out.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return out


def ensemble(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Seed means of the forecasts and of alpha. The class is the mixed change against the +-0.125 band (the rule of
    the regression study); the class probabilities come from the mean CORAL levels."""
    ens = frames[0][["date", "y_true", "lag", "class_true"]].copy()
    for c in ("y_pred", "y_reg", "y_coral", "lp1", "lp2", "alpha"):
        ens[c] = np.mean([f[c].to_numpy(float) for f in frames], axis=0)
    p = class_probs(ens[["lp1", "lp2"]].to_numpy(float))
    ens["p_cut"], ens["p_hold"], ens["p_hike"] = p[:, 0], p[:, 1], p[:, 2]
    ens["class_pred"] = classes_from_change(ens["y_pred"] - ens["lag"])
    return ens


def small_sample(base: pd.DataFrame, new: pd.DataFrame) -> dict:
    """Small-sample checks (one-sided p, positive differences favour `new`): fixed-b and fixed-m on the squared
    rate error and on the log loss; Newey-West on the ranked probability score."""
    y = base["y_true"].to_numpy(float)
    d_sq = (y - base["y_pred"].to_numpy(float)) ** 2 - (y - new["y_pred"].to_numpy(float)) ** 2
    d_ll = log_loss(base) - log_loss(new)
    return {"sq_fixed_b_p": st.fixed_b(d_sq)["p_one_sided"], "sq_fixed_m_p": st.fixed_m(d_sq)["p_one_sided"],
            "logloss_fixed_b_p": st.fixed_b(d_ll)["p_one_sided"], "logloss_fixed_m_p": st.fixed_m(d_ll)["p_one_sided"],
            "rps_nw_p": st.newey_west(st.rps(base) - st.rps(new))["p_one_sided"]}


def primary_holm(comp: pd.DataFrame) -> pd.Series:
    """Holm over the primary tests (real tones against no tone, squared error, scheduled meetings from 2010)."""
    prim = comp[(comp["model"] == TONE) & (comp["against"] == BASE) & (comp["period"] == "from2010")
                & (comp["subset"] == "scheduled")]
    adj = holm({i: p for i, p in prim["sq_nw_p"].items()})
    return pd.Series({i: adj.get(i, np.nan) for i in comp.index}, dtype=float)


def reading(comp: pd.DataFrame, period: str = "from2010") -> dict[str, bool]:
    """Placebo reading on the squared error: the gain is from the tones' content when the real tones beat the
    placebo at p < 0.05 and the placebo does not beat no tone at p < 0.05 (scheduled meetings)."""
    c = comp[(comp["period"] == period) & (comp["subset"] == "scheduled")]
    get = lambda a, m, b: float(c[(c["arch"] == a) & (c["model"] == m) & (c["against"] == b)]["sq_nw_p"].iloc[0])
    return {a: bool(get(a, TONE, PLACEBO) < config.ALPHA and get(a, PLACEBO, BASE) >= config.ALPHA)
            for a in c["arch"].unique()}


def best_arch(settings: dict[str, dict]) -> str:
    """The network type whose real-tone arm has the lowest mean validation squared error over the tuning points."""
    return min(settings, key=lambda a: np.mean([s["cv_mse"] for s in settings[a].values()]))


def _keep(dates: pd.Series, start, sub_mask) -> np.ndarray:
    """Rows of the subset that are dated `start` or later (all rows if start is None)."""
    return sub_mask & ((dates >= pd.Timestamp(start)).to_numpy() if start else True)


def parents(preds: dict, pan: dict, archs) -> pd.DataFrame:
    """The real-tone blend against each parent's saved forecasts, same network type, scheduled meetings: the CORAL
    network with the distilled tone (all tests) and, if the other repository is next to this one, the regression
    network with the ChronoBERT tone (squared error and accuracy only: it has no class probabilities)."""
    rows = []
    for arch in archs:
        new = preds[(arch, TONE)]
        for name, path in (("coral S+M+Hd", dn.OUT / f"preds_{arch}_S+M+Hd.csv"),
                           ("regression S+M+C", V2_PREDS / f"preds_{arch}_S+M+C.csv")):
            if not path.exists():
                continue
            old = pd.read_csv(path, parse_dates=["date"])
            both = np.intersect1d(old["date"], new["date"])
            o, n = (f[f["date"].isin(both)].reset_index(drop=True) for f in (old, new))
            for period, start in PERIODS.items():
                keep = _keep(o["date"], start, rn._subsets(rn.T2, pan, o["date"])["scheduled"])
                ok, nk = o[keep].reset_index(drop=True), n[keep].reset_index(drop=True)
                stats = rn.pair_stats_coral(ok, nk) if "p_cut" in ok else {
                    "sq_nw_p" if k == "nw_p" else k: v for k, v in pair_stats(ok, nk, rn.REFIT_EVERY).items()}
                rows.append({"arch": arch, "model": TONE, "against": name, "period": period,
                             "rmse_parent": metrics(ok)["rmse"], "rmse_blend": metrics(nk)["rmse"],
                             "accuracy_parent": metrics(ok)["accuracy"], "accuracy_blend": metrics(nk)["accuracy"],
                             **stats})
    return pd.DataFrame(rows)


def forecast(pan: dict, device: str, out: Path = OUT, archs=rn.ARCHS, seeds=rn.SEEDS) -> dict:
    """Forecast every arm (seed mean), score, test, compare with the parents; write the tables and the summary."""
    preds, tone_settings = {}, {}
    for arch in archs:
        for arm, (which, _, _) in ARMS.items():
            df = pan[which]
            settings = _load_settings(out, arch, arm, df)
            missing = sorted(set(rn.tuning_points(df.index)) - set(settings))
            if missing:
                raise ValueError(f"{arch} {arm}: tuning incomplete, missing tuning points {missing}")
            ens = ensemble([seed_forecast(df, arch, arm, settings, s, device,
                                          out / f"preds_{arch}_{arm}_seed{s}.csv") for s in seeds])
            ens.to_csv(out / f"preds_{arch}_{arm}.csv", index=False)
            preds[(arch, arm)] = ens
            if arm == TONE:
                tone_settings[arch] = settings
    rows, comps, small = [], [], []
    for (arch, arm), p in preds.items():
        for period, start in PERIODS.items():
            for sub, mask in rn._subsets(rn.T2, pan, p["date"]).items():
                q = p[_keep(p["date"], start, mask)]
                rows.append({"arch": arch, "features": arm, "period": period, "subset": sub, **metrics(q),
                             "log_loss": float(log_loss(q).mean()), "alpha_mean": float(q["alpha"].mean()),
                             "rmse_reg_branch": float(np.sqrt(np.mean((q["y_reg"] - q["y_true"]) ** 2))),
                             "rmse_coral_branch": float(np.sqrt(np.mean((q["y_coral"] - q["y_true"]) ** 2)))})
    for arch in archs:
        for new, old in PAIRS:
            b, n = preds[(arch, old)], preds[(arch, new)]
            for period, start in PERIODS.items():
                for sub, mask in rn._subsets(rn.T2, pan, b["date"]).items():
                    keep = _keep(b["date"], start, mask)
                    bk, nk = b[keep].reset_index(drop=True), n[keep].reset_index(drop=True)
                    comps.append({"arch": arch, "model": new, "against": old, "period": period, "subset": sub,
                                  **rn.pair_stats_coral(bk, nk)})
                    if sub == "scheduled":
                        small.append({"arch": arch, "model": new, "against": old, "period": period, "n": len(bk),
                                      **small_sample(bk, nk)})
    met, comp, small, par = pd.DataFrame(rows), pd.DataFrame(comps), pd.DataFrame(small), parents(preds, pan, archs)
    comp["sq_nw_p_holm"] = primary_holm(comp)
    for name, table in (("metrics", met), ("comparison", comp), ("small_sample", small), ("parents", par)):
        table.to_csv(out / f"{name}.csv", index=False)
    best = best_arch(tone_settings)
    at = lambda t, **kw: t[np.logical_and.reduce([t[k] == v for k, v in kw.items()])]
    one = lambda t, **kw: at(t, **kw).drop(columns=list(kw)).iloc[0].to_dict()
    sched = dict(period="from2010", subset="scheduled")
    summary = {
        "note": "exploratory (attempt 15); log loss and the tests on it use the CORAL branch's probabilities",
        "primary_from2010_scheduled": {r.arch: {"sq_nw_p": r.sq_nw_p, "sq_nw_p_holm": r.sq_nw_p_holm,
                                                "sq_diff_mean": r.sq_diff_mean}
                                       for r in comp[comp["sq_nw_p_holm"].notna()].itertuples()},
        "gain_from_tone_content_from2010": reading(comp, "from2010"),
        "gain_from_tone_content_full": reading(comp, "full"),
        "best_arch_by_validation": best,
        "best": {"scores": {arm: one(met, arch=best, features=arm, **sched) for arm in ARMS},
                 "tests": {f"{new} vs {old}": one(comp, arch=best, model=new, against=old, **sched)
                           | one(small, arch=best, model=new, against=old, period="from2010")
                           for new, old in PAIRS},
                 "vs_parents": {r["against"]: {k: v for k, v in r.items() if k not in ("arch", "model", "against")}
                                for r in at(par, arch=best, period="from2010").to_dict("records")} if len(par) else {}}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float), flush=True)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["tune", "forecast"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=OUT, help="results folder (use another one for a trial run)")
    ap.add_argument("--trials", type=int, default=rn.N_TRIALS)
    ap.add_argument("--archs", nargs="+", default=list(rn.ARCHS), choices=rn.ARCHS)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(rn.SEEDS))
    args = ap.parse_args()
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")
    p = panels()
    if args.step == "tune":
        tune(p, args.device, args.out, args.trials, args.archs)
    else:
        forecast(p, args.device, args.out, args.archs, args.seeds)
