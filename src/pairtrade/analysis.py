"""Pair-trading analytics: hedge ratio, spread, z-score, signals, cointegration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


@dataclass
class PairStats:
    a: str
    b: str
    n_obs: int
    beta: float
    mean_corr: float
    coint_pvalue: float
    zscore_min: float
    zscore_max: float
    signal_changes: int

    @property
    def cointegrated(self) -> bool:
        return self.coint_pvalue < 0.05


def hedge_ratio(y: pd.Series, x: pd.Series) -> float:
    """OLS β from y = α + β·x (use log prices)."""
    X = sm.add_constant(x.values)
    return float(sm.OLS(y.values, X).fit().params[1])


def _signals(zscore: np.ndarray, entry_z: float, exit_z: float) -> np.ndarray:
    """State machine: +1 long spread, -1 short spread, 0 flat."""
    pos = np.zeros(len(zscore))
    state = 0
    for i, z in enumerate(zscore):
        if np.isnan(z):
            pos[i] = 0
            continue
        if state == 0:
            if z > entry_z:
                state = -1
            elif z < -entry_z:
                state = 1
        elif state == 1 and z >= -exit_z:
            state = 0
        elif state == -1 and z <= exit_z:
            state = 0
        pos[i] = state
    return pos


def analyze(
    df: pd.DataFrame,
    a: str,
    b: str,
    window: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> pd.DataFrame:
    """Add log prices, returns, rolling corr, spread, z-score, and position columns."""
    out = df.copy()
    out[f"log_{a}"] = np.log(out[a])
    out[f"log_{b}"] = np.log(out[b])
    out[f"ret_{a}"] = out[f"log_{a}"].diff()
    out[f"ret_{b}"] = out[f"log_{b}"].diff()

    out["rolling_corr"] = out[f"ret_{a}"].rolling(window).corr(out[f"ret_{b}"])

    beta = hedge_ratio(out[f"log_{a}"], out[f"log_{b}"])
    out["spread"] = out[f"log_{a}"] - beta * out[f"log_{b}"]
    out["spread_mean"] = out["spread"].rolling(window).mean()
    out["spread_std"] = out["spread"].rolling(window).std()
    out["zscore"] = (out["spread"] - out["spread_mean"]) / out["spread_std"]
    out["position"] = _signals(out["zscore"].values, entry_z, exit_z)

    out.attrs.update(
        a=a, b=b, beta=beta, window=window, entry_z=entry_z, exit_z=exit_z
    )
    return out


def summarize(out: pd.DataFrame) -> PairStats:
    a, b = out.attrs["a"], out.attrs["b"]
    df = out.dropna(subset=["zscore"])
    _, pvalue, _ = coint(df[f"log_{a}"], df[f"log_{b}"])
    return PairStats(
        a=a,
        b=b,
        n_obs=len(df),
        beta=out.attrs["beta"],
        mean_corr=float(df["rolling_corr"].mean()),
        coint_pvalue=float(pvalue),
        zscore_min=float(df["zscore"].min()),
        zscore_max=float(df["zscore"].max()),
        signal_changes=int((df["position"].diff().abs() > 0).sum()),
    )


def print_summary(stats: PairStats) -> None:
    flag = "(cointegrated)" if stats.cointegrated else "(NOT cointegrated)"
    print(f"\nPair: {stats.a} / {stats.b}")
    print(f"  Observations:        {stats.n_obs:,}")
    print(f"  beta (hedge ratio):  {stats.beta:.4f}")
    print(f"  Mean rolling corr:   {stats.mean_corr:.3f}")
    print(f"  Cointegration p:     {stats.coint_pvalue:.4f}   {flag}")
    print(f"  z-score range:       [{stats.zscore_min:.2f}, {stats.zscore_max:.2f}]")
    print(f"  Signal changes:      {stats.signal_changes}")
