"""Scan a basket of coins and rank every pair by cointegration."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from .analysis import hedge_ratio
from .data import load_pair


def screen(
    coins: list[str],
    data_dir: str | Path = "data",
    window: int = 60,
    min_obs: int = 180,
) -> pd.DataFrame:
    """For every (a, b) pair, compute β, mean rolling correlation, cointegration p-value.

    Returns a DataFrame sorted by p-value (ascending = most cointegrated first).
    """
    rows = []
    for a, b in combinations(coins, 2):
        try:
            df = load_pair(a, b, data_dir)
        except FileNotFoundError:
            continue
        if len(df) < min_obs:
            continue

        log_a = np.log(df[a])
        log_b = np.log(df[b])
        ret_a = log_a.diff()
        ret_b = log_b.diff()

        beta = hedge_ratio(log_a, log_b)
        try:
            _, pvalue, _ = coint(log_a, log_b)
        except Exception:
            pvalue = float("nan")
        corr = ret_a.rolling(window).corr(ret_b).mean()

        rows.append(
            {
                "a": a,
                "b": b,
                "n_obs": len(df),
                "beta": beta,
                "mean_corr": float(corr),
                "coint_pvalue": float(pvalue),
            }
        )

    result = pd.DataFrame(rows).sort_values("coint_pvalue", kind="stable").reset_index(drop=True)
    result["cointegrated"] = result["coint_pvalue"] < 0.05
    return result
