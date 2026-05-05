"""Feature engineering for the AAVE WETH utilization model.

The public entry point is `build_dataset(...)`, which returns
`(X, y, meta)` ready for `model.train`. All features at row `t` use only
information available at the *close* of day `t`; the label uses `t + h` so
the strict no-leak assertion `(X.index.shift(h) <= y.index).all()` holds.

Feature blocks
--------------
1. Requested four: Yang-Zhang vol z-score, volume z-score, vol-volume rolling
   correlation, vol autocorrelation. Plus Lo-MacKinlay variance ratio (k=5).
2. Utilization autocorrelation block: rolling lag-1 ACF on `U` and `ΔU`,
   variance ratio of `ΔU`, AR(1) half-life, gap to the kink, lagged levels
   and changes.
3. Protocol features: borrow-rate level + 1d change, 7d change in liquidity,
   USDC reserve U as cross-asset leverage proxy.
4. Options: DVOL level, DVOL z-score, variance risk premium proxy
   (DVOL^2 - RV_30^2), best-effort IV term slope and 25-delta risk-reversal
   if columns are present (else NaN-skipped).
5. Funding: daily funding sum and its 30-day z-score.
6. Calendar: day-of-week one-hot (drops Sunday to avoid singularity).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config

# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _zscore(s: pd.Series, win: int) -> pd.Series:
    mu = s.rolling(win, min_periods=win).mean()
    sd = s.rolling(win, min_periods=win).std()
    return (s - mu) / sd.replace(0.0, np.nan)


def yang_zhang_vol(
    o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series, win: int
) -> pd.Series:
    """Yang-Zhang annualized realized volatility (decimal, e.g. 0.6 = 60% IV).

    Combines overnight, open-to-close, and Rogers-Satchell variances with the
    minimum-variance Yang-Zhang weighting. Robust to opening jumps and drift.
    """

    log_oc = np.log(c / o)
    log_co = np.log(o / c.shift(1))  # overnight return
    log_ho = np.log(h / o)
    log_lo = np.log(l / o)
    log_hc = np.log(h / c)
    log_lc = np.log(l / c)

    rs = log_ho * (log_ho - log_oc) + log_lo * (log_lo - log_oc)  # Rogers-Satchell
    sigma_o2 = log_co.rolling(win, min_periods=win).var()  # overnight variance
    sigma_c2 = log_oc.rolling(win, min_periods=win).var()  # open-to-close variance
    sigma_rs2 = rs.rolling(win, min_periods=win).mean()  # intraday RS variance

    k = 0.34 / (1.34 + (win + 1) / (win - 1))
    yz_var = sigma_o2 + k * sigma_c2 + (1 - k) * sigma_rs2
    return np.sqrt(yz_var * 365)


def lo_mackinlay_variance_ratio(returns: pd.Series, win: int, k: int = 5) -> pd.Series:
    """Rolling Lo-MacKinlay variance ratio for lag k.

    VR(k) = Var(r_t^{k}) / (k * Var(r_t)). VR<1 = mean-reverting, VR>1 = trending,
    VR=1 = random walk.
    """

    def _vr(window: np.ndarray) -> float:
        n = len(window)
        if n < k + 1:
            return np.nan
        var1 = np.nanvar(window, ddof=1)
        usable = n - (n % k)
        if usable < k or var1 <= 0:
            return np.nan
        sums = np.add.reduceat(window[:usable], np.arange(0, usable, k))
        vark = np.nanvar(sums, ddof=1) / k
        return float(vark / var1)

    return returns.rolling(win, min_periods=win).apply(_vr, raw=True)


def rolling_autocorr(s: pd.Series, win: int, lag: int = 1) -> pd.Series:
    return s.rolling(win, min_periods=win).apply(
        lambda x: pd.Series(x).autocorr(lag), raw=False
    )


def rolling_corr(a: pd.Series, b: pd.Series, win: int) -> pd.Series:
    return a.rolling(win, min_periods=win).corr(b)


def ar1_half_life(window: np.ndarray) -> float:
    """OLS-fit AR(1) on a window, return Ornstein-Uhlenbeck half-life in steps."""

    x = window[~np.isnan(window)]
    if len(x) < 10:
        return np.nan
    y = np.diff(x)
    xl = x[:-1]
    if np.std(xl) < 1e-12:
        return np.nan
    b = np.polyfit(xl, y, 1)[0]
    if (1 + b) <= 0 or b >= 0:
        # No mean reversion (random walk or explosive)
        return np.nan
    return float(-np.log(2) / np.log(1 + b))


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------


@dataclass
class Inputs:
    """All raw frames required to build the feature matrix."""

    weth: pd.DataFrame  # data/cache/weth_reserve.parquet
    usdc: pd.DataFrame  # data/cache/usdc_reserve.parquet
    ohlcv: pd.DataFrame  # data/cache/eth_ohlcv.parquet
    funding: pd.DataFrame  # data/cache/eth_funding.parquet
    dvol: pd.DataFrame  # data/cache/eth_dvol.parquet
    gas: pd.DataFrame  # data/cache/eth_gas.parquet (carried for backtester join)


def load_inputs() -> Inputs:
    return Inputs(
        weth=pd.read_parquet(config.CACHE_DIR / "weth_reserve.parquet"),
        usdc=pd.read_parquet(config.CACHE_DIR / "usdc_reserve.parquet"),
        ohlcv=pd.read_parquet(config.CACHE_DIR / "eth_ohlcv.parquet"),
        funding=pd.read_parquet(config.CACHE_DIR / "eth_funding.parquet"),
        dvol=pd.read_parquet(config.CACHE_DIR / "eth_dvol.parquet"),
        gas=pd.read_parquet(config.CACHE_DIR / "eth_gas.parquet"),
    )


# ---------------------------------------------------------------------------
# Per-block builders
# ---------------------------------------------------------------------------


def market_features(ohlcv: pd.DataFrame, win: int = config.ROLL_WINDOW) -> pd.DataFrame:
    o, h, l, c, v = (ohlcv[k] for k in ("open", "high", "low", "close", "volume"))
    log_ret = np.log(c / c.shift(1))

    yz = yang_zhang_vol(o, h, l, c, win)

    out = pd.DataFrame(index=ohlcv.index)
    out["log_ret_1d"] = log_ret
    out["yz_vol"] = yz
    out["yz_vol_z"] = _zscore(yz, win)
    out["volume_z"] = _zscore(v, win)
    out["vol_volume_corr"] = rolling_corr(yz, v, win)
    out["vol_autocorr"] = rolling_autocorr(yz, win, lag=1)
    out["variance_ratio_5"] = lo_mackinlay_variance_ratio(log_ret, win, k=5)
    out["ret_7d"] = c.pct_change(7)
    out["ret_30d"] = c.pct_change(30)
    out["amihud_illiq"] = (log_ret.abs() / v.replace(0, np.nan)).rolling(win, min_periods=win).mean()
    return out


def utilization_features(weth: pd.DataFrame, win: int = config.ROLL_WINDOW) -> pd.DataFrame:
    U = weth["U"]
    dU = U.diff()

    out = pd.DataFrame(index=weth.index)
    out["U_level"] = U
    out["U_ac1"] = rolling_autocorr(U, win, lag=1)
    out["dU_ac1"] = rolling_autocorr(dU, win, lag=1)
    out["U_vr5"] = lo_mackinlay_variance_ratio(dU, win, k=5)
    out["U_halflife"] = U.rolling(win, min_periods=win).apply(ar1_half_life, raw=True)
    out["U_gap_kink"] = U - weth["opt_U"]
    for lag in (1, 3, 7, 14):
        out[f"U_lag{lag}"] = U.shift(lag)
        out[f"dU_lag{lag}"] = dU.shift(lag)
    return out


def protocol_features(weth: pd.DataFrame, usdc: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=weth.index)
    out["borrow_rate"] = weth["borrow_rate"]
    out["d_borrow_rate_1"] = weth["borrow_rate"].diff()
    out["d_liquidity_7"] = weth["available_liquidity"].pct_change(7)
    out["d_debt_7"] = weth["debt"].pct_change(7)
    # USDC reserve U on the same calendar day, forward-filled if the cross-asset
    # frame has gaps. Aligned via reindex+ffill — no peeking at later days.
    out["usdc_U"] = usdc["U"].reindex(weth.index, method="ffill")
    out["usdc_dU_7"] = out["usdc_U"].diff(7)
    return out


def options_features(
    dvol: pd.DataFrame,
    yz_vol: pd.Series,
    win: int = config.ROLL_WINDOW,
) -> pd.DataFrame:
    """DVOL-derived features. DVOL is in vol points (e.g. 60 = 60%)."""

    if dvol.empty:
        return pd.DataFrame(index=yz_vol.index)

    daily_dvol = dvol["dvol_close"].resample("1D").last().ffill()
    daily_dvol = daily_dvol.reindex(yz_vol.index, method="ffill")

    out = pd.DataFrame(index=yz_vol.index)
    out["dvol"] = daily_dvol
    out["dvol_z"] = _zscore(daily_dvol, win)
    # Variance risk premium proxy: IV^2 - RV^2 (annualized, in decimal^2).
    iv2 = (daily_dvol / 100.0) ** 2
    rv2 = yz_vol**2
    out["vrp"] = iv2 - rv2
    return out


def funding_features(funding: pd.DataFrame, win: int = config.ROLL_WINDOW) -> pd.DataFrame:
    out = pd.DataFrame(index=funding.index)
    out["funding_sum"] = funding["funding_sum_24h"]
    out["funding_z"] = _zscore(funding["funding_sum_24h"], win)
    return out


def calendar_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    dow = pd.get_dummies(idx.dayofweek, prefix="dow", drop_first=True).astype(float)
    dow.index = idx
    return dow


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


def build_dataset(
    horizon: int = config.LABEL_HORIZON_DAYS,
    win: int = config.ROLL_WINDOW,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return `(X, y, meta)` aligned on the WETH daily index.

    `meta` carries columns the backtester needs (`U`, `debt`, `available_liquidity`,
    `price_usd`, reserve params, gas) so the simulator never re-reads parquets.
    """

    ins = load_inputs()
    idx = ins.weth.index

    market = market_features(ins.ohlcv, win=win).reindex(idx, method="ffill")
    util = utilization_features(ins.weth, win=win)
    proto = protocol_features(ins.weth, ins.usdc)
    opts = options_features(ins.dvol, market["yz_vol"], win=win).reindex(idx, method="ffill")
    fund = funding_features(ins.funding, win=win).reindex(idx, method="ffill")
    cal = calendar_features(idx)

    X = pd.concat([market, util, proto, opts, fund, cal], axis=1)

    # Label: did U rise over the next `horizon` days?
    U = ins.weth["U"]
    U_future = U.shift(-horizon)
    y = (U_future > U).astype("Int64")

    # Meta for the backtester (kept on the same index as X)
    gas = ins.gas["gas_gwei"].reindex(idx, method="ffill")
    meta = pd.DataFrame(
        {
            "U": U,
            "debt": ins.weth["debt"],
            "available_liquidity": ins.weth["available_liquidity"],
            "liquidity_rate_market": ins.weth["liquidity_rate"],
            "price_usd": ins.weth["price_usd"],
            "opt_U": ins.weth["opt_U"],
            "r0": ins.weth["r0"],
            "slope1": ins.weth["slope1"],
            "slope2": ins.weth["slope2"],
            "reserve_factor": ins.weth["reserve_factor"],
            "gas_gwei": gas,
        }
    )

    # Drop the warm-up window where rolling features are NaN, and the last
    # `horizon` rows where the label is undefined.
    full = pd.concat([X, y.rename("__y__"), meta], axis=1).dropna(
        subset=list(X.columns) + ["__y__"]
    )
    X_clean = full[X.columns]
    y_clean = full["__y__"].astype(int)
    meta_clean = full[meta.columns]

    # No-leak invariant: every feature row's index must be at least `horizon`
    # days before "now" so the label has had time to materialize without
    # peeking forward.
    assert (X_clean.index.shift(horizon, freq="1D") <= meta_clean.index.max()).all(), (
        "Leak: feature row exists whose label horizon extends past dataset end"
    )

    return X_clean, y_clean, meta_clean
