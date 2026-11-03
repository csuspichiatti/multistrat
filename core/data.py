"""
core.data — Abstracción de fuentes de datos para multistrat.

Provee fetch unificado de:
  - US stocks / ETFs / índices (yfinance)
  - FX majors (yfinance, ticker tipo USDCLP=X)
  - Commodities futures (yfinance, ticker tipo HG=F)
  - Macro CLP (FRED si tenés API key, fallback yfinance)

Cache local en parquet por ticker+intervalo para no repetir requests.

Uso típico:
    from core.data import fetch_ohlcv, fetch_close
    df = fetch_ohlcv("AAPL", interval="1h", start="2024-09-01")  # OHLCV
    s  = fetch_close("AAPL", interval="1d", start="2018-01-01")  # solo close
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


# Cache root
CACHE_DIR = Path(os.getenv("MULTISTRAT_CACHE",
                           str(Path(__file__).resolve().parent.parent / "data_cache")))
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# yfinance es opcional para permitir tests sin red
try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    yf = None
    _HAS_YF = False


def _cache_path(ticker: str, interval: str) -> Path:
    safe = ticker.replace("/", "_").replace("=", "_").replace("^", "_")
    return CACHE_DIR / f"{safe}_{interval}.parquet"


def _load_cache(ticker: str, interval: str) -> Optional[pd.DataFrame]:
    p = _cache_path(ticker, interval)
    if p.is_file():
        try:
            return pd.read_parquet(p)
        except Exception:
            return None
    return None


def _save_cache(ticker: str, interval: str, df: pd.DataFrame):
    if df is None or df.empty:
        return
    p = _cache_path(ticker, interval)
    try:
        df.to_parquet(p, index=True)
    except Exception as e:
        print(f"[data] no se pudo cachear {ticker}/{interval}: {e}")


def fetch_ohlcv(ticker: str,
                interval: str = "1h",
                start: Optional[str] = None,
                end: Optional[str] = None,
                use_cache: bool = True,
                refresh: bool = False) -> pd.DataFrame:
    """
    Devuelve DataFrame con index DatetimeIndex (UTC) y columnas
    open, high, low, close, volume.

    interval: '1m','5m','15m','30m','1h','1d','1wk','1mo'
    Limitaciones de yfinance:
      - 1m: solo últimos 7 días
      - 5m,15m,30m: solo últimos 60 días
      - 1h: solo últimos 730 días
      - 1d en adelante: histórico completo
    """
    if use_cache and not refresh:
        cached = _load_cache(ticker, interval)
        if cached is not None and len(cached) > 0:
            # filtrar por rango si se pide
            df = cached.copy()
            if start: df = df[df.index >= pd.to_datetime(start, utc=True)]
            if end:   df = df[df.index <= pd.to_datetime(end,   utc=True)]
            if len(df) > 0:
                return df

    if not _HAS_YF:
        raise RuntimeError("yfinance no está instalado. pip install yfinance")

    # yfinance recomienda usar period='2y' para 1h en lugar de start/end
    kwargs = {"interval": interval, "auto_adjust": False, "progress": False}
    if interval == "1h" and not start:
        kwargs["period"] = "730d"
    elif start:
        kwargs["start"] = start
        if end: kwargs["end"] = end
    else:
        kwargs["period"] = "max"

    df = yf.download(ticker, **kwargs)
    if df is None or df.empty:
        return pd.DataFrame()

    # Aplanar MultiIndex si yfinance lo devuelve
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    keep = ["open", "high", "low", "close", "volume"]
    df = df[[c for c in keep if c in df.columns]]

    # Index a UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df = df.sort_index()

    if use_cache:
        _save_cache(ticker, interval, df)
    return df


def fetch_close(ticker: str, interval: str = "1d",
                start: Optional[str] = None,
                end: Optional[str] = None) -> pd.Series:
    """Atajo para series de close."""
    df = fetch_ohlcv(ticker, interval=interval, start=start, end=end)
    if df.empty:
        return pd.Series(dtype=float, name=ticker)
    return df["close"].rename(ticker)


def fetch_panel(tickers: list[str],
                interval: str = "1h",
                start: Optional[str] = None,
                end: Optional[str] = None) -> pd.DataFrame:
    """
    Devuelve DataFrame con columnas = tickers, valores = close,
    índice = timestamps comunes.
    """
    series = {}
    for t in tickers:
        s = fetch_close(t, interval=interval, start=start, end=end)
        if not s.empty:
            series[t] = s
    if not series:
        return pd.DataFrame()
    panel = pd.concat(series, axis=1)
    return panel.dropna(how="all")


# ──────────────────────────────────────────────────────────────────
# Datos sintéticos para tests / demo sin red
# ──────────────────────────────────────────────────────────────────
def synth_cointegrated_pair(n: int = 1000,
                             beta: float = 1.0,
                             alpha: float = 0.0,
                             noise_a: float = 0.5,
                             noise_b: float = 0.4,
                             mean_rev_speed: float = 0.05,
                             seed: int = 42) -> pd.DataFrame:
    """
    Genera dos series cointegradas: A es random walk; B = alpha + beta*A + spread,
    donde spread es un proceso AR(1) mean-reverting.
    Útil para validar que el motor de pairs trading detecta cointegración.
    """
    rng = np.random.default_rng(seed)
    A = np.cumsum(rng.normal(0, noise_a, n)) + 100
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = (1 - mean_rev_speed) * spread[i-1] + rng.normal(0, noise_b)
    B = alpha + beta * A + spread
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"A": A, "B": B}, index=idx)


def synth_random_walk(n: int = 1000, drift: float = 0.0,
                       sigma: float = 0.5, start: float = 100,
                       seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, sigma, n)
    px = start * np.exp(np.cumsum(rets) / 100)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.Series(px, index=idx, name="rw")
