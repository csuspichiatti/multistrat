"""
strategies.pairs_us — Pairs trading sobre acciones US correlacionadas,
con overlay de VIX como filtro de volatilidad.

Lógica:
  1. Para cada par (A, B), corre test de cointegración Engle-Granger
     en ventana rolling. Si p-value > umbral, el par no opera.
  2. Estima beta hedge con OLS sobre la misma ventana.
  3. Calcula spread = B - alpha - beta*A
  4. Z-score del spread en ventana corta.
  5. Entry: |z| > z_entry → posición opuesta al spread (vender el caro,
     comprar el barato). Sizing dollar-neutral.
  6. Exit: |z| < z_exit → cierra posición.
  7. Stop: |z| > z_stop → cierra (cointegración rota o régimen cambió).
  8. VIX overlay: si VIX > vix_max, cierra todo y no abre nuevas (entorno
     de stress, las correlaciones colapsan).

Pares candidatos (mismo sector, históricamente cointegrados):
  - KO / PEP    — bebidas
  - V  / MA     — payment networks
  - MSFT / AAPL — mega tech
  - JPM / BAC   — bancos US
  - XOM / CVX   — energía majors
  - HD  / LOW   — home improvement
  - WMT / TGT   — retail mass-market
  - GS  / MS    — investment banks
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm


# Lista de pares candidatos por defecto
DEFAULT_PAIRS = [
    ("KO",   "PEP"),
    ("V",    "MA"),
    ("MSFT", "AAPL"),
    ("JPM",  "BAC"),
    ("XOM",  "CVX"),
    ("HD",   "LOW"),
    ("WMT",  "TGT"),
    ("GS",   "MS"),
]


@dataclass
class PairsConfig:
    # Ventanas
    coint_window:  int = 252 * 4    # ~4 días para datos 1h (66 horas/día * 4)
    beta_window:   int = 252 * 4
    zscore_window: int = 100        # ~15 días en 1h

    # Umbrales
    p_value_max: float = 0.05       # p-value máx para considerar cointegración
    z_entry:     float = 2.0
    z_exit:      float = 0.5
    z_stop:      float = 4.0

    # VIX overlay
    vix_max: float = 28.0           # cierra todo si VIX > 28
    vix_use: bool   = True

    # Risk
    gross_exposure_per_pair: float = 0.10   # 10% del equity por par
    max_open_pairs:          int   = 5
    fee_bps:     float = 5.0
    slippage_bps: float = 2.0


@dataclass
class PairState:
    """Estado de un par específico durante el backtest."""
    a: str
    b: str
    beta:    float = 0.0
    alpha:   float = 0.0
    p_value: float = 1.0
    spread_mean: float = 0.0
    spread_std:  float = 0.0
    z:       float = 0.0
    position: int = 0           # +1 long spread, -1 short spread, 0 flat
    entry_z:  float = 0.0
    last_qty_a: float = 0.0
    last_qty_b: float = 0.0


def estimate_beta(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    """OLS y = alpha + beta*x. Retorna (alpha, beta)."""
    X = sm.add_constant(x.values)
    model = sm.OLS(y.values, X, missing="drop").fit()
    return float(model.params[0]), float(model.params[1])


def cointegration_pvalue(a: pd.Series, b: pd.Series) -> float:
    """Engle-Granger p-value. Más bajo = más cointegrados."""
    if len(a) < 30 or len(b) < 30:
        return 1.0
    try:
        _t, p, _ = coint(a.dropna(), b.dropna())
        return float(p)
    except Exception:
        return 1.0


def update_pair_signal(state: PairState,
                       a: pd.Series, b: pd.Series,
                       cfg: PairsConfig) -> PairState:
    """
    Recibe series de A y B truncadas hasta el momento actual.
    Actualiza state.beta/alpha/spread/z.
    """
    if len(a) < cfg.coint_window or len(b) < cfg.coint_window:
        return state

    # Cointegración rolling
    a_w = a.iloc[-cfg.coint_window:]
    b_w = b.iloc[-cfg.coint_window:]
    state.p_value = cointegration_pvalue(a_w, b_w)

    # Beta hedge (recalculado cada vez sobre misma ventana)
    state.alpha, state.beta = estimate_beta(b_w, a_w)

    # Spread y z-score sobre ventana corta
    spread_full = b - state.alpha - state.beta * a
    spread_recent = spread_full.iloc[-cfg.zscore_window:]
    state.spread_mean = float(spread_recent.mean())
    state.spread_std  = float(spread_recent.std())
    if state.spread_std > 0:
        state.z = float((spread_full.iloc[-1] - state.spread_mean) / state.spread_std)
    else:
        state.z = 0.0
    return state


def decide_target_position(state: PairState, cfg: PairsConfig,
                            vix: Optional[float] = None) -> int:
    """
    Devuelve +1, -1 o 0 según el estado del par.
    +1 = long spread (long A, short B*beta) — esto se hace cuando z < -z_entry
         (B cayó vs A, esperamos que vuelva)
    -1 = short spread (short A, long B*beta) — cuando z > +z_entry
    """
    # VIX kill switch: si volatilidad alta, cerrar todo
    if cfg.vix_use and vix is not None and vix > cfg.vix_max:
        return 0

    # Si la cointegración se rompió, no operar
    if state.p_value > cfg.p_value_max:
        # Pero si tenemos posición abierta, cerrarla
        return 0 if state.position == 0 else 0

    # Si estamos flat: chequear entry
    if state.position == 0:
        if state.z > cfg.z_entry:
            return -1   # short spread
        elif state.z < -cfg.z_entry:
            return +1   # long spread
        else:
            return 0

    # Si estamos en posición:
    # Stop loss: spread se sigue ampliando contra nosotros
    if abs(state.z) > cfg.z_stop:
        return 0
    # Take profit: spread volvió cerca de la media
    if abs(state.z) < cfg.z_exit:
        return 0
    # Si no, mantener
    return state.position


def compute_qty(equity: float,
                price_a: float, price_b: float,
                beta: float,
                target_position: int,
                gross_exposure_per_pair: float) -> tuple[float, float]:
    """Dollar-neutral sizing: misma plata long que short."""
    if target_position == 0:
        return 0.0, 0.0
    half_notional = equity * gross_exposure_per_pair / 2
    qty_a = (half_notional / price_a) * target_position
    qty_b = -(half_notional / price_b) * beta * target_position
    return qty_a, qty_b
