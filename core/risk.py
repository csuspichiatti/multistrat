"""
core.risk — Sizing y caps de riesgo a nivel cuenta.

Métodos comunes de sizing usados por las 3 estrategias:
  - vol-targeted sizing (Kelly fraccional, vol-parity)
  - dollar-neutral sizing para pairs
  - cap por estrategia y cap agregado de exposición
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def vol_target_size(equity: float,
                    target_vol_annual: float,
                    realized_vol_annual: float,
                    cap_pct: float = 0.20) -> float:
    """
    Devuelve el % de equity a asignar a una posición para que su contribución
    a la volatilidad del portafolio sea target_vol_annual.

    Si la vol realizada del activo es 60% y querés target 20%, asignás 33%.
    Cap por defecto 20% del equity por posición.
    """
    if realized_vol_annual <= 0:
        return 0.0
    raw = target_vol_annual / realized_vol_annual
    return float(min(raw, cap_pct))


def dollar_neutral_qty(equity: float,
                       price_a: float,
                       price_b: float,
                       beta: float,
                       gross_exposure_pct: float = 0.10) -> tuple[float, float]:
    """
    Para un par (A, B) con relación de cointegración B = alpha + beta*A:
    si la posición es +1 en spread (long A, short B*beta), las cantidades en
    notional son iguales y opuestas en sentido del beta.

    Retorna (qty_a, qty_b) signed, con la convención long=spread.
    gross_exposure_pct: % de equity comprometido en el gross (long+short).
    """
    half_notional = equity * gross_exposure_pct / 2
    qty_a =  half_notional / price_a
    qty_b = -half_notional * beta / price_b
    return qty_a, qty_b


def realized_vol_annual(returns: pd.Series, periods_per_year: float) -> float:
    """Vol anualizada a partir de retornos de cualquier frecuencia."""
    if len(returns) < 5:
        return 0.0
    return float(returns.std() * np.sqrt(periods_per_year))


def daily_loss_cap_breached(equity: pd.Series, cap_pct: float = 0.05) -> bool:
    """True si el equity cayó más de cap_pct desde el último máximo del día."""
    if len(equity) < 2:
        return False
    today = equity[equity.index.date == equity.index[-1].date()]
    if len(today) < 2:
        return False
    high_today = today.max()
    current = today.iloc[-1]
    drawdown_today = (current - high_today) / high_today
    return bool(drawdown_today < -cap_pct)
