"""
strategies.pairs_us — Pairs trading sobre acciones US correlacionadas,
con overlay de VIX como filtro de volatilidad.

Lógica:
  1. Para cada par (A, B), corre test de cointegración Engle-Granger
     en ventana rolling. Si p-value > umbral, el par no opera.
  2. Estima beta hedge con OLS sobre la misma ventana.
  3. Calcula spread = B - alpha - beta*A
  4. Z-score del spread en ventana corta.
  5. Entry: |z| > z_entry y p-value <= p_value_max → posición opuesta al
     spread (vender el caro, comprar el barato).
  6. Al entrar se CONGELAN alpha, beta, media y desviación del spread. Todas
     las salidas se miden contra esa foto, así el stop no se diluye cuando la
     desviación móvil crece junto con la pérdida, y el z refleja la posición
     que realmente se tiene (misma beta con la que se dimensionó).
  7. Salidas, en orden de prioridad:
       a. Stop duro en USD: pérdida del par >= max_loss_pct_per_pair del equity.
       b. Stop en z congelado: |z_pos| > z_stop.
       c. Stop por tiempo: barras en posición > time_stop_halflives × vida
          media del spread (con piso y techo).
       d. Ruptura de cointegración con histéresis: p-value > p_value_exit
          durante p_exit_bars barras seguidas.
       e. Take profit: el z congelado volvió a la zona |z_pos| < z_exit
          (o cruzó al otro lado de la media).
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
    # Ventanas (en barras). Con datos 1h de acciones US hay ~7 barras por día.
    coint_window:  int = 1000       # ~140 días de mercado en 1h
    beta_window:   int = 1000
    zscore_window: int = 100        # ~14 días de mercado en 1h

    # Umbrales de señal
    p_value_max:  float = 0.05      # p-value máx para ABRIR
    p_value_exit: float = 0.20      # p-value que se considera ruptura real
    p_exit_bars:  int   = 10        # barras seguidas sobre p_value_exit para cerrar
    p_entry_bars: int   = 20        # barras seguidas con p <= p_value_max para habilitar entradas
    half_life_min: float = 2.0      # vida media del spread aceptable (barras)
    half_life_max: float = 100.0    # más lento que esto no es operable con estas ventanas

    # Rendimiento: cointegración, beta y vida media se re-estiman cada
    # `recalc_every` barras (7 ≈ una vez por día de mercado en 1h). El z-score
    # y los stops se evalúan en TODAS las barras con la última estimación.
    recalc_every: int = 7
    coint_maxlag: Optional[int] = 1   # lags del ADF; None = autolag AIC (≈50x más lento)
    z_entry:      float = 2.0
    z_exit:       float = 0.5
    z_stop:       float = 4.0       # medido con estadísticas congeladas a la entrada

    # Stops de riesgo
    max_loss_pct_per_pair: float = 0.01   # stop duro: 1% del equity por par
    time_stop_halflives:   float = 3.0    # salir tras 3 vidas medias sin revertir
    time_stop_min_bars:    int   = 20
    time_stop_max_bars:    int   = 500
    daily_loss_cap_pct:    float = 0.03   # kill switch diario sobre el equity

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
    # Estimación vigente (se actualiza cada barra; se usa para ENTRAR)
    beta:    float = 0.0
    alpha:   float = 0.0
    p_value: float = 1.0
    spread_mean: float = 0.0
    spread_std:  float = 0.0
    half_life:   float = float("nan")
    z:       float = 0.0
    # Posición
    position: int = 0           # +1 long spread, -1 short spread, 0 flat
    entry_z:  float = 0.0
    last_qty_a: float = 0.0
    last_qty_b: float = 0.0
    # Foto congelada al entrar (se usa para SALIR)
    entry_alpha: float = 0.0
    entry_beta:  float = 0.0
    entry_mean:  float = 0.0
    entry_std:   float = 0.0
    entry_equity: float = 0.0
    entry_cost:  float = 0.0    # costo signed de abrir las dos patas (USD)
    max_bars:    int   = 0      # stop por tiempo calculado al entrar
    bars_held:   int   = 0
    z_pos:       float = 0.0    # z del spread de la posición, con foto congelada
    p_bad_bars:  int   = 0      # barras seguidas con p-value > p_value_exit
    p_good_bars: int   = 0      # barras seguidas con p-value <= p_value_max
    exit_reason: str   = ""


def estimate_beta(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    """OLS y = alpha + beta*x. Retorna (alpha, beta)."""
    X = sm.add_constant(np.asarray(x, dtype=float))
    model = sm.OLS(np.asarray(y, dtype=float), X, missing="drop").fit()
    return float(model.params[0]), float(model.params[1])


def cointegration_pvalue(y, x, maxlag: Optional[int] = None) -> float:
    """
    Engle-Granger p-value de y sobre x (misma dirección que el OLS de beta).
    Más bajo = más cointegrados. maxlag=None usa autolag AIC (lento);
    un entero fija los lags del ADF (rápido).
    """
    y = np.asarray(y, dtype=float); x = np.asarray(x, dtype=float)
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    if len(y) < 30:
        return 1.0
    try:
        if maxlag is None:
            _t, p, _ = coint(y, x)
        else:
            _t, p, _ = coint(y, x, maxlag=maxlag, autolag=None)
        return float(p)
    except Exception:
        return 1.0


def spread_half_life(spread: pd.Series) -> float:
    """
    Vida media de reversión del spread (en barras) vía AR(1):
    Δs_t = λ·s_{t-1} + ε  →  half_life = -ln(2)/λ. NaN si no revierte.
    """
    s = np.asarray(spread, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) < 30:
        return float("nan")
    lag = s[:-1]
    delta = np.diff(s)
    lam = np.polyfit(lag - lag.mean(), delta, 1)[0]
    if lam >= 0:
        return float("nan")
    return float(-np.log(2) / lam)


def update_pair_signal(state: PairState,
                       a, b,
                       cfg: PairsConfig,
                       recompute: bool = True) -> PairState:
    """
    Recibe A y B (Series o arrays) truncadas hasta el momento actual, sin
    datos futuros. Basta con pasar las últimas max(coint_window, beta_window)
    barras.

    recompute=True: re-estima p-value, alpha, beta y vida media (caro).
    recompute=False: reutiliza la última estimación y solo actualiza z,
    z_pos y contadores (barato). El runner alterna según cfg.recalc_every.
    """
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if len(a) < cfg.coint_window or len(b) < cfg.coint_window:
        return state

    if recompute:
        a_w = a[-cfg.coint_window:]
        b_w = b[-cfg.coint_window:]
        state.p_value = cointegration_pvalue(b_w, a_w, maxlag=cfg.coint_maxlag)
        state.alpha, state.beta = estimate_beta(b_w, a_w)
        hl_n = cfg.beta_window
        state.half_life = spread_half_life(b[-hl_n:] - state.alpha - state.beta * a[-hl_n:])

    zw = cfg.zscore_window
    spread_recent = b[-zw:] - state.alpha - state.beta * a[-zw:]
    state.spread_mean = float(np.nanmean(spread_recent))
    state.spread_std  = float(np.nanstd(spread_recent, ddof=1))
    state.z = (float((spread_recent[-1] - state.spread_mean) / state.spread_std)
               if state.spread_std > 0 else 0.0)
    state.p_good_bars = state.p_good_bars + 1 if state.p_value <= cfg.p_value_max else 0

    if state.position != 0:
        s_pos = float(b[-1] - state.entry_alpha - state.entry_beta * a[-1])
        state.z_pos = ((s_pos - state.entry_mean) / state.entry_std
                       if state.entry_std > 0 else 0.0)
        state.p_bad_bars = state.p_bad_bars + 1 if state.p_value > cfg.p_value_exit else 0
    return state


def freeze_entry(state: PairState, cfg: PairsConfig, equity: float) -> PairState:
    """Congela la foto del spread al abrir. Llamar justo al ejecutar la entrada."""
    state.entry_alpha = state.alpha
    state.entry_beta  = state.beta
    state.entry_mean  = state.spread_mean
    state.entry_std   = state.spread_std
    state.entry_z     = state.z
    state.z_pos       = state.z
    state.entry_equity = equity
    state.bars_held   = 0
    state.p_bad_bars  = 0
    hl = state.half_life
    if np.isfinite(hl) and hl > 0:
        mb = int(round(cfg.time_stop_halflives * hl))
    else:
        mb = cfg.time_stop_max_bars
    state.max_bars = int(min(max(mb, cfg.time_stop_min_bars), cfg.time_stop_max_bars))
    return state


def decide_target_position(state: PairState, cfg: PairsConfig,
                            vix: Optional[float] = None,
                            pair_pnl: float = 0.0) -> int:
    """
    Devuelve +1, -1 o 0 y deja el motivo en state.exit_reason.
    +1 = long spread (long B, short beta·A) — cuando z < -z_entry
    -1 = short spread (short B, long beta·A) — cuando z > +z_entry
    pair_pnl: PnL no realizado del par en USD (incluye costos de entrada).
    """
    state.exit_reason = ""

    # VIX kill switch: si volatilidad alta, cerrar todo y no abrir
    if cfg.vix_use and vix is not None and vix > cfg.vix_max:
        if state.position != 0:
            state.exit_reason = "vix"
        return 0

    # Flat: solo se abre con cointegración ESTABLE y reversión a velocidad operable.
    # Un solo p-value < 0.05 en ventana corta es frecuente entre random walks
    # independientes (falso positivo); exigir persistencia y vida media razonable
    # filtra buena parte de los pares espurios.
    if state.position == 0:
        if state.p_good_bars < cfg.p_entry_bars:
            return 0
        hl = state.half_life
        if not (np.isfinite(hl) and cfg.half_life_min <= hl <= cfg.half_life_max):
            return 0
        if state.z > cfg.z_entry:
            return -1
        if state.z < -cfg.z_entry:
            return +1
        return 0

    # En posición: salidas por prioridad
    if state.entry_equity > 0 and pair_pnl <= -cfg.max_loss_pct_per_pair * state.entry_equity:
        state.exit_reason = "stop_usd"
        return 0
    if abs(state.z_pos) > cfg.z_stop:
        state.exit_reason = "stop_z"
        return 0
    if state.max_bars and state.bars_held >= state.max_bars:
        state.exit_reason = "stop_tiempo"
        return 0
    if state.p_bad_bars >= cfg.p_exit_bars:
        state.exit_reason = "coint_rota"
        return 0
    # Take profit: el spread volvió a la media (o la cruzó a nuestro favor).
    # position=-1 entró con z>0 → gana cuando z_pos baja.
    if state.position == -1 and state.z_pos < cfg.z_exit:
        state.exit_reason = "tp"
        return 0
    if state.position == +1 and state.z_pos > -cfg.z_exit:
        state.exit_reason = "tp"
        return 0
    return state.position


def compute_qty(equity: float,
                price_a: float, price_b: float,
                beta: float,
                target_position: int,
                gross_exposure_per_pair: float) -> tuple[float, float]:
    """
    Cantidades para tener `target_position` unidades del spread S = B - beta*A.

    +1 (long spread)  = long B, short beta·A  → gana si S sube (z venía < 0).
    -1 (short spread) = short B, long beta·A  → gana si S baja (z venía > 0).

    Nota: la versión anterior tenía los signos invertidos (con z>0 compraba B,
    la pata cara), por eso perdía incluso en un par cointegrado sintético.
    """
    if target_position == 0:
        return 0.0, 0.0
    half_notional = equity * gross_exposure_per_pair / 2
    qty_b = (half_notional / price_b) * target_position
    qty_a = -beta * qty_b
    return qty_a, qty_b
