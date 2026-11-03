"""
backtest.runner_pairs — Orquestador del backtest de pairs trading.

Recibe:
  - panel: DataFrame con columnas = tickers (incluye los pares + opcionalmente '^VIX')
  - pairs: lista de tuplas (a, b)
  - config: PairsConfig

Itera bar por bar, actualiza estado de cada par, decide posición, ejecuta
en el PaperTrader, marka equity al cierre.

Output: el PaperTrader poblado, listo para extraer métricas y graficar.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from core.paper_trader import PaperTrader
from strategies.pairs_us import (PairsConfig, PairState, update_pair_signal,
                                  decide_target_position, compute_qty)


def run_pairs_backtest(panel: pd.DataFrame,
                        pairs: list[tuple[str, str]],
                        cfg: PairsConfig,
                        initial_cash: float = 100_000.0,
                        warmup_bars: int = None,
                        vix_col: str = "^VIX",
                        verbose: bool = False) -> tuple[PaperTrader, dict]:
    """
    Backtest event-driven bar-por-bar.

    panel: DataFrame con DatetimeIndex y columnas con los closes de cada ticker.
           Debe contener todos los tickers de los pares + opcionalmente vix_col.
    """
    if warmup_bars is None:
        warmup_bars = max(cfg.coint_window, cfg.beta_window, cfg.zscore_window) + 5

    trader = PaperTrader(initial_cash=initial_cash,
                         fee_bps=cfg.fee_bps,
                         slippage_bps=cfg.slippage_bps)

    # Inicializar estados por par
    states: dict[tuple[str, str], PairState] = {
        (a, b): PairState(a=a, b=b) for (a, b) in pairs
    }

    # Filtrar pares cuyos tickers no están en el panel
    pairs_valid = [(a, b) for (a, b) in pairs
                   if a in panel.columns and b in panel.columns]
    if not pairs_valid:
        return trader, {"error": "Ningún par tiene datos en el panel."}

    has_vix = vix_col in panel.columns

    # Iterar
    for i in range(warmup_bars, len(panel)):
        ts = panel.index[i]
        prices = {col: float(panel[col].iloc[i]) for col in panel.columns
                  if pd.notna(panel[col].iloc[i])}
        vix = prices.get(vix_col) if has_vix else None

        # Contar pares abiertos para enforce max_open_pairs
        n_open = sum(1 for s in states.values() if s.position != 0)

        for (a, b) in pairs_valid:
            state = states[(a, b)]
            a_ser = panel[a].iloc[:i+1].dropna()
            b_ser = panel[b].iloc[:i+1].dropna()
            if len(a_ser) < cfg.coint_window or len(b_ser) < cfg.coint_window:
                continue

            # Actualizar señal
            state = update_pair_signal(state, a_ser, b_ser, cfg)

            # Decidir target
            target = decide_target_position(state, cfg, vix=vix)

            # Si vamos a abrir nuevo y ya estamos en el cap, skip
            if state.position == 0 and target != 0 and n_open >= cfg.max_open_pairs:
                continue

            if target == state.position:
                states[(a, b)] = state
                continue

            # Ejecutar el cambio
            px_a = prices.get(a); px_b = prices.get(b)
            if px_a is None or px_b is None:
                continue

            qty_a, qty_b = compute_qty(
                equity=trader.cash + sum(p.qty * prices.get(p.symbol, 0)
                                          for p in trader.positions.values()),
                price_a=px_a, price_b=px_b,
                beta=state.beta,
                target_position=target,
                gross_exposure_per_pair=cfg.gross_exposure_per_pair,
            )
            note = f"pair={a}/{b} z={state.z:.2f} target={target}"
            trader.submit_target(ts, a, qty_a, px_a, note=note)
            trader.submit_target(ts, b, qty_b, px_b, note=note)

            if state.position == 0 and target != 0:
                state.entry_z = state.z
                n_open += 1
            elif target == 0 and state.position != 0:
                n_open -= 1

            state.position = target
            state.last_qty_a = qty_a
            state.last_qty_b = qty_b
            states[(a, b)] = state

            if verbose and target != 0:
                print(f"[{ts}] {a}/{b} z={state.z:+.2f} pval={state.p_value:.3f} "
                      f"target={target:+d} qty_a={qty_a:.2f} qty_b={qty_b:.2f}")

        # Marcar equity al close de la barra
        trader.mark_to_market(ts, prices)

    # Cerrar posiciones al final del backtest
    last_ts = panel.index[-1]
    last_prices = {col: float(panel[col].iloc[-1]) for col in panel.columns
                   if pd.notna(panel[col].iloc[-1])}
    trader.close_all(last_ts, last_prices)
    trader.mark_to_market(last_ts, last_prices)

    # Calcular métricas (asume datos 1h con 6.5h * 252 días = 1638)
    inferred_periods = 252 * 6.5 if pd.infer_freq(panel.index[:50]) in ("h", "1h", "H") else 252
    metrics = trader.metrics(periods_per_year=inferred_periods)

    # Estado final de los pares
    states_summary = {f"{a}/{b}": {
        "p_value": s.p_value, "beta": s.beta,
        "spread_std": s.spread_std, "z_final": s.z,
        "position_final": s.position,
    } for (a, b), s in states.items()}

    metrics["pairs_states_final"] = states_summary
    return trader, metrics


# ── Demo end-to-end con datos sintéticos ────────────────────────────────
def demo_synthetic():
    """Corre un backtest sobre un par cointegrado sintético + un par espurio.
    El motor debe ganar en el cointegrado y NO operar en el espurio."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core.data import synth_cointegrated_pair, synth_random_walk

    # Par cointegrado
    pair = synth_cointegrated_pair(n=2000, beta=1.2, alpha=10,
                                    noise_a=0.5, noise_b=0.3,
                                    mean_rev_speed=0.04, seed=42)
    pair.columns = ["AAA", "BBB"]

    # Par espurio (dos random walks independientes, no cointegrados)
    rw1 = synth_random_walk(n=2000, seed=1).rename("CCC")
    rw2 = synth_random_walk(n=2000, seed=2).rename("DDD")

    # VIX sintético: uniforme bajo 20 con algunos picos
    rng = np.random.default_rng(7)
    vix_vals = 15 + rng.normal(0, 3, 2000).clip(-5, 20)
    vix_vals[800:830] = 35   # spike de stress en medio
    vix = pd.Series(vix_vals, index=pair.index, name="^VIX")

    panel = pd.concat([pair, rw1, rw2, vix], axis=1)

    cfg = PairsConfig(
        coint_window=300,
        beta_window=300,
        zscore_window=100,
        z_entry=1.5,
        z_exit=0.3,
        z_stop=4.0,
        vix_max=28,
        gross_exposure_per_pair=0.10,
        max_open_pairs=3,
    )

    trader, metrics = run_pairs_backtest(
        panel=panel,
        pairs=[("AAA", "BBB"), ("CCC", "DDD")],
        cfg=cfg,
        initial_cash=100_000,
        warmup_bars=320,
        verbose=False,
    )
    return trader, metrics


if __name__ == "__main__":
    trader, metrics = demo_synthetic()
    print("=" * 60)
    print("BACKTEST DEMO — Pairs Trading sintético")
    print("=" * 60)
    for k, v in metrics.items():
        if k == "pairs_states_final":
            print(f"\n{k}:")
            for pk, pv in v.items():
                print(f"  {pk}: {pv}")
        else:
            if isinstance(v, float):
                print(f"  {k:25s}: {v:>10.3f}")
            else:
                print(f"  {k:25s}: {v}")
