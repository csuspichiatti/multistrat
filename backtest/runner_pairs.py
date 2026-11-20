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

import time

import pandas as pd
import numpy as np

from core.paper_trader import PaperTrader
from strategies.pairs_us import (PairsConfig, PairState, update_pair_signal,
                                  decide_target_position, compute_qty,
                                  freeze_entry)


def infer_periods_per_year(index: pd.DatetimeIndex) -> float:
    """
    Barras por año = mediana de barras por día × 252.
    No usa pd.infer_freq: con datos 1h de acciones (huecos nocturnos y fines
    de semana) infer_freq devuelve None y el Sharpe quedaba mal anualizado.
    """
    if len(index) < 2:
        return 252.0
    per_day = pd.Series(1, index=index).groupby(index.date).sum()
    return float(per_day.median() * 252)


def run_pairs_backtest(panel: pd.DataFrame,
                        pairs: list[tuple[str, str]],
                        cfg: PairsConfig,
                        initial_cash: float = 100_000.0,
                        warmup_bars: int = None,
                        vix_col: str = "^VIX",
                        open_panel: pd.DataFrame = None,
                        periods_per_year: float = None,
                        verbose: bool = False,
                        progress: bool = True) -> tuple[PaperTrader, dict]:
    """
    Backtest event-driven bar-por-bar, sin look-ahead.

    Secuencia por barra i:
      1. Se ejecutan las órdenes decididas en la barra i-1, al OPEN de i
         (open_panel) o, si no hay opens, al CLOSE de i (más conservador).
      2. Se marca el equity al close de i y se chequea el kill switch diario.
      3. Con datos hasta el close de i se actualizan señales y se deciden
         órdenes para la barra i+1.

    panel:      closes, DatetimeIndex, columnas = tickers (+ vix_col opcional).
    open_panel: opens con el mismo índice/columnas (opcional, recomendado).
    """
    if warmup_bars is None:
        warmup_bars = max(cfg.coint_window, cfg.beta_window, cfg.zscore_window) + 5

    trader = PaperTrader(initial_cash=initial_cash,
                         fee_bps=cfg.fee_bps,
                         slippage_bps=cfg.slippage_bps)

    pairs_valid = [(a, b) for (a, b) in pairs
                   if a in panel.columns and b in panel.columns]
    if not pairs_valid:
        return trader, {"error": "Ningún par tiene datos en el panel."}
    states: dict[tuple[str, str], PairState] = {
        (a, b): PairState(a=a, b=b) for (a, b) in pairs_valid}
    pair_cash: dict[tuple[str, str], float] = {k: 0.0 for k in states}
    round_trips: list[dict] = []
    pending: dict[tuple[str, str], tuple[int, str]] = {}   # par -> (target, motivo)

    has_vix = vix_col in panel.columns
    blocked_day = None          # fecha bloqueada por kill switch diario
    kill_days: list = []
    day_key, day_high = None, None

    # Pre-cálculo para no copiar la historia completa en cada barra (era O(n²)):
    # por cada par, filas donde ambos precios existen y un puntero por barra.
    hist_n = max(cfg.coint_window, cfg.beta_window, cfg.zscore_window)
    pair_data = {}
    for key in states:
        a, b = key
        both = panel[[a, b]].dropna()
        pos_in_pair = np.searchsorted(both.index.values, panel.index.values, side="right")
        pair_data[key] = (both[a].to_numpy(float), both[b].to_numpy(float), pos_in_pair)
    last_recalc = {key: -10**9 for key in states}

    n_bars = len(panel) - warmup_bars
    t0 = time.time()
    next_report = 0.0

    def row_prices(df, i):
        return {c: float(df[c].iloc[i]) for c in df.columns if pd.notna(df[c].iloc[i])}

    def equity_now(prices):
        return trader.cash + sum(p.qty * prices.get(p.symbol, 0.0)
                                 for p in trader.positions.values())

    def execute(key, target, reason, ts, exec_px, mark_px):
        a, b = key
        st = states[key]
        px_a, px_b = exec_px.get(a), exec_px.get(b)
        if px_a is None or px_b is None:
            return False
        if target == 0:
            qa, qb = 0.0, 0.0
        else:
            qa, qb = compute_qty(equity_now(mark_px), px_a, px_b, st.beta,
                                 target, cfg.gross_exposure_per_pair)
        note = f"pair={a}/{b} z={st.z:.2f} target={target} motivo={reason}"
        flow = 0.0
        for sym, q, px in ((a, qa, px_a), (b, qb, px_b)):
            prev = trader.positions[sym].qty if sym in trader.positions else 0.0
            t = trader.submit_target(ts, sym, q, px, note=note)
            if t is not None:
                delta = q - prev
                flow += -delta * t.price - t.fee_usd
        if st.position == 0 and target != 0:
            pair_cash[key] = flow
            freeze_entry(st, cfg, equity_now(mark_px))
            st.entry_cost = flow
        else:
            pair_cash[key] += flow
            if target == 0:
                round_trips.append({"pair": f"{a}/{b}", "exit_ts": ts,
                                    "pnl_usd": pair_cash[key],
                                    "bars_held": st.bars_held,
                                    "motivo": reason})
                pair_cash[key] = 0.0
        st.position = target
        st.last_qty_a, st.last_qty_b = qa, qb
        if verbose:
            print(f"[{ts}] {a}/{b} {reason or 'entrada'} target={target:+d} z={st.z:+.2f} "
                  f"z_pos={st.z_pos:+.2f} p={st.p_value:.3f}")
        return True

    for i in range(warmup_bars, len(panel)):
        ts = panel.index[i]
        prices = row_prices(panel, i)
        exec_prices = row_prices(open_panel, i) if open_panel is not None else prices

        # 1. Ejecutar lo decidido en la barra anterior
        for key, (target, reason) in list(pending.items()):
            if execute(key, target, reason, ts, exec_prices, prices):
                pending.pop(key)

        # 2. Mark-to-market y kill switch diario (máximo intradía en O(1))
        equity = trader.mark_to_market(ts, prices)
        today = ts.date()
        if today != day_key:
            day_key, day_high = today, equity
        day_high = max(day_high, equity)
        if blocked_day is not None and blocked_day != today:
            blocked_day = None
        if blocked_day is None and day_high > 0 and \
                (equity - day_high) / day_high < -cfg.daily_loss_cap_pct:
            blocked_day = today
            kill_days.append(str(today))
            for key, st in states.items():
                if st.position != 0:
                    pending[key] = (0, "kill_diario")

        vix = prices.get(vix_col) if has_vix else None
        n_open = sum(1 for k, s in states.items()
                     if (s.position != 0) or (k in pending and pending[k][0] != 0))

        # 3. Señales y decisiones para la próxima barra
        for key, state in states.items():
            if key in pending:
                continue
            a, b = key
            a_arr, b_arr, ptr = pair_data[key]
            k = int(ptr[i])                      # filas válidas hasta la barra i inclusive
            if k < cfg.coint_window:
                continue
            if state.position != 0:
                state.bars_held += 1
            recompute = (k - last_recalc[key]) >= cfg.recalc_every
            if recompute:
                last_recalc[key] = k
            lo = max(0, k - hist_n)
            update_pair_signal(state, a_arr[lo:k], b_arr[lo:k], cfg, recompute=recompute)

            pnl = 0.0
            if state.position != 0 and a in prices and b in prices:
                pnl = pair_cash[key] + state.last_qty_a * prices[a] + state.last_qty_b * prices[b]
            target = decide_target_position(state, cfg, vix=vix, pair_pnl=pnl)

            if target == state.position:
                continue
            if state.position == 0 and target != 0:
                if blocked_day is not None or n_open >= cfg.max_open_pairs:
                    continue
                n_open += 1
            pending[key] = (target, state.exit_reason)

        if progress and n_bars > 0:
            frac = (i - warmup_bars + 1) / n_bars
            if frac >= next_report or i == len(panel) - 1:
                el = time.time() - t0
                eta = el / frac * (1 - frac) if frac > 0 else 0
                n_pos = sum(1 for s_ in states.values() if s_.position != 0)
                print(f"  [{frac*100:5.1f}%] barra {i+1}/{len(panel)} {ts:%Y-%m-%d} "
                      f"equity=${equity:,.0f} abiertos={n_pos} "
                      f"trades={len(round_trips)} | {el:,.0f}s, faltan ~{eta:,.0f}s",
                      flush=True)
                next_report += 0.05

    # Cerrar posiciones al final del backtest
    last_ts = panel.index[-1]
    last_prices = row_prices(panel, len(panel) - 1)
    for key, st in states.items():
        if st.position != 0:
            execute(key, 0, "fin_backtest", last_ts, last_prices, last_prices)
    trader.mark_to_market(last_ts, last_prices)

    ppy = periods_per_year or infer_periods_per_year(panel.index)
    metrics = trader.metrics(periods_per_year=ppy)
    metrics["periods_per_year"] = ppy

    rt = pd.DataFrame(round_trips)
    if not rt.empty:
        metrics["n_round_trips"] = int(len(rt))
        metrics["hit_rate_pairs_pct"] = float((rt["pnl_usd"] > 0).mean() * 100)
        metrics["pnl_by_exit"] = rt.groupby("motivo")["pnl_usd"].agg(["count", "sum"]).round(2).to_dict("index")
        metrics["pnl_by_pair"] = rt.groupby("pair")["pnl_usd"].agg(["count", "sum"]).round(2).to_dict("index")
    else:
        metrics["n_round_trips"] = 0
    metrics["kill_switch_days"] = kill_days
    metrics["round_trips"] = rt

    metrics["pairs_states_final"] = {f"{a}/{b}": {
        "p_value": s.p_value, "beta": s.beta,
        "half_life": s.half_life, "z_final": s.z,
        "position_final": s.position,
    } for (a, b), s in states.items()}
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
        if k == "round_trips":
            continue
        if k == "pairs_states_final":
            print(f"\n{k}:")
            for pk, pv in v.items():
                print(f"  {pk}: {pv}")
        else:
            if isinstance(v, float):
                print(f"  {k:25s}: {v:>10.3f}")
            else:
                print(f"  {k:25s}: {v}")
