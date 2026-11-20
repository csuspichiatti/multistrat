"""
Corre el backtest de pairs trading con datos REALES de yfinance.

Uso:
    cd multistrat
    pip install -r requirements.txt
    python -m backtest.run_real_pairs

Por defecto: 8 pares estándar, 1h, últimos 730 días.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from core.data import fetch_panel
from strategies.pairs_us import DEFAULT_PAIRS, PairsConfig
from backtest.runner_pairs import run_pairs_backtest


def main(interval: str = "1h",
         start: str = None,
         pairs: list = None,
         initial_cash: float = 100_000):
    pairs = pairs or DEFAULT_PAIRS
    tickers = sorted({t for p in pairs for t in p} | {"^VIX"})

    print(f"Descargando {len(tickers)} tickers, interval={interval}...")
    panel = fetch_panel(tickers, interval=interval, start=start)
    # Opens para ejecutar la orden en la apertura de la barra siguiente
    open_panel = fetch_panel(tickers, interval=interval, start=start, field="open")
    open_panel = open_panel.reindex(index=panel.index, columns=panel.columns)
    print(f"Panel: {panel.shape[0]} barras × {panel.shape[1]} cols")
    print(f"Rango: {panel.index[0]} → {panel.index[-1]}")

    cfg = PairsConfig(
        coint_window=1000, beta_window=1000, zscore_window=100,
        z_entry=2.0, z_exit=0.5, z_stop=4.0,
        p_value_max=0.05, p_value_exit=0.20, p_exit_bars=10, p_entry_bars=20,
        half_life_min=2, half_life_max=100,
        max_loss_pct_per_pair=0.01, time_stop_halflives=3.0,
        daily_loss_cap_pct=0.03,
        vix_max=28, vix_use=True,
        gross_exposure_per_pair=0.10, max_open_pairs=5,
        fee_bps=5.0, slippage_bps=2.0,
    )

    trader, metrics = run_pairs_backtest(
        panel=panel, pairs=pairs, cfg=cfg,
        initial_cash=initial_cash, open_panel=open_panel, verbose=False,
    )

    print()
    print("=" * 60)
    print("RESULTADO BACKTEST — Pairs Trading + VIX overlay (datos reales)")
    print("=" * 60)
    for k, v in metrics.items():
        if k == "round_trips":
            continue
        if k == "pairs_states_final":
            print(f"\nEstado final por par:")
            for pk, pv in v.items():
                print(f"  {pk}: p={pv['p_value']:.3f} β={pv['beta']:+.3f} hl={pv['half_life']:.1f} "
                      f"z={pv['z_final']:+.2f} pos={pv['position_final']:+d}")
        else:
            if isinstance(v, float):
                print(f"  {k:25s}: {v:>10.3f}")
            else:
                print(f"  {k:25s}: {v}")

    # Guardar resultados
    outdir = Path(__file__).resolve().parent.parent / "reports"
    outdir.mkdir(exist_ok=True)
    eq = trader.equity_series()
    eq.to_csv(outdir / "equity_pairs.csv")
    trader.trades_df().to_csv(outdir / "trades_pairs.csv", index=False)
    if not metrics["round_trips"].empty:
        metrics["round_trips"].to_csv(outdir / "round_trips_pairs.csv", index=False)
    print(f"\nGuardado: {outdir / 'equity_pairs.csv'} y trades_pairs.csv")
    return trader, metrics


if __name__ == "__main__":
    main()
