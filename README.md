# multistrat — Multi-strategy quantitative framework

A modular Python framework for systematic strategies across equities, FX and macro pairs. Built around three independent alpha sources, all sharing one risk-managed execution core.

## Strategies

**1. US equities pairs trading + VIX overlay** (`strategies/pairs_us.py`)
Engle-Granger cointegration on rolling windows, OLS hedge ratio estimation, z-score entries on the residual spread, dollar-neutral sizing. VIX threshold acts as a kill switch during stress regimes when correlation structure typically breaks down. Universe: 8 large-cap pairs (KO/PEP, V/MA, MSFT/AAPL, JPM/BAC, XOM/CVX, HD/LOW, WMT/TGT, GS/MS).

**2. Cross-asset macro: USD/CLP vs copper (HG)** (planned, not implemented yet)
The Chilean peso is structurally exposed to copper (Chile produces ~28% of world copper). Rolling regression of USDCLP on HG futures, residual z-score signals dislocations, mean-reversion trade in the spread.

**3. Selective FX carry with VIX filter** (planned, not implemented yet)
High-yielding crosses (AUDJPY, NZDJPY, MXNJPY) entered when implied vol is low (risk-on), exited when VIX spikes above threshold. Carry capture with embedded risk-off circuit breaker.

### Risk management in Strategy 1

Entries require **stable** cointegration (p-value ≤ 0.05 for 20 consecutive bars) and a spread half-life between 2 and 100 bars, which filters most spurious pairs between independent random walks.

At entry the bot **freezes** alpha, beta, spread mean and spread std. Every exit is measured against that snapshot, so the z-score stop cannot be diluted by a rolling std that widens together with the loss, and the z reflects the position actually held. Exits, in priority order:

| Exit | Rule (defaults) |
|---|---|
| Hard USD stop | pair loss ≥ 1% of equity at entry |
| z stop | frozen \|z\| > 4 |
| Time stop | 3 × spread half-life (floor 20, cap 500 bars) |
| Cointegration break | p-value > 0.20 for 10 consecutive bars (hysteresis) |
| Take profit | frozen z back inside ±0.5, or crossed the mean |
| VIX kill switch | VIX > 28 closes everything and blocks entries |
| Daily kill switch | intraday drawdown > 3% closes everything until next day |

Cointegration, beta and half-life are re-estimated every `recalc_every` bars (default 7, about once per US trading day on 1h data) with a fixed-lag ADF (`coint_maxlag=1`); z-scores and all stops are evaluated on every bar. A full 3-year 1h run over 8 pairs takes about a minute and prints progress every 5%.

Orders decided on the close of bar *t* are filled on the **open of bar t+1** (or its close when opens are not available), never on the signal bar.

## Architecture

```
multistrat/
├── core/
│   ├── data.py          Unified market data (yfinance, FRED, parquet cache)
│   ├── paper_trader.py  Simulated execution with realistic fees & slippage
│   └── risk.py          Vol-targeted sizing, dollar-neutral, exposure caps
├── strategies/
│   └── pairs_us.py      Strategy 1 (strategies 2 and 3: planned)
├── backtest/
│   ├── runner_pairs.py  Event-driven bar-by-bar backtest
│   └── run_real_pairs.py  Backtest with real yfinance data
└── reports/             Equity curves, trade logs, metrics
```

The strategies expose a uniform interface so the same `runner` orchestrates backtesting and (in a later phase) paper-live execution.

## Usage

```bash
pip install -r requirements.txt

# End-to-end demo with synthetic cointegrated data (no network needed)
python -m backtest.runner_pairs

# Backtest with real yfinance data (1h, last ~24 months)
python -m backtest.run_real_pairs
```

## Design choices and notes

The framework chooses **modularity over monolith**: every component (data, execution, risk, strategy) has a single responsibility and can be swapped. The `PaperTrader` accepts target positions, not deltas, which lets each strategy express intent at the portfolio level without coordinating with peers.

Cointegration is checked **per-pair on every bar** within a rolling window, so spurious pairs that drift out of cointegration get auto-disabled. The hedge ratio (β) is re-estimated continuously rather than fixed at calibration, which adapts to slow structural drift.

Costs are explicit: `fee_bps` and `slippage_bps` are configured per backtest and applied to every fill. This matters because pairs strategies often look profitable in zero-cost backtests but die under realistic execution.

## Performance metrics

Output of every backtest includes Sharpe (annualized with bars-per-day inferred from the data, e.g. 7 × 252 for US 1h bars), max drawdown, total/annualized return, hit rate per completed pair round trip, PnL by exit reason and by pair, total fees paid, kill-switch days, and end-of-backtest state of every pair (cointegration p-value, β, half-life, final z-score).

## Status

- Core: complete
- Strategy 1: backtest engine complete with risk layer. **No edge on real data with the default parameters.** Run of 2026-11-20 on the 8 default pairs, yfinance 1h, 2024-07 to 2026-11 after warm-up: total return −0.64%, Sharpe −0.81, max drawdown −0.67%, 38 pair round trips, 42% hit rate. z-stops lost $1,446 in 19 exits vs $978 earned by 14 take-profits; fees $393. Exposure is small (10% gross per pair, usually 0–1 pairs open; annual vol 0.34%), so the absolute numbers are small too; the useful reading is the sign and the stop/TP balance, not the size. On synthetic data the engine is roughly breakeven after costs. Treat this as a research framework, not a trading system.
- Strategies 2 and 3: not implemented
- Live execution via MetaTrader5 (Capitaria CFDs): planned phase 2. Before going live it needs leg-risk handling (unwind the filled leg if the other one fails), position reconciliation after reconnects, and an earnings blackout.

### Changelog

- 2026-11: fixed inverted leg signs in `compute_qty` (the strategy was buying the expensive leg and lost money even on a synthetic cointegrated pair); added frozen-snapshot exits, hard USD stop, time stop, p-value hysteresis, stable-cointegration and half-life entry filters, next-bar execution, daily kill switch, correct annualization and per-pair hit rate.

## Author

Carlos Suspichiatti. CFA Levels I and II passed. 16+ years in finance and investments.
