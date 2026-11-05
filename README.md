# multistrat — Multi-strategy quantitative framework

A modular Python framework for systematic strategies across equities, FX and macro pairs. Built around three independent alpha sources, all sharing one risk-managed execution core.

## Strategies

**1. US equities pairs trading + VIX overlay** (`strategies/pairs_us.py`)
Engle-Granger cointegration on rolling windows, OLS hedge ratio estimation, z-score entries on the residual spread, dollar-neutral sizing. VIX threshold acts as a kill switch during stress regimes when correlation structure typically breaks down. Universe: 8 large-cap pairs (KO/PEP, V/MA, MSFT/AAPL, JPM/BAC, XOM/CVX, HD/LOW, WMT/TGT, GS/MS).

**2. Cross-asset macro: USD/CLP vs copper (HG)** (in progress)
The Chilean peso is structurally exposed to copper (Chile produces ~28% of world copper). Rolling regression of USDCLP on HG futures, residual z-score signals dislocations, mean-reversion trade in the spread.

**3. Selective FX carry with VIX filter** (in progress)
High-yielding crosses (AUDJPY, NZDJPY, MXNJPY) entered when implied vol is low (risk-on), exited when VIX spikes above threshold. Carry capture with embedded risk-off circuit breaker.

## Architecture

```
multistrat/
├── core/
│   ├── data.py          Unified market data (yfinance, FRED, parquet cache)
│   ├── paper_trader.py  Simulated execution with realistic fees & slippage
│   └── risk.py          Vol-targeted sizing, dollar-neutral, exposure caps
├── strategies/
│   ├── pairs_us.py      Strategy 1
│   ├── clp_copper.py    Strategy 2
│   └── fx_carry.py      Strategy 3
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

Output of every backtest includes Sharpe (annualized), max drawdown, total/annualized return, hit rate on closed trades, total fees paid, and end-of-backtest state of every pair (cointegration p-value, β, final z-score).

## Status

- Core: complete
- Strategy 1: complete (backtest engine), pending validation on real data
- Strategies 2 and 3: skeleton, lógica pendiente
- Live execution via MetaTrader5 (Capitaria CFDs): planned phase 2

## Author

Carlos Suspichiatti. CFA Levels I and II passed. 16+ years in finance and investments.
