"""
core.paper_trader — Trader simulado para backtest y paper live.

Diseño:
  - Mantiene posiciones por símbolo con qty signed (+long, -short).
  - Acepta órdenes "target_qty" (la nueva cantidad deseada) y calcula el delta.
  - Modela fees y slippage configurables.
  - Lleva ledger en memoria (lista de trades) + equity curve.
  - Se reusa idéntico para backtest histórico y para paper live.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import numpy as np


@dataclass
class Position:
    symbol: str
    qty: float = 0.0          # signed: + long, - short
    avg_price: float = 0.0


@dataclass
class Trade:
    ts: pd.Timestamp
    symbol: str
    side: str                 # 'buy' o 'sell'
    qty: float                # absoluto
    price: float              # con slippage aplicado
    fee_usd: float
    pnl_realized: float       # PnL realizado al cerrar/reducir posición
    note: str = ""


class PaperTrader:
    """
    Simulador minimalista pero realista.

    fee_bps:      basis points por trade (Capitaria CFD ~ 5-15 bps via spread)
    slippage_bps: basis points adicionales por slippage simulado
    initial_cash: USD inicial
    """
    def __init__(self,
                 initial_cash: float = 100_000.0,
                 fee_bps: float = 5.0,
                 slippage_bps: float = 2.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.equity_curve: list[tuple[pd.Timestamp, float]] = []

    def _apply_slippage(self, price: float, side: str) -> float:
        slip = price * self.slippage_bps / 10_000
        return price + slip if side == "buy" else price - slip

    def submit_target(self, ts: pd.Timestamp, symbol: str,
                      target_qty: float, ref_price: float, note: str = "") -> Optional[Trade]:
        """
        Mueve la posición de su qty actual a target_qty al precio ref_price.
        target_qty signed: positivo long, negativo short, 0 flat.
        """
        if ref_price <= 0 or not np.isfinite(ref_price):
            return None

        pos = self.positions.setdefault(symbol, Position(symbol=symbol))
        delta = target_qty - pos.qty
        if abs(delta) < 1e-9:
            return None

        side = "buy" if delta > 0 else "sell"
        qty_abs = abs(delta)
        fill_px = self._apply_slippage(ref_price, side)
        notional = qty_abs * fill_px
        fee = notional * self.fee_bps / 10_000

        # PnL realizado: solo cuando se reduce o invierte la posición
        pnl_realized = 0.0
        if pos.qty != 0 and (
            (pos.qty > 0 and delta < 0) or (pos.qty < 0 and delta > 0)
        ):
            qty_closing = min(abs(pos.qty), qty_abs) * (1 if pos.qty > 0 else -1)
            # qty_closing tiene el signo de la posición vieja
            pnl_realized = qty_closing * (fill_px - pos.avg_price)

        # Actualizar avg_price y qty
        if pos.qty == 0 or np.sign(pos.qty) == np.sign(delta):
            # Abriendo o aumentando posición en el mismo lado
            new_qty = pos.qty + delta
            if new_qty != 0:
                pos.avg_price = (pos.qty * pos.avg_price + delta * fill_px) / new_qty
            pos.qty = new_qty
        else:
            # Reduciendo / invirtiendo
            new_qty = pos.qty + delta
            if abs(new_qty) < 1e-9:
                pos.qty = 0.0
                pos.avg_price = 0.0
            elif np.sign(new_qty) != np.sign(pos.qty):
                # Cruzó cero, abre nueva posición en el otro lado
                pos.qty = new_qty
                pos.avg_price = fill_px
            else:
                # Solo redujo, mantiene avg_price
                pos.qty = new_qty

        # Cash impact
        self.cash -= delta * fill_px + fee
        # Realizado se acumula en cash automáticamente por el delta * fill_px

        trade = Trade(ts=ts, symbol=symbol, side=side, qty=qty_abs,
                      price=fill_px, fee_usd=fee, pnl_realized=pnl_realized,
                      note=note)
        self.trades.append(trade)
        return trade

    def mark_to_market(self, ts: pd.Timestamp, prices: dict[str, float]) -> float:
        """Calcula equity = cash + valor de posiciones a mark-to-market."""
        equity = self.cash
        for sym, pos in self.positions.items():
            px = prices.get(sym)
            if px is None or not np.isfinite(px):
                continue
            equity += pos.qty * px
        self.equity_curve.append((ts, equity))
        return equity

    def close_all(self, ts: pd.Timestamp, prices: dict[str, float]):
        for sym in list(self.positions.keys()):
            px = prices.get(sym)
            if px is None: continue
            self.submit_target(ts, sym, 0.0, px, note="close_all")

    # ── Métricas de performance ────────────────────────────────
    def equity_series(self) -> pd.Series:
        if not self.equity_curve:
            return pd.Series(dtype=float)
        ts, eq = zip(*self.equity_curve)
        return pd.Series(eq, index=pd.DatetimeIndex(ts), name="equity")

    def trades_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        rows = []
        for t in self.trades:
            rows.append({
                "ts": t.ts, "symbol": t.symbol, "side": t.side,
                "qty": t.qty, "price": t.price, "fee_usd": t.fee_usd,
                "pnl_realized": t.pnl_realized, "note": t.note,
            })
        return pd.DataFrame(rows)

    def metrics(self, periods_per_year: int = 252 * 6.5) -> dict:
        """
        Calcula Sharpe, max DD, return total, hit rate, etc.
        periods_per_year: 252 días * 6.5 horas para datos 1h US market.
                         252 para daily, 12 para mensual.
        """
        eq = self.equity_series()
        if len(eq) < 2:
            return {"error": "insufficient data"}
        rets = eq.pct_change().dropna()
        total_return = eq.iloc[-1] / eq.iloc[0] - 1
        ann_return = (1 + total_return) ** (periods_per_year / len(rets)) - 1
        ann_vol = rets.std() * (periods_per_year ** 0.5)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        # Max drawdown
        cummax = eq.cummax()
        dd = (eq - cummax) / cummax
        max_dd = dd.min()
        # Hit rate sobre trades realizados (con pnl_realized != 0)
        tdf = self.trades_df()
        realized_trades = tdf[tdf["pnl_realized"] != 0] if not tdf.empty else tdf
        wins = (realized_trades["pnl_realized"] > 0).sum() if not realized_trades.empty else 0
        n_realized = len(realized_trades)
        hit_rate = wins / n_realized if n_realized > 0 else 0
        return {
            "initial_equity":    self.initial_cash,
            "final_equity":      float(eq.iloc[-1]),
            "total_return_pct":  float(total_return * 100),
            "ann_return_pct":    float(ann_return * 100),
            "ann_vol_pct":       float(ann_vol * 100),
            "sharpe":            float(sharpe),
            "max_drawdown_pct":  float(max_dd * 100),
            "n_trades":          len(self.trades),
            "n_realized":        int(n_realized),
            "hit_rate_pct":      float(hit_rate * 100),
            "total_fees_usd":    float(sum(t.fee_usd for t in self.trades)),
        }
