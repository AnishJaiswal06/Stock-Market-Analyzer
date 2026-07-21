"""
backtest.py — Backtesting engine for evaluating ML-driven trading strategies.

Provides:
    - run_backtest: Simulates a simple long-only trading strategy using model
      predictions, tracking portfolio state day-by-day.
    - calculate_performance_metrics: Computes real risk/return metrics from the
      portfolio history produced by ``run_backtest``.

All calculations are deterministic — no random/mocked values.
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backtest simulation
# ---------------------------------------------------------------------------

def run_backtest(
    close_prices: pd.Series,
    predictions: np.ndarray,
    dates: pd.DatetimeIndex,
    initial_capital: float = 10000.0,
    signal_threshold: float = 0.5,
    trend_data: pd.DataFrame = None,
    commission_pct: float = 0.0,
    slippage_pct: float = 0.0,
    use_atr_risk: bool = False,
    atr_stop_mult: float = 2.0,
    atr_tp_mult: float = 3.0,
    risk_per_trade: float = 0.02,
) -> pd.DataFrame:
    """Simulate a long-only trading strategy with multi-factor confirmation.

    Trading rules
    -------------
    * **BUY** — When ALL of the following are true:
        1. Model probability ≥ *signal_threshold*
        2. We hold zero shares
        3. If *trend_data* is provided:
           - Close > SMA_20  (short-term uptrend)
           - SMA_20 > SMA_50 (medium-term uptrend / golden cross)
           - 40 ≤ RSI_14 ≤ 68  (healthy momentum, not overbought)
      Position size: 90 % of cash, or ATR-risk-based when *use_atr_risk*.
    * **SELL** — When we hold shares AND either:
        - Price hit the take-profit or stop-loss level (fixed +3 %/−12 %
          of the entry fill, or ATR multiples when *use_atr_risk*), OR
        - (Model probability < signal_threshold − 0.20 OR trend broken
          below SMA_50/SMA_20) while the position is in profit
    * **HOLD** — In every other case.

    Execution model
    ---------------
    Fills occur at ``close * (1 + slippage_pct)`` on buys and
    ``close * (1 - slippage_pct)`` on sells; a commission of
    ``commission_pct`` of trade value is charged on each side.

    Args:
        close_prices: Daily closing prices as ``pd.Series``.
        predictions: 1-D array of predicted probabilities.
        dates: ``pd.DatetimeIndex`` aligned with prices/predictions.
        initial_capital: Starting cash balance (USD).
        signal_threshold: Probability cutoff for buy signals.
        trend_data: Optional ``pd.DataFrame`` with technical indicator
            columns (Close, SMA_20, SMA_50, RSI_14, ATR_14) for
            multi-factor trade confirmation and ATR risk management.
        commission_pct: Fractional commission per side (0.0005 = 0.05 %).
        slippage_pct: Fractional slippage per side.
        use_atr_risk: If True and ATR_14 is available in *trend_data*,
            size positions so that a stop-out loses ``risk_per_trade`` of
            capital, with stop/target at ATR multiples of the entry.
        atr_stop_mult: Stop-loss distance in ATR multiples.
        atr_tp_mult: Take-profit distance in ATR multiples.
        risk_per_trade: Fraction of capital risked per trade (ATR mode).

    Returns:
        A ``pd.DataFrame`` with daily portfolio state, including
        ``Fill_Price`` (execution price incl. slippage on trade days)
        and ``Cum_Fees`` (running commission total).

    Raises:
        ValueError: If array lengths are inconsistent.
    """
    if len(predictions) != len(dates):
        raise ValueError(
            f"predictions length ({len(predictions)}) must equal "
            f"dates length ({len(dates)})"
        )

    logger.info(
        "Backtest started — days=%d, initial_capital=%.2f, threshold=%.2f",
        len(dates),
        initial_capital,
        signal_threshold,
    )

    # Sell threshold has a wider gap to avoid premature model-based exits
    sell_threshold = signal_threshold - 0.20

    # Per-day ATR values for risk-based sizing/stops
    atr_values = np.full(len(dates), np.nan)
    if trend_data is not None and "ATR_14" in trend_data.columns:
        atr_series = trend_data["ATR_14"]
        atr_series = atr_series[~atr_series.index.duplicated(keep="first")]
        atr_values = atr_series.reindex(dates).to_numpy(dtype=float)

    # Pre-compute multi-factor confirmation flags for each day
    buy_confirmed = np.ones(len(dates), dtype=bool)
    sell_trend_break = np.zeros(len(dates), dtype=bool)

    has_trend_filter = (
        trend_data is not None
        and "SMA_20" in trend_data.columns
    )

    if has_trend_filter:
        has_sma50 = "SMA_50" in trend_data.columns
        has_rsi = "RSI_14" in trend_data.columns
        logger.info(
            "Asymmetric trend filter ENABLED (BUY: Close>SMA_20 + golden cross + RSI, "
            "SELL: Close<SMA_%s)",
            "50" if has_sma50 else "20",
        )
        for i, d in enumerate(dates):
            if d not in trend_data.index:
                buy_confirmed[i] = False
                continue

            row = trend_data.loc[d]
            # Handle potential duplicate index
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            close_val = float(row["Close"])
            sma20 = float(row["SMA_20"]) if pd.notna(row["SMA_20"]) else None

            if sma20 is None:
                buy_confirmed[i] = False
                continue

            # Entry requires: Close > SMA_20 (short-term uptrend)
            trend_ok = close_val > sma20

            # Golden cross confirmation: SMA_20 > SMA_50
            if has_sma50:
                sma50 = float(row["SMA_50"]) if pd.notna(row["SMA_50"]) else None
                if sma50 is not None:
                    trend_ok = trend_ok and (sma20 > sma50)

            # RSI filter: avoid overbought entries (only buy 40-68)
            rsi_ok = True
            if has_rsi and pd.notna(row.get("RSI_14")):
                rsi = float(row["RSI_14"])
                rsi_ok = 40.0 <= rsi <= 68.0

            buy_confirmed[i] = trend_ok and rsi_ok

            # Sell on trend break: use SMA_50 if available (stronger filter),
            # otherwise fall back to SMA_20
            if has_sma50:
                sma50_val = float(row["SMA_50"]) if pd.notna(row.get("SMA_50")) else None
                if sma50_val is not None:
                    sell_trend_break[i] = close_val < sma50_val
                else:
                    sell_trend_break[i] = close_val < sma20
            else:
                sell_trend_break[i] = close_val < sma20

    # Generate binary signals from probabilities
    buy_signals: np.ndarray = (predictions >= signal_threshold).astype(int)
    sell_signals: np.ndarray = (predictions < sell_threshold).astype(int)

    capital: float = initial_capital
    shares_held: int = 0
    buy_fill: float = 0.0
    stop_price: float = 0.0
    tp_price: float = 0.0
    total_fees: float = 0.0

    records: List[Dict] = []

    for i in range(len(dates)):
        date = dates[i]
        signal = int(buy_signals[i])
        price = float(close_prices.iloc[i])

        action: str = "HOLD"
        fill_price: float = price

        if signal == 1 and shares_held == 0 and buy_confirmed[i]:
            fill_price = price * (1.0 + slippage_pct)
            atr = atr_values[i]

            if use_atr_risk and np.isfinite(atr) and atr > 0:
                # Risk-based sizing: a stop-out loses risk_per_trade of capital
                stop_dist = atr * atr_stop_mult
                shares_by_risk = int((capital * risk_per_trade) / stop_dist)
                shares_by_cash = int((capital * 0.9) / (fill_price * (1.0 + commission_pct)))
                shares_to_buy = max(0, min(shares_by_risk, shares_by_cash))
                stop_price = fill_price - stop_dist
                tp_price = fill_price + atr * atr_tp_mult
            else:
                # Fixed sizing: deploy 90 % of available capital
                shares_to_buy = int((capital * 0.9) / (fill_price * (1.0 + commission_pct)))
                stop_price = fill_price * 0.88   # 12% stop loss
                tp_price = fill_price * 1.03     # 3% take profit

            if shares_to_buy > 0:
                cost = shares_to_buy * fill_price
                fee = cost * commission_pct
                capital -= cost + fee
                total_fees += fee
                shares_held += shares_to_buy
                buy_fill = fill_price
                action = "BUY"
                logger.debug(
                    "BUY  %s — %d shares @ $%.2f (cost $%.2f, fee $%.2f)",
                    date, shares_to_buy, fill_price, cost, fee,
                )

        elif shares_held > 0:
            hit_tp = price >= tp_price
            hit_sl = price <= stop_price

            # Allow model or trend to close the trade early ONLY if we are in profit
            in_profit = price > buy_fill
            model_or_trend_sell = (sell_signals[i] or sell_trend_break[i]) and in_profit

            if hit_tp or hit_sl or model_or_trend_sell:
                # SELL: liquidate
                fill_price = price * (1.0 - slippage_pct)
                revenue = shares_held * fill_price
                fee = revenue * commission_pct
                capital += revenue - fee
                total_fees += fee
                logger.debug(
                    "SELL %s — %d shares @ $%.2f (revenue $%.2f, fee $%.2f)",
                    date, shares_held, fill_price, revenue, fee,
                )
                shares_held = 0
                buy_fill = 0.0
                action = "SELL"

        portfolio_value = capital + shares_held * price

        records.append({
            "Date": date,
            "Signal": signal,
            "Price": price,
            "Fill_Price": round(fill_price, 4),
            "Action": action,
            "Shares_Held": shares_held,
            "Capital": round(capital, 2),
            "Portfolio_Value": round(portfolio_value, 2),
            "Cum_Fees": round(total_fees, 2),
        })

    portfolio_df = pd.DataFrame(records)

    total_trades = len(portfolio_df[portfolio_df["Action"].isin(["BUY", "SELL"])])
    logger.info(
        "Backtest complete — final_value=%.2f, total_actions=%d",
        portfolio_df["Portfolio_Value"].iloc[-1] if len(portfolio_df) > 0 else initial_capital,
        total_trades,
    )

    return portfolio_df


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def calculate_performance_metrics(
    portfolio_df: pd.DataFrame,
    initial_capital: float,
    risk_free_rate: float = 0.02,
) -> Dict[str, float]:
    """Compute real risk-adjusted performance metrics from a portfolio ledger.

    All values are computed analytically from the portfolio time-series —
    **no random or mocked numbers** are used.

    Args:
        portfolio_df: DataFrame produced by :func:`run_backtest`.  Must contain
            at least the columns ``Portfolio_Value``, ``Action``, and ``Price``.
        initial_capital: The starting cash balance used in the backtest.
        risk_free_rate: Annualised risk-free rate used for the Sharpe Ratio
            calculation (default ``0.02`` = 2 %).

    Returns:
        A dict with the following keys:

        ====================== ==========================================
        Key                    Description
        ====================== ==========================================
        Total_Return_Pct       Total percentage return over the period
        Annualized_Return_Pct  Return annualised to 252 trading days
        Sharpe_Ratio           Annualised Sharpe Ratio
        Max_Drawdown_Pct       Maximum peak-to-trough drawdown (%)
        Win_Rate_Pct           Percentage of profitable round-trip trades
        Total_Trades           Count of completed round-trip trades
        Buy_Hold_Return_Pct    Benchmark buy-and-hold return (%)
        ====================== ==========================================
    """
    logger.info("Calculating performance metrics...")

    if portfolio_df is None or portfolio_df.empty:
        logger.warning("Empty portfolio DataFrame — returning zeroed metrics.")
        return {
            "Total_Return_Pct": 0.0,
            "Annualized_Return_Pct": 0.0,
            "Sharpe_Ratio": 0.0,
            "Max_Drawdown_Pct": 0.0,
            "Win_Rate_Pct": 0.0,
            "Total_Trades": 0,
            "Buy_Hold_Return_Pct": 0.0,
        }

    # ------------------------------------------------------------------
    # 1.  Total Return
    # ------------------------------------------------------------------
    final_value: float = float(portfolio_df["Portfolio_Value"].iloc[-1])
    total_return_pct: float = (final_value / initial_capital - 1.0) * 100.0

    # ------------------------------------------------------------------
    # 2.  Annualised Return
    # ------------------------------------------------------------------
    trading_days: int = len(portfolio_df)
    if trading_days > 1:
        annualized_return_pct: float = (
            (final_value / initial_capital) ** (252.0 / trading_days) - 1.0
        ) * 100.0
    else:
        annualized_return_pct = 0.0

    # ------------------------------------------------------------------
    # 3.  Sharpe Ratio (annualised)
    # ------------------------------------------------------------------
    portfolio_values = portfolio_df["Portfolio_Value"].values.astype(float)
    daily_returns: np.ndarray = np.diff(portfolio_values) / portfolio_values[:-1]

    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        daily_risk_free: float = risk_free_rate / 252.0
        excess_returns: np.ndarray = daily_returns - daily_risk_free
        sharpe_ratio: float = float(
            np.mean(excess_returns) / np.std(excess_returns, ddof=1) * np.sqrt(252)
        )
    else:
        sharpe_ratio = 0.0

    # ------------------------------------------------------------------
    # 4.  Maximum Drawdown
    # ------------------------------------------------------------------
    cumulative_max: np.ndarray = np.maximum.accumulate(portfolio_values)
    drawdowns: np.ndarray = (cumulative_max - portfolio_values) / cumulative_max
    max_drawdown_pct: float = float(np.max(drawdowns) * 100.0) if len(drawdowns) > 0 else 0.0

    # ------------------------------------------------------------------
    # 5.  Win Rate & Total Trades (round-trip analysis)
    # ------------------------------------------------------------------
    actions = portfolio_df["Action"].values
    # Use execution fill prices (incl. slippage) when available so the
    # win rate reflects realised trade P&L, not mark prices.
    price_col = "Fill_Price" if "Fill_Price" in portfolio_df.columns else "Price"
    prices = portfolio_df[price_col].values.astype(float)

    buy_price: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0

    for i in range(len(actions)):
        if actions[i] == "BUY":
            buy_price = prices[i]
        elif actions[i] == "SELL" and buy_price > 0:
            total_trades += 1
            if prices[i] > buy_price:
                winning_trades += 1
            buy_price = 0.0  # reset

    win_rate_pct: float = (
        (winning_trades / total_trades) * 100.0 if total_trades > 0 else 0.0
    )

    # ------------------------------------------------------------------
    # 6.  Buy-and-Hold benchmark
    # ------------------------------------------------------------------
    first_price: float = float(portfolio_df["Price"].iloc[0])
    last_price: float = float(portfolio_df["Price"].iloc[-1])
    buy_hold_return_pct: float = (
        (last_price / first_price - 1.0) * 100.0 if first_price > 0 else 0.0
    )

    metrics: Dict[str, float] = {
        "Total_Return_Pct": round(total_return_pct, 4),
        "Annualized_Return_Pct": round(annualized_return_pct, 4),
        "Sharpe_Ratio": round(sharpe_ratio, 4),
        "Max_Drawdown_Pct": round(max_drawdown_pct, 4),
        "Win_Rate_Pct": round(win_rate_pct, 4),
        "Total_Trades": total_trades,
        "Buy_Hold_Return_Pct": round(buy_hold_return_pct, 4),
    }

    if "Cum_Fees" in portfolio_df.columns:
        metrics["Total_Fees"] = round(float(portfolio_df["Cum_Fees"].iloc[-1]), 2)

    logger.info("Performance metrics: %s", metrics)
    return metrics