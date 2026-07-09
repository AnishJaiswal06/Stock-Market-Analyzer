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
) -> pd.DataFrame:
    """Simulate a simple long-only trading strategy driven by model predictions.

    Trading rules
    -------------
    * **BUY** — When the model's predicted probability ≥ *signal_threshold*
      and we currently hold zero shares.  We invest 90 % of available cash.
    * **SELL** — When the prediction drops below *signal_threshold* and we
      hold shares.  All shares are liquidated.
    * **HOLD** — In every other case.

    The function produces a *daily* portfolio ledger that records the signal,
    action, and full portfolio state for every trading day in the test period.

    Args:
        close_prices: A ``pd.Series`` of daily closing prices indexed
            consistently with *dates*.
        predictions: 1-D array of predicted probabilities (one per day in the
            test period).  Length must equal ``len(dates)``.
        dates: ``pd.DatetimeIndex`` aligned with *close_prices* and
            *predictions*.
        initial_capital: Starting cash balance (USD).
        signal_threshold: Probability cutoff above which the model signals
            a buy.

    Returns:
        A ``pd.DataFrame`` with one row per test-period day and columns:

        ============== ==============================================
        Column         Description
        ============== ==============================================
        Date           Trading date (datetime)
        Signal         1 (BUY signal) / 0 (SELL/HOLD signal)
        Price          Closing price on that day
        Action         ``'BUY'``, ``'SELL'``, or ``'HOLD'``
        Shares_Held    Cumulative shares held after the action
        Capital        Cash remaining after the action
        Portfolio_Value Cash + market value of held shares
        ============== ==============================================

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

    # Generate binary signals from probabilities.
    signals: np.ndarray = (predictions >= signal_threshold).astype(int)

    capital: float = initial_capital
    shares_held: int = 0

    records: List[Dict] = []

    for i in range(len(dates)):
        date = dates[i]
        signal = int(signals[i])
        price = float(close_prices.iloc[i]) if isinstance(close_prices.iloc[i], (int, float, np.floating, np.integer)) else float(close_prices.iloc[i])

        action: str = "HOLD"

        if signal == 1 and shares_held == 0:
            # BUY: deploy 90 % of available capital.
            buy_budget = capital * 0.9
            shares_to_buy = int(buy_budget / price)
            if shares_to_buy > 0:
                cost = shares_to_buy * price
                capital -= cost
                shares_held += shares_to_buy
                action = "BUY"
                logger.debug(
                    "BUY  %s — %d shares @ $%.2f (cost $%.2f)",
                    date, shares_to_buy, price, cost,
                )

        elif signal == 0 and shares_held > 0:
            # SELL: liquidate entire position.
            revenue = shares_held * price
            capital += revenue
            logger.debug(
                "SELL %s — %d shares @ $%.2f (revenue $%.2f)",
                date, shares_held, price, revenue,
            )
            shares_held = 0
            action = "SELL"

        portfolio_value = capital + shares_held * price

        records.append({
            "Date": date,
            "Signal": signal,
            "Price": price,
            "Action": action,
            "Shares_Held": shares_held,
            "Capital": round(capital, 2),
            "Portfolio_Value": round(portfolio_value, 2),
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
    prices = portfolio_df["Price"].values.astype(float)

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

    logger.info("Performance metrics: %s", metrics)
    return metrics