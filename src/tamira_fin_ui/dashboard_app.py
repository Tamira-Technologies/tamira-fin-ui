"""
Streamlit dashboard to manage portfolios, run fuzzy LSTM forecasts, visualize actual vs
predicted prices, display FMACD signals, and compute portfolio returns.

This app integrates:
- Curated blue chip lists across USA, India, China, Singapore.
- Portfolio creation and persistence (up to 6 tickers per portfolio).
- Parameter selection for model training: eps, min_samples, epochs, lr, start/end dates.
- Predictions for horizons t+1..t+13 per stock, with actual vs predicted visualization.
- FMACD signals (forecasted MACD based on predicted prices) and visualizations.
- Equal-weight portfolio returns based on FMACD signals.

Usage:
- Run: `streamlit run src/tamira_fin_ui/dashboard_app.py`
- Or via uv script: `uv run dashboard`

Notes:
- Forecasting trains one model per horizon (1..13), which can be compute-intensive.
  Reduce epochs for faster iteration.
- This is not financial advice. Forecasts are experimental and for research purposes.

"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from tamira_fin_ui.bluechips import get_bluechips, get_markets
from tamira_fin_ui.lstm_main_code import (
    compute_fmacd_portfolio_returns,
    forecast_macd_signals_for_ticker,
)
from tamira_fin_ui.portfolio import (
    Portfolio,
    add_portfolio,
    available_bluechips,
    get_portfolio,
    list_portfolios,
    remove_portfolio,
    update_portfolio,
)

# =====================
# Streamlit Caching
# =====================


@st.cache_data(show_spinner=False)
def cached_forecast_bundle(
    ticker: str,
    start_iso: str,
    end_iso: str,
    n_past: int,
    max_horizon: int,
    eps: float,
    min_samples: int,
    epochs: int,
    lr: float,
    quick_mode: bool,
) -> dict[str, object]:
    """
    Compute a single forecast bundle for a ticker:
    - Evaluation window (last 13 trading days up to end_iso): actual vs predicted aligned on dates
    - Future predictions (t+1..t+N) returned by the model
    - FMACD signals on the forecast path
    """
    # Quick mode constraints (kept for parity; does not alter model outputs)
    h = min(max_horizon, 5) if quick_mode else max_horizon
    e = min(epochs, 5) if quick_mode else epochs

    # Recent actuals (t-13..t) for quick reference (not used in the chart once eval window is available)
    recent = yf.download(ticker, end=end_iso, period=f"{max(n_past * 3, 60)}d")
    if "Close" not in recent or len(recent["Close"]) < n_past:
        raise ValueError("Not enough recent data to build the t-13..t window.")
    recent_actual = recent["Close"].values.astype(float)[-n_past:].tolist()

    # Forecast and FMACD (single training call per ticker)
    signals = forecast_macd_signals_for_ticker(
        ticker=ticker,
        start_date=start_iso,
        end_date=end_iso,
        n_past=n_past,
        max_horizon=h,
        eps=eps,
        min_samples=min_samples,
        epochs=e,
        lr=lr,
    )

    # Past-13 evaluation removed: only future predictions are computed and displayed
    eval_dates: list[str] = []
    eval_actual: list[float] = []
    eval_predicted: list[float] = []

    return {
        "recent_actual": recent_actual,
        "future_predictions": signals["predictions_by_day"],
        "macd": signals["macd"],
        "signal": signals["signal"],
        "hist": signals["hist"],
        "actions": signals["actions"],
        "eval_dates": eval_dates,
        "eval_actual": eval_actual,
        "eval_predicted": eval_predicted,
    }


@st.cache_data(show_spinner=False)
def cached_portfolio_returns(
    tickers: list[str],
    start_iso: str,
    end_iso: str,
    n_past: int,
    max_horizon: int,
    eps: float,
    min_samples: int,
    epochs: int,
    lr: float,
    quick_mode: bool,
) -> dict[int, float]:
    """
    Cache wrapper around `compute_fmacd_portfolio_returns` with optional quick mode.

    Parameters
    ----------
    tickers : list[str]
        Portfolio tickers.
    start_iso : str
        Training start date.
    end_iso : str
        Training end date.
    n_past : int
        Sliding window length.
    max_horizon : int
        Max forecast horizon.
    eps : float
        DBSCAN epsilon.
    min_samples : int
        DBSCAN min samples.
    epochs : int
        Training epochs.
    lr : float
        Learning rate.
    quick_mode : bool
        If True, reduce horizon/epochs to accelerate UI.

    Returns
    -------
    dict[int, float]
        Mapping day h -> portfolio return fraction.
    """
    h = min(max_horizon, 5) if quick_mode else max_horizon
    e = min(epochs, 5) if quick_mode else epochs

    return compute_fmacd_portfolio_returns(
        tickers=tickers,
        start_date=start_iso,
        end_date=end_iso,
        n_past=n_past,
        max_horizon=h,
        eps=eps,
        min_samples=min_samples,
        epochs=e,
        lr=lr,
    )


# =====================
# Visualization Helpers
# =====================


def build_future_predictions_chart(
    future_predictions: dict[int, float],
    title: str,
) -> go.Figure:
    """
    Future predictions chart:
    - Shows model-predicted close prices for the next n days (t+1..t+n)
    - Uses only the model outputs (no past overlay), ensuring clarity and speed
    """
    future_keys = sorted(future_predictions.keys())
    future_x = [f"t+{h}" for h in future_keys]
    future_y = [future_predictions[h] for h in future_keys]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=future_x,
            y=future_y,
            mode="lines+markers",
            name="Predicted (future)",
            line=dict(color="#FFB74D", width=3),
            marker=dict(size=7),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Horizon (t+h)",
        yaxis_title="Predicted Close",
        template="plotly_dark",
        legend=dict(orientation="h"),
    )
    return fig


def build_macd_chart(macd: list[float], signal: list[float], hist: list[float], title: str) -> go.Figure:
    """
    Create a Plotly MACD chart for forecasted price path.
    """
    x = list(range(1, len(macd) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=macd, mode="lines+markers", name="MACD", line=dict(color="#BA68C8", width=3), marker=dict(size=6)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=signal, mode="lines+markers", name="Signal", line=dict(color="#66BB6A", width=3), marker=dict(size=6)
        )
    )
    fig.add_trace(go.Bar(x=x, y=hist, name="Histogram", marker_color="#90A4AE", opacity=0.5))
    fig.update_layout(
        title=title,
        xaxis_title="Horizon (t+h)",
        yaxis_title="MACD",
        barmode="overlay",
        template="plotly_dark",
        legend=dict(orientation="h"),
    )
    return fig


def build_portfolio_returns_chart(returns_by_day: dict[int, float], title: str) -> go.Figure:
    """
    Create a Plotly chart showing per-day equal-weight portfolio returns and cumulative returns.
    """
    days = sorted(returns_by_day.keys())
    daily = [returns_by_day[h] for h in days]
    cumulative = list(np.cumsum(daily))

    fig = go.Figure()
    fig.add_trace(go.Bar(x=days, y=daily, name="Daily Returns", marker_color="#26A69A", opacity=0.9))
    fig.add_trace(
        go.Scatter(
            x=days,
            y=cumulative,
            name="Cumulative Returns",
            mode="lines+markers",
            line=dict(color="#EF5350", width=3),
            marker=dict(size=6),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Horizon (t+h)",
        yaxis_title="Return (fraction)",
        template="plotly_dark",
        legend=dict(orientation="h"),
    )
    return fig


# =====================
# UI Components
# =====================


def sidebar_parameters() -> dict[str, object]:
    """
    Render the sidebar inputs and return the selected parameters.
    """
    st.sidebar.header("Model Parameters")

    today = datetime.now(UTC).date()
    computed_end_iso = (pd.Timestamp(today) - pd.offsets.BDay(0)).date().isoformat()
    start_date = st.sidebar.text_input("Training Start Date (YYYY-MM-DD)", "2000-01-01")
    end_date = st.sidebar.text_input("Training End Date (YYYY-MM-DD)", computed_end_iso, disabled=True)
    n_past = st.sidebar.number_input("Window Length (n_past)", min_value=5, max_value=60, value=13, step=1)
    max_horizon = st.sidebar.slider("Max Horizon (days)", min_value=1, max_value=13, value=13, step=1)
    eps = st.sidebar.number_input("DBSCAN eps", min_value=0.05, max_value=5.0, value=0.2, step=0.05, format="%.2f")
    min_samples = st.sidebar.number_input("DBSCAN min_samples", min_value=2, max_value=50, value=4, step=1)
    epochs = st.sidebar.number_input("Training epochs per horizon", min_value=1, max_value=200, value=25, step=1)
    lr = st.sidebar.number_input(
        "Learning rate (Adam)", min_value=1e-5, max_value=1e-1, value=1e-3, step=1e-5, format="%.5f"
    )
    quick_mode = st.sidebar.checkbox("Quick Mode (faster UI: lower epochs/horizon)", value=True)

    st.sidebar.caption(
        "Training trains one model per horizon (1..N). Use fewer epochs to iterate faster. "
        "Forecasts and signals are computed out-of-sample using the most recent n_past closes."
    )

    return {
        "start_date": start_date,
        "end_date": computed_end_iso,
        "n_past": int(n_past),
        "max_horizon": int(max_horizon),
        "eps": float(eps),
        "min_samples": int(min_samples),
        "epochs": int(epochs),
        "lr": float(lr),
        "quick_mode": bool(quick_mode),
    }


def portfolio_manager() -> Portfolio | None:
    """
    Render portfolio management UI (create, list, select, delete) and return the selected portfolio.
    """
    st.subheader("Portfolio Manager")

    # Display curated blue chips by market
    st.markdown("Curated Blue Chips by Market")
    bc = available_bluechips()
    cols = st.columns(4)
    for i, market in enumerate(get_markets()):
        with cols[i]:
            st.markdown(f"**{market.capitalize()}**")
            st.write(", ".join(bc[market]))

    # Create or update portfolio
    st.markdown("---")
    st.markdown("Create or Update a Portfolio")

    name = st.text_input("Portfolio Name", value="My Portfolio")
    selected_tickers: list[str] = []

    sel_cols = st.columns(4)
    markets = get_markets()
    for i, market in enumerate(markets):
        with sel_cols[i]:
            ticks = get_bluechips(market)
            picks = st.multiselect(f"{market.capitalize()} tickers", ticks, max_selections=6)
            selected_tickers.extend(picks)

    # Enforce max 6 tickers across markets
    if len(selected_tickers) > 6:
        selected_tickers = selected_tickers[:6]
        st.warning("A portfolio cannot contain more than 6 tickers. Truncated to first 6 selections.")

    col_buttons = st.columns(3)
    with col_buttons[0]:
        if st.button("Save Portfolio"):
            try:
                existing = get_portfolio(name)
                if existing is None:
                    add_portfolio(name=name, tickers=selected_tickers)
                    st.success(f"Portfolio '{name}' created.")
                else:
                    update_portfolio(name=name, tickers=selected_tickers)
                    st.success(f"Portfolio '{name}' updated.")
            except Exception as exc:
                st.error(f"Failed to save portfolio: {exc}")

    with col_buttons[1]:
        if st.button("Delete Portfolio"):
            try:
                remove_portfolio(name=name)
                st.success(f"Portfolio '{name}' deleted.")
            except Exception as exc:
                st.error(f"Failed to delete portfolio: {exc}")

    with col_buttons[2]:
        st.caption("Portfolios are persisted locally (see settings).")

    # List existing portfolios and select one
    st.markdown("---")
    all_names = [p.name for p in list_portfolios()]
    selected_name = st.selectbox("Select Portfolio", options=all_names)
    if selected_name:
        return get_portfolio(selected_name)
    return None


def render_stock_section(
    ticker: str,
    params: dict[str, object],
) -> None:
    """
    Render forecasts, actual vs predicted, and FMACD signals for a single ticker using tabs.
    """
    start_iso = str(params["start_date"])
    end_iso = str(params["end_date"])
    n_past = int(params["n_past"])
    max_horizon = int(params["max_horizon"])
    eps = float(params["eps"])
    min_samples = int(params["min_samples"])
    epochs = int(params["epochs"])
    lr = float(params["lr"])
    quick_mode = bool(params.get("quick_mode", True))
    st.markdown(f"### {ticker}")

    tabs = st.tabs(["Actual vs Predicted", "FMACD", "Tables"])
    with st.spinner(f"Training & forecasting {ticker}..."):
        try:
            bundle = cached_forecast_bundle(
                ticker=ticker,
                start_iso=start_iso,
                end_iso=end_iso,
                n_past=n_past,
                max_horizon=max_horizon,
                eps=eps,
                min_samples=min_samples,
                epochs=epochs,
                lr=lr,
                quick_mode=quick_mode,
            )
        except Exception as exc:
            st.error(f"Failed to forecast for {ticker}: {exc}")
            return

    recent_actual: list[float] = bundle["recent_actual"]
    future_predictions: dict[int, float] = bundle["future_predictions"]
    macd = list(bundle["macd"])
    signal_line = list(bundle["signal"])
    hist = list(bundle["hist"])
    actions = list(bundle["actions"])

    with tabs[0]:
        fig_avp = build_future_predictions_chart(
            future_predictions=future_predictions,
            title=f"Future Predictions: {ticker}",
        )
        st.plotly_chart(fig_avp, use_container_width=True)
        if not bundle.get("eval_predicted"):
            st.info("Displaying only future predictions (t+1..t+n) based on current model outputs.")

    with tabs[1]:
        fig_macd = build_macd_chart(macd=macd, signal=signal_line, hist=hist, title=f"FMACD: {ticker}")
        st.plotly_chart(fig_macd, use_container_width=True)

    with tabs[2]:
        df_preds = pd.DataFrame(
            {
                "Day (t+h)": sorted(list(future_predictions.keys())),
                "Predicted Price": [future_predictions[h] for h in sorted(future_predictions.keys())],
            }
        )
        df_signals = pd.DataFrame(
            {
                "Day (t+h)": list(range(1, len(macd) + 1)),
                "MACD": macd,
                "Signal": signal_line,
                "Histogram": hist,
                "Action": actions,
            }
        )
        st.subheader("Predictions")
        st.dataframe(df_preds, use_container_width=True)
        st.subheader("Signals")
        st.dataframe(df_signals, use_container_width=True)


def render_portfolio_returns(portfolio: Portfolio, params: dict[str, object]) -> None:
    """
    Render equal-weight portfolio returns over the forecast horizon using FMACD signals.
    """
    if not portfolio.tickers:
        st.info("Portfolio has no tickers.")
        return

    start_iso = str(params["start_date"])
    end_iso = str(params["end_date"])
    n_past = int(params["n_past"])
    max_horizon = int(params["max_horizon"])
    eps = float(params["eps"])
    min_samples = int(params["min_samples"])
    epochs = int(params["epochs"])
    lr = float(params["lr"])
    quick_mode = bool(params.get("quick_mode", True))
    st.subheader(f"Portfolio Returns (FMACD strategy): {portfolio.name}")
    with st.spinner("Computing portfolio returns..."):
        try:
            returns_by_day = cached_portfolio_returns(
                tickers=portfolio.tickers,
                start_iso=start_iso,
                end_iso=end_iso,
                n_past=n_past,
                max_horizon=max_horizon,
                eps=eps,
                min_samples=min_samples,
                epochs=epochs,
                lr=lr,
                quick_mode=quick_mode,
            )
            if not returns_by_day:
                st.warning("No returns computed. Ensure forecasts were successful for selected tickers.")
                return

            fig_ret = build_portfolio_returns_chart(
                returns_by_day=returns_by_day, title=f"Equal-weight Portfolio Returns: {portfolio.name}"
            )
            st.plotly_chart(fig_ret, use_container_width=True)

            df_ret = pd.DataFrame(
                {
                    "Day (t+h)": sorted(returns_by_day.keys()),
                    "Return (fraction)": [returns_by_day[h] for h in sorted(returns_by_day.keys())],
                }
            )
            df_ret["Cumulative"] = df_ret["Return (fraction)"].cumsum()
            st.dataframe(df_ret, use_container_width=True)
        except Exception as exc:
            st.error(f"Failed to compute portfolio returns: {exc}")


# =====================
# Main App
# =====================


def main() -> None:
    """
    Entry point for the Streamlit dashboard.
    """
    st.set_page_config(page_title="Fuzzy LSTM Forecasts (OpenBB UI)", layout="wide")
    st.title("Fuzzy LSTM Forecast Dashboard")
    st.caption(
        "Curated blue chips, portfolio management, 1..13-day forecasts, FMACD signals, and portfolio returns. "
        "This is not financial advice."
    )

    # Sidebar model parameters
    params = sidebar_parameters()

    # Portfolio manager
    portfolio = portfolio_manager()
    if portfolio is None:
        st.info("Create a portfolio or select an existing one to view forecasts and signals.")
        return

    # Selected portfolio overview
    st.markdown("---")
    st.subheader(f"Selected Portfolio: {portfolio.name}")
    st.write(", ".join(portfolio.tickers))

    # Forecasts and signals per ticker (gated by Run button)
    st.markdown("---")
    run_clicked = st.button("Run Forecasts", type="primary")
    if not run_clicked:
        st.info("Select a portfolio and parameters, then click Run Forecasts to train and display results.")
        return

    st.subheader("Per-Stock Forecasts and FMACD Signals")
    for ticker in portfolio.tickers:
        with st.container():
            render_stock_section(ticker=ticker, params=params)
            st.markdown("---")

    # Portfolio returns
    render_portfolio_returns(portfolio=portfolio, params=params)

    # Footer
    st.markdown("---")
    st.caption(
        "Powered by Streamlit, Plotly, PyTorch, scikit-learn, yfinance, and Pydantic. "
        "Data availability and accuracy are subject to provider limitations."
    )


if __name__ == "__main__":
    # Allow running as a standard Python script (useful for debugging outside Streamlit).
    # In Streamlit, this block is ignored.
    main()
