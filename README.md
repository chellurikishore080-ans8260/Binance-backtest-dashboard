# Double SuperTrend Binance Spot Backtest

Streamlit dashboard for backtesting the Tanmay Kurtkoti Double SuperTrend rules on Binance spot candle data from Binance Vision.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

No Binance API keys are needed. The dashboard reads public Binance market data only and does not place live orders. Configure timeframe, dates, current Top USDT spot pairs, SuperTrend parameters, fixed USD notional per entry, and assumed fees in the sidebar.

## Deploy on Streamlit Community Cloud

1. Ensure this branch is pushed to GitHub (it is: `copilot/binance-vision-double-supertrend`).
2. Open Streamlit Community Cloud and select **Create app**.
3. Choose this repository: `chellurikishore080-ans8260/Binance-backtest-dashboard`.
4. Select branch `copilot/binance-vision-double-supertrend` and main file path `app.py`.
5. Click **Deploy**. Streamlit installs the dependencies from `requirements.txt` automatically.

If you later merge this branch into `main`, you can deploy `main` instead. The previous SMC app remains unchanged on the original `main` branch until you merge these changes.

## Important backtest limitations

- The Top 50 universe is ranked using the current 24-hour USDT quote volume. Applying it retrospectively introduces survivorship bias; historical membership is not reconstructed.
- Signals use completed candle values and execute at the following candle's open. Open positions are marked to the final close of the selected period.
- Fees are estimated at the configured rate on both entry and exit. Slippage, taxes, minimum notional/lot sizes, and exchange execution constraints are not modeled.
- Each coin position uses the configured fixed notional independently. Holding positions across many coins can require substantially more capital than one trade size.
- Short mode is provided only for strategy research; ordinary spot trading does not support borrowing/shorting by default.
- Results are hypothetical and are not investment advice.
