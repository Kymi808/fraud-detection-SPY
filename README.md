# Financial Fraud Detection Suite

Multi-metric fraud/anomaly detection for any stock or index available on Yahoo Finance. Combines statistical tests including Benford's Law, Altman Z-Score, earnings quality metrics, and more to produce a composite fraud risk score.

## Metrics

| # | Metric | What it detects |
|---|--------|----------------|
| 1 | **Benford's Law** | Digit distribution anomalies in financial figures |
| 2 | **Altman Z-Score** | Bankruptcy / financial distress risk |
| 3 | **Beneish M-Score** | Earnings manipulation likelihood |
| 4 | **Accrual Ratio** | Aggressive revenue recognition |
| 5 | **Cash Flow vs Net Income** | Earnings quality divergence |
| 6 | **Revenue/Receivables Divergence** | Channel stuffing signals |
| 7 | **Gross Margin Volatility** | Inconsistent cost reporting |
| 8 | **SGA-to-Revenue Trend** | Expense manipulation |
| 9 | **Debt-to-Equity Trend** | Hidden leverage build-up |
| 10 | **Volume Anomalies** | Unusual trading activity (pump & dump signals) |
| 11 | **Price Return Anomalies** | Statistical outlier detection via z-score |
| 12 | **Audit / Filing Signals** | Late filings, restatements (where available) |

## Quick Start

```bash
# Single ticker
python3 fraud_detector.py SPY

# Multiple tickers
python3 fraud_detector.py AAPL MSFT TSLA ENRON

# Full S&P 500 scan
python3 fraud_detector.py --sp500

# Custom tickers from file (one per line)
python3 fraud_detector.py --file tickers.txt

# JSON output
python3 fraud_detector.py AAPL --json

# Set risk threshold for alerts (0-100, default 50)
python3 fraud_detector.py --sp500 --threshold 60
```

## Output

Each ticker receives a **composite risk score (0–100)** with per-metric breakdowns and a risk rating (LOW / MEDIUM / HIGH / CRITICAL).

## Requirements

```
pip install yfinance numpy pandas scipy
```

## Disclaimer

This tool is for **educational and research purposes only**. A high risk score does not prove fraud — it flags statistical anomalies worthy of further investigation. Always consult qualified professionals before making financial decisions.
