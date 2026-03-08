#!/usr/bin/env python3
"""
Financial Fraud Detection Suite
Multi-metric anomaly detection for any Yahoo Finance ticker.
"""

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

# Benford's Law expected first-digit distribution
BENFORD_EXPECTED = {d: np.log10(1 + 1 / d) for d in range(1, 10)}

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


@dataclass
class MetricResult:
    name: str
    score: float  # 0-100 risk score
    detail: str
    raw_value: float = 0.0


@dataclass
class FraudReport:
    ticker: str
    composite_score: float = 0.0
    risk_rating: str = ""
    metrics: list = field(default_factory=list)
    error: str = ""


def get_sp500_tickers() -> list[str]:
    try:
        tables = pd.read_html(SP500_URL)
        return sorted(tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist())
    except Exception:
        print("Warning: Could not fetch S&P 500 list, falling back to top holdings")
        return [
            "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "UNH",
            "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV",
            "LLY", "PEP", "KO", "COST", "AVGO", "WMT", "MCD", "CSCO", "TMO",
        ]


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def benford_test(financials: pd.DataFrame) -> MetricResult:
    """Test leading digit distribution of financial line items against Benford's Law."""
    values = []
    for col in financials.columns:
        for val in financials[col].dropna():
            try:
                v = abs(float(val))
                if v >= 1:
                    values.append(v)
            except (ValueError, TypeError):
                continue

    if len(values) < 20:
        return MetricResult("Benford's Law", 0, "Insufficient data", 0)

    first_digits = [int(str(v).lstrip("0").replace(".", "")[0]) for v in values if v > 0]
    first_digits = [d for d in first_digits if 1 <= d <= 9]

    if not first_digits:
        return MetricResult("Benford's Law", 0, "No valid digits", 0)

    observed_counts = np.array([first_digits.count(d) for d in range(1, 10)])
    observed_freq = observed_counts / observed_counts.sum()
    expected_freq = np.array([BENFORD_EXPECTED[d] for d in range(1, 10)])

    # Chi-squared test
    chi2, p_value = stats.chisquare(observed_counts, f_exp=expected_freq * len(first_digits))

    # Mean absolute deviation from expected
    mad = np.mean(np.abs(observed_freq - expected_freq))

    # Score: low p-value or high MAD = suspicious
    if p_value < 0.01:
        score = min(100, 50 + mad * 500)
    elif p_value < 0.05:
        score = min(80, 30 + mad * 400)
    else:
        score = min(40, mad * 300)

    detail = f"chi2={chi2:.1f}, p={p_value:.4f}, MAD={mad:.4f}, n={len(first_digits)}"
    return MetricResult("Benford's Law", round(score, 1), detail, mad)


def altman_z_score(info: dict, bs: pd.DataFrame, inc: pd.DataFrame) -> MetricResult:
    """Altman Z-Score for bankruptcy risk."""
    try:
        total_assets = _latest(bs, "Total Assets")
        total_liab = _latest(bs, "Total Liabilities Net Minority Interest",
                             "Total Liabilities", "Total Debt")
        working_cap = _latest(bs, "Working Capital", default=None)
        if working_cap is None:
            ca = _latest(bs, "Current Assets")
            cl = _latest(bs, "Current Liabilities")
            working_cap = ca - cl

        retained_earnings = _latest(bs, "Retained Earnings")
        ebit = _latest(inc, "EBIT", "Operating Income")
        revenue = _latest(inc, "Total Revenue")
        market_cap = info.get("marketCap", 0) or 0

        if total_assets == 0:
            return MetricResult("Altman Z-Score", 50, "Total assets = 0", 0)

        a = working_cap / total_assets
        b = retained_earnings / total_assets
        c = ebit / total_assets
        d = market_cap / total_liab if total_liab else 0
        e = revenue / total_assets

        z = 1.2 * a + 1.4 * b + 3.3 * c + 0.6 * d + 1.0 * e

        if z > 2.99:
            score = max(0, 20 - (z - 2.99) * 5)
            zone = "Safe"
        elif z > 1.81:
            score = 40 + (2.99 - z) * 20
            zone = "Grey"
        else:
            score = min(100, 70 + (1.81 - z) * 10)
            zone = "Distress"

        detail = f"Z={z:.2f} ({zone})"
        return MetricResult("Altman Z-Score", round(score, 1), detail, z)
    except Exception as e:
        return MetricResult("Altman Z-Score", 0, f"Unavailable: {e}", 0)


def beneish_m_score(bs: pd.DataFrame, inc: pd.DataFrame, cf: pd.DataFrame) -> MetricResult:
    """Beneish M-Score — likelihood of earnings manipulation."""
    try:
        cols = inc.columns.tolist()
        if len(cols) < 2:
            return MetricResult("Beneish M-Score", 0, "Need 2+ years", 0)

        curr, prev = cols[0], cols[1]

        rev_c = _val(inc, "Total Revenue", curr)
        rev_p = _val(inc, "Total Revenue", prev)
        cogs_c = _val(inc, "Cost Of Revenue", curr)
        cogs_p = _val(inc, "Cost Of Revenue", prev)
        sga_c = _val(inc, "Selling General And Administration", curr)
        sga_p = _val(inc, "Selling General And Administration", prev)
        dep_c = _val(inc, "Reconciled Depreciation", curr)
        dep_p = _val(inc, "Reconciled Depreciation", prev)
        ni_c = _val(inc, "Net Income", curr)
        ta_c = _val(bs, "Total Assets", curr)
        ta_p = _val(bs, "Total Assets", prev)
        ca_c = _val(bs, "Current Assets", curr)
        ca_p = _val(bs, "Current Assets", prev)
        ppe_c = _val(bs, "Net PPE", curr)
        ppe_p = _val(bs, "Net PPE", prev)
        recv_c = _val(bs, "Accounts Receivable", curr)
        recv_p = _val(bs, "Accounts Receivable", prev)
        cfo_c = _val(cf, "Operating Cash Flow", curr)

        if rev_p == 0 or ta_p == 0 or ta_c == 0:
            return MetricResult("Beneish M-Score", 0, "Division by zero avoided", 0)

        # DSRI — Days Sales in Receivables Index
        dsri = (recv_c / rev_c) / (recv_p / rev_p) if rev_c and rev_p and recv_p else 1

        # GMI — Gross Margin Index
        gm_c = (rev_c - cogs_c) / rev_c if rev_c else 0
        gm_p = (rev_p - cogs_p) / rev_p if rev_p else 0
        gmi = gm_p / gm_c if gm_c else 1

        # AQI — Asset Quality Index
        aq_c = 1 - (ca_c + ppe_c) / ta_c if ta_c else 0
        aq_p = 1 - (ca_p + ppe_p) / ta_p if ta_p else 0
        aqi = aq_c / aq_p if aq_p else 1

        # SGI — Sales Growth Index
        sgi = rev_c / rev_p if rev_p else 1

        # DEPI — Depreciation Index
        dep_rate_c = dep_c / (dep_c + ppe_c) if (dep_c + ppe_c) else 0
        dep_rate_p = dep_p / (dep_p + ppe_p) if (dep_p + ppe_p) else 0
        depi = dep_rate_p / dep_rate_c if dep_rate_c else 1

        # SGAI — SGA Index
        sga_rate_c = sga_c / rev_c if rev_c else 0
        sga_rate_p = sga_p / rev_p if rev_p else 0
        sgai = sga_rate_c / sga_rate_p if sga_rate_p else 1

        # TATA — Total Accruals to Total Assets
        tata = (ni_c - cfo_c) / ta_c if ta_c else 0

        # LVGI — Leverage Index
        tl_c = _val(bs, "Total Liabilities Net Minority Interest", curr)
        tl_p = _val(bs, "Total Liabilities Net Minority Interest", prev)
        lev_c = tl_c / ta_c if ta_c else 0
        lev_p = tl_p / ta_p if ta_p else 0
        lvgi = lev_c / lev_p if lev_p else 1

        m = (-4.84 + 0.920 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi
             + 0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi)

        if m > -1.78:
            score = min(100, 60 + (m + 1.78) * 15)
            label = "Likely Manipulator"
        else:
            score = max(0, 30 + (m + 1.78) * 10)
            label = "Unlikely Manipulator"

        detail = f"M={m:.2f} ({label})"
        return MetricResult("Beneish M-Score", round(score, 1), detail, m)
    except Exception as e:
        return MetricResult("Beneish M-Score", 0, f"Unavailable: {e}", 0)


def accrual_ratio(bs: pd.DataFrame, cf: pd.DataFrame) -> MetricResult:
    """High accruals relative to assets suggest aggressive accounting."""
    try:
        cols = bs.columns.tolist()
        if len(cols) < 2:
            return MetricResult("Accrual Ratio", 0, "Need 2+ years", 0)

        ta_c = _val(bs, "Total Assets", cols[0])
        ta_p = _val(bs, "Total Assets", cols[1])
        cfo = _val(cf, "Operating Cash Flow", cols[0])
        ni = _val(cf, "Net Income From Continuing Operations", cols[0])

        avg_ta = (ta_c + ta_p) / 2
        if avg_ta == 0:
            return MetricResult("Accrual Ratio", 0, "Zero assets", 0)

        accrual = (ni - cfo) / avg_ta
        score = min(100, max(0, abs(accrual) * 300))
        detail = f"Accrual ratio={accrual:.4f}"
        return MetricResult("Accrual Ratio", round(score, 1), detail, accrual)
    except Exception as e:
        return MetricResult("Accrual Ratio", 0, f"Unavailable: {e}", 0)


def cashflow_vs_income(inc: pd.DataFrame, cf: pd.DataFrame) -> MetricResult:
    """Compare operating cash flow to net income — persistent divergence is a red flag."""
    try:
        col = inc.columns[0]
        ni = _val(inc, "Net Income", col)
        cfo = _val(cf, "Operating Cash Flow", col)

        if ni == 0:
            return MetricResult("CF vs Net Income", 0, "Net income = 0", 0)

        ratio = cfo / ni
        if ratio < 0:
            score = min(100, 70 + abs(ratio) * 10)
            label = "NEGATIVE divergence"
        elif ratio < 0.5:
            score = 60
            label = "Low CF quality"
        elif ratio > 2.0:
            score = 30
            label = "Unusually high CF"
        else:
            score = max(0, (1 - ratio) * 50)
            label = "Normal"

        detail = f"CFO/NI={ratio:.2f} ({label})"
        return MetricResult("CF vs Net Income", round(max(score, 0), 1), detail, ratio)
    except Exception as e:
        return MetricResult("CF vs Net Income", 0, f"Unavailable: {e}", 0)


def revenue_receivables_divergence(bs: pd.DataFrame, inc: pd.DataFrame) -> MetricResult:
    """Revenue growing much faster than receivables (or vice versa) can signal channel stuffing."""
    try:
        cols = inc.columns.tolist()
        if len(cols) < 2:
            return MetricResult("Rev/Recv Divergence", 0, "Need 2+ years", 0)

        rev_c = _val(inc, "Total Revenue", cols[0])
        rev_p = _val(inc, "Total Revenue", cols[1])
        recv_c = _val(bs, "Accounts Receivable", cols[0])
        recv_p = _val(bs, "Accounts Receivable", cols[1])

        if rev_p == 0 or recv_p == 0:
            return MetricResult("Rev/Recv Divergence", 0, "Zero baseline", 0)

        rev_growth = (rev_c - rev_p) / abs(rev_p)
        recv_growth = (recv_c - recv_p) / abs(recv_p)
        gap = recv_growth - rev_growth

        score = min(100, max(0, abs(gap) * 150))
        detail = f"Rev growth={rev_growth:.1%}, Recv growth={recv_growth:.1%}, gap={gap:.1%}"
        return MetricResult("Rev/Recv Divergence", round(score, 1), detail, gap)
    except Exception as e:
        return MetricResult("Rev/Recv Divergence", 0, f"Unavailable: {e}", 0)


def gross_margin_volatility(inc: pd.DataFrame) -> MetricResult:
    """High volatility in gross margins over time may indicate manipulation."""
    try:
        revs, cogs_vals = [], []
        for col in inc.columns:
            r = _val(inc, "Total Revenue", col)
            c = _val(inc, "Cost Of Revenue", col)
            if r and r != 0:
                revs.append(r)
                cogs_vals.append(c)

        if len(revs) < 3:
            return MetricResult("Gross Margin Vol", 0, "Need 3+ years", 0)

        margins = [(r - c) / r for r, c in zip(revs, cogs_vals)]
        vol = np.std(margins)
        score = min(100, max(0, vol * 500))
        detail = f"GM std={vol:.4f}, margins={[f'{m:.1%}' for m in margins]}"
        return MetricResult("Gross Margin Vol", round(score, 1), detail, vol)
    except Exception as e:
        return MetricResult("Gross Margin Vol", 0, f"Unavailable: {e}", 0)


def sga_trend(inc: pd.DataFrame) -> MetricResult:
    """SGA-to-revenue ratio trending unusually may indicate expense manipulation."""
    try:
        ratios = []
        for col in inc.columns:
            rev = _val(inc, "Total Revenue", col)
            sga = _val(inc, "Selling General And Administration", col)
            if rev and rev != 0 and sga:
                ratios.append(sga / rev)

        if len(ratios) < 3:
            return MetricResult("SGA Trend", 0, "Need 3+ years", 0)

        trend = np.polyfit(range(len(ratios)), ratios, 1)[0]
        vol = np.std(ratios)
        score = min(100, max(0, (abs(trend) * 1000 + vol * 300)))
        detail = f"SGA/Rev trend slope={trend:.5f}, vol={vol:.4f}"
        return MetricResult("SGA Trend", round(score, 1), detail, trend)
    except Exception as e:
        return MetricResult("SGA Trend", 0, f"Unavailable: {e}", 0)


def debt_equity_trend(bs: pd.DataFrame) -> MetricResult:
    """Rapidly increasing leverage can mask financial weakness."""
    try:
        ratios = []
        for col in bs.columns:
            debt = _val(bs, "Total Debt", col)
            equity = _val(bs, "Stockholders Equity", col)
            if equity and equity != 0:
                ratios.append(debt / equity)

        if len(ratios) < 2:
            return MetricResult("D/E Trend", 0, "Need 2+ years", 0)

        change = ratios[0] - ratios[-1]  # most recent minus oldest
        score = min(100, max(0, change * 40))
        detail = f"D/E ratios={[f'{r:.2f}' for r in ratios]}, change={change:.2f}"
        return MetricResult("D/E Trend", round(score, 1), detail, change)
    except Exception as e:
        return MetricResult("D/E Trend", 0, f"Unavailable: {e}", 0)


def volume_anomalies(hist: pd.DataFrame) -> MetricResult:
    """Detect unusual volume spikes that may signal pump-and-dump."""
    try:
        if hist.empty or len(hist) < 60:
            return MetricResult("Volume Anomalies", 0, "Insufficient history", 0)

        vol = hist["Volume"].dropna()
        rolling_mean = vol.rolling(20).mean()
        rolling_std = vol.rolling(20).std()

        z_scores = (vol - rolling_mean) / rolling_std
        z_scores = z_scores.dropna()

        extreme_days = (z_scores.abs() > 3).sum()
        pct_extreme = extreme_days / len(z_scores) * 100

        score = min(100, max(0, pct_extreme * 15))
        detail = f"{extreme_days} extreme volume days ({pct_extreme:.1f}% of trading days)"
        return MetricResult("Volume Anomalies", round(score, 1), detail, pct_extreme)
    except Exception as e:
        return MetricResult("Volume Anomalies", 0, f"Unavailable: {e}", 0)


def price_return_anomalies(hist: pd.DataFrame) -> MetricResult:
    """Flag statistically unusual daily returns."""
    try:
        if hist.empty or len(hist) < 60:
            return MetricResult("Return Anomalies", 0, "Insufficient history", 0)

        returns = hist["Close"].pct_change().dropna()
        # Test for normality
        if len(returns) > 20:
            _, p_normal = stats.normaltest(returns)
        else:
            p_normal = 1.0

        kurtosis = stats.kurtosis(returns)
        skew = stats.skew(returns)

        extreme = (returns.abs() > returns.std() * 3).sum()
        pct_extreme = extreme / len(returns) * 100

        score = min(100, max(0, pct_extreme * 12 + max(0, kurtosis - 3) * 5))
        detail = (f"kurtosis={kurtosis:.2f}, skew={skew:.2f}, "
                  f"extreme_days={extreme}, normality_p={p_normal:.4f}")
        return MetricResult("Return Anomalies", round(score, 1), detail, kurtosis)
    except Exception as e:
        return MetricResult("Return Anomalies", 0, f"Unavailable: {e}", 0)


def audit_signals(info: dict) -> MetricResult:
    """Check for audit-related red flags from available info."""
    score = 0
    flags = []

    # Check audit risk from info fields
    ar = info.get("auditRisk")
    if ar is not None and ar > 7:
        score += 30
        flags.append(f"auditRisk={ar}")

    # Overall risk
    overall = info.get("overallRisk")
    if overall is not None and overall > 7:
        score += 20
        flags.append(f"overallRisk={overall}")

    # Governance risk
    gov = info.get("governanceEpochDate")
    if gov is None:
        score += 5
        flags.append("No governance data")

    # Short interest
    short_pct = info.get("shortPercentOfFloat")
    if short_pct and short_pct > 0.10:
        score += 15
        flags.append(f"shortFloat={short_pct:.1%}")

    score = min(100, score)
    detail = "; ".join(flags) if flags else "No audit flags"
    return MetricResult("Audit Signals", round(score, 1), detail, score)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _latest(df: pd.DataFrame, *keys, default=0):
    """Get the most recent value for the first matching key."""
    for key in keys:
        if key in df.index:
            val = df.loc[key].dropna()
            if not val.empty:
                v = val.iloc[0]
                try:
                    return float(v)
                except (ValueError, TypeError):
                    continue
    if default is None:
        raise KeyError(f"None of {keys} found")
    return default


def _val(df: pd.DataFrame, key: str, col, default=0):
    """Get a specific value from a DataFrame."""
    try:
        if key in df.index and col in df.columns:
            v = df.loc[key, col]
            if pd.notna(v):
                return float(v)
    except (KeyError, TypeError, ValueError):
        pass
    return default


def rate_risk(score: float) -> str:
    if score >= 70:
        return "CRITICAL"
    elif score >= 50:
        return "HIGH"
    elif score >= 30:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_ticker(ticker_symbol: str) -> FraudReport:
    """Run all fraud detection metrics on a single ticker."""
    report = FraudReport(ticker=ticker_symbol)

    try:
        tk = yf.Ticker(ticker_symbol)
        info = tk.info or {}

        # Fetch financial statements
        inc = tk.financials
        bs = tk.balance_sheet
        cf = tk.cashflow
        hist = tk.history(period="2y")

        # Combine all financial data for Benford's test
        all_financials = pd.concat([inc, bs, cf], axis=0) if not any(
            df is None or df.empty for df in [inc, bs, cf]) else pd.DataFrame()

        # Run all metrics
        if not all_financials.empty:
            report.metrics.append(benford_test(all_financials))

        if bs is not None and not bs.empty and inc is not None and not inc.empty:
            report.metrics.append(altman_z_score(info, bs, inc))
            report.metrics.append(revenue_receivables_divergence(bs, inc))
            report.metrics.append(debt_equity_trend(bs))

        if (bs is not None and not bs.empty and inc is not None and not inc.empty
                and cf is not None and not cf.empty):
            report.metrics.append(beneish_m_score(bs, inc, cf))

        if bs is not None and not bs.empty and cf is not None and not cf.empty:
            report.metrics.append(accrual_ratio(bs, cf))

        if inc is not None and not inc.empty and cf is not None and not cf.empty:
            report.metrics.append(cashflow_vs_income(inc, cf))

        if inc is not None and not inc.empty:
            report.metrics.append(gross_margin_volatility(inc))
            report.metrics.append(sga_trend(inc))

        if not hist.empty:
            report.metrics.append(volume_anomalies(hist))
            report.metrics.append(price_return_anomalies(hist))

        report.metrics.append(audit_signals(info))

        # Composite score — weighted average
        scored = [m for m in report.metrics if m.score > 0 or m.detail != "Insufficient data"]
        if scored:
            # Weight fundamental metrics higher than market-based ones
            weights = {
                "Benford's Law": 2.0,
                "Beneish M-Score": 2.0,
                "Altman Z-Score": 1.5,
                "Accrual Ratio": 1.5,
                "CF vs Net Income": 1.5,
                "Rev/Recv Divergence": 1.2,
                "Gross Margin Vol": 1.0,
                "SGA Trend": 1.0,
                "D/E Trend": 1.0,
                "Volume Anomalies": 0.8,
                "Return Anomalies": 0.8,
                "Audit Signals": 1.2,
            }
            total_w = sum(weights.get(m.name, 1.0) for m in scored)
            weighted_sum = sum(m.score * weights.get(m.name, 1.0) for m in scored)
            report.composite_score = round(weighted_sum / total_w, 1)
        else:
            report.composite_score = 0

        report.risk_rating = rate_risk(report.composite_score)

    except Exception as e:
        report.error = str(e)
        report.risk_rating = "ERROR"

    return report


def print_report(report: FraudReport):
    """Pretty-print a single ticker report."""
    color = {
        "LOW": "\033[92m",       # green
        "MEDIUM": "\033[93m",    # yellow
        "HIGH": "\033[91m",      # red
        "CRITICAL": "\033[95m",  # magenta
        "ERROR": "\033[90m",     # grey
    }
    reset = "\033[0m"
    c = color.get(report.risk_rating, "")

    print(f"\n{'='*70}")
    print(f"  {report.ticker:8s}  |  Composite Risk: {c}{report.composite_score:5.1f}/100  "
          f"[{report.risk_rating}]{reset}")
    print(f"{'='*70}")

    if report.error:
        print(f"  ERROR: {report.error}")
        return

    for m in sorted(report.metrics, key=lambda x: x.score, reverse=True):
        bar_len = int(m.score / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {m.name:25s} {m.score:5.1f}  {bar}  {m.detail}")


def report_to_dict(report: FraudReport) -> dict:
    return {
        "ticker": report.ticker,
        "composite_score": report.composite_score,
        "risk_rating": report.risk_rating,
        "error": report.error,
        "metrics": [
            {"name": m.name, "score": m.score, "detail": m.detail, "raw_value": m.raw_value}
            for m in report.metrics
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Financial Fraud Detection Suite")
    parser.add_argument("tickers", nargs="*", help="Ticker symbols to analyze")
    parser.add_argument("--sp500", action="store_true", help="Scan all S&P 500 constituents")
    parser.add_argument("--file", type=str, help="File with one ticker per line")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--threshold", type=float, default=50,
                        help="Only show tickers above this risk score (default: 50)")
    args = parser.parse_args()

    tickers = list(args.tickers)

    if args.sp500:
        tickers.extend(get_sp500_tickers())

    if args.file:
        with open(args.file) as f:
            tickers.extend(line.strip() for line in f if line.strip())

    if not tickers:
        parser.print_help()
        sys.exit(1)

    tickers = list(dict.fromkeys(tickers))  # deduplicate, preserve order

    if not args.json:
        print(f"\nAnalyzing {len(tickers)} ticker(s)...\n")

    reports = []
    for i, t in enumerate(tickers, 1):
        if not args.json:
            print(f"[{i}/{len(tickers)}] {t}...", end="", flush=True)
        report = analyze_ticker(t)
        reports.append(report)
        if not args.json:
            c = "\033[91m" if report.composite_score >= args.threshold else "\033[92m"
            print(f" {c}{report.composite_score:.1f}{'\033[0m'}")

    if args.json:
        output = [report_to_dict(r) for r in reports]
        print(json.dumps(output, indent=2))
    else:
        # Print detailed reports for flagged tickers
        flagged = [r for r in reports if r.composite_score >= args.threshold]
        clean = [r for r in reports if r.composite_score < args.threshold and not r.error]
        errors = [r for r in reports if r.error]

        if flagged:
            print(f"\n{'#'*70}")
            print(f"  FLAGGED TICKERS (score >= {args.threshold}): {len(flagged)}")
            print(f"{'#'*70}")
            for r in sorted(flagged, key=lambda x: x.composite_score, reverse=True):
                print_report(r)
        else:
            print(f"\nNo tickers exceeded the risk threshold ({args.threshold}).")

        print(f"\n--- Summary: {len(flagged)} flagged, {len(clean)} clean, "
              f"{len(errors)} errors out of {len(reports)} analyzed ---\n")


if __name__ == "__main__":
    main()
