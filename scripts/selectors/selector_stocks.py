"""
selector_stocks.py - asset selector for STOCKS (course chapter 9: asset selection)

What it does
  1. downloads 3 years of daily prices (Yahoo Finance) and drops assets with less than 3 years of data
  2. computes the metrics: momentum P(t-21)/P(t-273)-1, volatility and beta over 252 days
  3. converts them into z-scores and a composite score = sum of weight x z-score
  4. removes redundant assets: an asset is rejected if its correlation with a better-ranked asset already
     kept is above MAX_CORR
  5. keeps the best N_KEEP assets, splits the ranking in two categories and saves everything in a CSV

Run it in Spyder: F5 for everything, or cell by cell with Ctrl+Enter.   pip install yfinance pandas numpy
"""
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

# %% 1. Parameters (change them here)
TICKERS = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", "AVGO": "Technology", "ORCL": "Technology",
    "CRM": "Technology", "AMD": "Technology", "ADBE": "Technology", "CSCO": "Technology", "IBM": "Technology",
    "QCOM": "Technology", "TXN": "Technology", "INTU": "Technology", "NOW": "Technology", "AMAT": "Technology",
    "LRCX": "Technology", "KLAC": "Technology", "MU": "Technology", "ANET": "Technology", "PANW": "Technology",
    "CRWD": "Technology", "PLTR": "Technology", "APH": "Technology", "MSI": "Technology", "FICO": "Technology",
    "APP": "Technology", "SNPS": "Technology", "CDNS": "Technology", "ADI": "Technology", "SHOP": "Technology",
    "GOOGL": "Communication", "META": "Communication", "NFLX": "Communication", "TMUS": "Communication",
    "DIS": "Communication", "VZ": "Communication", "T": "Communication", "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary", "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary",
    "BKNG": "Consumer Discretionary", "UBER": "Consumer Discretionary", "TJX": "Consumer Discretionary",
    "ROST": "Consumer Discretionary", "ORLY": "Consumer Discretionary", "AZO": "Consumer Discretionary",
    "CMG": "Consumer Discretionary", "RCL": "Consumer Discretionary", "NKE": "Consumer Discretionary",
    "LOW": "Consumer Discretionary", "WMT": "Consumer Staples", "COST": "Consumer Staples", "PG": "Consumer Staples",
    "KO": "Consumer Staples", "PEP": "Consumer Staples", "PM": "Consumer Staples", "MDLZ": "Consumer Staples",
    "MO": "Consumer Staples", "KMB": "Consumer Staples", "GIS": "Consumer Staples", "HSY": "Consumer Staples",
    "KR": "Consumer Staples", "CL": "Consumer Staples", "BRK-B": "Financials", "JPM": "Financials",
    "V": "Financials", "MA": "Financials", "BAC": "Financials", "WFC": "Financials", "GS": "Financials",
    "MS": "Financials", "AXP": "Financials", "BLK": "Financials", "SCHW": "Financials", "SPGI": "Financials",
    "ICE": "Financials", "CME": "Financials", "PGR": "Financials", "CB": "Financials", "MMC": "Financials",
    "KKR": "Financials", "IBKR": "Financials", "HOOD": "Financials", "AJG": "Financials", "LLY": "Health Care",
    "UNH": "Health Care", "JNJ": "Health Care", "ABBV": "Health Care", "MRK": "Health Care", "TMO": "Health Care",
    "ABT": "Health Care", "DHR": "Health Care", "ISRG": "Health Care", "VRTX": "Health Care", "REGN": "Health Care",
    "BSX": "Health Care", "SYK": "Health Care", "MCK": "Health Care", "COR": "Health Care", "CI": "Health Care",
    "AMGN": "Health Care", "GILD": "Health Care", "GE": "Industrials", "CAT": "Industrials", "RTX": "Industrials",
    "LMT": "Industrials", "NOC": "Industrials", "GD": "Industrials", "HON": "Industrials", "UNP": "Industrials",
    "ETN": "Industrials", "PH": "Industrials", "TT": "Industrials", "CTAS": "Industrials", "TDG": "Industrials",
    "HWM": "Industrials", "AXON": "Industrials", "URI": "Industrials", "PWR": "Industrials", "WM": "Industrials",
    "RSG": "Industrials", "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "WMB": "Energy", "KMI": "Energy",
    "OKE": "Energy", "TRGP": "Energy", "NEE": "Utilities", "SO": "Utilities", "DUK": "Utilities", "CEG": "Utilities",
    "VST": "Utilities", "NRG": "Utilities", "ED": "Utilities", "XEL": "Utilities", "WEC": "Utilities",
    "AEP": "Utilities", "LIN": "Materials", "NEM": "Materials", "FCX": "Materials", "ECL": "Materials",
}   # ticker -> tag (sector / role / family)

BENCH = "SPY"        # market used for the beta (S&P 500 proxy)
N_KEEP = 15          # number of assets kept
MAX_CORR = 0.75      # correlation above which an asset is considered redundant
PROFILE = "growth"   # "growth" = Velocity Fund (course weights) | "defensive" = Serenity Fund
WEIGHTS = {"growth": {"mom": 0.5, "vol": -0.3, "beta": -0.2},
           "defensive": {"mom": 0.2, "vol": -0.5, "beta": -0.3}}[PROFILE]

# %% 2. Data: 3 years of daily prices
extra = ["^TNX"] if "dur" in WEIGHTS else []                       # 10-year yield, for the duration
px = yf.download(list(TICKERS) + [BENCH] + extra, period="3y", auto_adjust=True, progress=False)["Close"]
px.index = pd.to_datetime(px.index).tz_localize(None)
px = px[px[BENCH].notna()]                                         # US trading days only
first = px.apply(lambda s: s.first_valid_index())
short = [t for t in first[first > px.index[0] + pd.Timedelta(days=10)].index if t != BENCH]
if short:
    print("Excluded (less than 3 years of data):", short)
px = px.drop(columns=short).dropna(axis=1, how="all").ffill(limit=3).dropna()
rets = px.pct_change().dropna()
assets = [t for t in TICKERS if t in px.columns]
print(f"{len(assets)} assets | {px.index[0]:%Y-%m-%d} -> {px.index[-1]:%Y-%m-%d} | {len(px)} trading days")

# %% 3. Metrics
m = pd.DataFrame(index=assets)
m["tag"] = pd.Series(TICKERS)
m["mom"] = px[assets].iloc[-22] / px[assets].iloc[-274] - 1        # momentum = P(t-21) / P(t-273) - 1
m["vol"] = rets[assets].iloc[-252:].std() * np.sqrt(252)           # volatility = std(daily returns, 252 d) x sqrt(252)
bench = rets[BENCH].iloc[-252:]
m["beta"] = rets[assets].iloc[-252:].apply(lambda r: r.cov(bench)) / bench.var()   # beta = cov(r, r_bench) / var(r_bench)

# %% 4. Z-scores and composite score
z = pd.DataFrame({k: (m[k] - m[k].mean()) / m[k].std() for k in WEIGHTS})   # z = (x - mean) / std
m["score"] = sum(w * z[k].fillna(0) for k, w in WEIGHTS.items())
res = m.join(z.add_prefix("z_")).sort_values("score", ascending=False)
res.insert(0, "rank", range(1, len(res) + 1))
res["category"] = np.where(res["rank"] <= len(res) / 2, "Top half", "Bottom half")   # two categories by ranking

# %% 5. Remove redundant assets and keep the best
corr = rets[res.index].corr()
selected, rejected = [], []
for t in res.index:                                                # best score first
    if len(selected) >= N_KEEP:
        break
    hits = [(s, corr.at[t, s]) for s in selected if corr.at[t, s] > MAX_CORR]
    if hits:
        rejected.append((t,) + max(hits, key=lambda h: h[1]))     # (asset, kept asset it duplicates, correlation)
    else:
        selected.append(t)
res["selected"] = res.index.isin(selected)

# %% 6. Results
pct = lambda x: "n/a" if pd.isna(x) else f"{x:.1%}"
show = res.loc[selected, ["rank", "tag", "score"] + list(WEIGHTS)].copy()
show["score"] = show["score"].map("{:+.2f}".format)
for k in ("mom", "vol"):
    show[k] = show[k].map(pct)
show["beta"] = show["beta"].map("{:.2f}".format)
print(f"\n===== STOCKS: the {len(selected)} selected assets (profile: {PROFILE}) =====")
print(show.to_string())
print("\nRejected as redundant (asset, better asset kept, correlation):")
for t, s, c in rejected[:10]:
    print(f"  {t:<10} ~ {s:<10} {c:.2f}")
print("\nTags of the selection:", res.loc[selected, "tag"].value_counts().to_dict())
print("Average correlation between the selected assets:",
      round(float(corr.loc[selected, selected].values[np.triu_indices(len(selected), 1)].mean()), 2))

out = (Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()) / "selection_stocks.csv"
res.to_csv(out, float_format="%.4f")
print(f"\nFull ranking saved in {out.name}")
