"""
selector_bonds.py - asset selector for BOND ETFs (course chapter 9: asset selection)

What it does
  1. downloads 3 years of daily prices (Yahoo Finance) and drops assets with less than 3 years of data
  2. computes the metrics: yield, duration and volatility (252 days)
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
    "BIL": "Cash & short Treasuries", "SGOV": "Cash & short Treasuries", "SHY": "Cash & short Treasuries",
    "VGSH": "Cash & short Treasuries", "SCHO": "Cash & short Treasuries", "IEI": "Intermediate & long Treasuries",
    "VGIT": "Intermediate & long Treasuries", "IEF": "Intermediate & long Treasuries",
    "GOVT": "Intermediate & long Treasuries", "TLH": "Intermediate & long Treasuries",
    "TLT": "Intermediate & long Treasuries", "VGLT": "Intermediate & long Treasuries",
    "AGG": "IG credit & aggregate", "BND": "IG credit & aggregate", "LQD": "IG credit & aggregate",
    "VCSH": "IG credit & aggregate", "VCIT": "IG credit & aggregate", "IGSB": "IG credit & aggregate",
    "MBB": "IG credit & aggregate", "MUB": "IG credit & aggregate", "TIP": "Inflation-linked",
    "VTIP": "Inflation-linked", "SCHP": "Inflation-linked", "STIP": "Inflation-linked",
    "BNDX": "International & high-yield bonds", "EMB": "International & high-yield bonds",
    "HYG": "International & high-yield bonds", "SJNK": "International & high-yield bonds",
}   # ticker -> tag (sector / role / family)

BENCH = "SPY"        # market used for the beta (S&P 500 proxy)
N_KEEP = 10          # number of assets kept
MAX_CORR = 0.95      # correlation above which an asset is considered redundant
PROFILE = "growth"   # "growth" = Velocity Fund (course weights) | "defensive" = Serenity Fund
WEIGHTS = {"growth": {"yield": 0.4, "dur": -0.3, "vol": -0.3},
           "defensive": {"yield": 0.3, "dur": -0.4, "vol": -0.3}}[PROFILE]

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
# bond ETFs: trailing yield (proxy for the yield to maturity) and empirical duration
m["yield"] = pd.Series({t: (yf.Ticker(t).info or {}).get("yield") for t in assets}, dtype=float)
dy = px["^TNX"].diff().loc[rets.index] / 100                       # daily change of the 10-year yield
m["dur"] = -rets[assets].apply(lambda r: r.cov(dy)) / dy.var()     # duration = - slope of returns on yield changes

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
show["vol"] = show["vol"].map(pct)
show["yield"] = show["yield"].map(lambda x: "n/a" if pd.isna(x) else f"{x:.2%}")
show["dur"] = show["dur"].map("{:.1f}".format)
print(f"\n===== BOND ETFs: the {len(selected)} selected assets (profile: {PROFILE}) =====")
print(show.to_string())
print("\nRejected as redundant (asset, better asset kept, correlation):")
for t, s, c in rejected[:10]:
    print(f"  {t:<10} ~ {s:<10} {c:.2f}")
print("\nTags of the selection:", res.loc[selected, "tag"].value_counts().to_dict())
print("Average correlation between the selected assets:",
      round(float(corr.loc[selected, selected].values[np.triu_indices(len(selected), 1)].mean()), 2))

out = (Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()) / "selection_bonds.csv"
res.to_csv(out, float_format="%.4f")
print(f"\nFull ranking saved in {out.name}")
