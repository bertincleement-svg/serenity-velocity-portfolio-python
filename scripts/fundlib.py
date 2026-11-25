"""
fundlib.py - shared engine for the Serenity and Velocity funds.

For one fund it:
  1. downloads 3 years of prices (Yahoo Finance) and fundamentals
  2. computes indicators and z-scores within each asset group
  3. builds base-100 and daily-return charts
  4. runs a Markowitz-Sharpe analysis and a constrained optimiser
  5. computes beta / alpha against the fund's benchmark
  6. computes the 10-day 99% historical VaR on $1bn
  7. writes the dashboard tiles (img/<n>/k-1.html), extra charts,
     the gauge and the analytics pages (analytics/<n>/k.html),
     including news on the 3 largest holdings.
"""
from __future__ import annotations

import datetime as dt
import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from scipy.optimize import minimize
from scipy.stats import norm

import config as C

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
DATA = SCRIPTS / "data"
CAPITAL = 1_000_000_000
TDAYS = 252
EQUITY_TYPES = {"equity", "equity_etf"}

PINK, PURPLE, GREEN, GREY = "#e2b857", "#35c6b5", "#6fcf97", "#b8c0cc"   # PINK = or, PURPLE = turquoise
GROUP_COLORS = ["#35c6b5", "#e2b857", "#5b8def", "#e8795a", "#9ad06b"]


# ---------------------------------------------------------------- data ----
def load_prices(refresh=False):
    """Daily adjusted closes for every ticker of both funds (cached for the day)."""
    DATA.mkdir(exist_ok=True)
    f = DATA / "prices.csv"
    wanted = C.all_tickers()
    if f.exists() and not refresh:
        cached = pd.read_csv(f, index_col=0, parse_dates=True)
        same_day = dt.date.fromtimestamp(f.stat().st_mtime) == dt.date.today()
        if same_day and set(wanted) <= set(cached.columns):
            print(f"Using cached prices ({f.name}, {cached.index[0]:%Y-%m-%d} -> {cached.index[-1]:%Y-%m-%d})")
            return cached[wanted]
    print(f"Downloading 3 years of prices for {len(wanted)} tickers from Yahoo Finance...")
    raw = yf.download(wanted, period="3y", auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(wanted[0])
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    missing = [t for t in wanted if t not in raw.columns or raw[t].dropna().empty]
    if missing:
        raise RuntimeError(f"No Yahoo data for: {missing}. Change these tickers in config.py.")
    raw = raw[wanted]
    raw.to_csv(f)
    return raw


def load_info(tickers, refresh=False):
    """A few fundamentals per ticker (ROE, yield, name), cached in data/info.json."""
    f = DATA / "info.json"
    info = {} if refresh or not f.exists() else json.loads(f.read_text())
    keep = ("shortName", "longName", "returnOnEquity", "trailingPE",
            "yield", "dividendYield", "quoteType")
    todo = [t for t in tickers if t not in info]
    if todo:
        print(f"Fetching fundamentals for {len(todo)} tickers...")
    for t in todo:
        try:
            i = yf.Ticker(t).info or {}
        except Exception as e:  # network hiccup: keep going
            print(f"  ! no fundamentals for {t}: {e}")
            continue
        if i:
            info[t] = {k: i.get(k) for k in keep}
    f.write_text(json.dumps(info, indent=1))
    return info


def fetch_news(ticker, n=4):
    """Latest Yahoo Finance news for a ticker (handles old and new yfinance formats)."""
    items = []
    try:
        items = yf.Ticker(ticker).news or []
    except Exception as e:
        print(f"  ! Ticker.news failed for {ticker}: {e}")
    if not items:                      # course exercise 9.2: yf.Search(ticker, news_count=n).news
        try:
            items = yf.Search(ticker, news_count=n).news or []
        except Exception as e:
            print(f"  ! no news for {ticker}: {e}")
            return []
    out = []
    for it in items:
        c = it.get("content", it) if isinstance(it, dict) else {}
        title = c.get("title")
        url = ((c.get("canonicalUrl") or {}).get("url")
               or (c.get("clickThroughUrl") or {}).get("url") or c.get("link"))
        publisher = (c.get("provider") or {}).get("displayName") or c.get("publisher") or ""
        date = c.get("pubDate") or c.get("displayTime") or ""
        if not date and c.get("providerPublishTime"):
            date = dt.datetime.fromtimestamp(c["providerPublishTime"]).isoformat()
        th = c.get("thumbnail") or {}
        res = th.get("resolutions") or [{}]
        thumb = th.get("originalUrl") or res[0].get("url")
        if title and url:
            out.append({"title": title, "url": url, "publisher": publisher,
                        "date": str(date)[:10], "image": thumb,
                        "summary": c.get("summary") or c.get("description") or ""})
        if len(out) >= n:
            break
    return out


def risk_free(prices):
    """Annualised 13-week T-bill rate (last close of ^IRX, in decimal)."""
    return float(prices[C.IRX].dropna().iloc[-1]) / 100


# ---------------------------------------------------------- indicators ----
METRICS = {   # indicator, sign (+1 higher is better / -1 lower is better / key in cfg)
    "equity":     [("Momentum", +1), ("Volatility", -1), ("Beta", "beta_sign"), ("ROE", +1)],
    "equity_etf": [("Momentum", +1), ("Volatility", -1), ("Beta", "beta_sign"), ("Yield", +1)],
    "bond":       [("Yield", +1), ("Duration", "duration_sign"), ("Volatility", -1), ("Momentum", +1)],
    "other":      [("Momentum", +1), ("Volatility", -1), ("Max drawdown", +1), ("Corr. S&P 500", -1)],
}
METRIC_FMT = {"Momentum": "pct", "Volatility": "pct", "Beta": "num", "ROE": "pct",
              "Yield": "pct", "Duration": "num", "Max drawdown": "pct", "Corr. S&P 500": "num"}
METRIC_DESC = {
    "Momentum": "P(t-21)/P(t-273) - 1: price change over a 13-month window, lagged by one month",
    "Volatility": "annualised standard deviation of daily returns over the last 252 trading days",
    "Beta": "cov(r, r_S&P 500) / var(r_S&P 500) over the last 252 trading days (SPY as S&P 500 proxy)",
    "ROE": "return on equity (Yahoo Finance fundamentals)",
    "Yield": "trailing distribution yield, used as a proxy for yield to maturity",
    "Duration": "empirical duration: -(regression slope of daily returns on daily changes in the 10y Treasury yield)",
    "Max drawdown": "worst peak-to-trough loss over 3 years",
    "Corr. S&P 500": "correlation of daily returns with the S&P 500",
}


def _yield(i):
    y = i.get("yield")
    if y is None:
        y = i.get("dividendYield")
        if y is not None and y > 0.25:     # some yfinance versions return percent
            y = y / 100
    return y


def indicators(px, rets, spx, dy, info):
    rows = {}
    for t in px.columns:
        p, r, i = px[t], rets[t], info.get(t, {})
        rows[t] = {
            "Momentum": p.iloc[-22] / p.iloc[-274] - 1 if len(p) > 274 else np.nan,
            "Volatility": r.iloc[-TDAYS:].std() * math.sqrt(TDAYS),
            "Beta": r.iloc[-TDAYS:].cov(spx.iloc[-TDAYS:]) / spx.iloc[-TDAYS:].var(),
            "ROE": i.get("returnOnEquity"),
            "Yield": _yield(i),
            "Duration": -r.cov(dy) / dy.var(),
            "Max drawdown": (p / p.cummax() - 1).min(),
            "Corr. S&P 500": r.corr(spx),
        }
    return pd.DataFrame(rows).T.astype(float)


def zscores(cfg, ind):
    """z-score of every indicator inside its group; Score = mean of oriented z-scores."""
    raw, zs, meta = {}, {}, {}
    for group, (gtype, ts) in cfg["groups"].items():
        for metric, sign in METRICS[gtype]:
            s = cfg[sign] if isinstance(sign, str) else sign
            x = ind.loc[ts, metric]
            sd = x.std()
            z = (x - x.mean()) / sd if x.notna().sum() >= 2 and sd > 0 else x * np.nan
            for t in ts:
                raw.setdefault(t, {})[metric] = x[t]
                zs.setdefault(t, {})[metric] = s * z[t]   # oriented: + is "good"
        for t in ts:
            meta[t] = (group, gtype)
    cols = list(dict.fromkeys(m for g in cfg["groups"].values() for m, _ in METRICS[g[0]]))
    raw = pd.DataFrame(raw).T[cols]
    zs = pd.DataFrame(zs).T[cols]
    out = pd.DataFrame({"Group": {t: m[0] for t, m in meta.items()},
                        "Type": {t: m[1] for t, m in meta.items()}})
    out = out.join(raw).join(zs.add_prefix("z "))
    out["Score"] = zs.mean(axis=1)
    out["Rank in group"] = out.groupby("Group")["Score"].rank(ascending=False).astype("Int64")
    return out, cols


# ----------------------------------------------------------- portfolio ----
def perf(w, mu, cov, rf):
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    return ret, vol, (ret - rf) / vol


def optimise(cfg, mu, cov, rf, eq_mask, symbols):
    n = len(mu)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if cfg.get("min_equity") is not None:
        cons.append({"type": "ineq", "fun": lambda w: w @ eq_mask - cfg["min_equity"]})
    if cfg.get("max_equity") is not None:
        cons.append({"type": "ineq", "fun": lambda w: cfg["max_equity"] - w @ eq_mask})
    if cfg.get("min_excess_return") is not None:
        cons.append({"type": "ineq", "fun": lambda w: w @ mu - rf - cfg["min_excess_return"]})
    if cfg.get("max_vol") is not None:
        cons.append({"type": "ineq", "fun": lambda w: cfg["max_vol"] ** 2 - w @ cov @ w})
    for sec, c in cfg.get("sector_caps", {}).items():          # course 9.3: maximum exposure per sector
        m = np.array([1.0 if cfg.get("sectors", {}).get(t) == sec else 0.0 for t in symbols])
        if m.sum() > 0:
            cons.append({"type": "ineq", "fun": lambda w, m=m, c=c: c - w @ m})
    mw = cfg.get("min_weight", 0.0)
    bounds = [(mw, cfg["max_weight"])] * n

    def neg_sharpe(w):
        return -(w @ mu - rf) / np.sqrt(w @ cov @ w)

    def feasible(w, tol=1e-4):
        return all((c["fun"](w) >= -tol) if c["type"] == "ineq" else abs(c["fun"](w)) < tol
                   for c in cons) and w.max() <= cfg["max_weight"] + tol and w.min() >= mw - tol

    rng = np.random.default_rng(42)
    starts = [np.ones(n) / n] + [np.minimum(x, cfg["max_weight"]) / np.minimum(x, cfg["max_weight"]).sum()
                                 for x in rng.dirichlet(np.ones(n), 30)]
    best = None
    for w0 in starts:
        res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 2000, "ftol": 1e-12})
        w = np.clip(res.x, 0, None)
        w[w < max(1e-4, mw * 0.5)] = 0 if mw == 0 else mw
        w /= w.sum()
        if feasible(w) and (best is None or neg_sharpe(w) < neg_sharpe(best)):
            best = w
    if best is None and cfg.get("max_vol") is not None:
        print(f"  ! no portfolio with volatility <= {cfg['max_vol']:.0%}: volatility cap removed")
        return optimise({**cfg, "max_vol": None}, mu, cov, rf, eq_mask)
    if best is None:
        raise RuntimeError(f"{cfg['title']}: no portfolio satisfies the constraints - relax them in config.py")
    return best


def efficient_frontier(mu, cov, points=40):
    """Long-only frontier (no other constraint), for the Markowitz chart."""
    n = len(mu)
    base = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    minvar = minimize(lambda w: w @ cov @ w, np.ones(n) / n, method="SLSQP",
                      bounds=[(0, 1)] * n, constraints=base).x
    vols, rets = [], []
    for target in np.linspace(minvar @ mu, mu.max(), points):
        cons = base + [{"type": "eq", "fun": lambda w, t=target: w @ mu - t}]
        r = minimize(lambda w: w @ cov @ w, np.ones(n) / n, method="SLSQP",
                     bounds=[(0, 1)] * n, constraints=cons)
        if r.success:
            vols.append(math.sqrt(r.x @ cov @ r.x))
            rets.append(target)
    return np.array(vols), np.array(rets)


def beta_alpha(rp, rb, rf):
    rfd = rf / TDAYS
    x, y = rb - rfd, rp - rfd
    beta, a = np.polyfit(x, y, 1)
    resid = y - (a + beta * x)
    r2 = 1 - resid.var() / y.var()
    te = (rp - rb).std() * math.sqrt(TDAYS)
    return {"beta": beta, "alpha": a * TDAYS, "r2": r2, "tracking_error": te,
            "information_ratio": (rp - rb).mean() * TDAYS / te if te > 0 else np.nan,
            "treynor": (rp.mean() * TDAYS - rf) / beta if beta != 0 else np.nan}


def var_10d(rp, level=0.99, horizon=10):
    """VaR / CVaR of the fund on $1bn (course chapter 10): historical and parametric, horizons 1..10 days."""
    q = 1 - level
    z = float(norm.ppf(level))
    logr = np.log1p(rp)
    sigma = logr.std()

    def hist(h):
        return -np.quantile(np.expm1(logr.rolling(h).sum()).dropna() * CAPITAL, q)

    hs = list(range(1, horizon + 1))
    curve = pd.DataFrame({
        "Historical VaR": [hist(h) for h in hs],
        "Parametric VaR": [z * sigma * math.sqrt(h) * CAPITAL for h in hs],
        "Parametric CVaR": [float(norm.pdf(z)) / q * sigma * math.sqrt(h) * CAPITAL for h in hs]}, index=hs)
    pnl10 = np.expm1(logr.rolling(horizon).sum()).dropna() * CAPITAL
    var_h = -np.quantile(pnl10, q)
    es_h = -pnl10[pnl10 <= -var_h].mean()
    var_1d = -np.quantile(rp * CAPITAL, q)
    return {"pnl10": pnl10, "var_hist": var_h, "es_hist": es_h, "var_1d": var_1d,
            "var_sqrt10": var_1d * math.sqrt(horizon), "var_param": curve.at[horizon, "Parametric VaR"],
            "cvar_param": curve.at[horizon, "Parametric CVaR"], "curve": curve,
            "breaches": int((pnl10 < -var_h).sum()), "windows": int(len(pnl10))}


# --------------------------------------------------------------- utils ----
def pct(x, d=1):
    return "n/a" if x is None or pd.isna(x) else f"{x * 100:.{d}f}%"


def num(x, d=2):
    return "n/a" if x is None or pd.isna(x) else f"{x:.{d}f}"


def usd_m(x):
    return f"${x / 1e6:,.1f}m"


def fmt_metric(metric, v):
    return pct(v) if METRIC_FMT[metric] == "pct" else num(v)


def esc(s):
    return html.escape(str(s), quote=True)


def save(fig, folder, name, par):
    fig.write_html(folder / name, config=par.config, include_plotlyjs="directory")


def zcolor(z):
    if pd.isna(z):
        return "#0f1b2f"
    a = min(abs(z) / 2, 1) * 0.65
    return f"rgba(46,204,113,{a:.2f})" if z > 0 else f"rgba(249,38,114,{a:.2f})"


# ---------------------------------------------------------------- main ----
def run(cfg, par, refresh=False):
    title, key, folder = cfg["title"], cfg["key"], cfg["folder"]
    img = ROOT / "img" / str(folder)
    ana = ROOT / "analytics" / str(folder)
    img.mkdir(parents=True, exist_ok=True)
    ana.mkdir(parents=True, exist_ok=True)
    print(f"\n################ {title} ################")

    symbols = C.tickers(cfg)
    assert len(symbols) == 25, f"{title} must hold 25 assets (has {len(symbols)})"
    types = {t: g[0] for g in cfg["groups"].values() for t in g[1]}
    groups = {t: name for name, g in cfg["groups"].items() for t in g[1]}
    bench = cfg["benchmark"]

    prices = load_prices(refresh)
    rf = risk_free(prices)
    cols = list(dict.fromkeys(symbols + [bench, C.SPX, C.TNX]))
    px = prices[cols]
    px = px[px[C.SPX].notna()].ffill(limit=3).dropna()      # US trading days only (crypto trades 7 days a week)
    starts = prices[symbols].apply(lambda s: s.first_valid_index())
    late = starts[starts > prices.index[0] + pd.Timedelta(days=10)]
    if len(late):
        print(f"  ! shorter history for {list(late.index)}: the sample starts {px.index[0]:%Y-%m-%d}")
    print(f"Sample: {px.index[0]:%Y-%m-%d} -> {px.index[-1]:%Y-%m-%d} ({len(px)} days) | risk-free (^IRX): {pct(rf, 2)}")

    rets_all = px.pct_change().dropna()
    rets = rets_all[symbols]
    dy = px[C.TNX].diff().loc[rets.index] / 100
    info = load_info(symbols + [bench])
    names = {t: (info.get(t, {}).get("shortName") or t) for t in symbols + [bench]}

    # ---- Part 1: indicators and z-scores
    ind = indicators(px[symbols], rets, rets_all[C.SPX], dy, info)
    vol3 = rets.std() * math.sqrt(TDAYS)          # 3-year volatility, consistent with the 3-year mean return
    zs, mcols = zscores(cfg, ind)
    zs.insert(0, "Name", pd.Series(names))
    zs.to_csv(DATA / f"zscores_{key}.csv", float_format="%.4f")

    # ---- Part 2: optimisation
    mu = (rets.mean() * TDAYS).values
    cov = (rets.cov() * TDAYS).values
    eq_mask = np.array([types[t] in EQUITY_TYPES for t in symbols], float)
    w = optimise(cfg, mu, cov, rf, eq_mask, symbols)
    ret, vol, sharpe = perf(w, mu, cov, rf)
    weights = pd.Series(w, index=symbols)
    rp = rets @ w
    rb = rets_all[bench]
    ba = beta_alpha(rp, rb, rf)
    ba_spx = beta_alpha(rp, rets_all[C.SPX], rf)
    vr = var_10d(rp)
    nav = (1 + rp).cumprod() * 100
    bnav = (1 + rb).cumprod() * 100
    b_vol = rb.std() * math.sqrt(TDAYS)
    b_ret = rb.mean() * TDAYS
    mdd = (nav / nav.cummax() - 1).min()
    rc = w * (cov @ w) / vol ** 2
    alloc = pd.DataFrame({"Name": pd.Series(names), "Group": pd.Series(groups), "Weight": weights,
                          "Amount": weights * CAPITAL, "Risk contribution": pd.Series(rc, index=symbols),
                          "Exp. return": pd.Series(mu, index=symbols),
                          "Volatility": vol3})
    alloc = alloc[alloc["Weight"] > 0].sort_values("Weight", ascending=False)
    eq_share = float(w @ eq_mask)
    top3 = list(alloc.index[:3])

    print(alloc[["Weight", "Amount", "Risk contribution"]].to_string(formatters={
        "Weight": "{:.2%}".format, "Amount": lambda x: f"{x / 1e6:,.1f}m",
        "Risk contribution": "{:.2%}".format}))
    print(f"Expected return {pct(ret, 2)} | volatility {pct(vol, 2)} | Sharpe {sharpe:.3f} | equity share {pct(eq_share)}")
    print(f"Beta vs {bench} {ba['beta']:.2f} | alpha {pct(ba['alpha'], 2)} | 10-day 99% VaR {usd_m(vr['var_hist'])}")
    if cfg.get("min_equity") is not None:
        ok1 = ret - rf >= 0.06
        ok2 = eq_share >= 0.70
        print(f"CHALLENGE CHECK -> excess return {pct(ret - rf, 2)} (>= 6.00%: {ok1}) | "
              f"equity {pct(eq_share, 2)} (>= 70%: {ok2}) | Sharpe {sharpe:.3f}")

    lay = par.layout

    # 1-1: z-score table (dashboard tile)
    order = zs.sort_values(["Group", "Score"], ascending=[True, False])
    order = pd.concat([order[order["Group"] == g] for g in cfg["groups"]])
    cells, fills = [order["Group"].tolist(), order.index.tolist()], [["rgb(50,50,50)"] * 25] * 2
    for m in mcols:
        cells.append([("" if pd.isna(order.at[t, m]) else
                       f"{fmt_metric(m, order.at[t, m])} ({order.at[t, 'z ' + m]:+.1f})"
                       if not pd.isna(order.at[t, 'z ' + m]) else fmt_metric(m, order.at[t, m]))
                      for t in order.index])
        fills.append([zcolor(order.at[t, "z " + m]) for t in order.index])
    cells.append([num(s) for s in order["Score"]])
    fills.append([zcolor(s * 2) for s in order["Score"]])
    fig = go.Figure(layout=lay)
    fig.add_table(header=dict(values=["Group", "Asset"] + mcols + ["Score"], fill_color=PURPLE,
                              font=dict(color="white", size=11), align="center"),
                  cells=dict(values=cells, fill_color=fills, font=dict(color="white", size=10),
                             align="center", height=22),
                  columnwidth=[2.2, 1] + [1.3] * len(mcols) + [0.8])
    fig.update_layout(title=f"{title} - indicators and z-scores (value, z) by group")
    save(fig, img, "1-2.html", par)

    # 1-1: heatmap of oriented z-scores
    zmat = order[["z " + m for m in mcols]]
    fig = go.Figure(layout=lay)
    fig.add_heatmap(z=zmat.values, x=mcols, y=[f"{t} · {g}" for t, g in zip(order.index, order["Group"])],
                    zmid=0, zmin=-2.5, zmax=2.5, colorscale=[[0, PINK], [0.5, "#0f1b2f"], [1, GREEN]],
                    colorbar=dict(title="z"), hovertemplate="%{y}<br>%{x}: z = %{z:.2f}<extra></extra>")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=f"{title} - oriented z-scores by group (green = favourable)")
    save(fig, img, "1-1.html", par)

    # 2-1: base 100
    base = px[symbols] / px[symbols].iloc[0] * 100
    fig = go.Figure(layout=lay)
    for t in symbols:
        fig.add_scatter(x=base.index, y=base[t], name=t, line=dict(width=1), opacity=0.75)
    nav100 = pd.concat([pd.Series([100.0], index=[px.index[0]]), nav])
    bnav100 = pd.concat([pd.Series([100.0], index=[px.index[0]]), bnav])
    fig.add_scatter(x=nav100.index, y=nav100, name=title, line=dict(width=4, color=PINK))
    fig.add_scatter(x=bnav100.index, y=bnav100, name=f"Benchmark ({bench})", line=dict(width=3, color="white", dash="dot"))
    fig.update_layout(title=f"{title} - assets, base 100", yaxis_title="Base 100")
    save(fig, img, "2-1.html", par)

    # 2-2: fund vs benchmark + drawdown
    fig = go.Figure(layout=lay)
    fig.add_scatter(x=nav100.index, y=nav100, name=title, line=dict(width=3, color=PINK))
    fig.add_scatter(x=bnav100.index, y=bnav100, name=bench, line=dict(width=2, color=GREY, dash="dot"))
    dd = nav100 / nav100.cummax() - 1
    fig.add_scatter(x=dd.index, y=dd, name="Fund drawdown", yaxis="y2", fill="tozeroy",
                    line=dict(width=1, color=PURPLE))
    fig.update_layout(title=f"{title} vs {bench} (base 100) and drawdown",
                      yaxis2=dict(overlaying="y", side="right", tickformat=".0%", showgrid=False),
                      legend=dict(orientation="h", y=-0.15))
    save(fig, img, "2-2.html", par)

    # 3-1: daily LOG returns of every asset (course: logarithmic daily returns)
    lrets = np.log1p(rets)
    fig = go.Figure(layout=lay)
    for t in symbols:
        fig.add_scatter(x=lrets.index, y=lrets[t], name=t, line=dict(width=1), opacity=0.7)
    fig.update_layout(title=f"{title} - daily log returns of each asset", yaxis_tickformat=".0%")
    save(fig, img, "3-1.html", par)

    # 3-2: distribution of daily returns per asset
    fig = go.Figure(layout=lay)
    for t in vol3[symbols].sort_values().index:
        fig.add_box(y=lrets[t], name=t, boxpoints=False, line=dict(width=1))
    fig.update_layout(title="Daily return distribution per asset (sorted by volatility)",
                      yaxis_tickformat=".1%", showlegend=False)
    save(fig, img, "3-2.html", par)

    # 3-3: portfolio daily returns histogram vs normal
    fig = go.Figure(layout=lay)
    fig.add_histogram(x=rp, nbinsx=80, histnorm="probability density", name="Fund daily returns",
                      marker_color=PURPLE)
    xs = np.linspace(rp.min(), rp.max(), 200)
    fig.add_scatter(x=xs, y=np.exp(-0.5 * ((xs - rp.mean()) / rp.std()) ** 2) / (rp.std() * math.sqrt(2 * math.pi)),
                    name="Normal fit", line=dict(color=PINK, width=2))
    fig.update_layout(title=f"{title} - daily returns vs normal distribution", xaxis_tickformat=".1%")
    save(fig, img, "3-3.html", par)

    # 3-4: correlation matrix of the 25 assets (ordered by group)
    order_c = [t for g in cfg["groups"].values() for t in g[1] if t in rets.columns]
    cm = rets[order_c].corr()
    fig = go.Figure(layout=lay)
    fig.add_heatmap(z=cm.values, x=order_c, y=order_c, zmin=-1, zmax=1, colorscale="RdBu_r", zmid=0,
                    colorbar=dict(title="corr."), hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=f"{title} - correlation matrix of daily returns (assets grouped by class)")
    save(fig, img, "3-4.html", par)

    # 4-1: Markowitz-Sharpe
    rng = np.random.default_rng(42)
    W = rng.dirichlet(np.ones(25) * 0.4, 10_000)
    sim_r = W @ mu
    sim_v = np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))
    fv, fr = efficient_frontier(mu, cov)
    fig = go.Figure(layout=lay)
    fig.add_scatter(x=sim_v, y=sim_r, mode="markers", name="Random portfolios",
                    marker=dict(size=3, color=(sim_r - rf) / sim_v, colorscale="viridis", showscale=True,
                                colorbar=dict(title="Sharpe")))
    fig.add_scatter(x=fv, y=fr, mode="lines", name="Efficient frontier (long only)", line=dict(color="white", width=2))
    xm = max(sim_v.max(), fv.max() if len(fv) else 0, vol3.max()) * 1.05
    fig.add_scatter(x=[0, xm], y=[rf, rf + sharpe * xm], mode="lines", name="Capital market line",
                    line=dict(color=PINK, dash="dash"))
    fig.add_scatter(x=vol3[symbols], y=mu, mode="markers+text", text=symbols,
                    textposition="top center", textfont=dict(size=9), name="Assets",
                    marker=dict(color=GREY, size=6))
    fig.add_scatter(x=[vol], y=[ret], mode="markers", name=f"{title} (Sharpe {sharpe:.2f})",
                    marker=dict(color=PINK, size=18, symbol="star", line=dict(color="white", width=1)))
    fig.update_layout(title=f"{title} - Markowitz-Sharpe analysis", xaxis_title="Volatility",
                      yaxis_title="Expected return", xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                      legend=dict(x=0.01, y=0.99, bgcolor="rgba(15,27,47,0.8)", font=dict(size=10)))
    save(fig, img, "4-1.html", par)

    # 4-2: allocation in $m
    gnames = list(cfg["groups"])
    fig = go.Figure(layout=lay)
    for k, g in enumerate(gnames):
        a = alloc[alloc["Group"] == g]
        if len(a):
            fig.add_bar(x=a.index, y=a["Amount"] / 1e6, name=g, marker_color=GROUP_COLORS[k % 5],
                        text=[pct(x) for x in a["Weight"]], textposition="outside")
    fig.update_layout(title=f"{title} - allocation of $1bn ($m)", yaxis_title="$m",
                      xaxis=dict(categoryorder="total descending"))
    save(fig, img, "4-2.html", par)

    # 4-3: weight vs risk contribution
    fig = go.Figure(layout=lay)
    fig.add_bar(x=alloc.index, y=alloc["Weight"], name="Weight", marker_color=PURPLE)
    fig.add_bar(x=alloc.index, y=alloc["Risk contribution"], name="Risk contribution", marker_color=PINK)
    fig.update_layout(title="Weight vs contribution to portfolio risk", yaxis_tickformat=".0%", barmode="group")
    save(fig, img, "4-3.html", par)

    # 5-1: beta / alpha regression
    rfd = rf / TDAYS
    xb, yp = rb - rfd, rp - rfd
    fig = go.Figure(layout=lay)
    fig.add_scatter(x=xb, y=yp, mode="markers", name="Daily excess returns",
                    marker=dict(size=4, color=PURPLE, opacity=0.6))
    xx = np.linspace(xb.min(), xb.max(), 50)
    fig.add_scatter(x=xx, y=ba["alpha"] / TDAYS + ba["beta"] * xx, mode="lines",
                    name=f"β = {ba['beta']:.2f}, α = {pct(ba['alpha'], 2)}/yr", line=dict(color=PINK, width=3))
    fig.update_layout(title=f"{title} - beta and alpha vs {bench}", xaxis_title=f"{bench} excess return",
                      yaxis_title="Fund excess return", xaxis_tickformat=".1%", yaxis_tickformat=".1%")
    save(fig, img, "5-1.html", par)

    # 5-2: rolling 6-month beta
    roll_beta = (rp.rolling(126).cov(rb) / rb.rolling(126).var()).dropna()
    fig = go.Figure(layout=lay)
    fig.add_scatter(x=roll_beta.index, y=roll_beta, name="Rolling 6-month beta", line=dict(color=PINK, width=2))
    fig.add_hline(y=ba["beta"], line_dash="dot", line_color=GREY, annotation_text=f"3-year beta {ba['beta']:.2f}")
    fig.update_layout(title=f"Rolling 6-month beta vs {bench}")
    save(fig, img, "5-2.html", par)

    # 6-1: VaR histogram
    pnl = vr["pnl10"] / 1e6
    fig = go.Figure(layout=lay)
    fig.add_histogram(x=pnl, nbinsx=80, name="10-day P&L", marker_color=PURPLE)
    fig.add_vline(x=-vr["var_hist"] / 1e6, line_color=PINK, line_width=3,
                  annotation_text=f"VaR 99% = {usd_m(vr['var_hist'])}", annotation_position="top left")
    fig.add_vline(x=-vr["es_hist"] / 1e6, line_color="orange", line_dash="dot",
                  annotation_text=f"ES = {usd_m(vr['es_hist'])}", annotation_position="bottom left")
    fig.update_layout(title=f"{title} - 10-day 99% historical VaR on $1bn", xaxis_title="10-day P&L ($m)",
                      showlegend=False)
    save(fig, img, "6-1.html", par)

    # 6-2: 10-day P&L through time with breaches
    br = pnl[pnl < -vr["var_hist"] / 1e6]
    fig = go.Figure(layout=lay)
    fig.add_scatter(x=pnl.index, y=pnl, name="Rolling 10-day P&L", line=dict(color=PURPLE, width=1))
    fig.add_hline(y=-vr["var_hist"] / 1e6, line_color=PINK, line_dash="dash", annotation_text="-VaR 99%")
    fig.add_scatter(x=br.index, y=br, mode="markers", name="Losses beyond VaR", marker=dict(color=PINK, size=6))
    fig.update_layout(title="Rolling 10-day P&L on $1bn ($m)", yaxis_title="$m")
    save(fig, img, "6-2.html", par)

    # 6-3: VaR methods
    meth = {"Historical 10-day": vr["var_hist"], "1-day × √10": vr["var_sqrt10"],
            "Parametric VaR (normal)": vr["var_param"], "Historical ES": vr["es_hist"],
            "Parametric CVaR": vr["cvar_param"]}
    fig = go.Figure(layout=lay)
    fig.add_bar(x=list(meth), y=[v / 1e6 for v in meth.values()], marker_color=[PINK, PURPLE, "#00b3ff", "orange", "#ffd54a"],
                text=[usd_m(v) for v in meth.values()], textposition="outside")
    fig.update_layout(title="10-day 99% VaR - method comparison ($m)", yaxis_title="$m")
    save(fig, img, "6-3.html", par)

    # 6-4: maximum loss over a horizon of 1 to 10 days
    cv = vr["curve"] / 1e6
    fig = go.Figure(layout=lay)
    fig.add_scatter(x=cv.index, y=cv["Historical VaR"], name="Historical VaR 99%", mode="lines+markers",
                    line=dict(color=PINK, width=3))
    fig.add_scatter(x=cv.index, y=cv["Parametric VaR"], name="Parametric VaR 99%", mode="lines+markers",
                    line=dict(color=PURPLE, width=2, dash="dash"))
    fig.add_scatter(x=cv.index, y=cv["Parametric CVaR"], name="Parametric CVaR 99%", mode="lines+markers",
                    line=dict(color="orange", width=2, dash="dot"))
    fig.update_layout(title=f"{title} - maximum loss over 1 to 10 days on $1bn ($m)", xaxis_title="Horizon (days)",
                      yaxis_title="$m", legend=dict(x=0.02, y=0.98))
    save(fig, img, "6-4.html", par)

    # gauge (index page)
    gmax = cfg["gauge_max"] * 100
    fig = go.Figure(layout=lay)
    fig.add_indicator(
        mode="gauge+number+delta", value=vol * 100,
        number={"suffix": "%", "valueformat": ".2f", "font": {"size": 40}},
        delta={"reference": b_vol * 100, "valueformat": ".2f", "suffix": " pts vs benchmark", "position": "bottom", "font": {"size": 14},
               "increasing": {"color": PINK}, "decreasing": {"color": GREEN}},
        title={"text": f"{title}<br><span style='font-size:0.7em'>annualised standard deviation</span>"},
        gauge={"axis": {"range": [0, gmax], "tickcolor": "#aaaaaa"},
               "bar": {"color": PURPLE}, "bgcolor": "rgba(8,14,26,1)", "borderwidth": 3,
               "bordercolor": "rgba(53,198,181,0.8)",
               "steps": [{"range": [0, gmax * 0.4], "color": "rgba(111,207,151,0.20)"},
                         {"range": [gmax * 0.4, gmax * 0.7], "color": "rgba(53,198,181,0.25)"},
                         {"range": [gmax * 0.7, gmax], "color": "rgba(232,121,90,0.45)"}],
               "threshold": {"line": {"color": "white", "width": 4}, "thickness": 0.8, "value": b_vol * 100}})
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=40, r=40, t=80, b=30), font={"color": "white", "family": "Helvetica"})
    save(fig, img, "gauge-1.html", par)

    # ---- news on the 3 largest holdings
    news = {t: fetch_news(t) for t in top3}
    (DATA / f"news_{key}.json").write_text(json.dumps(news, indent=1))

    summary = {
        "fund": title, "sample_start": f"{px.index[0]:%Y-%m-%d}", "sample_end": f"{px.index[-1]:%Y-%m-%d}",
        "risk_free": rf, "expected_return": ret, "volatility": vol, "sharpe": sharpe,
        "excess_return": ret - rf, "equity_share": eq_share, "max_drawdown": mdd,
        "benchmark": bench, "benchmark_return": b_ret, "benchmark_vol": b_vol,
        "benchmark_sharpe": (b_ret - rf) / b_vol, **{k: v for k, v in ba.items()}, "beta_spx": ba_spx["beta"], "alpha_spx": ba_spx["alpha"],
        "var_10d_99_hist": vr["var_hist"], "es_10d_99_hist": vr["es_hist"],
        "var_1d_99": vr["var_1d"], "var_sqrt10": vr["var_sqrt10"], "var_param": vr["var_param"], "cvar_param": vr["cvar_param"],
        "var_breaches": vr["breaches"], "var_windows": vr["windows"],
        "weights": {t: float(x) for t, x in alloc["Weight"].items()}, "top3": top3,
    }
    (DATA / f"summary_{key}.json").write_text(json.dumps(summary, indent=1, default=float))

    write_pages(cfg, ana, folder, zs, mcols, alloc, ind, rets, summary, vr, news, names)
    print(f"Done: img/{folder}/ and analytics/{folder}/ updated.")
    return summary


# ------------------------------------------------------------- HTML -------
def table(headers, rows, cls="kpi"):
    h = "".join(f"<th>{esc(x)}</th>" for x in headers)
    b = "".join("<tr>" + "".join(f"<td>{x}</td>" for x in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


def page(ana, folder, n, h1, charts, body):
    frames = "".join(f'<figure><iframe src="../../img/{folder}/{f}"></iframe><figcaption>{esc(c)}</figcaption></figure>'
                     for f, c in charts)
    doc = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
    <title>{esc(h1)}</title>
    <link rel="stylesheet" href="../../css/reset.css">
    <link rel="stylesheet" href="../../css/customize.css">
  </head>
  <body class="analytics">
    <a class="logo2" href="../../index.html"></a>
    <nav>
      <a href="../../{folder}.html"></a>
    </nav>
    <section>
      <h1>{esc(h1)}</h1>
      <div class="charts">{frames}</div>
      <article class="report">
{body}
      </article>
    </section>
  </body>
</html>
"""
    (ana / f"{n}.html").write_text(doc, encoding="utf-8")


def write_pages(cfg, ana, folder, zs, mcols, alloc, ind, rets, s, vr, news, names):
    T, bench, rf = cfg["title"], s["benchmark"], s["risk_free"]
    period = f"{s['sample_start']} to {s['sample_end']}"

    # ---------- 1. universe and z-scores
    groups_txt = "".join(
        f"<li><strong>{esc(g)}</strong> ({len(ts)}): {esc(', '.join(ts))}</li>" for g, (_, ts) in cfg["groups"].items())
    best = []
    for g in cfg["groups"]:
        top = zs[zs["Group"] == g].sort_values("Score", ascending=False).head(2)
        best.append([esc(g)] + [f"{esc(t)} ({num(top.at[t, 'Score'])})" for t in top.index])
    used = {}
    for g, (gt, _) in cfg["groups"].items():
        for m, sign in METRICS[gt]:
            sg = cfg[sign] if isinstance(sign, str) else sign
            used.setdefault(m, (sg, []))[1].append(g)
    ztab = []
    for g in cfg["groups"]:
        sub = zs[zs["Group"] == g].sort_values("Score", ascending=False)
        for t, r in sub.iterrows():
            cells = [("" if pd.isna(r[m]) else fmt_metric(m, r[m]) +
                      ("" if pd.isna(r["z " + m]) else f' <span class="z" style="background:{zcolor(r["z " + m])}">{r["z " + m]:+.2f}</span>'))
                     for m in mcols]
            ztab.append([esc(g), f"<strong>{esc(t)}</strong>", esc(r["Name"])] + cells + [num(r["Score"]), str(r["Rank in group"])])
    defs = [[esc(m), esc(METRIC_DESC[m]), "higher" if sg > 0 else "lower", esc(", ".join(gs))]
            for m, (sg, gs) in used.items()]
    body = f"""
<h2>Investment universe</h2>
<p>The {esc(T)} ({esc(cfg['tagline'].lower())}) selects 25 assets, split into {len(cfg['groups'])} groups. All data come from Yahoo Finance over the last three years ({period}).</p>
<ul class="list">{groups_txt}</ul>
<h2>Method</h2>
<p>Each group is scored on its own indicators, so that a bond ETF is only compared with other bond ETFs. For every indicator we compute the z-score inside the group, z = (x − mean) / standard deviation, and orient it so that a positive value is favourable for this fund. The <strong>score</strong> is the average of the oriented z-scores. In the table, each cell shows the raw value and its oriented z-score in brackets.</p>
{table(["Indicator", "Definition", "Preferred", "Groups"], defs)}
<h2>Best-ranked assets per group</h2>
{table(["Group", "1st (score)", "2nd (score)"], best)}
<h2>Full z-score table</h2>
{table(["Group", "Ticker", "Name"] + mcols + ["Score", "Rank"], ztab)}
"""
    page(ana, folder, 1, f"{T} - asset selection and z-scores",
         [("1-1.html", "Heatmap of oriented z-scores"), ("1-2.html", "Indicators and z-scores by group")], body)

    # ---------- 2. base 100
    cagr = (rets + 1).prod() ** (TDAYS / len(rets)) - 1
    perf_rows = [[esc(t), esc(names.get(t, t)), pct(cagr[t]), pct(ind.at[t, "Max drawdown"])]
                 for t in cagr.sort_values(ascending=False).index]
    body = f"""
<h2>On the graph</h2>
<p>All 25 assets are rebased to 100 on {s['sample_start']}. The thick gold line is the {esc(T)} with its optimised weights (rebalanced daily); the dotted line is its benchmark, {esc(cfg['benchmark_name'])}.</p>
<p>Best performer over the period: <strong>{esc(cagr.idxmax())}</strong> ({pct(cagr.max())} per year). Weakest: <strong>{esc(cagr.idxmin())}</strong> ({pct(cagr.min())} per year). The fund's maximum drawdown was {pct(s['max_drawdown'])}.</p>
<h2>Annualised performance of each asset</h2>
{table(["Ticker", "Name", "CAGR (3y)", "Max drawdown"], perf_rows)}
"""
    page(ana, folder, 2, f"{T} - performance (base 100)",
         [("2-1.html", "Assets, fund and benchmark - base 100"), ("2-2.html", "Fund vs benchmark and drawdown")], body)

    # ---------- 3. daily returns
    vol_s = rets.std() * math.sqrt(TDAYS)
    worst_day = rets.min()
    rp = rets @ pd.Series(s["weights"]).reindex(rets.columns).fillna(0)
    skew, kurt = rp.skew(), rp.kurt()
    body = f"""
<h2>On the graph</h2>
<p>The first chart shows the daily returns of every asset; the box plots compare their distributions. The most volatile asset is <strong>{esc(vol_s.idxmax())}</strong> ({pct(vol_s.max())} annualised) and the least volatile is <strong>{esc(vol_s.idxmin())}</strong> ({pct(vol_s.min())}). The worst single day in the universe was {pct(worst_day.min(), 2)} for {esc(worst_day.idxmin())}.</p>
<p>The fund's own daily returns have a skewness of {num(skew)} and an excess kurtosis of {num(kurt)}: {"the tails are fatter than a normal distribution, which is why the VaR is computed historically rather than only with a normal model." if kurt > 0.5 else "the distribution is close to normal over the period."}</p>
"""
    cmx = rets.corr().values
    n_a = len(cmx)
    avg_c = (cmx.sum() - n_a) / (n_a * (n_a - 1))
    iu = np.triu_indices(n_a, 1)
    pairs = sorted(zip(cmx[iu], np.array(rets.columns)[iu[0]], np.array(rets.columns)[iu[1]]))
    lo, hi = pairs[0], pairs[-1]
    body += f"""<h2>Correlation between assets</h2>
<p>The correlation matrix shows how the 25 assets move together (+1: identical moves, 0: unrelated, -1: opposite moves). The average pairwise correlation is <strong>{num(avg_c)}</strong>: the lower it is, the more diversified the fund. The most correlated pair is <strong>{esc(hi[1])} / {esc(hi[2])}</strong> ({num(hi[0])}) and the least correlated pair is <strong>{esc(lo[1])} / {esc(lo[2])}</strong> ({num(lo[0])}). This is the effect of the correlation filter applied when the assets were selected.</p>
"""
    page(ana, folder, 3, f"{T} - daily returns",
         [("3-1.html", "Daily log returns of each asset"), ("3-2.html", "Distribution per asset"),
          ("3-3.html", "Fund daily returns vs normal"),
          ("3-4.html", "Correlation matrix of the 25 assets")], body)

    # ---------- 4. Markowitz + allocation + news
    cons = [f"maximum {pct(cfg['max_weight'], 0)} per asset"]
    if cfg.get("max_equity") is not None:
        cons.append(f"at most {pct(cfg['max_equity'], 0)} in equities")
    if cfg.get("max_vol") is not None:
        cons.append(f"volatility below {pct(cfg['max_vol'], 0)}")
    if cfg.get("sector_caps"):
        cons.append(f"maximum {pct(max(cfg['sector_caps'].values()), 0)} per sector")
    if cfg.get("min_equity") is not None:
        cons.append("at least 70% in equities and equity ETFs")
    if cfg.get("min_excess_return") is not None:
        cons.append("expected return at least 6 points above the risk-free rate")
    rows = [[esc(t), esc(a["Name"]), esc(a["Group"]), pct(a["Weight"]), usd_m(a["Amount"]), pct(a["Risk contribution"])]
            for t, a in alloc.iterrows()]
    kpis = [["Expected return (annualised)", pct(s["expected_return"], 2)], ["Volatility", pct(s["volatility"], 2)],
            ["Risk-free rate (13-week T-bill)", pct(rf, 2)], ["Excess return", pct(s["excess_return"], 2)],
            ["Sharpe ratio", num(s["sharpe"], 3)], ["Equity share", pct(s["equity_share"])],
            ["Benchmark Sharpe ratio", num(s["benchmark_sharpe"], 3)]]
    news_html = ""
    for t in s["top3"]:
        cards = ""
        for it in news.get(t, []):
            img = f'<img src="{esc(it["image"])}" alt="">' if it.get("image") else ""
            cards += (f'<a class="card" href="{esc(it["url"])}" target="_blank" rel="noopener">{img}<div>'
                      f'<small>{esc(it["publisher"])} · {esc(it["date"])}</small><h3>{esc(it["title"])}</h3>'
                      f'<p>{esc(it["summary"][:220])}</p></div></a>')
        vids = "".join(f'<div class="video"><iframe src="https://www.youtube.com/embed/{esc(v)}" allowfullscreen></iframe></div>'
                       for v in cfg.get("videos", {}).get(t, []))
        news_html += (f'<h2>{esc(t)} - {esc(names.get(t, t))} ({pct(s["weights"][t])})</h2>'
                      f'<div class="news">{cards or "<p>No recent news returned by Yahoo Finance.</p>"}</div>{vids}')
    body = f"""
<h2>Markowitz-Sharpe analysis</h2>
<p>10,000 random long-only portfolios of the 25 assets are simulated and coloured by Sharpe ratio; the white curve is the long-only efficient frontier. The fund (gold star) maximises the Sharpe ratio, (expected return − risk-free rate) / volatility, under the following constraints: {esc('; '.join(cons))}. Expected returns and the covariance matrix are annualised from three years of daily returns.</p>
{table(["Indicator", "Value"], kpis)}
<h2>Allocation of $1bn</h2>
{table(["Ticker", "Name", "Group", "Weight", "Amount", "Risk contribution"], rows)}
<h2>News on our 3 largest holdings</h2>
{news_html}
"""
    page(ana, folder, 4, f"{T} - Markowitz-Sharpe allocation",
         [("4-1.html", "Random portfolios, frontier and optimal fund"), ("4-2.html", "Allocation of $1bn"),
          ("4-3.html", "Weight vs risk contribution")], body)

    # ---------- 5. beta / alpha
    b = s["beta"]
    spx_rows = [] if bench == C.SPX else [["Beta vs S&P 500 (SPY)", num(s["beta_spx"])],
                                          ["Alpha vs S&P 500 (annualised)", pct(s["alpha_spx"], 2)]]
    body = f"""
<h2>Benchmark</h2>
<p>The {esc(T)} is compared with the {esc(cfg['benchmark_name'])}, which matches its risk profile. Daily excess returns of the fund (over the daily risk-free rate) are regressed on those of the benchmark: R<sub>p</sub> − R<sub>f</sub> = α + β (R<sub>b</sub> − R<sub>f</sub>) + ε.</p>
{table(["Indicator", "Value"], [["Beta", num(b)], ["Alpha (annualised, Jensen)", pct(s["alpha"], 2)], ["R²", num(s["r2"])], ["Tracking error", pct(s["tracking_error"], 2)], ["Information ratio", num(s["information_ratio"])], ["Treynor ratio", num(s["treynor"], 3)], ["Benchmark return / volatility", f'{pct(s["benchmark_return"], 2)} / {pct(s["benchmark_vol"], 2)}']] + spx_rows)}
<h2>Reading</h2>
<p>A beta of {num(b)} means that when {esc(bench)} moves by 1%, the fund moves on average by {num(b)}%{": the fund is less sensitive to its market than the benchmark" if b < 0.95 else ": the fund amplifies the moves of its benchmark" if b > 1.05 else ", in line with its benchmark"}. The annualised alpha of {pct(s["alpha"], 2)} is the return {"earned on top of" if s["alpha"] > 0 else "lost compared with"} what this level of market risk would explain. With an R² of {num(s["r2"])}, {pct(s["r2"], 0)} of the fund's daily variance is explained by the benchmark. The rolling beta shows how stable this exposure has been.</p>
"""
    page(ana, folder, 5, f"{T} - beta and alpha vs {bench}",
         [("5-1.html", "Regression of excess returns"), ("5-2.html", "Rolling 6-month beta")], body)

    # ---------- 6. VaR
    body = f"""
<h2>Method</h2>
<p>We apply the fund's weights to three years of daily returns and compound them over every overlapping 10-day window ({vr['windows']} windows). The 10-day 99% historical VaR is the loss exceeded in only 1% of these windows, applied to $1bn. The expected shortfall is the average loss beyond the VaR.</p>
{table(["Measure (10 days, 99%, $1bn)", "Value"], [["Historical VaR", f"<strong>{usd_m(s['var_10d_99_hist'])}</strong>"], ["Expected shortfall", usd_m(s["es_10d_99_hist"])], ["1-day historical VaR", usd_m(s["var_1d_99"])], ["1-day VaR × √10", usd_m(s["var_sqrt10"])], ["Parametric (normal) VaR", usd_m(s["var_param"])], ["Parametric CVaR", usd_m(s["cvar_param"])], ["Windows beyond the VaR", f"{vr['breaches']} / {vr['windows']}"]])}
<h2>Reading</h2>
<p>Over the last three years, the {esc(T)} would have lost more than {usd_m(s['var_10d_99_hist'])} over 10 trading days in only 1% of cases, i.e. {pct(s['var_10d_99_hist'] / CAPITAL, 2)} of its capital. {"The historical VaR is above the normal VaR, which confirms fat tails in the return distribution." if s['var_10d_99_hist'] > s['var_param'] else "The historical VaR is close to or below the normal VaR over this period."}</p>
"""
    page(ana, folder, 6, f"{T} - 10-day 99% VaR",
         [("6-1.html", "Distribution of 10-day P&L"), ("6-2.html", "10-day P&L through time"),
          ("6-3.html", "VaR methods compared"), ("6-4.html", "Maximum loss over 1 to 10 days")], body)
