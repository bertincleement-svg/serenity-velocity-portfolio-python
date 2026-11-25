"""
config.py - investment universes and constraints for the two funds.

Each fund holds 25 assets split into groups. The group "type" decides which
indicators are used for the z-scores and whether the asset counts as equity:
    equity      -> single stocks             (counts as equity)
    equity_etf  -> equity ETFs               (counts as equity)
    bond        -> bond ETFs
    other       -> gold, commodities, REITs, cash-like ETFs
"""

SERENITY = {
    "key": "serenity",
    "folder": 1,                      # writes to img/1 and analytics/1
    "title": "Serenity Fund",
    "tagline": "Capital preservation for risk-averse clients",
    "benchmark": "AOK",
    "benchmark_name": "iShares Core Conservative Allocation ETF (AOK, ~30% equity / 70% bonds)",
    "groups": {
        "Defensive equities": ("equity",
            ["JNJ", "PG", "KO", "PEP", "WMT", "MCD", "DUK", "SO", "MRK", "CL"]),
        "Low-volatility equity ETFs": ("equity_etf",
            ["USMV", "SPLV", "VIG"]),
        "Bond ETFs": ("bond",
            ["SHY", "IEI", "IEF", "TLT", "AGG", "LQD", "VCSH", "TIP", "MBB"]),
        "Alternatives & cash": ("other",
            ["GLD", "BIL", "VNQ"]),
    },
    # z-score orientation for the indicators whose "good" direction depends on the fund
    "beta_sign": -1,          # a low beta is preferred
    "duration_sign": -1,      # a short duration (low rate risk) is preferred
    # optimiser: max Sharpe under the conservative constraints below
    "max_weight": 0.15,       # max 15% per line
    "max_equity": 0.35,       # at most 35% in equities / equity ETFs
    "max_vol": 0.06,          # annualised volatility capped at 6%
    "gauge_max": 0.20,        # gauge scale: 0-20% volatility
    # optional YouTube video IDs shown on analytics page 4, e.g. {"GLD": ["xxxxxxxxxxx"]}
    "videos": {},
}

VELOCITY = {
    "key": "velocity",
    "folder": 2,
    "title": "Velocity Fund",
    "tagline": "Long-term growth for clients who accept higher risk",
    "benchmark": "SPY",
    "benchmark_name": "SPDR S&P 500 ETF (SPY)",
    "groups": {
        "Large-cap growth equities": ("equity",
            ["NVDA", "MSFT", "AAPL", "AMZN", "META", "GOOGL", "AVGO", "TSLA",
             "NFLX", "PLTR", "LLY", "ANET", "GE", "JPM", "COST"]),
        "Equity ETFs": ("equity_etf",
            ["QQQ", "SMH", "XLK", "VUG", "EEM"]),
        "Bond ETFs": ("bond",
            ["TLT", "HYG"]),
        "Alternatives": ("other",
            ["GLD", "SLV", "VNQ"]),
    },
    "beta_sign": +1,          # growth fund: higher beta accepted
    "duration_sign": -1,
    # optimiser: max Sharpe under the challenge rules (small safety buffers included)
    "max_weight": 0.25,       # concentration cap; raise it to push the Sharpe higher
    "min_equity": 0.705,      # rule: >= 70% in equities / equity ETFs
    "min_excess_return": 0.062,  # rule: expected return >= risk-free + 6 pts
    "gauge_max": 0.40,
    "videos": {},
}

FUNDS = [SERENITY, VELOCITY]

# market series used by every fund
SPX = "SPY"        # equity market proxy for betas
TNX = "^TNX"       # 10-year Treasury yield, for empirical bond duration
IRX = "^IRX"       # 13-week T-bill rate (annualised, %), risk-free rate


def tickers(cfg):
    return [t for _, (_, ts) in cfg["groups"].items() for t in ts]


def all_tickers():
    out = []
    for cfg in FUNDS:
        out += tickers(cfg) + [cfg["benchmark"]]
    out += [SPX, TNX, IRX]
    return list(dict.fromkeys(out))   # unique, order kept


# ---- universes chosen by the selectors replace the groups / constraints above (if the files exist):
#      scripts/1/selector_serenity.py -> universe_serenity.json ; scripts/2/selector_velocity.py -> universe_velocity.json
import json as _json
from pathlib import Path as _Path

for _cfg, _name in ((SERENITY, "serenity"), (VELOCITY, "velocity")):
    _f = _Path(__file__).with_name(f"universe_{_name}.json")
    if _f.exists():
        _d = _json.loads(_f.read_text())
        _cfg["groups"] = {g: (k, ts) for g, (k, ts) in _d["groups"].items()}
        for _k in ("max_weight", "min_weight", "max_equity", "min_equity", "max_vol", "min_excess_return"):
            if _d["rules"].get(_k) is None:
                _cfg.pop(_k, None)
            else:
                _cfg[_k] = _d["rules"][_k]
        _cfg["sector_caps"] = _d["rules"].get("sector_caps", {})
        _cfg["sectors"] = _d.get("sectors", {})
