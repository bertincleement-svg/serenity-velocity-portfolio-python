# -*- coding: utf-8 -*-
"""
compose_funds.py  -  Etape 2 : prendre les meilleurs de chaque classe, composer
les 25 actifs de Serenity et de Velocity, puis optimiser les poids (Sharpe max).

A placer dans le MEME dossier que les 5 fichiers selection_*.csv. Lancer avec F5.
Sorties : universe_serenity.csv, universe_velocity.csv, weights_*.csv
"""
import os, warnings
import numpy as np, pandas as pd
from scipy.optimize import minimize
import yfinance as yf
warnings.filterwarnings("ignore")

# ============================ 1. PARAMETRES ============================
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
BENCH, N_FUND, MAX_CORR = "SPY", 25, 0.85

# nb d'actifs par classe (somme = 25)
QUOTAS = {
    "Serenity": {"bonds": 9, "stocks": 8, "etfs": 5, "commodities": 3, "crypto": 0},
    "Velocity": {"stocks": 13, "etfs": 7, "commodities": 2, "bonds": 2, "crypto": 1},
}
# Serenity : on choisit les moins volatils parmi la moitie haute du classement
# Velocity : on suit le score du selecteur (momentum eleve, beta faible...)
RANKING = {"Serenity": "low_vol", "Velocity": "score"}

# contraintes d'optimisation
RULES = {
    "Serenity": dict(min_w=0.01, max_w=0.10, eq_min=0.00, eq_max=0.35, excess_min=0.0),
    "Velocity": dict(min_w=0.01, max_w=0.12, eq_min=0.70, eq_max=1.00, excess_min=0.06),  # defi : >= rf + 6 pts
}
# ETF "actions" (comptent dans les 70 % du defi) : tout sauf matieres/immo/obligations
NON_EQUITY_ETF_TAGS = {"Gold", "Silver", "Commodities", "Real estate"}

# ============================ 2. CHARGEMENT DES SELECTIONS ============================
def load_class(cls):
    df = pd.read_csv(os.path.join(HERE, f"selection_{cls}.csv"), index_col=0)
    df["cls"] = cls
    return df

def is_equity(cls, ticker, tag):
    if cls == "stocks":
        return True
    if cls == "etfs":
        return tag not in NON_EQUITY_ETF_TAGS
    return False

def shortlist(fund):
    """Candidats par classe selon le quota (avec marge pour le filtre de correlation)."""
    out = {}
    for cls, n in QUOTAS[fund].items():
        if n == 0:
            continue
        d = load_class(cls)
        if RANKING[fund] == "low_vol":
            d = d[d["category"] == "Top half"].sort_values("vol")
        else:
            d = d.sort_values("score", ascending=False)
        out[cls] = d.head(n * 3)        # 3x le quota : reserve pour le filtre de correlation
    return out

# ============================ 3. DONNEES ============================
def download(tickers):
    px = yf.download(sorted(set(tickers + [BENCH])), period="3y", auto_adjust=True, progress=False)["Close"]
    px = px[px[BENCH].notna()].ffill(limit=3)          # jours de bourse US uniquement (crypto 24/7)
    px = px.dropna(axis=1, thresh=int(len(px) * 0.98)).dropna()
    return px

def risk_free():
    r = yf.download("^IRX", period="5d", auto_adjust=True, progress=False)["Close"].dropna()
    return float(np.squeeze(r.iloc[-1])) / 100

# ============================ 4. COMPOSITION (filtre de correlation inter-classes) ============================
def compose(fund, cands, rets):
    chosen, counts = [], {c: 0 for c in QUOTAS[fund]}
    # 1re passe : respecter les quotas avec filtre de correlation
    for cls, d in cands.items():
        for t in d.index:
            if t not in rets.columns or counts[cls] >= QUOTAS[fund][cls]:
                continue
            if all(abs(rets[t].corr(rets[k])) <= MAX_CORR for k in chosen):
                chosen.append(t); counts[cls] += 1
    # 2e passe : completer jusqu'a 25 avec les meilleurs restants (classes avec quota)
    for cls, d in cands.items():
        for t in d.index:
            if len(chosen) >= N_FUND: break
            if t in chosen or t not in rets.columns: continue
            if all(abs(rets[t].corr(rets[k])) <= MAX_CORR for k in chosen):
                chosen.append(t)
    return chosen[:N_FUND]

# ============================ 5. OPTIMISATION ============================
def optimise(fund, tickers, rets, rf, eq_mask):
    R = RULES[fund]
    mu, cov = rets.mean().values * 252, rets.cov().values * 252
    n = len(tickers)

    def neg_sharpe(w):
        return -(w @ mu - rf) / np.sqrt(w @ cov @ w)

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1},
            {"type": "ineq", "fun": lambda w: eq_mask @ w - R["eq_min"]},
            {"type": "ineq", "fun": lambda w: R["eq_max"] - eq_mask @ w},
            {"type": "ineq", "fun": lambda w: w @ mu - (rf + R["excess_min"])}]
    bounds = [(R["min_w"], R["max_w"])] * n   # min_w = 0 -> actifs a poids nul autorises
    best = None
    rng = np.random.default_rng(0)
    for i in range(30):                                  # multistart
        w0 = rng.dirichlet(np.ones(n)) if i else np.ones(n) / n
        r = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=cons,
                     options=dict(maxiter=500, ftol=1e-10))
        if r.success and (best is None or r.fun < best.fun):
            best = r
    if best is None and R["min_w"] > 0:
        print(f"  [{fund}] infaisable avec min_w={R['min_w']:.0%} -> relance sans poids minimum")
        RULES[fund]["min_w"] = 0.0
        return optimise(fund, tickers, rets, rf, eq_mask)
    if best is None:
        raise RuntimeError(f"{fund} : optimisation infaisable (contraintes trop strictes pour cet univers)")
    w = np.where(best.x < 1e-4, 0, best.x); w /= w.sum()
    return w, mu, cov

# ============================ 6. PIPELINE ============================
def run(fund, rf, all_rets, meta):
    cands = shortlist(fund)
    pool = [t for d in cands.values() for t in d.index]
    universe = compose(fund, cands, all_rets[[t for t in pool if t in all_rets.columns]])
    rets = all_rets[universe]
    eq_mask = np.array([1.0 if meta.loc[t, "eq"] else 0.0 for t in universe])
    w, mu, cov = optimise(fund, universe, rets, rf, eq_mask)

    ret, vol = w @ mu, np.sqrt(w @ cov @ w)
    df = pd.DataFrame({"classe": meta.loc[universe, "cls"], "tag": meta.loc[universe, "tag"],
                       "poids": w, "rend_ann": mu, "vol_ann": np.sqrt(np.diag(cov))}, index=universe)
    df.sort_values("poids", ascending=False).to_csv(os.path.join(HERE, f"weights_{fund.lower()}.csv"))
    pd.DataFrame({"ticker": universe}).to_csv(os.path.join(HERE, f"universe_{fund.lower()}.csv"), index=False)

    print("=" * 70); print(f" {fund.upper()}  ({len(universe)} actifs)"); print("=" * 70)
    print(df.sort_values("poids", ascending=False).round(4).to_string())
    print("\nPoids par classe :"); print(df.groupby("classe")["poids"].sum().round(3).to_string())
    print(f"\nActions + ETF actions : {eq_mask @ w:.1%}")
    print(f"Rendement annualise   : {ret:.2%}   (rf = {rf:.2%} -> excedent {ret - rf:+.2%})")
    print(f"Volatilite annualisee : {vol:.2%}")
    print(f"Sharpe                : {(ret - rf) / vol:.3f}")
    corr = rets.corr().values; iu = np.triu_indices(len(universe), 1)
    print(f"Correlation moyenne entre actifs : {corr[iu].mean():.2f}")
    if fund == "Velocity":
        ok = (ret >= rf + 0.06) and (eq_mask @ w >= 0.70)
        print("DEFI (rend >= rf+6 pts et >= 70 % actions) :", "VALIDE" if ok else "NON VALIDE")
    print()

if __name__ == "__main__":
    metas, cands_all = [], []
    for fund in QUOTAS:
        for cls, d in shortlist(fund).items():
            cands_all += list(d.index)
    for cls in ["stocks", "etfs", "bonds", "crypto", "commodities"]:
        d = load_class(cls); d["eq"] = [is_equity(cls, t, g) for t, g in zip(d.index, d["tag"])]
        metas.append(d[["cls", "tag", "eq"]])
    meta = pd.concat(metas)
    meta = meta[~meta.index.duplicated()]

    print("Telechargement des prix (3 ans)...")
    px = download(sorted(set(cands_all)))
    all_rets = np.log(px / px.shift(1)).dropna()
    all_rets = all_rets.drop(columns=[BENCH], errors="ignore") if BENCH not in cands_all else all_rets
    rf = risk_free()
    print(f"{all_rets.shape[1]} actifs, {len(all_rets)} jours, rf (13 sem.) = {rf:.2%}\n")
    for fund in QUOTAS:
        run(fund, rf, all_rets, meta)
