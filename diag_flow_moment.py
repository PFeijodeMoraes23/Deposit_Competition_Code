"""diag_flow_moment.py -- D11: a gross-flow-style second measure of 1-phi from Pix keys.

Author: Pedro Feijo de Moraes

Egan et al. (2025) identify sleepiness twice: from deposit autocorrelation AND from a
gross flow (new account openings, their eq. 8). We have no account microdata
(scrape_14 documents that BCB publishes no per-institution account counts), so this
diagnostic builds the closest on-disk analogue: per-institution Pix KEY stocks
(shared/PIX/pix_participant_keys_panel.csv, ISPB x month, 2020-10+). A depositor
registering a key at an institution is an observable client-relationship event; the
implied quarterly relationship-formation rate

    a_jt = max(0, delta keys_jt) / keys_jt-1

is a flow-based activity measure to set against the awake share 1-phi-hat from the
autocorrelation step.

HONESTY BLOCK (printed and written into the CSV header): this is an order-of-magnitude
triangulation, NOT a moment condition. (i) Delta keys is a NET flow (gross - removals);
(ii) keys != accounts (one person can hold up to 5 keys per account); (iii) coverage
starts 2020-10 (Pix launch) and early quarters are adoption ramp, not steady state;
(iv) PF keys proxy retail depositors. All four push a_jt UP relative to a true
new-relationship rate in the ramp years and make it net-censored afterwards.

Outputs -> DIAG_PHI_SEPARATION/d11_flow_moment.csv + d11_flow_moment.png.
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

from utils import paths as _paths

OUT_DIR = _paths.PROCESSED / "ESTIMATION_OUTPUT" / "DIAG_PHI_SEPARATION"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PIX_DIR = _paths.OPEN_FINANCE / "shared" / "PIX"

# Anchors from the battery (sources printed with the output):
PHI_E2_AVG = 0.918          # D4 full-sample phi-hat(avg market), spec 12 (d_augmented_spreadlevel)
PHI_K = {1: 0.7244, 2: 0.8504, 4: 0.9093, 5: 0.9687}   # D5 type contrast (d_interactions types)
RAMP_END = 202312           # quarters <= this are flagged "adoption ramp"


USERS_CSV = PIX_DIR / "pix_users_dict_national.csv"
USERS_URL = ("https://olinda.bcb.gov.br/olinda/servico/Pix_DadosAbertos/versao/v1/odata/"
             "PixUsuariosCadastradosDICT?$format=json")


def load_national_users():
    """Registered DICT USERS (people), national monthly. Cached after first fetch.

    Why this matters: the institution-level series counts KEYS, and one person may hold up
    to five keys per account, so key growth overstates relationship growth whenever keys per
    user is rising. This endpoint is the only published users series -- it carries NO
    institution dimension (columns are date + PF/PJ/total counts), so it cannot replace the
    per-conglomerate key panel. It is used here to (i) measure the national keys-per-user
    ratio and deflate the key-based rate into user-equivalent units, and (ii) provide a
    national user-growth benchmark.
    """
    if USERS_CSV.exists():
        u = pd.read_csv(USERS_CSV, parse_dates=["date"])
        print(f"  [users] cached {USERS_CSV.name}: {len(u)} months")
        return u
    try:
        import requests
        r = requests.get(USERS_URL, timeout=90, headers={"Accept": "application/json"})
        r.raise_for_status()
        v = r.json().get("value", [])
    except Exception as e:                                        # noqa: BLE001
        print(f"  [users] fetch failed ({type(e).__name__}: {e}); "
              "user-equivalent correction skipped")
        return None
    if not v:
        print("  [users] endpoint returned no rows; correction skipped")
        return None
    u = pd.DataFrame(v).rename(columns={
        "DataGraficosPix": "date", "qtdUsuariosPessoaFisica": "users_pf",
        "qtdUsuariosPessoaJuridica": "users_pj", "qtdUsuariosCadastradosDICTTotal": "users_total"})
    u["date"] = pd.to_datetime(u["date"])
    u = u.sort_values("date")
    try:
        u.to_csv(USERS_CSV, index=False)
        print(f"  [users] fetched {len(u)} months -> {USERS_CSV.name}")
    except OSError as e:
        print(f"  [users] cache not written ({e})")
    return u


def load_keys_to_congl():
    keys = pd.read_csv(PIX_DIR / "pix_participant_keys_panel.csv",
                       dtype={"ISPB": str, "AnoMes": int})
    tl = pd.read_csv(PIX_DIR / "pix_participant_timeline.csv", dtype={"ispb": str})
    # 'CNPJ' is the name build_cnpj_to_congl/map_foundation_to_congl require (8-digit root,
    # zfilled) -- same convention as panel_10_estban_instruments.py:63-67.
    tl["CNPJ"] = (tl["cnpj"].astype(str).str.replace(r"\D", "", regex=True)
                  .str[:8].str.zfill(8))
    keys = keys.merge(tl.loc[tl["CNPJ"].str.len() == 8, ["ispb", "CNPJ"]]
                      .dropna().drop_duplicates("ispb"),
                      left_on="ISPB", right_on="ispb", how="left")
    n0 = len(keys)
    keys = keys.dropna(subset=["CNPJ"])
    print(f"  ISPB->CNPJ: kept {len(keys):,}/{n0:,} rows ({keys['ISPB'].nunique():,} ISPBs)")

    # CNPJ root -> prudential conglomerate, as-of by month (panel_10 pattern)
    from cosif_process_2_calibrate import build_cnpj_to_congl, map_foundation_to_congl
    cmap = build_cnpj_to_congl()
    found = keys[["CNPJ", "AnoMes"]].drop_duplicates()
    mapped = map_foundation_to_congl(found, cmap)[
        ["CNPJ", "AnoMes", "CodConglomeradoPrudencial"]]
    keys = keys.merge(mapped, on=["CNPJ", "AnoMes"], how="left").rename(
        columns={"CodConglomeradoPrudencial": "congl"})
    n1 = len(keys)
    keys = keys.dropna(subset=["congl"])
    print(f"  CNPJ->congl: kept {len(keys):,}/{n1:,} rows ({keys['congl'].nunique():,} conglomerates)")
    return keys


def main():
    print("=== D11: Pix-keys relationship-formation rate vs the awake share 1-phi ===")
    keys = load_keys_to_congl()

    # quarter-end months only -> congl x quarter stocks
    keys["month"] = keys["AnoMes"] % 100
    q = keys[keys["month"].isin([3, 6, 9, 12])].copy()
    q["yq"] = (q["AnoMes"] // 100).astype(str) + "Q" + (q["month"] // 3).astype(str)
    agg = q.groupby(["congl", "AnoMes", "yq"], as_index=False)[["qtd_chaves", "qtd_pf"]].sum()
    agg.sort_values(["congl", "AnoMes"], inplace=True)

    # D/B classification from the market panel's national rows
    mp = pd.read_parquet(_paths.PROCESSED / "market_panel.parquet",
                         columns=["CodConglomeradoPrudencial", "CODMUN_IBGE"])
    d_set = set(mp.loc[mp["CODMUN_IBGE"].astype(str) == "0",
                       "CodConglomeradoPrudencial"].astype(str))
    agg["congl"] = agg["congl"].astype(str)
    agg["group"] = np.where(agg["congl"].isin(d_set), "D", "B")

    for col in ("qtd_chaves", "qtd_pf"):
        lag = agg.groupby("congl")[col].shift(1)
        agg[f"a_{col}"] = np.maximum(0.0, agg[col] - lag) / lag.replace(0, np.nan)
    agg["ramp"] = agg["AnoMes"] <= RAMP_END

    # ---- keys -> USERS correction ------------------------------------------------------
    # a_jt counts KEY formation. What we want is RELATIONSHIP (person) formation. Nationally,
    # keys/user is observable, so the user-equivalent rate is a_jt adjusted for how much of
    # key growth is just people adding more keys:  a_user ~ a_key - dlog(keys per user).
    users = load_national_users()
    kpu = None
    if users is not None:
        nat_keys = (keys[keys["month"].isin([3, 6, 9, 12])]
                    .groupby("AnoMes", as_index=False)[["qtd_chaves", "qtd_pf"]].sum())
        u = users.copy()
        u["AnoMes"] = u["date"].dt.year * 100 + u["date"].dt.month
        m = nat_keys.merge(u[["AnoMes", "users_pf", "users_total"]], on="AnoMes", how="inner")
        if len(m) >= 4:
            m["keys_per_user_pf"] = m["qtd_pf"] / m["users_pf"]
            m["kpu_growth"] = m["keys_per_user_pf"].pct_change()
            kpu = m
            print(f"\n  keys per PF user (national): {m['keys_per_user_pf'].iloc[0]:.3f} "
                  f"({int(m['AnoMes'].iloc[0])}) -> {m['keys_per_user_pf'].iloc[-1]:.3f} "
                  f"({int(m['AnoMes'].iloc[-1])})")
            mature = m[m["AnoMes"] > RAMP_END]
            drift = float(mature["kpu_growth"].median()) if len(mature) else float("nan")
            print(f"  median quarterly growth in keys/user, mature period: {drift:+.4f}")
            agg = agg.merge(m[["AnoMes", "kpu_growth"]], on="AnoMes", how="left")
            # user-equivalent formation rate: strip the part of key growth that is just
            # existing users registering additional keys
            agg["a_user_equiv"] = (agg["a_qtd_pf"] - agg["kpu_growth"]).clip(lower=0.0)
            m.to_csv(OUT_DIR / "d11_keys_per_user.csv", index=False)

    hdr = ("# D11 CAVEATS: net-not-gross; 2020Q4+ only (ramp<=2023 flagged); PF keys proxy "
           "retail. a_qtd_pf counts KEYS; a_user_equiv strips national keys-per-user growth "
           "to approximate PERSON formation (the users series is national-only -- no ISPB -- "
           "so no per-institution user rate exists). Triangulation, not a moment condition.\n")
    with open(OUT_DIR / "d11_flow_moment.csv", "w", encoding="utf-8") as fh:
        fh.write(hdr)
        agg.to_csv(fh, index=False)

    one_m_phi = 1.0 - PHI_E2_AVG
    print(f"\n  quarterly relationship-formation rate a_jt = max(0,dKeys)/Keys_(t-1), PF keys:")
    for grp in ("B", "D"):
        for ramp, tag in ((True, "2020Q4-2023 (ramp)"), (False, "2024+ (mature)")):
            s = agg.loc[(agg["group"] == grp) & (agg["ramp"] == ramp), "a_qtd_pf"].dropna()
            if len(s) == 0:
                continue
            print(f"    {grp} {tag:<20s} n={len(s):>5,}  median={s.median():.4f}  "
                  f"p25={s.quantile(.25):.4f}  p75={s.quantile(.75):.4f}")
    print(f"\n  anchors: 1-phi(E2 avg mkt)={one_m_phi:.3f} quarterly; by type 1-phi_k = "
          + ", ".join(f"k{k}:{1-v:.3f}" for k, v in PHI_K.items()))

    # figure: mature-period distributions with the 1-phi rules
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for grp, color in (("B", "#1565C0"), ("D", "#E64A19")):
        s = agg.loc[(agg["group"] == grp) & (~agg["ramp"]), "a_qtd_pf"].dropna()
        s = s.clip(upper=np.nanquantile(s, 0.99) if len(s) else 1.0)
        if len(s):
            ax.hist(s, bins=40, alpha=0.45, density=True, color=color,
                    label=f"{grp} firms (2024+, n={len(s):,})")
    ax.axvline(one_m_phi, color="#4F4F4F", lw=1.6, ls="--",
               label=f"1−φ̂ (E2 avg mkt) = {one_m_phi:.3f}")
    for k, v in PHI_K.items():
        ax.axvline(1 - v, color="#9E9E9E", lw=0.8, ls=":")
    ax.set_xlabel("quarterly PF-key formation rate  max(0,Δkeys)/keys$_{t-1}$")
    ax.set_ylabel("density")
    ax.set_title("Pix-key relationship formation vs the awake share", loc="left", fontsize=10.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(OUT_DIR / "d11_flow_moment.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)

    s_mat = agg.loc[~agg["ramp"], "a_qtd_pf"].dropna()
    med = float(s_mat.median()) if len(s_mat) else float("nan")
    print(f"\n  VERDICT: mature-period median KEY-formation rate = {med:.4f} "
          f"vs 1-phi-hat = {one_m_phi:.3f}.")
    if "a_user_equiv" in agg.columns:
        su = agg.loc[~agg["ramp"], "a_user_equiv"].dropna()
        if len(su):
            print(f"           USER-EQUIVALENT rate (keys/user growth removed) = "
                  f"{float(su.median()):.4f} -- this is the number to quote, since it counts")
            print("           people forming a relationship rather than keys being registered.")
    print("\n  HOW MUCH THIS TEST CAN ACTUALLY SAY (read before quoting it):")
    print("  Delta-keys is a NET flow. In a stationary market a firm's gross inflow equals its")
    print("  gross outflow, so the net rate tends to ZERO whatever phi is -- the observed")
    print("  positive net rate here reflects the market still expanding, not the awake share.")
    print("  So this moment BOUNDS the gross rate from below and cannot pin it down. It has")
    print("  power against exactly one alternative: a measured formation rate ABOVE 1-phi-hat")
    print("  would contradict the estimate, since no more than 1-phi of depositors can act.")
    print(f"  Observed {float(su.median()) if 'a_user_equiv' in agg.columns and len(su) else med:.4f}")
    print(f"  < 1-phi-hat {one_m_phi:.3f}: the estimate SURVIVES, but treat this as a sanity")
    print("  check, not as the second identification moment Egan et al. get from microdata.")
    print(f"\nresults -> {OUT_DIR / 'd11_flow_moment.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
