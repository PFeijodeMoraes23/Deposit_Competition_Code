"""Why the panel's demand deposits (ESTBAN) are 0.67-0.76 of IF.Data's for the same institutions.

Read-only diagnostic. It reads the raw shared files and market_panel.parquet (column subsets) and writes
only under --out.

The chain, per prudential conglomerate and quarter-end, 2016Q1-2024Q4 (R$):
  I    IF.Data type 1, report 3, account 78282. IF.Data's column composition: (+)[41100000] Depositos a vista.
  P    Cosif document 4060, the prudential conglomerate's consolidated balancete (BLOPRUDENCIAL files),
       account 41100000. For an independent institution (no conglomerate) P is its own document 4020 or 4010.
  S20  the members' individual balancetes with their units abroad: document 4020 where the member files it
       (banks with dependencies abroad; the public files carry it up to 2022), else document 4010.
  S10  the members' individual balancetes, document 4010: the document ESTBAN's verbetes reconcile to.
  E    ESTBAN (document 4500) by municipality, column 401+402+403+404+411+...+419, quarter-end month,
       summed over the members' CNPJs (members = the List's institutions with that prudential code in
       the quarter).
  Es   the same ESTBAN sum over the CNPJs the panel files under the code (build_cnpj_conglomerate_map:
       one code per CNPJ for its whole history, 'CNPJ_<cnpj>' when the map lacks it).
  N    market_panel.parquet dep_a1 summed over the code's rows (B rows from ESTBAN, D rows from IF.Data).

  I - N = (I - P) + (P - S20) + (S20 - S10) + (S10 - E) + (E - Es) + (Es - N)
          source     abroad:      abroad:       units in     membership   panel
          check      subsidiaries branches      Brazil not   (List vs     construction
                     net of       (4020 - 4010) in ESTBAN    static map)
                     eliminations

The same chain is built for savings (41200003, 78283, verbete 420, dep_a2) and time deposits including
judicial deposits (41500002, 78286, verbete 432, dep_a4_incl_judicial) as controls.

Steps:
  --step cache       read the raw files once into <out>/cache/*.parquet
  --step tables      the reconciliation tables (needs the cache)
  --step titles2025  holders of demand deposits from the 2025 titles of 4.1.1 (outside the sample window;
                     a composition check only)

Usage (repo root):
  C:/venvs/egan/Scripts/python.exe diagnostics/diag_estban_demand_gap.py --out <dir> --step cache,tables,titles2025
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import glob
import hashlib
import io
import json
import os
import re
import time
import zipfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from utils import paths, raw_paths
from utils.conglomerates import build_conglomerate_canonical_map, build_cnpj_conglomerate_map

T0 = time.time()
BANK = "CodConglomeradoPrudencial"
Q0, Q1 = 2016 * 4, 2024 * 4 + 3                      # 2016Q1 .. 2024Q4, q = year*4 + quarter - 1
QEND = (3, 6, 9, 12)
COSIF_DIR = Path(paths.COSIF_RAW)
ACC_COSIF = {"41100000": "dem", "41200003": "sav", "41500002": "time", "41300006": "interb", "41000007": "dep"}
ACC_IFD = {"78282": "dem", "78283": "sav", "78286": "time", "78284": "interb", "78285": "other", "78287": "dep"}
EST_COLS = {"V400_401": "dem", "V420": "sav", "V432": "time"}
PANEL_COLS = {"dep_a1": "dem", "dep_a2": "sav", "dep_a4_incl_judicial": "time"}
TYPES = ("dem", "sav", "time")
PANEL = paths.PROCESSED / "market_panel.parquet"
FGC = paths.RAW / "FGC_CENSO" / "censo_produto.parquet"
AGG_DIR = raw_paths.BCB_MONETARY_AGGREGATES_SGS


def stamp(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:5.0f}s] {msg}", flush=True)


def md5_lf(p):
    return hashlib.md5(Path(p).read_bytes().replace(b"\r", b"")).hexdigest()


def qlab(q):
    return f"{int(q) // 4}Q{int(q) % 4 + 1}"


def num(s):
    """Brazilian decimal strings ('1.234,56' or '1234,56') to float."""
    return pd.to_numeric(s.astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
                         errors="coerce")


# ------------------------------------------------------------------------------------------- raw readers
def cosif_file(ym, kind):
    """One file per month and kind (BANCOS, BLOPRUDENCIAL): zipped preferred, ' (1)' copies skipped."""
    rx = re.compile(rf"^{ym}{kind}\.(zip|csv\.zip|csv)$", re.IGNORECASE)
    cands = [p for p in os.listdir(COSIF_DIR) if rx.match(p)]
    if not cands:
        return None
    cands.sort(key=lambda p: 0 if p.lower().endswith("zip") else 1)
    return COSIF_DIR / cands[0]


def read_cosif(kind):
    rows = []
    for q in range(Q0, Q1 + 1):
        y, m = q // 4, (q % 4 + 1) * 3
        f = cosif_file(f"{y}{m:02d}", kind)
        if f is None:
            stamp(f"  Cosif {kind} {y}-{m:02d}: no file")
            continue
        if f.suffix.lower() == ".zip":
            with zipfile.ZipFile(f) as z:
                name = [n for n in z.namelist() if n.upper().endswith(".CSV")][0]
                raw = z.read(name)
        else:
            raw = f.read_bytes()
        d = pd.read_csv(io.BytesIO(raw), sep=";", skiprows=3, encoding="latin-1", dtype=str, on_bad_lines="skip")
        d.columns = [c.upper().lstrip("#").strip() for c in d.columns]
        d = d[d["CONTA"].str.strip().isin(ACC_COSIF)].copy()
        d["acc"] = d["CONTA"].str.strip().map(ACC_COSIF)
        d["v"] = num(d["SALDO"])
        d["q"] = q
        d["CNPJ"] = d["CNPJ"].astype(str).str.strip().str.zfill(8)
        d["file"] = f.name
        keep = ["q", "DOCUMENTO", "CNPJ", "NOME_INSTITUICAO", "COD_CONGL", "acc", "v", "file"]
        rows.append(d[[c for c in keep if c in d.columns]])
    out = pd.concat(rows, ignore_index=True)
    out["DOCUMENTO"] = out["DOCUMENTO"].astype(str).str.strip()
    return out


def read_ifdata(folder):
    rows = []
    for q in range(Q0, Q1 + 1):
        y, m = q // 4, (q % 4 + 1) * 3
        f = Path(paths.IF_DATA_ROOT) / folder / f"IF_DATA_Values_{y}_{m}.csv"
        d = pd.read_csv(f, encoding="latin-1", dtype=str, usecols=["CodInst", "NumeroRelatorio", "Conta", "Saldo"])
        d = d[(d["NumeroRelatorio"] == "3") & d["Conta"].isin(ACC_IFD)].copy()
        d["acc"] = d["Conta"].map(ACC_IFD)
        d["v"] = num(d["Saldo"])
        d["q"] = q
        rows.append(d[["q", "CodInst", "acc", "v"]])
    return pd.concat(rows, ignore_index=True)


def read_list():
    rows = []
    for q in range(Q0, Q1 + 1):
        y, m = q // 4, (q % 4 + 1) * 3
        L = pd.read_csv(paths.IF_DATA_LIST / f"IF_DATA_List_{y}_{m}.csv", dtype=str,
                        usecols=["CodInst", "NomeInstituicao", "Tcb", "Td", "Tc", "Sr", "CodConglomeradoPrudencial",
                                 "CnpjInstituicaoLider"])
        L = L.replace({"null": np.nan})
        L["q"] = q
        rows.append(L)
    return pd.concat(rows, ignore_index=True)


def read_estban():
    """Quarter-end months of ESTBAN.csv, read in chunks (the file is 349 MB), summed by CNPJ and quarter."""
    cols = ["CNPJ", "NOME_INSTITUICAO", "CODMUN_IBGE", "YEAR", "MONTH", "AGEN_PROCESSADAS"] + list(EST_COLS)
    parts = []
    for ch in pd.read_csv(paths.ESTBAN_CSV, encoding="latin-1", dtype={"CNPJ": str, "NOME_INSTITUICAO": str},
                          usecols=cols, low_memory=False, chunksize=400_000):
        ch["YEAR"] = pd.to_numeric(ch["YEAR"], errors="coerce")
        ch["MONTH"] = pd.to_numeric(ch["MONTH"], errors="coerce")
        parts.append(ch[ch["MONTH"].isin(QEND) & ch["YEAR"].between(2016, 2024)])
    e = pd.concat(parts, ignore_index=True)
    e["q"] = (e["YEAR"] * 4 + e["MONTH"] // 3 - 1).astype(int)
    e["CNPJ"] = e["CNPJ"].astype(str).str.strip().str.zfill(8)
    for c in list(EST_COLS) + ["AGEN_PROCESSADAS", "CODMUN_IBGE"]:
        e[c] = pd.to_numeric(e[c], errors="coerce")
    e["unfiled"] = e["AGEN_PROCESSADAS"].fillna(0) <= 0
    e["no_ibge"] = e["CODMUN_IBGE"].isna() | (e["CODMUN_IBGE"] == 0)
    g = e.groupby(["CNPJ", "q"])
    out = g[list(EST_COLS)].sum(min_count=1)
    out["n_cells"] = g.size()
    out["n_unfiled"] = g["unfiled"].sum()
    out["n_no_ibge"] = g["no_ibge"].sum()
    for c in EST_COLS:
        out[f"{c}_no_ibge"] = e[e["no_ibge"]].groupby(["CNPJ", "q"])[c].sum().reindex(out.index).fillna(0.0)
    out["name"] = g["NOME_INSTITUICAO"].first()
    return out.reset_index()


def read_panel():
    cols = [BANK, "NomeInstituicao", "mca_code", "is_B", "Source", "year", "quarter"] + list(PANEL_COLS) + ["dep_a4"]
    p = pq.read_table(PANEL, columns=cols).to_pandas()
    p = p[p["year"].between(2016, 2024)].copy()
    p["q"] = p["year"].astype(int) * 4 + p["quarter"].astype(int) - 1
    p[BANK] = p[BANK].astype(str)
    dup = int(p.duplicated(subset=[BANK, "mca_code", "q"]).sum())
    return p, dup


def read_fgc():
    t = pq.read_table(FGC, columns=["sr", "cod_competencia", "num_produto", "num_titularidade", "val_deposito"]).to_pandas()
    t = t[(t["num_produto"] == 1) & (t["cod_competencia"] % 100 % 3 == 0) & (t["cod_competencia"] <= 202412)]
    t["q"] = (t["cod_competencia"] // 100) * 4 + (t["cod_competencia"] % 100) // 3 - 1
    return t.groupby(["q", "sr", "num_titularidade"])["val_deposito"].sum().reset_index()


def read_m1_demand():
    f = Path(AGG_DIR) / "sgs_27790_2015-2024.json"
    if not f.exists():
        return None
    j = json.loads(f.read_text(encoding="utf-8"))
    s = pd.Series({pd.to_datetime(r["data"], dayfirst=True): float(r["valor"]) * 1e3 for r in j})
    s = s[s.index.month.isin(QEND)]
    s.index = s.index.year * 4 + s.index.month // 3 - 1
    return s.rename("m1_demand")


def step_cache(out):
    c = out / "cache"
    c.mkdir(parents=True, exist_ok=True)
    stamp("Cosif BANCOS (4010/4016/4020/4026) ...")
    read_cosif("BANCOS").to_parquet(c / "cosif_bancos.parquet")
    stamp("Cosif BLOPRUDENCIAL (4060/4066) ...")
    read_cosif("BLOPRUDENCIAL").to_parquet(c / "cosif_prud.parquet")
    stamp("IF.Data type 1 ...")
    read_ifdata("Prudential Conglomerates").to_parquet(c / "ifd_t1.parquet")
    stamp("IF.Data type 3 ...")
    read_ifdata("Individual Institutions").to_parquet(c / "ifd_t3.parquet")
    stamp("List ...")
    read_list().to_parquet(c / "list.parquet")
    stamp("ESTBAN ...")
    read_estban().to_parquet(c / "estban.parquet")
    stamp("panel ...")
    p, dup = read_panel()
    p.to_parquet(c / "panel.parquet")
    stamp("FGC census ...")
    read_fgc().to_parquet(c / "fgc_demand.parquet")
    m1 = read_m1_demand()
    if m1 is not None:
        m1.to_frame().to_parquet(c / "m1_demand.parquet")
    meta = {"panel_md5": hashlib.md5(PANEL.read_bytes()).hexdigest(), "panel_duplicate_cells": dup,
            "estban_csv": str(paths.ESTBAN_CSV), "cosif_dir": str(COSIF_DIR), "ifdata_root": str(paths.IF_DATA_ROOT)}
    (c / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    stamp("cache done")


# ------------------------------------------------------------------------------------------------ tables
def load_cache(out):
    c = out / "cache"
    r = {k: pd.read_parquet(c / f"{k}.parquet") for k in
         ("cosif_bancos", "cosif_prud", "ifd_t1", "ifd_t3", "list", "estban", "panel", "fgc_demand")}
    r["m1"] = pd.read_parquet(c / "m1_demand.parquet")["m1_demand"] if (c / "m1_demand.parquet").exists() else None
    r["meta"] = json.loads((c / "meta.json").read_text(encoding="utf-8"))
    return r


def panel_code_of_cnpj(cnpjs, static):
    """The code panel_deposits files an ESTBAN CNPJ under (static map, else 'CNPJ_<int>')."""
    out = {}
    for c in pd.Series(cnpjs).dropna().unique():
        k = int(c)
        out[c] = static[k][0] if k in static and static[k][0] is not None else f"CNPJ_{k}"
    return out


def build(r):
    canon, _ = build_conglomerate_canonical_map()
    static = build_cnpj_conglomerate_map()
    L = r["list"]
    # ---- IF.Data type 1, wide by acc; code -> canonical (independents: the panel's code of their CNPJ)
    t1 = r["ifd_t1"].pivot_table(index=["CodInst", "q"], columns="acc", values="v", aggfunc="sum").reset_index()
    is_c = t1["CodInst"].str.startswith("C")
    ind_map = panel_code_of_cnpj(t1.loc[~is_c, "CodInst"], static)
    t1["code"] = np.where(is_c, t1["CodInst"].map(lambda c: canon.get(c, c)), t1["CodInst"].map(ind_map))
    Lq = L.set_index(["CodInst", "q"])
    t1 = t1.join(Lq[["Tcb", "Td", "Sr", "Tc", "NomeInstituicao"]], on=["CodInst", "q"])
    # ---- members per (reported prudential code, q) from the List; independents are their own member
    mem = L[L["CodInst"].str.fullmatch(r"\d{8}") & L["CodConglomeradoPrudencial"].notna()][
        ["CodInst", "q", "CodConglomeradoPrudencial"]].rename(columns={"CodInst": "CNPJ", "CodConglomeradoPrudencial": "CodInst"})
    ind = t1.loc[~is_c, ["CodInst", "q"]].assign(CNPJ=lambda d: d["CodInst"])
    mem = pd.concat([mem, ind[["CNPJ", "q", "CodInst"]]], ignore_index=True).drop_duplicates(["CNPJ", "q"])
    # ---- Cosif per CNPJ: 4010 and 4020 (4020 falls back to 4010)
    b = r["cosif_bancos"]
    b10 = b[b["DOCUMENTO"] == "4010"].pivot_table(index=["CNPJ", "q"], columns="acc", values="v", aggfunc="sum")
    b20 = b[b["DOCUMENTO"] == "4020"].pivot_table(index=["CNPJ", "q"], columns="acc", values="v", aggfunc="sum")
    has20 = pd.Series(True, index=b20.index, name="has4020")
    # ---- ESTBAN per CNPJ
    e = r["estban"].set_index(["CNPJ", "q"])
    # ---- per CNPJ table on the union of keys
    keys = sorted(set(b10.index) | set(e.index) | set(mem.set_index(["CNPJ", "q"]).index))
    cn = pd.DataFrame(index=pd.MultiIndex.from_tuples(keys, names=["CNPJ", "q"]))
    for k in TYPES:
        cn[f"S10_{k}"] = b10[k].reindex(cn.index) if k in b10 else np.nan
        cn[f"S20_{k}"] = (b20[k].reindex(cn.index) if k in b20 else pd.Series(np.nan, index=cn.index))
        cn[f"S20_{k}"] = cn[f"S20_{k}"].fillna(cn[f"S10_{k}"])
        cn[f"E_{k}"] = e[{"dem": "V400_401", "sav": "V420", "time": "V432"}[k]].reindex(cn.index)
    cn["has4020"] = has20.reindex(cn.index).fillna(False).astype(bool)
    cn["in_estban"] = e["n_cells"].reindex(cn.index).notna()
    cn = cn.reset_index()
    cn = cn.merge(mem, on=["CNPJ", "q"], how="left")                      # CodInst = the reported conglomerate
    ind_codes = cn["CodInst"].dropna()
    ind_pc = panel_code_of_cnpj(ind_codes[~ind_codes.str.startswith("C")], static)
    cn["code_list"] = cn["CodInst"].map(
        lambda c: (canon.get(c, c) if c.startswith("C") else ind_pc.get(c, np.nan)) if isinstance(c, str) else np.nan)
    pc = panel_code_of_cnpj(cn.loc[cn["in_estban"], "CNPJ"], static)
    cn["code_static"] = cn["CNPJ"].map(pc)
    # ---- Cosif 4060 per conglomerate (canonical)
    pr = r["cosif_prud"]
    pr = pr[pr["DOCUMENTO"] == "4060"].copy()
    pr["code"] = pr["COD_CONGL"].map(lambda c: canon.get(c, c))
    P = pr.pivot_table(index=["code", "q"], columns="acc", values="v", aggfunc="sum")
    # ---- panel per code
    p = r["panel"]
    N = p.groupby([BANK, "q"])[list(PANEL_COLS)].sum(min_count=1).rename(columns=PANEL_COLS)
    Nsrc = p.groupby([BANK, "q"])["Source"].agg(lambda s: "+".join(sorted(set(s.dropna()))))
    NB = p[p["Source"] == "ESTBAN"].groupby([BANK, "q"])[list(PANEL_COLS)].sum(min_count=1).rename(columns=PANEL_COLS)
    return dict(t1=t1, cn=cn, P=P, N=N, NB=NB, Nsrc=Nsrc, canon=canon, static=static, has20=has20)


def chain(B):
    """Per (code, q): I, P, S20, S10, E (List members), Es (static map), N, by type."""
    t1, cn, P, N, NB = B["t1"], B["cn"], B["P"], B["N"], B["NB"]
    I = t1.groupby(["code", "q"])[[k for k in TYPES]].sum(min_count=1).add_prefix("I_")
    td = t1.groupby(["code", "q"])["Td"].first()
    tcb = t1.groupby(["code", "q"])["Tcb"].first()
    sr = t1.groupby(["code", "q"])["Sr"].first()
    nm = t1.groupby(["code", "q"])["NomeInstituicao"].first()
    agg_l = cn[cn["code_list"].notna()].groupby(["code_list", "q"])[
        [f"{s}_{k}" for s in ("S20", "S10", "E") for k in TYPES]].sum(min_count=1)
    agg_l.index.names = ["code", "q"]
    agg_s = cn[cn["code_static"].notna()].groupby(["code_static", "q"])[[f"E_{k}" for k in TYPES]].sum(min_count=1)
    agg_s.index.names = ["code", "q"]
    agg_s.columns = [c.replace("E_", "Es_") for c in agg_s.columns]
    n4020 = cn[cn["code_list"].notna()].groupby(["code_list", "q"])["has4020"].sum()
    n4020.index.names = ["code", "q"]
    idx = I.index
    T = pd.DataFrame(index=idx)
    T = T.join(I).join(agg_l).join(agg_s)
    Pk = P.reindex(idx)
    for k in TYPES:
        T[f"P_{k}"] = Pk[k] if k in Pk else np.nan
        # independents and conglomerates without a 4060 in the public file: P is their own S20
        T[f"P_{k}"] = T[f"P_{k}"].fillna(T[f"S20_{k}"])
        T[f"N_{k}"] = N[k].reindex(idx)
        T[f"NB_{k}"] = NB[k].reindex(idx)
    T["P_src"] = np.where(Pk["dem"].notna() if "dem" in Pk else False, "4060", "own S20")
    T["n_mem_4020"] = n4020.reindex(idx).fillna(0)
    T["Td"], T["Tcb"], T["Sr"], T["name"] = td, tcb, sr, nm
    T["source"] = B["Nsrc"].reindex(idx)
    T = T.reset_index()
    T["in_panel"] = T["N_dem"].notna() | T["N_sav"].notna() | T["N_time"].notna()
    T["in_panel_step1"] = T["code"].isin(set(B["N"].index.get_level_values(0)))
    return T


def decompose(T, k, sel):
    """National sums over the rows `sel` by quarter: I, N and the five terms (R$)."""
    t = T[sel].copy()
    f = lambda c: t[f"{c}_{k}"].fillna(0.0)
    t["I"], t["N"] = f("I"), f("N")
    t["src_check"] = f("I") - f("P")
    t["abroad_subs_net"] = f("P") - f("S20")
    t["abroad_branches"] = f("S20") - f("S10")
    t["brazil_not_in_estban"] = f("S10") - f("E")
    t["membership"] = f("E") - f("Es")
    t["panel_construction"] = f("Es") - f("N")
    cols = ["I", "N", "src_check", "abroad_subs_net", "abroad_branches", "brazil_not_in_estban", "membership",
            "panel_construction"]
    g = t.groupby("q")[cols].sum()
    g["gap"] = g["I"] - g["N"]
    g["ratio_N_over_I"] = g["N"] / g["I"]
    g["check_sum_terms_minus_gap"] = g[cols[2:]].sum(axis=1) - g["gap"]
    return g


def step_tables(out):
    r = load_cache(out)
    stamp("build ...")
    B = build(r)
    T = chain(B)
    T.to_csv(out / "chain_by_code_q.csv", index=False)
    res = {"meta": r["meta"]}
    # ---- (0) step 1's in-both ratio, reproduced, and the same on the canonical/independent mapping
    p = r["panel"]
    t1raw = r["ifd_t1"]
    # step 1 (coverage.py): IF.Data per reported CodInst x q, inner-merged with the panel per code x q
    xi = t1raw.pivot_table(index=["CodInst", "q"], columns="acc", values="v", aggfunc="sum").reset_index()
    pb = p.groupby([BANK, "q"])[list(PANEL_COLS)].sum().reset_index().rename(columns={BANK: "CodInst"})
    mm = xi.merge(pb, on=["CodInst", "q"], how="inner").groupby("q").sum(numeric_only=True)
    rep = pd.DataFrame({"step1_dem": mm["dep_a1"] / mm["dem"], "step1_sav": mm["dep_a2"] / mm["sav"],
                        "step1_time_incl_jud": mm["dep_a4_incl_judicial"] / mm["time"]})
    sel_inpanel = T["in_panel"]
    for k in TYPES:
        g = T[sel_inpanel].groupby("q")[[f"I_{k}", f"N_{k}"]].sum()
        rep[f"canon_{k}"] = g[f"N_{k}"] / g[f"I_{k}"]
    rep.index = [qlab(q) for q in rep.index]
    rep.to_csv(out / "ratio_reproduction.csv")
    res["ratio_reproduction"] = {c: [float(rep[c].min()), float(rep[c].max())] for c in rep.columns}
    stamp("ratios: " + json.dumps({c: [round(v, 3) for v in res['ratio_reproduction'][c]] for c in rep.columns}))
    # ---- (1) national decomposition by type, on the in-panel set
    for k in TYPES:
        g = decompose(T, k, sel_inpanel)
        g.index = [qlab(q) for q in g.index]
        g.to_csv(out / f"decomp_national_{k}.csv")
        res[f"decomp_{k}_max_abs_check"] = float(g["check_sum_terms_minus_gap"].abs().max())
        # the same on the codes the panel takes from ESTBAN (B tier) and from IF.Data (D tier)
        for lab, src in (("estban_tier", "ESTBAN"), ("ifdata_tier", "IFDATA")):
            g = decompose(T, k, sel_inpanel & (T["source"] == src))
            g.index = [qlab(q) for q in g.index]
            g.to_csv(out / f"decomp_national_{k}_{lab}.csv")
    # by year (Q4), R$ bn, demand, and the D/B split
    # ---- (2) conglomerates carrying the demand gap (pooled over quarters and Q4s)
    k = "dem"
    t = T[sel_inpanel].copy()
    for c in ("I", "P", "S20", "S10", "E", "Es", "N"):
        t[c] = t[f"{c}_{k}"].fillna(0.0)
    t["gap"] = t["I"] - t["N"]
    t["abroad"] = t["P"] - t["S10"]
    pooled = t.groupby("code").agg(name=("name", "last"), Tcb=("Tcb", "last"), Sr=("Sr", "last"), source=("source", "last"),
                                   I=("I", "sum"), N=("N", "sum"), gap=("gap", "sum"), abroad=("abroad", "sum"),
                                   S10=("S10", "sum"), E=("E", "sum"), n_q=("q", "nunique"))
    tot_gap = pooled["gap"].sum()
    pooled["share_of_gap"] = pooled["gap"] / tot_gap
    pooled["ratio"] = pooled["N"] / pooled["I"]
    pooled = pooled.sort_values("gap", ascending=False)
    pooled.to_csv(out / "gap_by_conglomerate_pooled_dem.csv")
    yr = t[t["q"] % 4 == 3].pivot_table(index="code", columns="q", values="gap", aggfunc="sum") / 1e9
    yr.columns = [qlab(q) for q in yr.columns]
    yr.to_csv(out / "gap_by_conglomerate_q4_dem_bn.csv")
    # ---- (3) per-CNPJ check of ESTBAN against 4010 (Brazil units), all ESTBAN banks, by type and quarter
    cn = B["cn"]
    ce = cn[cn["in_estban"]]
    chk = ce.groupby("q")[[f"{s}_{k}" for s in ("S10", "S20", "E") for k in TYPES]].sum()
    for k2 in TYPES:
        chk[f"E_over_S10_{k2}"] = chk[f"E_{k2}"] / chk[f"S10_{k2}"]
    chk.index = [qlab(q) for q in chk.index]
    chk.to_csv(out / "estban_vs_4010_all_estban_banks.csv")
    # 4010 filers that do not file ESTBAN: their demand deposits
    ne = cn[~cn["in_estban"] & cn["S10_dem"].notna()].groupby("q")["S10_dem"].sum()
    ne.index = [qlab(q) for q in ne.index]
    ne.rename("S10_dem_not_in_estban").to_csv(out / "s10_demand_not_in_estban.csv")
    # per bank, where E and S10 differ most (pooled abs)
    ce2 = ce.assign(d=ce["E_dem"].fillna(0) - ce["S10_dem"].fillna(0))
    bybank = ce2.groupby("CNPJ").agg(E=("E_dem", "sum"), S10=("S10_dem", "sum"), d=("d", "sum"),
                                     absd=("d", lambda s: s.abs().sum())).sort_values("absd", ascending=False)
    names = r["estban"].groupby("CNPJ")["name"].last()
    bybank["name"] = names.reindex(bybank.index)
    bybank.head(40).to_csv(out / "estban_minus_4010_by_bank_dem.csv")
    # ---- (4) IF.Data type 3 (individual): does it follow 4010 or 4020?
    t3 = r["ifd_t3"][r["ifd_t3"]["acc"] == "dem"].rename(columns={"CodInst": "CNPJ", "v": "ifd3_dem"})[["CNPJ", "q", "ifd3_dem"]]
    m3 = cn[["CNPJ", "q", "S10_dem", "S20_dem", "has4020"]].merge(t3, on=["CNPJ", "q"], how="inner")
    m3 = m3[m3["has4020"]]
    m3["eq_4010"] = (m3["ifd3_dem"] - m3["S10_dem"]).abs() <= 1e-6 * m3["ifd3_dem"].abs().clip(lower=1e6)
    m3["eq_4020"] = (m3["ifd3_dem"] - m3["S20_dem"]).abs() <= 1e-6 * m3["ifd3_dem"].abs().clip(lower=1e6)
    res["ifd_type3_vs_docs_for_4020_filers"] = {"rows": int(len(m3)), "equals_4010": int(m3["eq_4010"].sum()),
                                               "equals_4020": int(m3["eq_4020"].sum())}
    # IF.Data type 1 against Cosif 4060
    tt = T[T["P_src"] == "4060"]
    rel = ((tt["I_dem"] - tt["P_dem"]).abs() / tt["I_dem"].abs().clip(lower=1e6))
    res["ifd_type1_vs_4060_dem"] = {"rows": int(len(tt)), "share_rel_le_1e-6": float((rel <= 1e-6).mean()),
                                    "max_abs_bn": float((tt["I_dem"] - tt["P_dem"]).abs().max() / 1e9)}
    # ---- (5) holders: M1 demand deposits and the FGC census against the panel's domestic demand
    agg = {}
    if r["m1"] is not None:
        agg["m1_demand"] = r["m1"]
    nip = T[~sel_inpanel]
    agg["ifd_dem_not_in_panel_B3S"] = nip[nip["Tcb"] == "B3S"].groupby("q")["I_dem"].sum()
    agg["ifd_dem_not_in_panel_other"] = nip[nip["Tcb"] != "B3S"].groupby("q")["I_dem"].sum()
    agg["ifd_dem_in_panel"] = T[sel_inpanel].groupby("q")["I_dem"].sum()
    agg["panel_dem"] = r["panel"].groupby("q")["dep_a1"].sum()
    agg["panel_dem_B_estban"] = r["panel"][r["panel"]["Source"] == "ESTBAN"].groupby("q")["dep_a1"].sum()
    agg["estban_dem_all_cnpj"] = r["estban"].groupby("q")["V400_401"].sum()
    agg["cosif4010_dem_all_banks"] = cn[cn["S10_dem"].notna()].groupby("q")["S10_dem"].sum()
    fg = r["fgc_demand"]
    for h, lab in ((1, "persons"), (2, "firms"), (3, "firms_no_cover"), (4, "transferable")):
        agg[f"fgc_dem_{lab}"] = fg[fg["num_titularidade"] == h].groupby("q")["val_deposito"].sum()
    A = pd.DataFrame(agg).loc[Q0:Q1]
    A["m1_minus_cu"] = (A["m1_demand"] if "m1_demand" in A else np.nan) - A["ifd_dem_not_in_panel_B3S"]
    A["m1_minus_cu_over_estban_all"] = A["m1_minus_cu"] / A["estban_dem_all_cnpj"]
    A["fgc_total"] = A[[c for c in A.columns if c.startswith("fgc_dem_")]].sum(axis=1, min_count=1)
    A["fgc_persons_firms_over_estban_all"] = (A["fgc_dem_persons"] + A["fgc_dem_firms"]) / A["estban_dem_all_cnpj"]
    A["fgc_total_over_estban_all"] = A["fgc_total"] / A["estban_dem_all_cnpj"]
    A["fgc_total_over_ifd_in_panel"] = A["fgc_total"] / A["ifd_dem_in_panel"]
    A.index = [qlab(q) for q in A.index]
    A.to_csv(out / "holders_national_dem.csv")
    # FGC by segment against IF.Data and ESTBAN by segment (Sr of the conglomerate in IF.Data)
    seg = T[sel_inpanel & T["Sr"].notna()].groupby(["q", "Sr"])[["I_dem", "E_dem", "Es_dem", "S10_dem", "P_dem"]].sum()
    fgs = fg.groupby(["q", "sr"])["val_deposito"].sum().rename("fgc_total")
    fgs.index.names = ["q", "Sr"]
    fpf = fg[fg["num_titularidade"].isin([1, 2])].groupby(["q", "sr"])["val_deposito"].sum().rename("fgc_persons_firms")
    fpf.index.names = ["q", "Sr"]
    seg = seg.join(fgs, how="outer").join(fpf, how="outer").reset_index()
    seg = seg[seg["q"].between(Q0, Q1)]
    seg["q"] = seg["q"].map(qlab)
    seg.to_csv(out / "holders_by_segment_dem.csv", index=False)
    (out / "summary.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    stamp("tables done")


# ------------------------------------------------------------- holders, from the 2025 titles of 4.1.1
# From 2025-01 the public balancetes carry the titles of 4.1.1 (Cosif plan of IN BCB 429/2023; Annex I maps
# every title to an ESTBAN verbete in 401-419). 2025 lies outside the sample (2016-2024): these shares are a
# composition check only.
TITLES_2025 = {
    "4110000006": "total", "4110500001": "related_parties", "4111000005": "persons", "4112000004": "firms",
    "4113000003": "financial_institutions", "4114000002": "governments", "4116000000": "non_residents",
    "4116500005": "treasury_special", "4117500004": "obligatory", "4117700000": "obligatory_related",
    "4118000008": "fiscal_incentives", "4118500003": "linked", "4119000007": "credit_balances_loans",
    "4119800001": "closed_accounts"}


def read_titles_2025(kind, doc):
    rows = []
    for m in QEND:
        f = cosif_file(f"2025{m:02d}", kind)
        if f is None:
            continue
        with zipfile.ZipFile(f) as z:
            raw = z.read([n for n in z.namelist() if n.upper().endswith(".CSV")][0])
        d = pd.read_csv(io.BytesIO(raw), sep=";", skiprows=3, encoding="latin-1", dtype=str, on_bad_lines="skip")
        d.columns = [c.upper().lstrip("#").strip() for c in d.columns]
        d = d[(d["DOCUMENTO"].str.strip() == doc) & d["CONTA"].str.strip().isin(TITLES_2025)].copy()
        d["t"] = d["CONTA"].str.strip().map(TITLES_2025)
        d["v"] = num(d["SALDO"])
        d["month"] = m
        d["CNPJ"] = d["CNPJ"].astype(str).str.strip().str.zfill(8)
        rows.append(d[["month", "CNPJ", "NOME_INSTITUICAO", "COD_CONGL", "t", "v"]])
    return pd.concat(rows, ignore_index=True)


def step_titles2025(out):
    canon, _ = build_conglomerate_canonical_map()
    est = pd.read_parquet(out / "cache" / "estban.parquet")
    estban_cnpjs = set(est["CNPJ"])
    b = read_titles_2025("BANCOS", "4010")
    p = read_titles_2025("BLOPRUDENCIAL", "4060")
    b["in_estban"] = b["CNPJ"].isin(estban_cnpjs)
    sub = [t for t in TITLES_2025.values() if t != "total"]
    # national composition of the ESTBAN banks' 4010 demand deposits (the panel's universe), by quarter-end
    w = b[b["in_estban"]].pivot_table(index="month", columns="t", values="v", aggfunc="sum")
    w["sum_titles"] = w[[c for c in sub if c in w]].sum(axis=1)
    comp = w.div(w["total"], axis=0)
    comp.to_csv(out / "titles2025_composition_national_4010_estban_banks.csv")
    # by bank (ESTBAN banks), mean over the four 2025 quarter-ends, R$ and shares
    wb = b[b["in_estban"]].pivot_table(index=["CNPJ", "month"], columns="t", values="v", aggfunc="sum").groupby("CNPJ").mean()
    nm = b.groupby("CNPJ")["NOME_INSTITUICAO"].last()
    wb.insert(0, "name", nm.reindex(wb.index))
    wb = wb.sort_values("total", ascending=False)
    wb.to_csv(out / "titles2025_by_bank_4010.csv")
    # prudential conglomerates: 4060 by title, and 4060 minus the members' 4010 by title (the part abroad)
    p["code"] = p["COD_CONGL"].map(lambda c: canon.get(c, c))
    wp = p.pivot_table(index=["code", "month"], columns="t", values="v", aggfunc="sum").groupby("code").mean()
    L = pd.read_parquet(out / "cache" / "list.parquet")
    L4 = L[(L["q"] == Q1) & L["CodInst"].str.fullmatch(r"\d{8}") & L["CodConglomeradoPrudencial"].notna()].copy()
    L4["code"] = L4["CodConglomeradoPrudencial"].map(lambda c: canon.get(c, c))
    mb = b.merge(L4[["CodInst", "code"]].rename(columns={"CodInst": "CNPJ"}), on="CNPJ", how="inner")
    w10 = mb.pivot_table(index=["code", "month"], columns="t", values="v", aggfunc="sum").groupby("code").mean()
    top = wp.sort_values("total", ascending=False).head(12).index
    diff = (wp.reindex(top) - w10.reindex(top).fillna(0.0))
    pd.concat({"c4060": wp.reindex(top), "members_4010": w10.reindex(top), "c4060_minus_4010": diff}, axis=1).to_csv(
        out / "titles2025_prudential_vs_members.csv")
    stamp("titles2025 done")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", default="cache,tables,titles2025")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    steps = a.step.split(",")
    if "cache" in steps:
        step_cache(out)
    if "tables" in steps:
        step_tables(out)
    if "titles2025" in steps:
        step_titles2025(out)
    files = sorted(p for p in out.glob("*.csv")) + sorted(out.glob("*.json"))
    (out / "MD5SUMS.txt").write_text("".join(f"{md5_lf(f)}  {f.name}\n" for f in files), encoding="utf-8")


if __name__ == "__main__":
    main()
