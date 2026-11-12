"""
make_lambda_registry.py
=======================
lambda, the share of a bank's time deposits that can be redeemed before maturity, from the BCB's
answer to registry request 18810.025399/2026-18 (CDB stock by prudential segment, early-redemption
flag and holder type, month-end 2017-06..2024-12), in five versions, by panel conglomerate and
quarter, 2016Q1-2024Q4. Data construction only: no estimate reads these series (K1 keeps the
statements' lambda until Pedro switches it; claude/MAIN_REPLY_2026-11-05.md, 11-11 18:30 and 18:38).

Inputs (read only)
  DATA_RAW/BCB/registry_025399/raw/Estoque captações segmento liquidez nat detentor.xlsx
      sheet Dados: ANO_MES (YYYYMM), SEG (S1-S5), RESGATE (S/N), DETENTOR_NATUREZA, VALOR (R$)
  DATA_RAW/BCB/segmentacao_prudencial/ifdata_cadastro/IF_DATA_List_<y>_<m>.csv, 2016-03..2024-12
      the BCB's IF.data register (Olinda service IFDATA, IfDataCadastro), one file per quarter-end;
      field Sr = "Segmento Resolução nº 4.553/2017 com classificação entre S1 e S5" (the service's
      documentation); empty before 2017-06, no S5 before 2018-03
  processed/market_panel.parquet (pyarrow column subsets): conglomerate, name, is_B, dep_a4
      (time deposits), the panel's own segment column (compared, not used)
  Rout/k0_omega.csv: the statements' share_no_maturity by conglomerate-year (K0 (e)), for S-stmt

Registry shares
  share_S(s, month, holders) = the stock with RESGATE = S over the whole stock of segment s in the
  month, for holders pfpj (DETENTOR_NATUREZA = PF/PJ, the headline) or all (every holder type).
  S = the CDB can be redeemed before maturity, N = it cannot (Pedro, 11-11 18:38): the S share is an
  upper bound on daily liquidity. The quarterly value is the quarter's last month. Fill: before a
  segment's data start its first available value is held back (2017-06 for S1-S4). S5 phases into
  the registry over 2018-02..2019-02, so its months before FIRST_FULL (2019-03, the first full-stock
  month) are not used and 2019-03 is held back (Pedro, 11-11 21:25). The keys leave those months
  out as well: reg.lambda.pfpj.all.min counts S5 from its first full-stock December, 2019-12 (main
  thread, 11-11 22:55); the rule is recorded as first_full_month in lambda_registry_numbers.json.

Segment map (conglomerate j, quarter t), each assignment with its source (seg_src)
  code        the IF.data row of j itself (CodInst = j) in t, its Sr
  leader      j has no row in t (a code the panel keeps after the BCB retired it, e.g. Banco PAN once
              it joined BTG Pactual's prudential conglomerate): the row in t of j's leader institution
              (CnpjInstituicaoLider of j's last own row), with the Sr of the prudential conglomerate
              that institution belongs to in t, else its own Sr
  held_back   no Sr in t from either route (every quarter before 2017Q2, when the segmentation starts):
              j's first available assignment, for quarters before it
  carried     the same after j's first assignment: the last one before t
  none        j never has an Sr (its R-* lambdas are missing)

Versions (claude/MAIN_REPLY_2026-11-05.md, 11-11 18:38)
  Rq (R-q, headline)  share_S(segment of j in t, last month of t)
  Rconst (R-const)    one value per segment, the mean of its quarterly shares over 2018Q1-2024Q4
                      (S5: 2019Q1-2024Q4, its full-stock quarters); membership by quarter
  Rfixmem (R-fixmem)  quarterly shares as Rq, j's membership fixed at 2019Q4 (its first panel quarter
                      if it enters later; its last panel quarter if it leaves before 2019Q4)
  Rannual (R-annual)  the segment's December share applied to the four quarters of the next year (the
                      statements' timing), December 2017 also to 2017 and 2016; where that December
                      precedes the segment's first December (S5: 2019, its first full-stock December)
                      the first December is held back; membership by quarter
  Sstmt (S-stmt)      the statements' lambda K1 uses now (kappa/k1_netflow_nls.py, clock_table and
                      omega_lambda, re-implemented here): share_no_maturity for a B bank with a
                      statement, the nearest year's B median of it otherwise, 1 for a D bank; a
                      conglomerate-year absent from k0_omega.csv takes the B median; the statement of
                      year y applies to the quarters of y + 1, and 2016 quarters take 2016
  Every R version comes in pfpj (headline) and all-holders form; Sstmt has one form.

B rows are the panel's B conglomerate-quarters (is_B = 1; dep_a4 summed over the conglomerate's
markets); weights are dep_a4.

Outputs (Rout/lambda_registry/, LF line ends)
  registry_segment_month.csv          month, segment, holders: stock S, N, total (R$), share_S
  registry_segment_quarter.csv        year, quarter, segment, holders: Rq (with month_used and
                                      share_filled), Rconst, Rannual (with december_used, filled)
  conglomerate_segment_quarter.csv    the segment map: conglomerate-quarter, is_B, dep_a4, segment,
                                      seg_src, leader CNPJ, the panel's segment, the R-fixmem quarter
                                      and segment
  lambda_conglomerate_quarter.csv     the five versions (pfpj and all) by conglomerate-quarter, with
                                      the fill flags and Sstmt's source
  report_*.csv                        the report tables (segment map by year, B segment changes,
                                      quarter-on-quarter share changes above 0.10 with the members
                                      that moved, B time-deposit-weighted means by year, the
                                      2020Q3->2021Q1 change, correlations, fill counts)
  lambda_registry_numbers.json        inputs and md5s, counts, the four reg.lambda keys
                                      (make_paper_numbers.py collect_lambda_registry), every report
  md5_manifest.csv                    md5 of each output above

Usage:  python make_lambda_registry.py [--any-panel]
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse
import hashlib
import json
from datetime import datetime

import numpy as np
import openpyxl
import pandas as pd
import pyarrow.parquet as pq

from utils import paths as _paths
from utils import raw_paths

REG_XLSX = (raw_paths.DATA_RAW / "BCB" / "registry_025399" / "raw"
            / "Estoque captações segmento liquidez nat detentor.xlsx")
SEG_DIR = raw_paths.DATA_RAW / "BCB" / "segmentacao_prudencial" / "ifdata_cadastro"
PANEL = _paths.PROCESSED / "market_panel.parquet"
OMEGA = _paths.rout_dir() / "k0_omega.csv"
OUT = _paths.rout_dir() / "lambda_registry"

PANEL_EXPECT = "fa38794d"                 # I17, 11-05: the final panel
YEARS = range(2016, 2025)
SEGS = ("S1", "S2", "S3", "S4", "S5")
FIRST_FULL = {"S5": 201903}               # first full-stock month of a segment that phases in; the
                                          # quarterly and December series start there
HOLDERS = ("pfpj", "all")
VERSIONS = ("Rq", "Rconst", "Rfixmem", "Rannual", "Sstmt")
CONST_FROM, CONST_TO = (2018, 1), (2024, 4)
FIXMEM_Q = (2019, 4)
LAUNCH = ((2020, 3), (2021, 1))
KEY_DECEMBERS = range(2018, 2025)
KEY_TDW_Q = (2019, 4)
JUMP = 0.10                               # quarter-on-quarter change in a segment's share to report
MOVER_SHARE = 0.10                        # the movers of a share change are "large movers" when they
                                          # hold this share of the segment's panel time deposits
REG_HEADER = ("ANO_MES", "SEG", "RESGATE", "DETENTOR_NATUREZA", "VALOR")
CAD_HEADER = ("CodInst", "Data", "NomeInstituicao", "DataInicioAtividade", "Tcb", "Td", "Tc", "SegmentoTb",
              "Atividade", "Uf", "Municipio", "Sr", "CodConglomeradoFinanceiro", "CodConglomeradoPrudencial",
              "CnpjInstituicaoLider", "Situacao")


def qi(y, q):
    return np.asarray(y, dtype=int) * 4 + np.asarray(q, dtype=int) - 1


def qlabel(i) -> str:
    return f"{int(i) // 4}Q{int(i) % 4 + 1}"


def qmonth(i) -> int:
    """The last month of quarter index i, as YYYYMM."""
    return int(i) // 4 * 100 + 3 * (int(i) % 4 + 1)


def md5_file(p) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def write_csv(df: pd.DataFrame, name: str, written: list):
    p = OUT / name
    df.to_csv(p, index=False, encoding="utf-8", lineterminator="\n")
    written.append(p)


def write_text(name: str, text: str, written: list):
    p = OUT / name
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    written.append(p)


# ------------------------------------------------------------------------------------------- registry
def read_registry():
    wb = openpyxl.load_workbook(REG_XLSX, read_only=True, data_only=True)
    rows = list(wb["Dados"].iter_rows(values_only=True))
    if tuple(rows[0]) != REG_HEADER:
        raise SystemExit(f"{REG_XLSX.name}: header {rows[0]}, expected {REG_HEADER}")
    notas = [r[0] for r in wb["Notas"].iter_rows(values_only=True) if r and r[0]]
    d = pd.DataFrame(rows[1:], columns=list(REG_HEADER))
    d["ANO_MES"] = d["ANO_MES"].astype(int)
    d["VALOR"] = d["VALOR"].astype(float)
    bad = (~d["SEG"].isin(SEGS)) | (~d["RESGATE"].isin(["S", "N"])) | d["VALOR"].isna()
    if bad.any() or d.duplicated(["ANO_MES", "SEG", "RESGATE", "DETENTOR_NATUREZA"]).any():
        raise SystemExit(f"{REG_XLSX.name}: unexpected codes, missing values or repeated keys")
    info = dict(rows=int(len(d)), months=int(d["ANO_MES"].nunique()), first=int(d["ANO_MES"].min()),
                last=int(d["ANO_MES"].max()), holders=sorted(d["DETENTOR_NATUREZA"].unique().tolist()),
                notas=notas)
    return d, info


def segment_month(d: pd.DataFrame) -> pd.DataFrame:
    out = []
    for h in HOLDERS:
        x = d if h == "all" else d[d["DETENTOR_NATUREZA"] == "PF/PJ"]
        p = x.pivot_table(index=["ANO_MES", "SEG"], columns="RESGATE", values="VALOR", aggfunc="sum")
        p = p.reindex(columns=["S", "N"]).reset_index()
        p["holders"] = h
        out.append(p)
    m = pd.concat(out, ignore_index=True).rename(columns={"ANO_MES": "month", "SEG": "segment",
                                                          "S": "stock_S", "N": "stock_N"})
    m[["stock_S", "stock_N"]] = m[["stock_S", "stock_N"]].fillna(0.0)
    m["stock_total"] = m["stock_S"] + m["stock_N"]
    m["share_S"] = m["stock_S"] / m["stock_total"]
    return m[["holders", "segment", "month", "stock_S", "stock_N", "stock_total", "share_S"]].sort_values(
        ["holders", "segment", "month"]).reset_index(drop=True)


def segment_quarter(m: pd.DataFrame):
    """Rq, Rconst and Rannual by segment, holders and quarter 2016Q1-2024Q4, with the fill flags."""
    qe = m[m["month"] % 100 % 3 == 0].copy()
    qe["year"], qe["quarter"] = qe["month"] // 100, qe["month"] % 100 // 3
    qe["qi"] = qi(qe["year"], qe["quarter"])
    rows, const, decs = [], {}, {}
    for h in HOLDERS:
        for s in SEGS:
            g = qe[(qe["holders"] == h) & (qe["segment"] == s)].set_index("qi")["share_S"].dropna()
            ff = FIRST_FULL.get(s)
            if ff is not None:
                g = g[g.index >= int(qi(ff // 100, (ff % 100 - 1) // 3 + 1))]
            first = int(g.index.min())
            lo = max(int(qi(*CONST_FROM)), first)
            win = g[(g.index >= lo) & (g.index <= qi(*CONST_TO))]
            if len(win) != qi(*CONST_TO) - lo + 1:
                raise SystemExit(f"{s} {h}: {len(win)} quarterly shares in the R-const window")
            const[(h, s)] = float(win.mean())
            dec = m[(m["holders"] == h) & (m["segment"] == s) & (m["month"] % 100 == 12)].dropna(subset=["share_S"])
            if ff is not None:
                dec = dec[dec["month"] >= ff]
            dec = dec.set_index(dec["month"] // 100)["share_S"]
            decs[(h, s)] = dec
            for y in YEARS:
                # the December the statements' timing calls for is y - 1; where it is not in the data
                # (2016-2017 for S1-S4, 2016-2019 for S5) the segment's first December is held back
                dfill = (y - 1) not in dec.index
                du = int(dec.index.min()) if dfill else y - 1
                for q in range(1, 5):
                    i = int(qi(y, q))
                    fill = i < first
                    used = first if fill else i
                    rows.append(dict(holders=h, segment=s, year=y, quarter=q,
                                     Rq=float(g.loc[used]), Rq_month_used=qmonth(used),
                                     Rq_share_filled=int(fill), Rconst=const[(h, s)],
                                     Rannual=float(dec.loc[du]), Rannual_december_used=du,
                                     Rannual_filled=int(dfill)))
    return pd.DataFrame(rows), const, decs


# ------------------------------------------------------------------------------------------- segments
def read_cadastro():
    """The 36 quarter-end IF.data registers, named one by one (no glob)."""
    out, md5s = [], {}
    for y in YEARS:
        for mo in (3, 6, 9, 12):
            p = SEG_DIR / f"IF_DATA_List_{y}_{mo}.csv"
            d = pd.read_csv(p, dtype=str, keep_default_na=False)
            if tuple(d.columns) != CAD_HEADER:
                raise SystemExit(f"{p.name}: header {tuple(d.columns)}, expected {CAD_HEADER}")
            if (d["Data"] != f"{y}{mo:02d}").any():
                raise SystemExit(f"{p.name}: Data field differs from the file's quarter")
            d["qi"] = int(qi(y, mo // 3))
            out.append(d)
            md5s[p.name] = md5_file(p)
    c = pd.concat(out, ignore_index=True)
    c["Sr"] = c["Sr"].where(c["Sr"].isin(SEGS))
    if c.duplicated(["CodInst", "qi"]).any():
        raise SystemExit("IF.data register: a CodInst repeats within a quarter")
    return c, md5s


def segment_map(cq: pd.DataFrame, cad: pd.DataFrame):
    """Segment of every panel conglomerate in every quarter 2016Q1-2024Q4 (module docstring)."""
    own = cad.set_index(["CodInst", "qi"])
    sr = own["Sr"].to_dict()
    pc = own["CodConglomeradoPrudencial"].to_dict()
    last_own = cad.sort_values("qi").drop_duplicates("CodInst", keep="last").set_index("CodInst")
    lead_of = last_own["CnpjInstituicaoLider"].where(~last_own["CnpjInstituicaoLider"].isin(["", "null"])).to_dict()
    quarters = range(int(qi(YEARS[0], 1)), int(qi(YEARS[-1], 4)) + 1)
    recs = []
    for j in sorted(cq["code"].unique()):
        lead = lead_of.get(j)
        obs = {}
        for t in quarters:
            if (j, t) in sr:
                if isinstance(sr[(j, t)], str):
                    obs[t] = (sr[(j, t)], "code")
                continue
            if lead and (lead, t) in sr:
                p = pc.get((lead, t))
                s = sr.get((p, t)) if isinstance(p, str) and p.startswith("C") else None
                s = s if isinstance(s, str) else sr[(lead, t)]
                if isinstance(s, str):
                    obs[t] = (s, "leader")
        ts = sorted(obs)
        for t in quarters:
            if t in obs:
                s, src = obs[t]
            elif not ts:
                s, src = None, "none"
            elif t < ts[0]:
                s, src = obs[ts[0]][0], "held_back"
            else:
                s, src = obs[max(u for u in ts if u < t)][0], "carried"
            recs.append(dict(code=j, qi=t, seg=s, seg_src=src, leader_cnpj=lead))
    return pd.DataFrame(recs)


# ------------------------------------------------------------------------------------------- panel
def panel_cq():
    cols = ["CodConglomeradoPrudencial", "NomeInstituicao", "mca_code", "is_B", "year", "quarter", "dep_a4",
            "segment"]
    t = pq.read_table(PANEL, columns=cols).to_pandas()
    t = t[(t["year"] >= YEARS[0]) & (t["year"] <= YEARS[-1])]
    t["code"] = t["CodConglomeradoPrudencial"].astype(str)
    g = t.groupby(["code", "year", "quarter"])
    if (g["is_B"].nunique() > 1).any() or (g["segment"].nunique() > 1).any():
        raise SystemExit("is_B or segment varies within a conglomerate-quarter of the panel")
    cq = g.agg(name=("NomeInstituicao", "first"), is_B=("is_B", "max"), dep_a4=("dep_a4", "sum"),
               panel_rows=("mca_code", "size"), panel_segment=("segment", "first")).reset_index()
    cq["qi"] = qi(cq["year"], cq["quarter"])
    return cq, int(len(t))


def fixmem_quarter(cq: pd.DataFrame) -> pd.DataFrame:
    """The R-fixmem membership quarter of each conglomerate (module docstring)."""
    f = int(qi(*FIXMEM_Q))
    out = []
    for j, g in cq.groupby("code"):
        ts = np.sort(g["qi"].unique())
        if f in ts:
            u, rule = f, "2019Q4"
        elif (ts > f).any():
            u, rule = int(ts[ts > f].min()), "first panel quarter after 2019Q4"
        else:
            u, rule = int(ts[ts < f].max()), "last panel quarter before 2019Q4"
        out.append(dict(code=j, fixmem_qi=u, fixmem_rule=rule))
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------- S-stmt
def statements_lambda(cq: pd.DataFrame):
    """K1's statements' lambda (kappa/k1_netflow_nls.py clock_table + omega_lambda), re-implemented."""
    W = pd.read_csv(OMEGA, dtype={"conglomerate": str})
    own = W[(W["imputed"] == 0) & (W["is_B"] == 1)]
    med_l = own.groupby("year")["share_no_maturity"].median()

    def nearest(s, y):
        return float(s.loc[min(s.index, key=lambda z: (abs(z - y), z))])

    lam = np.where(W["is_B"] == 1, W["share_no_maturity"], 1.0).astype(float)
    miss = np.isnan(lam)
    src = np.where(W["is_B"] == 1, "statement", "D one").astype(object)
    lam[miss] = [nearest(med_l, y) for y in W.loc[miss, "year"]]
    src[miss] = "B median"
    tab = {(j, int(y)): (float(v), s) for j, y, v, s in zip(W["conglomerate"], W["year"], lam, src)}
    fill = {int(y): nearest(med_l, y) for y in YEARS}
    yy = np.clip(cq["qi"].to_numpy() // 4 - 1, YEARS[0], YEARS[-1])
    got = [tab.get((j, int(y)), (fill[int(y)], "absent: B median")) for j, y in zip(cq["code"], yy)]
    info = dict(rows=int(len(W)), B_with_statement=int(own["share_no_maturity"].notna().sum()),
                B_median_by_year={int(k): float(v) for k, v in med_l.items()})
    return np.array([g[0] for g in got]), np.array([g[1] for g in got], dtype=object), yy, info


# ------------------------------------------------------------------------------------------- reports
def wmean(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float)
    ok = ~np.isnan(x) & (w > 0)
    return float(np.sum(x[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else float("nan")


def wcorr(a, b, w):
    a, b, w = (np.asarray(v, float) for v in (a, b, w))
    ok = ~np.isnan(a) & ~np.isnan(b) & (w > 0)
    a, b, w = a[ok], b[ok], w[ok] / w[ok].sum()
    ca, cb = a - np.sum(w * a), b - np.sum(w * b)
    den = np.sqrt(np.sum(w * ca * ca) * np.sum(w * cb * cb))
    return float(np.sum(w * ca * cb) / den) if den > 0 else float("nan")


def runs(seq):
    """[(segment, first qi, last qi), ...] of a conglomerate's quarterly segment path."""
    out = []
    for t, s in seq:
        if out and out[-1][0] == s and out[-1][2] == t - 1:
            out[-1][2] = t
        else:
            out.append([s, t, t])
    return out


def top_level(cad: pd.DataFrame) -> pd.DataFrame:
    """The units the segmentation classifies: prudential conglomerates (their own row) and the
    institutions outside any prudential conglomerate."""
    conf = cad["CodInst"] == cad["CodConglomeradoPrudencial"]
    indep = cad["CodConglomeradoPrudencial"].isin(["", "null"]) & ~cad["CodInst"].str.startswith("C")
    return cad[(conf | indep) & cad["Sr"].notna()][["CodInst", "NomeInstituicao", "qi", "Sr"]]


def jump_report(sq: pd.DataFrame, m: pd.DataFrame, cad: pd.DataFrame, lam: pd.DataFrame):
    tl = top_level(cad)
    seg_by = {t: g.set_index("CodInst")["Sr"] for t, g in tl.groupby("qi")}
    name_by = tl.drop_duplicates("CodInst", keep="last").set_index("CodInst")["NomeInstituicao"].to_dict()
    dep = lam.set_index(["code", "qi"])["dep_a4"].to_dict()
    segdep = lam.dropna(subset=["seg"]).groupby(["qi", "seg"])["dep_a4"].sum().to_dict()
    obs = sq[sq["Rq_share_filled"] == 0]
    rows = []
    for (h, s), g in obs.groupby(["holders", "segment"]):
        g = g.sort_values(["year", "quarter"])
        t = qi(g["year"], g["quarter"])
        v = g["Rq"].to_numpy()
        for k in range(1, len(g)):
            if abs(v[k] - v[k - 1]) <= JUMP:
                continue
            t0, t1 = int(t[k - 1]), int(t[k])
            a, b = seg_by.get(t0, pd.Series(dtype=str)), seg_by.get(t1, pd.Series(dtype=str))
            into = sorted(set(b[b == s].index) - set(a[a == s].index))
            outof = sorted(set(a[a == s].index) - set(b[b == s].index))
            dep_in = sum(dep.get((c, t1), 0.0) for c in into)
            dep_out = sum(dep.get((c, t0), 0.0) for c in outof)
            base0, base1 = segdep.get((t0, s), 0.0), segdep.get((t1, s), 0.0)
            mshare = max(dep_out / base0 if base0 > 0 else 0.0, dep_in / base1 if base1 > 0 else 0.0)
            mm = m[(m["holders"] == h) & (m["segment"] == s)].set_index("month")["stock_total"]
            m0, m1 = qmonth(t0), qmonth(t1)

            def top(cs, tt):
                cs = sorted(cs, key=lambda c: -dep.get((c, tt), 0.0))
                return "; ".join(f"{name_by.get(c, c)} ({c}, dep_a4 R$ {dep.get((c, tt), 0.0) / 1e9:.1f} bn)"
                                 for c in cs[:4] if dep.get((c, tt), 0.0) > 0) or "-"
            rows.append(dict(holders=h, segment=s, from_q=qlabel(t0), to_q=qlabel(t1), share_from=v[k - 1],
                             share_to=v[k], change=v[k] - v[k - 1],
                             segment_stock_change=float(mm.get(m1) / mm.get(m0) - 1.0),
                             n_in=len(into), n_out=len(outof),
                             movers_panel_dep_share=mshare,
                             membership=("no movers" if not into and not outof else
                                         "large movers" if mshare >= MOVER_SHARE else "small movers"),
                             largest_in=top(into, t1), largest_out=top(outof, t0)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--any-panel", action="store_true",
                    help=f"run on a panel other than {PANEL_EXPECT} (the outputs record its md5)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    written: list = []

    panel_md5 = md5_file(PANEL)
    if not panel_md5.startswith(PANEL_EXPECT) and not a.any_panel:
        raise SystemExit(f"{PANEL.name}: md5 {panel_md5[:8]}, expected {PANEL_EXPECT} (--any-panel to override)")

    reg, reg_info = read_registry()
    m = segment_month(reg)
    sq, const, decs = segment_quarter(m)
    cad, cad_md5 = read_cadastro()
    cq, n_panel_rows = panel_cq()
    smap = segment_map(cq, cad)
    fx = fixmem_quarter(cq)

    lam = cq.merge(smap, on=["code", "qi"], how="left").merge(fx, on="code", how="left")
    fseg = smap.rename(columns={"qi": "fixmem_qi", "seg": "fixmem_seg", "seg_src": "fixmem_src"})
    lam = lam.merge(fseg[["code", "fixmem_qi", "fixmem_seg", "fixmem_src"]], on=["code", "fixmem_qi"], how="left")
    for h in HOLDERS:
        s_h = sq[sq["holders"] == h]
        key = s_h.set_index(["segment", "year", "quarter"])
        rq = key["Rq"].to_dict()
        rqf = key["Rq_share_filled"].to_dict()
        ra = key["Rannual"].to_dict()
        raf = key["Rannual_filled"].to_dict()
        yq = list(zip(lam["year"], lam["quarter"]))
        lam[f"Rq_{h}"] = [rq.get((s, y, q), np.nan) for s, (y, q) in zip(lam["seg"], yq)]
        lam[f"Rconst_{h}"] = [const.get((h, s), np.nan) for s in lam["seg"]]
        lam[f"Rfixmem_{h}"] = [rq.get((s, y, q), np.nan) for s, (y, q) in zip(lam["fixmem_seg"], yq)]
        lam[f"Rannual_{h}"] = [ra.get((s, y, q), np.nan) for s, (y, q) in zip(lam["seg"], yq)]
        if h == "pfpj":
            lam["Rq_share_filled"] = [rqf.get((s, y, q), 0) for s, (y, q) in zip(lam["seg"], yq)]
            lam["Rfixmem_share_filled"] = [rqf.get((s, y, q), 0) for s, (y, q) in zip(lam["fixmem_seg"], yq)]
            lam["Rannual_filled"] = [raf.get((s, y, q), 0) for s, (y, q) in zip(lam["seg"], yq)]
    lam["Sstmt"], lam["Sstmt_src"], lam["Sstmt_statement_year"], s_info = statements_lambda(lam)
    lam["mem_filled"] = lam["seg_src"].isin(["held_back", "carried"]).astype(int)
    lam["fixmem_mem_filled"] = lam["fixmem_src"].isin(["held_back", "carried"]).astype(int)
    lam["fill_Rq"] = ((lam["mem_filled"] == 1) | (lam["Rq_share_filled"] == 1)).astype(int)
    lam["fill_Rconst"] = lam["mem_filled"]
    lam["fill_Rfixmem"] = ((lam["fixmem_mem_filled"] == 1) | (lam["Rfixmem_share_filled"] == 1)).astype(int)
    lam["fill_Rannual"] = ((lam["mem_filled"] == 1) | (lam["Rannual_filled"] == 1)).astype(int)
    lam["fill_Sstmt"] = (lam["Sstmt_src"] != "statement").astype(int) * (lam["is_B"] == 1)
    lam = lam.sort_values(["code", "qi"]).reset_index(drop=True)
    lam["quarter_label"] = [qlabel(t) for t in lam["qi"]]
    lam["fixmem_quarter"] = [qlabel(t) for t in lam["fixmem_qi"]]

    # ---- outputs: data
    write_csv(m, "registry_segment_month.csv", written)
    write_csv(sq, "registry_segment_quarter.csv", written)
    seg_cols = ["code", "name", "is_B", "year", "quarter", "quarter_label", "dep_a4", "panel_rows", "seg", "seg_src",
                "leader_cnpj", "panel_segment", "fixmem_quarter", "fixmem_rule", "fixmem_seg", "fixmem_src"]
    write_csv(lam[seg_cols], "conglomerate_segment_quarter.csv", written)
    lam_cols = (["code", "name", "is_B", "year", "quarter", "quarter_label", "dep_a4", "seg", "seg_src", "fixmem_seg"]
                + [f"{v}_{h}" for h in HOLDERS for v in VERSIONS[:4]]
                + ["Sstmt", "Sstmt_src", "Sstmt_statement_year", "Rq_share_filled", "Rfixmem_share_filled",
                   "Rannual_filled", "mem_filled", "fixmem_mem_filled"] + [f"fill_{v}" for v in VERSIONS])
    write_csv(lam[lam_cols], "lambda_conglomerate_quarter.csv", written)

    B = lam[lam["is_B"] == 1]
    depB = B["dep_a4"].sum()
    vcols = {v: (f"{v}_pfpj" if v != "Sstmt" else "Sstmt") for v in VERSIONS}

    # ---- report 1: segment map by year (Q4) and the B conglomerates that change segment
    r1 = []
    for y in YEARS:
        for blk, g in (("B", B), ("D", lam[lam["is_B"] == 0])):
            gq = g[(g["year"] == y) & (g["quarter"] == 4)]
            tot = gq["dep_a4"].sum()
            for s in SEGS + (None,):
                x = gq[gq["seg"] == s] if s else gq[gq["seg"].isna()]
                r1.append(dict(year=y, quarter=4, block=blk, segment=s or "none", n_conglomerates=int(len(x)),
                               dep_a4_brl_bn=x["dep_a4"].sum() / 1e9,
                               dep_a4_share_of_block=(x["dep_a4"].sum() / tot) if tot > 0 else np.nan,
                               n_leader_route=int((x["seg_src"] == "leader").sum()),
                               n_filled=int(x["seg_src"].isin(["held_back", "carried"]).sum())))
    rep_map = pd.DataFrame(r1)
    write_csv(rep_map, "report_segment_map_by_year.csv", written)

    r2 = []
    for j, g in B.groupby("code"):
        g = g.sort_values("qi")
        rr = runs(list(zip(g["qi"], g["seg"].fillna("none"))))
        segs = [r[0] for r in rr]
        changes = [qlabel(rr[k][1]) for k in range(1, len(rr)) if rr[k][0] != rr[k - 1][0]]
        gaps = [qlabel(rr[k][1]) for k in range(1, len(rr)) if rr[k][0] == rr[k - 1][0]]
        q19 = g[g["qi"] == qi(*FIXMEM_Q)]["dep_a4"].sum()
        r2.append(dict(code=j, name=g["name"].iloc[-1], first_q=qlabel(g["qi"].min()), last_q=qlabel(g["qi"].max()),
                       path=" > ".join(f"{r[0]} {qlabel(r[1])}-{qlabel(r[2])}" for r in rr),
                       n_changes=len(changes), change_quarters="; ".join(changes),
                       srcs="; ".join(sorted(g["seg_src"].unique())),
                       dep_a4_share_B_2016_2024=g["dep_a4"].sum() / depB,
                       dep_a4_share_B_2019Q4=q19 / B[B["qi"] == qi(*FIXMEM_Q)]["dep_a4"].sum(),
                       panel_gaps=len(gaps)))
    rep_chg = pd.DataFrame(r2).sort_values(["n_changes", "dep_a4_share_B_2016_2024"], ascending=[False, False])
    write_csv(rep_chg, "report_B_segment_paths.csv", written)

    # ---- report 2: quarter-on-quarter share changes above JUMP
    rep_jump = jump_report(sq, m, cad, lam)
    write_csv(rep_jump, "report_share_jumps.csv", written)

    # ---- report 3: B time-deposit-weighted means by year, and the launch-window change
    r3 = []
    for h in HOLDERS:
        for v in VERSIONS:
            c = vcols[v] if (h == "pfpj" or v == "Sstmt") else f"{v}_{h}"
            if h == "all" and v == "Sstmt":
                continue
            row = dict(holders=h, version=v)
            for y in YEARS:
                gy = B[B["year"] == y]
                row[str(y)] = wmean(gy[c], gy["dep_a4"])
            row["all_years"] = wmean(B[c], B["dep_a4"])
            q0, q1 = (B[B["qi"] == qi(*LAUNCH[0])], B[B["qi"] == qi(*LAUNCH[1])])
            row["2020Q3"], row["2021Q1"] = wmean(q0[c], q0["dep_a4"]), wmean(q1[c], q1["dep_a4"])
            row["change_2020Q3_2021Q1"] = row["2021Q1"] - row["2020Q3"]
            r3.append(row)
    rep_tdw = pd.DataFrame(r3)
    write_csv(rep_tdw, "report_B_tdw_by_year.csv", written)

    # ---- report 4: correlations across versions at the B-row level (B conglomerate-quarters, dep_a4 > 0)
    Bp = B[B["dep_a4"] > 0]
    cols = [vcols[v] for v in VERSIONS]
    r4 = []
    for wt in ("unweighted", "dep_a4"):
        for c1 in cols:
            row = dict(weight=wt, version=c1)
            for c2 in cols:
                w = Bp["dep_a4"] if wt == "dep_a4" else pd.Series(1.0, index=Bp.index)
                row[c2] = wcorr(Bp[c1], Bp[c2], w)
            r4.append(row)
    rep_corr = pd.DataFrame(r4)
    write_csv(rep_corr, "report_B_corr.csv", written)

    # ---- report 5: fill counts (B rows and B time-deposit share)
    r5 = []
    for v in VERSIONS:
        f = B[f"fill_{v}"] == 1
        r5.append(dict(version=v, B_rows=int(len(B)), B_rows_filled=int(f.sum()),
                       B_rows_filled_with_dep=int((f & (B["dep_a4"] > 0)).sum()),
                       dep_a4_share_filled=B.loc[f, "dep_a4"].sum() / depB,
                       B_rows_missing=int(B[vcols[v]].isna().sum())))
    pre = B["qi"] < qi(2017, 2)
    for lab, msk in (("before 2017Q2 (no segmentation)", pre),
                     ("membership by leader route", B["seg_src"] == "leader"),
                     ("membership carried or held back, 2017Q2 on", ~pre & B["seg_src"].isin(["held_back", "carried"])),
                     ("segment share held back (S5 before 2019Q1), 2017Q2 on", ~pre & (B["Rq_share_filled"] == 1))):
        r5.append(dict(version=lab, B_rows=int(len(B)), B_rows_filled=int(msk.sum()),
                       B_rows_filled_with_dep=int((msk & (B["dep_a4"] > 0)).sum()),
                       dep_a4_share_filled=B.loc[msk, "dep_a4"].sum() / depB, B_rows_missing=0))
    rep_fill = pd.DataFrame(r5)
    write_csv(rep_fill, "report_fill.csv", written)

    # ---- the four reg.lambda keys (claude/MAIN_REPLY_2026-11-05.md 11-11 18:30)
    dec = m[(m["holders"] == "pfpj") & (m["month"] % 100 == 12) & (m["month"] // 100).isin(list(KEY_DECEMBERS))]
    dec = dec[dec["month"] >= dec["segment"].map(FIRST_FULL).fillna(0)]
    s1 = dec[dec["segment"] == "S1"]
    i_lo, i_hi, i_all = s1["share_S"].idxmin(), s1["share_S"].idxmax(), dec["share_S"].idxmin()
    q4 = B[B["qi"] == qi(*KEY_TDW_Q)]
    by_seg = {s: dict(min=float(dec.loc[dec["segment"] == s, "share_S"].min()),
                      max=float(dec.loc[dec["segment"] == s, "share_S"].max())) for s in SEGS}
    keys = {"reg.lambda.pfpj.S1.min": dict(value=float(s1.loc[i_lo, "share_S"]), month=int(s1.loc[i_lo, "month"])),
            "reg.lambda.pfpj.S1.max": dict(value=float(s1.loc[i_hi, "share_S"]), month=int(s1.loc[i_hi, "month"])),
            "reg.lambda.pfpj.all.min": dict(value=float(dec.loc[i_all, "share_S"]), segment=str(dec.loc[i_all, "segment"]),
                                            month=int(dec.loc[i_all, "month"])),
            "reg.lambda.B.tdw.2019Q4": dict(
                value=wmean(q4["Rq_pfpj"], q4["dep_a4"]), quarter="2019Q4", B_conglomerates=int(len(q4)),
                B_with_dep=int((q4["dep_a4"] > 0).sum()),
                dep_share_by_segment={s: float(q4.loc[q4["seg"] == s, "dep_a4"].sum() / q4["dep_a4"].sum()) for s in SEGS},
                seg_src_counts={k: int(v) for k, v in q4["seg_src"].value_counts().items()}),
            "by_segment_decembers_2018_2024": by_seg}

    agree = lam[lam["panel_segment"].notna()]
    numbers = dict(
        generated=datetime.now().isoformat(timespec="seconds"), generator="make_lambda_registry.py",
        first_full_month=FIRST_FULL,
        inputs=dict(registry=dict(path=str(REG_XLSX), md5=md5_file(REG_XLSX), **reg_info),
                    panel=dict(path=str(PANEL), md5=panel_md5, rows_2016_2024=n_panel_rows),
                    k0_omega=dict(path=str(OMEGA), md5=md5_file(OMEGA)),
                    ifdata_cadastro=dict(dir=str(SEG_DIR), files=cad_md5)),
        versions={v: d for v, d in zip(VERSIONS, (
            "segment S share at the quarter's last month, membership by quarter (headline)",
            "segment mean of the quarterly S shares 2018Q1-2024Q4 (S5 2019Q1-2024Q4), membership by quarter",
            "quarterly segment S shares, membership fixed at 2019Q4 (first panel quarter if later, last if earlier)",
            "segment December S share applied to the next year's quarters (Dec 2017 also to 2016-2017; "
            "first December held back where missing), membership by quarter",
            "statements' lambda as K1 reads it (k0_omega.csv; B median where missing; D = 1)"))},
        counts=dict(conglomerate_quarters=int(len(lam)), conglomerates=int(lam["code"].nunique()),
                    B_conglomerate_quarters=int(len(B)), B_conglomerates=int(B["code"].nunique()),
                    seg_src={k: int(v) for k, v in lam["seg_src"].value_counts().items()},
                    seg_src_B={k: int(v) for k, v in B["seg_src"].value_counts().items()},
                    fixmem_rule={k: int(v) for k, v in lam.drop_duplicates("code")["fixmem_rule"].value_counts().items()},
                    panel_segment_agreement=dict(compared=int(len(agree)),
                                                 equal=int((agree["panel_segment"] == agree["seg"]).sum())),
                    Sstmt_src={k: int(v) for k, v in lam["Sstmt_src"].value_counts().items()},
                    Sstmt_src_B={k: int(v) for k, v in B["Sstmt_src"].value_counts().items()},
                    D_rows_Sstmt_not_one=int(((lam["is_B"] == 0) & (lam["Sstmt"] != 1.0)).sum())),
        statements=s_info,
        rconst={f"{h}.{s}": const[(h, s)] for h in HOLDERS for s in SEGS},
        keys=keys,
        reports=dict(segment_map_by_year=rep_map.to_dict("records"), B_segment_paths=rep_chg.to_dict("records"),
                     share_jumps=rep_jump.to_dict("records"), B_tdw_by_year=rep_tdw.to_dict("records"),
                     B_corr=rep_corr.to_dict("records"), fill=rep_fill.to_dict("records")))

    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (float, np.floating)):
            return float(x) if np.isfinite(x) else None
        return x
    write_text("lambda_registry_numbers.json", json.dumps(clean(numbers), indent=1, ensure_ascii=False) + "\n", written)
    man = pd.DataFrame([dict(file=p.name, bytes=p.stat().st_size, md5=md5_file(p)) for p in written])
    write_csv(man, "md5_manifest.csv", [])

    print(f"panel {panel_md5[:8]}; registry {reg_info['rows']} rows, {reg_info['months']} months; "
          f"{len(lam)} conglomerate-quarters ({len(B)} B)")
    for k in ("reg.lambda.pfpj.S1.min", "reg.lambda.pfpj.S1.max", "reg.lambda.pfpj.all.min", "reg.lambda.B.tdw.2019Q4"):
        print(f"  {k}: {keys[k]['value']:.4f}")
    for p in written + [OUT / "md5_manifest.csv"]:
        print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
