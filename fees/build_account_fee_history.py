###─────────────────────────────────────────────────────────────────────────────
# fees/build_account_fee_history.py
#
# Author: Pedro Feijó de Moraes
#
# PURPOSE
#   One dated listed-price series per institution for the fees of KEEPING an account, and a
#   measure of where 2016-2024 is still unknown:
#     F 1501-1504  Pacote Padronizado de Serviços I-IV (individuals)
#     J 0402/0403  Manutenção de conta ativa / inativa (firms)
#
#   Two BCB sources are joined:
#     census    the monthly census of listed tariffs, 2015-01 to 2024-12
#               (fees/scrape_bcb_tarifas_monthly.py -> tariff_history_bcb_monthly.csv)
#     snapshot  the Olinda schedule of 2026-06-22, one price per service with the date it
#               took effect (fees/scrape_bcb_tariff_vigencia.py -> tariff_vigencia_history.csv)
#
#   Each listed price is a VERSION: a price with its "vigência" (date in force), seen from one
#   date to another. A version proves its price over [vigência, last date seen]: the
#   institution reports the date of its last change, so a later change would have shown a new
#   vigência. The price on a date is KNOWN only when the date falls in such an interval. When
#   intervals overlap with different prices, the version with the later vigência wins (it is
#   the later change); a version listed with a vigência after the date it was seen (an
#   announced price) proves nothing until it is seen in force.
#
#   One bridge: across a price change between two consecutive monthly census files, the old
#   price is taken to run until the day before the new vigência (labelled census_bridged; it
#   assumes no second change inside that month).
#
#   Months are evaluated on their 15th day (the census files are dated at month-end, the
#   24th in Feb 2017). Listed prices are ceilings; the price a client pays can be lower, and
#   many clients hold only the free essential services.
#
#   Reasons a month is unknown: before_first_version (not listed yet, or the current price
#   took effect later), gap_census_to_snapshot (after the institution's last census listing
#   and before the snapshot price took effect), dropped_during_census (the service stopped
#   being listed before the census ended and is not in the snapshot),
#   after_census_not_in_snapshot, gap_between_versions (absent from some census months between
#   two listed prices), after_last_version.
#
#   Not a pipeline step, but stage 7 reads its output: fees/panel_fee_merge.py takes the six
#   account-keeping services' versions from account_fee_listed_history.csv, builds the other
#   listed services with load_versions(), and turns both into the panel's quarterly
#   listed_fee_* and tarifa_stickiness_* columns. Run this script before stage 7.
#
# OUTPUT  (utils.paths.TARIFAS_PROC)
#   account_fee_listed_history.csv   one row per version: cnpj8, pessoa, codigo_servico,
#       vigencia, valor_maximo, unidade, periodicidade, source, seen_from, seen_to,
#       in_force_to, bridged, first_file, last_file, n_files, instituicao, servico
#   account_fee_listed_by_year.csv   cnpj8 x pessoa x codigo_servico x year 2016-2024:
#       months_known (0-12), known_all_year, price_dec, vigencia_dec, source_dec,
#       reason_dec (why December is unknown), last_census_seen, snapshot_vigencia
#   account_fee_listed_coverage.csv  share of institution-years known, by service and year,
#       on two denominators (every institution that ever lists the service; institutions in
#       the 2026 snapshot), with the snapshot-only share for comparison
#
# USAGE
#   python fees/build_account_fee_history.py
###─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: utils/

from utils import paths

PROC = paths.TARIFAS_PROC
CENSUS = PROC / "tariff_history_bcb_monthly.csv"
SNAPSHOT = PROC / "tariff_vigencia_history.csv"

SERVICES = {
    ("F", "1501"): "Pacote Padronizado I",
    ("F", "1502"): "Pacote Padronizado II",
    ("F", "1503"): "Pacote Padronizado III",
    ("F", "1504"): "Pacote Padronizado IV",
    ("J", "0402"): "Manutenção de conta ativa",
    ("J", "0403"): "Manutenção de conta inativa",
}
YEARS = range(2016, 2025)
BIG_BANKS = {
    "60701190": "Itaú", "00000000": "Banco do Brasil", "60746948": "Bradesco",
    "90400888": "Santander", "00360305": "Caixa", "92702067": "Banrisul",
    "00000208": "BRB", "00416968": "Inter",
}
KEY = ["cnpj8", "pessoa", "codigo_servico"]


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------

def load_versions(services: dict | None = None) -> pd.DataFrame:
    """Census spells and snapshot rows of the given services as one table of versions.

    services maps (pessoa, codigo_servico) to the label written to the servico column; the
    default is SERVICES, the six account-keeping services. fees/panel_fee_merge.py passes
    its other listed services through the same rule.

    A version seen in both sources (same vigência and price) is one row, seen from its first
    census file to the snapshot date."""
    services = SERVICES if services is None else services
    # Only prices listed in reais: a few small institutions list a package or account fee as
    # a percentage (moeda "Percentual"), which is not a price in R$.
    c = pd.read_csv(CENSUS, dtype=str)
    c = c[np.array([(p, s) in services for p, s in zip(c["pessoa"], c["codigo_servico"])])
          & (c["moeda"] == "Real").to_numpy()].copy()
    c["seen_from"] = c["first_file_date"]
    c["seen_to"] = c["census_seen_to"] = c["last_file_date"]
    c["source"] = "census"

    s = pd.read_csv(SNAPSHOT, dtype=str)
    s = s.rename(columns={"cnpj": "cnpj8", "customer_type": "pessoa", "data_vigencia": "vigencia",
                          "nome_instituicao": "instituicao"})
    s["cnpj8"] = s["cnpj8"].str.zfill(8)
    s["codigo_servico"] = s["codigo_servico"].str.zfill(4)
    s = s[np.array([(p, k) in services for p, k in zip(s["pessoa"], s["codigo_servico"])])
          & (s["tipo_valor"] == "Real").to_numpy()].copy()
    s["seen_from"] = s["seen_to"] = s["data_coleta"]
    s["source"] = "snapshot"
    s["first_file"] = s["last_file"] = s["n_files"] = s["census_seen_to"] = ""

    cols = KEY + ["vigencia", "valor_maximo", "unidade", "periodicidade", "source", "seen_from",
                  "seen_to", "census_seen_to", "first_file", "last_file", "n_files", "instituicao"]
    v = pd.concat([c[cols], s[cols]], ignore_index=True)
    for col in ("vigencia", "seen_from", "seen_to", "census_seen_to"):
        v[col] = pd.to_datetime(v[col].str[:10], errors="coerce")
    v["valor_maximo"] = pd.to_numeric(v["valor_maximo"], errors="coerce").round(2)

    # One row when census and snapshot show the same version.
    merged = (v.sort_values("seen_from")
              .groupby(KEY + ["vigencia", "valor_maximo"], dropna=False, sort=False)
              .agg(unidade=("unidade", "last"), periodicidade=("periodicidade", "last"),
                   source=("source", lambda x: "+".join(dict.fromkeys(x))),
                   seen_from=("seen_from", "min"), seen_to=("seen_to", "max"),
                   census_seen_to=("census_seen_to", "max"),
                   first_file=("first_file", "first"), last_file=("last_file", "first"),
                   n_files=("n_files", "first"), instituicao=("instituicao", "last"))
              .reset_index())
    merged["servico"] = [services[(p, s)] for p, s in zip(merged["pessoa"], merged["codigo_servico"])]
    merged = merged.sort_values(KEY + ["vigencia", "seen_from"]).reset_index(drop=True)
    return bridge(merged)


def census_months() -> list[str]:
    """The census months on disk, in order (tariff_census_bcb_monthly.csv)."""
    c = pd.read_csv(PROC / "tariff_census_bcb_monthly.csv", dtype=str)
    return sorted(c["yyyymm"].unique())


def bridge(v: pd.DataFrame) -> pd.DataFrame:
    """in_force_to: the last day each version is known to be in force.

    It is seen_to, except across a price change between two consecutive census files: when
    version B is first listed in the file right after version A's last file and took effect
    after A was last seen, A was in force until the day before B's vigência. Only a second
    change inside the same month, never listed, could break this. Such versions carry
    bridged=True, and the months they add are labelled census_bridged."""
    months = census_months()
    nxt = dict(zip(months[:-1], months[1:]))
    v["in_force_to"] = v["seen_to"]
    v["bridged"] = False
    cen = v[v["source"].str.contains("census")]
    for _, g in cen.groupby(KEY, sort=False):
        g = g.sort_values(["first_file", "vigencia"])
        for (ia, a), (_, b) in zip(g.iterrows(), list(g.iterrows())[1:]):
            if (nxt.get(a["last_file"]) == b["first_file"]
                    and pd.notna(b["vigencia"]) and a["seen_to"] < b["vigencia"] <= b["seen_from"]):
                v.at[ia, "in_force_to"] = b["vigencia"] - pd.Timedelta(days=1)
                v.at[ia, "bridged"] = True
    return v


# ---------------------------------------------------------------------------
# Price in force on a date
# ---------------------------------------------------------------------------

def evaluate(versions: pd.DataFrame) -> pd.DataFrame:
    """Price in force on the 15th of every month 2016-01..2024-12 for every institution x
    service, with the reason when it is unknown."""
    dates = (pd.date_range("2016-01-01", "2024-12-01", freq="MS") + pd.Timedelta(days=14)).to_numpy()
    census_end = pd.to_datetime(pd.read_csv(PROC / "tariff_census_bcb_monthly.csv", dtype=str)
                                ["file_date"], format="%d/%m/%Y").max()
    out = []
    for key, g0 in versions.groupby(KEY, sort=False):
        last_census = g0["census_seen_to"].max()
        g = g0[g0["vigencia"].notna() & (g0["vigencia"] <= g0["seen_to"])]
        snap = g[g["source"].str.contains("snapshot")]
        snap_vig = snap["vigencia"].max() if len(snap) else pd.NaT
        vig, end, seen = (g["vigencia"].to_numpy(), g["in_force_to"].to_numpy(),
                          g["seen_to"].to_numpy())
        price, src = g["valor_maximo"].to_numpy(), g["source"].to_numpy()
        # hits[i, j]: version i proves its price on date j.
        hits = (vig[:, None] <= dates[None, :]) & (end[:, None] >= dates[None, :])
        # Among the versions in force, the one with the latest vigência (then latest seen_to).
        order = np.lexsort((end, vig)) if len(g) else np.array([], dtype=int)
        rank = np.empty(len(g), dtype=int)
        rank[order] = np.arange(len(g))
        best = np.where(hits, rank[:, None], -1).max(axis=0) if len(g) else np.full(len(dates), -1)
        inv = order  # rank -> row
        first_start = vig.min() if len(g) else None
        last_end = end.max() if len(g) else None
        for j, d in enumerate(dates):
            base = dict(zip(KEY, key), date=d, last_census_seen=last_census,
                        snapshot_vigencia=snap_vig)
            if best[j] >= 0:
                i = inv[best[j]]
                n_prices = len(set(price[hits[:, j]]))
                source = "census_bridged" if d > seen[i] else src[i]
                out.append({**base, "price": price[i], "vigencia": vig[i], "source": source,
                            "reason": "", "conflict": n_prices > 1})
                continue
            d_ts = pd.Timestamp(d)
            if not len(g):
                reason = "no_version_in_force"
            elif d < first_start:
                reason = "before_first_version"
            elif (pd.notna(last_census) and pd.isna(snap_vig) and d_ts > last_census
                  and last_census < census_end - pd.Timedelta(days=40)):
                reason = "dropped_during_census"
            elif pd.notna(last_census) and d_ts > last_census and (pd.isna(snap_vig) or d_ts < snap_vig):
                reason = ("gap_census_to_snapshot" if pd.notna(snap_vig)
                          else "after_census_not_in_snapshot")
            elif d > last_end:
                reason = "after_last_version"
            else:
                reason = "gap_between_versions"
            out.append({**base, "price": np.nan, "vigencia": pd.NaT, "source": "",
                        "reason": reason, "conflict": False})
    return pd.DataFrame(out)


def by_year(monthly: pd.DataFrame) -> pd.DataFrame:
    m = monthly.assign(year=monthly["date"].dt.year, known=monthly["price"].notna())
    dec = m[m["date"].dt.month == 12].set_index(KEY + ["year"])
    y = (m.groupby(KEY + ["year"])
         .agg(months_known=("known", "sum"), conflicts=("conflict", "sum"),
              last_census_seen=("last_census_seen", "first"),
              snapshot_vigencia=("snapshot_vigencia", "first")))
    y["known_all_year"] = y["months_known"] == 12
    y["price_dec"] = dec["price"]
    y["vigencia_dec"] = dec["vigencia"]
    y["source_dec"] = dec["source"]
    y["reason_dec"] = dec["reason"]
    y = y.reset_index()
    y["servico"] = [SERVICES[(p, s)] for p, s in zip(y["pessoa"], y["codigo_servico"])]
    return y


def coverage(year: pd.DataFrame, versions: pd.DataFrame) -> pd.DataFrame:
    """Share of institution-years with the December price known, per service and year.

    snapshot_only: the same December test using the snapshot versions alone, i.e. what was
    measurable before the census was added."""
    snap = versions[versions["source"].str.contains("snapshot")]
    in_snap = set(map(tuple, snap[KEY].to_numpy()))
    snap_start = snap.groupby(KEY)["vigencia"].min()
    rows = []
    for (p, s), g in year.groupby(["pessoa", "codigo_servico"]):
        for yr, gy in g.groupby("year"):
            dec = pd.Timestamp(f"{yr}-12-15")
            keys = list(map(tuple, gy[KEY].to_numpy()))
            snap_mask = np.array([k in in_snap for k in keys])
            snap_known = np.array([k in in_snap and snap_start.get(k, pd.NaT) <= dec for k in keys])
            known = gy["price_dec"].notna().to_numpy()
            rows.append({
                "pessoa": p, "codigo_servico": s, "servico": SERVICES[(p, s)], "year": yr,
                "n_ever_listed": len(gy), "known_dec_ever_listed": known.mean().round(3),
                "n_in_snapshot": int(snap_mask.sum()),
                "known_dec_in_snapshot": known[snap_mask].mean().round(3) if snap_mask.any() else np.nan,
                "snapshot_only_in_snapshot": snap_known[snap_mask].mean().round(3) if snap_mask.any() else np.nan,
                "mean_months_known_ever_listed": gy["months_known"].mean().round(2),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    versions = load_versions()
    versions.to_csv(PROC / "account_fee_listed_history.csv", index=False, date_format="%Y-%m-%d")
    print(f"versions: {len(versions):,} rows, {versions['cnpj8'].nunique():,} institutions; "
          f"by source {versions['source'].value_counts().to_dict()}")

    monthly = evaluate(versions)
    year = by_year(monthly)
    year.to_csv(PROC / "account_fee_listed_by_year.csv", index=False, date_format="%Y-%m-%d")
    cov = coverage(year, versions)
    cov.to_csv(PROC / "account_fee_listed_coverage.csv", index=False)
    print(f"by year: {len(year):,} institution x service x year rows; "
          f"months with overlapping conflicting versions: {int(monthly['conflict'].sum()):,}; "
          f"bridged versions: {int(versions['bridged'].sum()):,}")
    print("institution x service x months by source:",
          monthly["source"].replace("", "unknown").value_counts().to_dict())
    print("unknown months by reason:", monthly.loc[monthly["reason"] != "", "reason"].value_counts().to_dict())

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\nShare of institution-years with the December price known")
        print(cov.to_string(index=False))
        big = year[year["cnpj8"].isin(BIG_BANKS)].copy()
        big["bank"] = big["cnpj8"].map(BIG_BANKS)
        tab = big.pivot_table(index=["bank", "codigo_servico"], columns="year",
                              values="months_known", aggfunc="first")
        print("\nBig banks: months known per year (of 12)")
        print(tab.to_string())
        print("\nBig banks: reason December is unknown")
        r = big[big["reason_dec"] != ""].groupby(["bank", "reason_dec"])["year"].agg(
            lambda x: f"{x.min()}-{x.max()} ({len(x)})")
        print(r.to_string())


if __name__ == "__main__":
    main()
