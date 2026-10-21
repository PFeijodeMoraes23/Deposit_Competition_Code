###─────────────────────────────────────────────────────────────────────────────
# make_consumidor_ibge_crosswalk.py
#
# Author: Pedro Feijó de Moraes
#
# PURPOSE
#   Attach IBGE municipality codes to the consumidor.gov.br complaint files, which carry the
#   consumer's city only as a NAME (`Cidade`) plus the state (`UF`). Writes a crosswalk
#   (UF, city name as served) -> 7-digit IBGE code. The raw complaint files are not modified;
#   nothing is aggregated beyond counting complaints per (UF, city) to show each pair's weight.
#
#   Not a pipeline step. Inputs come from scrape_consumidor_gov.py (complaint zips) and the
#   IBGE localities API list saved next to them.
#
# MATCHING  (each pair records which rule matched it in `match`; no fuzzy matching)
#   1 exact_normalised  (UF, normalised name) equals an IBGE municipality; normalising =
#                       strip accents, upper case, every non-alphanumeric run -> one space
#                       ("D'OESTE" and "D OESTE" agree)
#   2 wildcard_unique   the source lost accented letters to '?' ("Ibirit?" for Ibirité):
#                       '?' = any one letter, accepted only if EXACTLY one municipality of
#                       the UF fits
#   3 district          the name is an IBGE district, not a municipality ("Bonfim Paulista",
#                       district of Ribeirão Preto): the code is the parent municipality's,
#                       accepted only if exactly one district of the UF has that name (rules
#                       1 and 2 applied to district names)
#   Anything else stays `unmatched`, listed with its complaint count for review.
#
# INPUT   DATA_RAW/consumidor_gov/data/*.zip            (one CSV inside each, ';'-separated;
#                                                        UTF-8 in recent years, Latin-1 earlier)
#         DATA_RAW/consumidor_gov/docs/ibge_localidades_municipios_<date>.json
#         (https://servicodados.ibge.gov.br/api/v1/localidades/municipios)
#         DATA_RAW/consumidor_gov/docs/ibge_localidades_distritos_<date>.json
#         (https://servicodados.ibge.gov.br/api/v1/localidades/distritos)
# OUTPUT  DATA_RAW/consumidor_gov/crosswalk/consumidor_cidade_ibge.csv
#           uf, cidade, cidade_norm, n_complaints, first_file, last_file, ibge_code,
#           ibge_nome, match
#
# USAGE
#   python make_consumidor_ibge_crosswalk.py
###─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import re
import unicodedata
import zipfile

import pandas as pd

from utils import paths

BASE = paths.OPEN_FINANCE / "DATA_RAW" / "consumidor_gov"


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Z0-9]+", " ", s.upper()).strip()


def norm_wild(s: str) -> str:
    """norm() that keeps '?' (a letter the source lost) as a wildcard."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Z0-9?]+", " ", s.upper()).strip()


def uf_of(m: dict) -> str | None:
    if m.get("microrregiao"):
        return m["microrregiao"]["mesorregiao"]["UF"]["sigla"]
    if m.get("regiao-imediata"):
        return m["regiao-imediata"]["regiao-intermediaria"]["UF"]["sigla"]
    return None


def ibge_table(path: str) -> pd.DataFrame:
    rows = [{"uf": uf_of(m), "ibge_code": int(m["id"]), "ibge_nome": m["nome"], "cidade_norm": norm(m["nome"])}
            for m in json.load(open(path, encoding="utf-8"))]
    t = pd.DataFrame(rows)
    dup = t.duplicated(["uf", "cidade_norm"], keep=False)
    if dup.any():   # two municipalities of one state sharing a normalised name: refuse to match them
        print("ambiguous IBGE names (left unmatched):", t[dup][["uf", "ibge_nome"]].values.tolist())
        t = t[~dup]
    return t


def district_table(path: str) -> pd.DataFrame:
    """IBGE districts -> parent municipality. Names shared by two districts of one UF are dropped."""
    rows = []
    for d in json.load(open(path, encoding="utf-8")):
        mun = d["municipio"]
        rows.append({"uf": uf_of(mun), "cidade_norm": norm(d["nome"]),
                     "ibge_code": int(mun["id"]), "ibge_nome": mun["nome"]})
    t = pd.DataFrame(rows).drop_duplicates()
    return t[~t.duplicated(["uf", "cidade_norm"], keep=False)]


def unique_wildcard(uf: str, cidade: str, table: pd.DataFrame):
    """(code, nome) if exactly one name of the UF fits the '?'-wildcard pattern, else None."""
    pat = norm_wild(cidade)
    if "?" not in pat:
        return None
    rx = re.compile("^" + re.escape(pat).replace(r"\?", "[A-Z]") + "$")
    hit = table[(table["uf"] == uf) & table["cidade_norm"].str.match(rx)]
    if hit["ibge_code"].nunique() == 1:
        return int(hit["ibge_code"].iloc[0]), hit["ibge_nome"].iloc[0]
    return None


def read_zip(path: str) -> pd.DataFrame:
    z = zipfile.ZipFile(path)
    name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
    raw = z.read(name)
    for enc in ("utf-8", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    d = pd.read_csv(io.StringIO(text), sep=";", dtype=str, on_bad_lines="skip", low_memory=False)
    d.columns = [c.strip() for c in d.columns]
    return d[["UF", "Cidade"]]


def main() -> None:
    ap = argparse.ArgumentParser(description="IBGE codes for consumidor.gov.br city names")
    ap.add_argument("--base", default=str(BASE))
    args = ap.parse_args()

    ibge_json = sorted(glob.glob(os.path.join(args.base, "docs", "ibge_localidades_municipios_*.json")))[-1]
    ibge = ibge_table(ibge_json)
    dist_json = sorted(glob.glob(os.path.join(args.base, "docs", "ibge_localidades_distritos_*.json")))[-1]
    dist = district_table(dist_json)
    files = sorted(glob.glob(os.path.join(args.base, "data", "*.zip")))
    print(f"{len(files)} complaint files; IBGE list {os.path.basename(ibge_json)} ({len(ibge)} usable names)")

    parts = []
    for f in files:
        d = read_zip(f)
        g = d.groupby(["UF", "Cidade"], dropna=False).size().rename("n").reset_index()
        g["file"] = os.path.basename(f)
        parts.append(g)
    allp = pd.concat(parts, ignore_index=True)
    pairs = (allp.groupby(["UF", "Cidade"], dropna=False)
                 .agg(n_complaints=("n", "sum"), first_file=("file", "min"), last_file=("file", "max"))
                 .reset_index().rename(columns={"UF": "uf", "Cidade": "cidade"}))
    pairs["uf"] = pairs["uf"].astype(str).str.strip().str.upper()
    pairs["cidade_norm"] = pairs["cidade"].map(norm)
    out = pairs.merge(ibge, on=["uf", "cidade_norm"], how="left")
    out["match"] = out["ibge_code"].notna().map({True: "exact_normalised", False: "unmatched"})
    out["ibge_code"] = out["ibge_code"].astype("Int64")

    # rules 2 and 3 on the pairs still unmatched
    for i in out.index[out["match"] == "unmatched"]:
        uf, cid, cn = out.at[i, "uf"], out.at[i, "cidade"], out.at[i, "cidade_norm"]
        hit = unique_wildcard(uf, cid, ibge)
        rule = "wildcard_unique"
        if hit is None:
            d = dist[(dist["uf"] == uf) & (dist["cidade_norm"] == cn)]
            hit = (int(d["ibge_code"].iloc[0]), d["ibge_nome"].iloc[0]) if len(d) == 1 else unique_wildcard(uf, cid, dist)
            rule = "district"
        if hit is not None:
            out.at[i, "ibge_code"], out.at[i, "ibge_nome"], out.at[i, "match"] = hit[0], hit[1], rule
    print(out.groupby("match")["n_complaints"].agg(["size", "sum"]).to_string())

    os.makedirs(os.path.join(args.base, "crosswalk"), exist_ok=True)
    target = os.path.join(args.base, "crosswalk", "consumidor_cidade_ibge.csv")
    out.sort_values(["uf", "cidade"]).to_csv(target, index=False, encoding="utf-8")
    tot = out["n_complaints"].sum()
    um = out[out["match"] == "unmatched"]
    print(f"pairs {len(out):,}; matched {len(out) - len(um):,}; complaints matched "
          f"{100 * (1 - um['n_complaints'].sum() / tot):.2f}% of {tot:,}")
    print("largest unmatched:", um.nlargest(15, "n_complaints")[["uf", "cidade", "n_complaints"]].values.tolist())
    print("written:", target)


if __name__ == "__main__":
    main()
