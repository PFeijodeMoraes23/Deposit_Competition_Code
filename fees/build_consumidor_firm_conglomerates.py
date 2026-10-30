###─────────────────────────────────────────────────────────────────────────────
# fees/build_consumidor_firm_conglomerates.py
#
# Author: Pedro Feijó de Moraes
#
# PURPOSE
#   Put the consumidor.gov.br firm labels (`Nome Fantasia`, a brand) on the panel's key, the
#   prudential conglomerate, and sum the fee-complaint counts of
#   build_consumidor_fee_complaints.py to conglomerate x quarter and conglomerate x
#   municipality x year.
#
#   A label reaches a conglomerate only through a CNPJ root, and the code always comes from
#   utils.conglomerates.build_cnpj_conglomerate_map (the IF Data List, canonical code per
#   CNPJ), never from this file. Two rules:
#     table       LABEL_CNPJ below: the CNPJ root of the institution the label names
#                 ("entity") or whose brand it is ("brand", e.g. Ourocard -> Banco do
#                 Brasil). Each entry carries a fragment that must appear in a name the
#                 BCB files on disk give that CNPJ (IF Data List or the tariff census), so a
#                 wrong root fails the run instead of joining the wrong firm.
#     name_exact  the label, without legal suffixes, equals the name of exactly one
#                 institution on disk, and that institution has a code.
#   Every other label stays unresolved on purpose: cooperative systems (Sicredi), labels
#   whose two entities carry different codes (Crefisa), and brands no file on disk names.
#
#   The code is the CNPJ's canonical one for the whole sample (for an ESTBAN bank that is
#   also the code the panel files its deposits under in every quarter).
#
# INPUT   utils.paths.TARIFAS_PROC: consumidor_fee_complaints_firms.csv,
#         consumidor_fee_complaints_firm_quarter.csv,
#         consumidor_fee_complaints_firm_municipality_year.csv
#         BCB/Tarifas/processed/tariff_history_bcb_monthly.csv (census names)
# OUTPUT  (utils.paths.TARIFAS_PROC)
#   consumidor_firm_conglomerate_crosswalk.csv  one row per label: cnpj8, rule, link,
#       name_on_disk, cod_cong_prudencial, counts 2016-2024
#   consumidor_fee_complaints_conglomerate_quarter.csv          code x quarter
#   consumidor_fee_complaints_conglomerate_municipality_year.csv code x ibge_code x year
#   (counts summed over the labels of a code; shares recomputed)
#
# USAGE
#   python fees/build_consumidor_firm_conglomerates.py   (after build_consumidor_fee_complaints.py)
###─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: utils/

from utils import paths
from utils.conglomerates import build_cnpj_conglomerate_map

PROC = paths.TARIFAS_PROC
COUNTS = ["n_fin", "n_fee", "n_account", "n_fee_account"]

# label key -> (CNPJ root, fragment of a name on disk for that root, entity | brand).
# Keys are key(label); the fragment is matched against key(name).
LABEL_CNPJ = {
    "CAIXA ECONOMICA FEDERAL": ("00360305", "CAIXA ECONOMICA FEDERAL", "entity"),
    "BANCO BRADESCO": ("60746948", "BRADESCO", "entity"),
    "MERCADO PAGO": ("10573521", "MERCADO PAGO", "entity"),
    "BANCO SANTANDER": ("90400888", "SANTANDER", "entity"),
    "BANCO PAN": ("59285411", "BANCO PAN", "entity"),
    "BANCO DO BRASIL": ("00000000", "BANCO DO BRASIL", "entity"),
    "BANCO ITAU UNIBANCO": ("60701190", "ITAU UNIBANCO", "entity"),
    "NUBANK": ("18236120", "NU PAGAMENTOS", "brand"),
    "BANCO BMG": ("61186680", "BANCO BMG", "entity"),
    "BRADESCO CARTOES": ("59438325", "BANCO BRADESCO CART", "entity"),
    "BANCO INTER BANCO INTERMEDIUM": ("00416968", "BANCO INTER", "entity"),
    "BRADESCARD": ("04184779", "BRADESCARD", "entity"),
    "ITAUCARD": ("17192451", "ITAUCARD", "entity"),
    "BANCO CETELEM": ("00558456", "BANCO CETELEM", "entity"),
    "C6 BANK": ("31872495", "BANCO C6", "brand"),
    "BANCO SANTANDER CARTOES": ("90400888", "SANTANDER", "brand"),
    "CARTOES ITAU": ("60701190", "ITAU UNIBANCO", "brand"),
    "OUROCARD": ("00000000", "BANCO DO BRASIL", "brand"),             # BB's card brand
    "BANCO SAFRA": ("58160789", "BANCO SAFRA", "entity"),
    "BANCO OLE CONSIGNADO": ("71371686", "OLE", "entity"),
    "BANCO C6 CONSIGNADO BANCO FICSA": ("61348538", "C6 CONSIGNADO", "entity"),
    "BANCO DAYCOVAL": ("62232889", "BANCO DAYCOVAL", "entity"),
    "PAGSEGURO": ("08561701", "PAGSEGURO", "entity"),
    "CARTAO CARREFOUR": ("08357240", "BANCO CSF", "brand"),           # Carrefour's bank, CSF
    "SANTANDER FINANCIAMENTOS AYMORE": ("07707650", "AYMORE", "entity"),
    "BANCO AGIBANK AGIPLAN": ("10664513", "BANCO AGIBANK", "entity"),
    "ITAU CONSIGNADO": ("33885724", "ITAU CONSIGNADO", "entity"),
    "PICPAY": ("22896431", "PICPAY", "entity"),
    "BRADESCO PROMOTORA": ("60746948", "BRADESCO", "brand"),          # Bradesco's sales arm
    "BRADESCO FINANCIAMENTOS": ("07207996", "BRADESCO FINANCIAMENTOS", "entity"),
    "BANCO BV ANTIGO BANCO VOTORANTIM": ("01858774", "BANCO BV", "entity"),
    "PERNAMBUCANAS CARTOES": ("43180355", "PERNAMBUCANAS", "brand"),
    "BANCO MERCANTIL DO BRASIL": ("17184037", "BANCO MERCANTIL DO BRASIL", "entity"),
    "BRB BANCO DE BRASILIA": ("00000208", "BANCO DE BRASILIA", "entity"),
    "MIDWAY RIACHUELO": ("09464032", "MIDWAY", "entity"),
    "BANRISUL": ("92702067", "ESTADO DO RIO GRANDE DO SUL", "entity"),
    "SEM PARAR": ("04088208", "SEM PARAR", "entity"),
    "PORTO SEGURO CARTOES E FINANCIAMENTOS": ("04862600", "PORTOSEG", "brand"),
    "FACTA FINANCEIRA": ("15581638", "FACTA FINANCEIRA", "entity"),
    "CREDICARD": ("34098442", "CREDICARD", "entity"),
    "CARTOES RENNER REALIZE CFI": ("27351731", "REALIZE", "entity"),
    "LUIZACRED": ("02206577", "LUIZACRED", "entity"),
    "CARTOES RIACHUELO MIDWAY FINANCEIRA": ("09464032", "MIDWAY", "entity"),
    "BANCO ORIGINAL": ("92894922", "BANCO ORIGINAL", "entity"),
    "BV FINANCEIRA": ("01149953", "BV FINANCEIRA", "entity"),
    "BRBCARD": ("00000208", "BANCO DE BRASILIA", "brand"),            # BRB's card company
    "PARANA BANCO": ("14388334", "PARANA BANCO", "entity"),
    "ITAU UNIBANCO CONSIGNADO": ("33885724", "ITAU CONSIGNADO", "entity"),
    "HIPERCARD": ("03012230", "HIPERCARD", "entity"),
    "LOSANGO": ("33254319", "LOSANGO", "entity"),
    "BANCO DIGIO": ("27098060", "BANCO DIGIO", "entity"),
    "ITAULEASING": ("49925225", "ITAULEASING", "entity"),
    "BANCO MASTER ANTIGO BANCO MAXIMA": ("33923798", "MAXIMA", "entity"),
    "AGIBANK FINANCEIRA": ("13660104", "AGIBANK FINANCEIRA", "entity"),
    "AME DIGITAL": ("32778350", "AME DIGITAL", "entity"),
    "OMNI FINANCEIRA": ("92228410", "OMNI", "entity"),
    "BANCO VOTORANTIM": ("59588111", "BANCO VOTORANTIM", "entity"),
    "CIELO": ("01027058", "CIELO", "entity"),
    "CREDSYSTEM": ("04670195", "CREDSYSTEM", "entity"),
    "BANCO NEXT": ("60746948", "BRADESCO", "brand"),                  # Bradesco's digital bank
    "BRASILCARD": ("03130170", "BRASIL CARD", "entity"),
    "GETNET": ("10440482", "GETNET", "entity"),
    "WILL BANK ANTIGO MEU PAG": ("23862762", "WILL FINANCEIRA", "brand"),
    "WILL BANK": ("23862762", "WILL FINANCEIRA", "brand"),
    "SENFF": ("11970623", "SENFF", "entity"),
    "EBANX": ("21018182", "EBANX", "entity"),
    "PAYPAL DO BRASIL": ("10878448", "PAYPAL", "entity"),
    "99PAY": ("24313102", "99PAY", "entity"),
    "BANCO VOLKSWAGEN": ("59109165", "BANCO VOLKSWAGEN", "entity"),
    "BANCO XP": ("33264668", "BANCO XP", "entity"),
    "BANCO DIGIMAIS ANTIGO BANCO RENNER": ("92874270", "DIGIMAIS", "entity"),
    "BANCO BTG PACTUAL": ("30306294", "BANCO BTG PACTUAL", "entity"),
    "ITAU UNIBANCO CREDITO IMOBILIARIO": ("60701190", "ITAU UNIBANCO", "brand"),
    "CREDITAS": ("32997490", "CREDITAS", "entity"),
    "RECARGAPAY": ("11275560", "RECARGAPAY", "entity"),
    "BANCO DO NORDESTE": ("07237373", "BANCO DO NORDESTE", "entity"),
    "TRICARD": ("05045717", "TRICARD", "entity"),
    "BANCO BNP PARIBAS": ("01522368", "BNP PARIBAS", "entity"),
    "AFINZ SOROCRED": ("04814563", "SOROCRED", "entity"),
    "PEFISA": ("43180355", "PEFISA", "entity"),
    "BANCO BS2": ("71027866", "BANCO BS2", "entity"),
    "BANESE BANCO DO ESTADO DE SERGIPE": ("13009717", "ESTADO DE SERGIPE", "entity"),
    "ALELO VELOE PEDE PRONTO": ("04740876", "ALELO", "entity"),
    "PORTOCRED FINANCEIRA": ("01800019", "PORTOCRED", "entity"),
    "BANCO HONDA": ("03634220", "BANCO HONDA", "entity"),
    "FORTBRASIL": ("02732968", "FORTBRASIL", "entity"),
    "BANESTES": ("28127603", "BANESTES", "entity"),
}
# Labels that name two retired brands now inside Banco Santander keep that bank's root.
LABEL_CNPJ["BANCO SANTANDER CARTOES DESATIVADA ATUAL BANCO SANTANDER"] = ("90400888", "SANTANDER", "brand")
LABEL_CNPJ["BANCO OLE CONSIGNADO DESATIVADA ATUAL BANCO SANTANDER"] = ("71371686", "OLE", "entity")

LEGAL_SUFFIX = re.compile(r"( S ?A| LTDA| ME| EPP)$")


def key(s: str) -> str:
    """Accent-free upper-case key, every non-alphanumeric run as one space; a letter the
    source lost (U+FFFD) is dropped."""
    s = unicodedata.normalize("NFKD", str(s).replace("�", "")).encode("ascii", "ignore")
    return re.sub(r"[^A-Z0-9]+", " ", s.decode("ascii").upper()).strip()


def names_on_disk(cmap: dict) -> pd.DataFrame:
    """Every (CNPJ root, name) the BCB files on disk give: IF Data List and tariff census."""
    ifd = pd.DataFrame([(c, v[2]) for c, v in cmap.items()], columns=["cnpj", "name"])
    cen = pd.read_csv(PROC / "tariff_history_bcb_monthly.csv", usecols=["cnpj8", "instituicao"],
                      dtype=str).drop_duplicates()
    cen = pd.DataFrame({"cnpj": cen["cnpj8"].astype(int),
                        "name": cen["instituicao"].str.split(" - ").str[0]})
    out = pd.concat([ifd, cen], ignore_index=True)
    out["name_key"] = out["name"].map(key)
    return out.drop_duplicates(["cnpj", "name_key"])


def resolve(labels: pd.Series, cmap: dict, names: pd.DataFrame) -> pd.DataFrame:
    """One row per label: cnpj8, rule, link, name_on_disk, cod_cong_prudencial."""
    by_cnpj = names.groupby("cnpj")["name_key"].apply(list).to_dict()
    exact = names.assign(stripped=names["name_key"].str.replace(LEGAL_SUFFIX, "", regex=True))
    exact["code"] = exact["cnpj"].map(lambda c: cmap.get(c, (None,))[0])
    exact = exact[exact["code"].notna()]
    rows = []
    for label in labels:
        k = key(label)
        row = {"firm": label, "label_key": k, "cnpj8": None, "rule": "unresolved", "link": None,
               "name_on_disk": None}
        if k in LABEL_CNPJ:
            cnpj8, frag, link = LABEL_CNPJ[k]
            hits = [n for n in by_cnpj.get(int(cnpj8), []) if frag in n]
            if not hits:
                raise ValueError(f"{label!r}: no name on disk for {cnpj8} contains {frag!r}")
            row.update(cnpj8=cnpj8, rule="table", link=link, name_on_disk=hits[0])
        else:
            hit = exact[exact["stripped"] == LEGAL_SUFFIX.sub("", k)]
            if hit["cnpj"].nunique() == 1:
                row.update(cnpj8=f"{int(hit['cnpj'].iloc[0]):08d}", rule="name_exact",
                           link="entity", name_on_disk=hit["name_key"].iloc[0])
        if row["cnpj8"] is not None:
            row["cod_cong_prudencial"] = cmap.get(int(row["cnpj8"]), (None,))[0]
            if row["cod_cong_prudencial"] is None:
                row["rule"] = row["rule"] + "_no_code"
        rows.append(row)
    return pd.DataFrame(rows)


def to_codes(t: pd.DataFrame, xw: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Sum a firm-level table to conglomerate level and recompute the shares."""
    t = t.merge(xw[["firm", "cod_cong_prudencial"]], on="firm", how="inner")
    t = t[t["cod_cong_prudencial"].notna()]
    out = t.groupby(["cod_cong_prudencial"] + keys)[COUNTS].sum().reset_index()
    out["n_labels"] = t.groupby(["cod_cong_prudencial"] + keys)["firm"].nunique().to_numpy()
    out["share_fee"] = out["n_fee"] / out["n_fin"]
    out["share_fee_account"] = (out["n_fee_account"] / out["n_account"]).where(out["n_account"] > 0)
    return out


def main() -> None:
    firms = pd.read_csv(PROC / "consumidor_fee_complaints_firms.csv")
    cmap = build_cnpj_conglomerate_map()
    xw = resolve(firms["firm"], cmap, names_on_disk(cmap))
    unused = set(LABEL_CNPJ) - set(xw["label_key"])
    if unused:
        print("table entries matching no label:", sorted(unused))
    xw = xw.merge(firms[["firm"] + COUNTS], on="firm", how="left")
    xw.to_csv(PROC / "consumidor_firm_conglomerate_crosswalk.csv", index=False)

    fq = pd.read_csv(PROC / "consumidor_fee_complaints_firm_quarter.csv")
    to_codes(fq, xw, ["year", "quarter"]).to_csv(
        PROC / "consumidor_fee_complaints_conglomerate_quarter.csv", index=False, float_format="%.6g")
    fmy = pd.read_csv(PROC / "consumidor_fee_complaints_firm_municipality_year.csv")
    to_codes(fmy, xw, ["ibge_code", "year"]).to_csv(
        PROC / "consumidor_fee_complaints_conglomerate_municipality_year.csv", index=False,
        float_format="%.6g")

    has = xw["cod_cong_prudencial"].notna()
    print(xw.groupby("rule").agg(labels=("firm", "size"), n_fin=("n_fin", "sum"),
                                 n_account=("n_account", "sum")).to_string())
    for c in COUNTS:
        print(f"{c}: {100 * xw.loc[has, c].sum() / xw[c].sum():.1f}% on a conglomerate")
    print(f"labels {len(xw)}, resolved {int(has.sum())}, codes {xw.loc[has, 'cod_cong_prudencial'].nunique()}")
    top = xw[~has].nlargest(12, "n_fin")[["firm", "n_fin", "n_account"]]
    print("largest unresolved:\n" + top.to_string(index=False))
    print(f"wrote {PROC}")


if __name__ == "__main__":
    main()
