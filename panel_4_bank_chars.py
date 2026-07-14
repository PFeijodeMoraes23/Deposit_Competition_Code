import os
import unicodedata
import logging
try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None

if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import numpy as np
import pandas as pd
from pathlib import Path

try:
    from utils.toon_runtime import resolve_script_paths
except Exception:
    resolve_script_paths = None

from utils import paths

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

BASE = str(paths.OPEN_FINANCE)
IF_AGG_DIR  = str(paths.IF_DATA_AGG)
IF_LIST_DIR = paths.IF_DATA_LIST   # Path object; List files have IPs, Prudential report does not
OUTPUT_DIR  = str(paths.PROCESSED)
OUT_CSV     = os.path.join(OUTPUT_DIR, 'bank_chars_panel.csv')

if resolve_script_paths is not None:
    _paths = resolve_script_paths(
        'bank_chars_panel_build',
        {
            'if_agg_dir': IF_AGG_DIR,
            'output_dir': OUTPUT_DIR,
            'out_csv': OUT_CSV,
        },
        script_dir=os.path.dirname(os.path.abspath(__file__)),
    )
    IF_AGG_DIR = _paths['if_agg_dir']
    OUTPUT_DIR = _paths['output_dir']
    OUT_CSV = _paths['out_csv']

os.makedirs(OUTPUT_DIR, exist_ok=True)


def _has_ip_from_list_files(list_dir: Path) -> dict[str, int]:
    """
    Build a {CodConglomeradoPrudencial -> has_ip} lookup from IF-Data List files.

    The Prudential Conglomerate report (Report 1) contains NO Payment Institutions
    (they are not reported under the prudential consolidation). Only the List files
    include them, identified by SegmentoTb containing 'Instituição de Pagamento'.
    has_ip=1 if ANY member institution across ALL available quarterly List files is a PI.

    Returns an empty dict on failure (fallback: has_ip stays as derived from Report 1).
    """
    def _norm(s: str) -> str:
        return (unicodedata.normalize("NFKD", str(s))
                .encode("ascii", "ignore").decode("ascii").lower())

    try:
        files = sorted(list_dir.glob("IF_DATA_List_*.csv"))
        if not files:
            logging.warning(f"[has_ip] No List files found in {list_dir}; has_ip may be all-zero.")
            return {}
        frames = []
        for f in files:
            try:
                d = pd.read_csv(f, dtype=str, encoding="utf-8")
            except UnicodeDecodeError:
                d = pd.read_csv(f, dtype=str, encoding="latin-1")
            d.columns = [c.strip() for c in d.columns]
            need = {"CodInst", "CodConglomeradoPrudencial", "SegmentoTb"}
            if not need.issubset(d.columns):
                continue
            if {"Td", "Situacao"}.issubset(d.columns):
                d = d[(d["Td"] == "I") & (d["Situacao"] == "A")]
            cong = d["CodConglomeradoPrudencial"].astype(str)
            congl = np.where(cong.isna() | (cong == "null") | (cong == "nan"),
                             d["CodInst"].astype(str), cong)
            seg = d["SegmentoTb"].map(_norm)
            is_ip = (seg.str.contains("institui", na=False)
                     & seg.str.contains("pagamento", na=False)).astype(int)
            frames.append(pd.DataFrame({"congl": congl.astype(str), "is_ip": is_ip.values}))
        if not frames:
            return {}
        congl_ip = pd.concat(frames, ignore_index=True).groupby("congl")["is_ip"].max()
        lookup = {str(k).replace(".0", ""): int(v) for k, v in congl_ip.items()}
        n_ip = sum(v for v in lookup.values() if v)
        logging.info(f"[has_ip] List files: {len(files)} files, {n_ip} IP conglomerates identified.")
        return lookup
    except Exception as exc:
        logging.warning(f"[has_ip] List-file derivation failed ({exc}); falling back to Prudential report.")
        return {}


# ── 2025 IF-Data account renumbering ─────────────────────────────────────────────
# In 2025 the BCB renumbered every IF-Data report we read (1, 2, 3, 4) — and for
# reports 2 and 4 it also RESTRUCTURED the lines. The old 78xxx codes stop at 2024Q4
# and 140xxx/141xxx codes take over from 2025Q1, so hardcoding the old codes silently
# returned NaN for all of 2025. Report 5 (capital) was NOT renumbered.
#
# The two vintages never coexist within a quarter, so both are merged into a single
# {code -> canonical_name} lookup and the pivot's groupby-sum resolves whichever code
# that period actually carries. Mappings were derived empirically by matching the raw
# `NomeColuna` labels between 2024 and 2025 — no code was guessed.
#
# This is a pure code->name re-resolution: the canonical series keep their pre-2025
# economic definitions (where 2025 SPLIT an old line into components, every component
# maps to the same canonical name so the sum re-assembles the original aggregate).

def _accounts(pre_2025: dict, from_2025: dict) -> dict:
    """Merge the pre-2025 and 2025+ account-code vintages into one lookup."""
    overlap = set(pre_2025) & set(from_2025)
    if overlap:
        raise ValueError(f'Account code(s) {sorted(overlap)} claimed by both vintages.')
    return {**pre_2025, **from_2025}


def load_and_pivot(report_num, accounts_dict):
    csv_path = os.path.join(IF_AGG_DIR, f'IF_DATA_type_1_report_{report_num}.csv')
    if not os.path.exists(csv_path):
        logging.warning(f'IF Data report {report_num} not found: {csv_path}')
        return pd.DataFrame()

    logging.info(f'Loading IF Data Report {report_num} ...')
    df = pd.read_csv(csv_path, encoding='latin1', low_memory=False)

    if 'CodConglomeradoPrudencial' not in df.columns:
        return pd.DataFrame()

    # Fallback to CNPJ if missing
    mask_no_cong = df['CodConglomeradoPrudencial'].isna() | (df['CodConglomeradoPrudencial'].astype(str).str.strip() == '')
    if mask_no_cong.any():
        df.loc[mask_no_cong, 'CodConglomeradoPrudencial'] = 'CNPJ_' + df.loc[mask_no_cong, 'CNPJ'].astype(str)
    
    # Strip .0 if parsed as float string
    df['CodConglomeradoPrudencial'] = df['CodConglomeradoPrudencial'].astype(str).str.replace(r'\.0$', '', regex=True)

    # Time variables
    df['Month'] = pd.to_numeric(df.get('Month'), errors='coerce')
    df['Year']  = pd.to_numeric(df.get('Year'),  errors='coerce')
    df = df.dropna(subset=['Month', 'Year']).copy()
    df['Quarter'] = df['Month'].map({3: 1, 6: 2, 9: 3, 12: 4})
    df = df.dropna(subset=['Quarter']).copy()
    df['Quarter'] = df['Quarter'].astype(int)
    df['Year']    = df['Year'].astype(int)

    # Filter needed accounts
    df['Conta'] = pd.to_numeric(df.get('NumeroConta'), errors='coerce').astype('Int64')
    df['Value'] = pd.to_numeric(df.get('Value'), errors='coerce')
    df_sub = df[df['Conta'].isin(accounts_dict.keys())].copy()

    if df_sub.empty:
        return pd.DataFrame()

    df_sub['prop_name'] = df_sub['Conta'].map(accounts_dict)

    # Pivot accounts
    pivot = (
        df_sub.groupby(['CodConglomeradoPrudencial', 'Year', 'Quarter', 'prop_name'])['Value']
        .sum().unstack('prop_name').reset_index()
    )
    pivot.columns.name = None

    # For Report 1, also keep segment details
    if report_num == 1:
        if 'SegmentoTb' in df.columns:
            df['is_ip'] = (
                df['SegmentoTb'].astype(str).str.contains('Institu', case=False, na=False) &
                df['SegmentoTb'].astype(str).str.contains('Pagamento', case=False, na=False)
            )
            df['is_coop'] = df['SegmentoTb'].astype(str).str.contains('Cooperativa', case=False, na=False)
        else:
            df['is_ip'] = False
            df['is_coop'] = False

        if 'NomeInstituicao' in df.columns:
            df['is_state_owned'] = (
                df['SegmentoTb'].astype(str).str.contains('Caixa Econômica', case=False, na=False) |
                df['NomeInstituicao'].astype(str).str.contains('Banco do Brasil|BNDES|BANRISUL|NORDESTE|AMAZONIA|BANZ|BANPARA|BANESE|BRB|BANDES', case=False, na=False)
            )
        else:
            df['is_state_owned'] = False

        if 'Atividade' in df.columns:
            df['is_captive'] = df['Atividade'].astype(str).str.contains('Mútuo PF sem conta corrente', case=False, na=False)
        else:
            df['is_captive'] = False

        if 'Sr' in df.columns:
            df['segment_raw'] = df['Sr'].astype(str).str.strip().str.upper()
            df['segment_raw'] = df['segment_raw'].where(df['segment_raw'].isin(['S1', 'S2', 'S3', 'S4', 'S5']), np.nan)
        else:
            df['segment_raw'] = np.nan

        cat_agg = (
            df.groupby(['CodConglomeradoPrudencial', 'Year', 'Quarter'])
            .agg(has_ip=('is_ip', 'max'), 
                 is_coop=('is_coop', 'max'),
                 is_state_owned=('is_state_owned', 'max'),
                 is_captive=('is_captive', 'max'),
                 segment=('segment_raw', 'first'))
            .reset_index()
        )
        pivot = pivot.merge(cat_agg, on=['CodConglomeradoPrudencial', 'Year', 'Quarter'], how='right')

    return pivot

def build_panel() -> pd.DataFrame:
    # 1. Base (Resumo) size attributes
    p1 = load_and_pivot(1, _accounts(
        {78182: 'total_assets',    # Ativo Total
         78186: 'equity'},         # Patrimonio Liquido
        {140220: 'total_assets',   # 2025: Ativo Total
         140246: 'equity'},        # 2025: Patrimonio Liquido
    ))
    if p1.empty: return pd.DataFrame()
    for col in ['total_assets', 'equity']:
        if col not in p1.columns: p1[col] = np.nan

    # 2. Asset quality (Ativo) — NPL provision + the EARNING-ASSET stocks.
    #    The earning-asset stocks feed the bank's asset return r^j (V_Main eq 16, ψ1 row): the
    #    return on what deposits actually fund. Disponibilidades (78188) is deliberately EXCLUDED
    #    — it is non-earning cash. Compulsórios are not broken out separately in IF-Data's condensed
    #    Ativo; they sit inside the asset base earning little, so the realized yield below is already
    #    reserve-drag-adjusted (see counterfactuals_plan.md §9.4).
    p2 = load_and_pivot(2, _accounts(
        {78192: 'npl_provision',    # Provisao sobre Operacoes de Credito
         78189: 'aplic_interfin',   # Aplicacoes Interfinanceiras de Liquidez
         78190: 'tvm',              # TVM e Instrumentos Financeiros Derivativos
         78193: 'credit_net',       # Operacoes de Credito Liquidas de Provisao
         78198: 'leasing_net'},     # Arrendamento Mercantil Liquido de Provisao
        # 2025 renumbered AND restructured the Ativo (IFRS-9 style "Perda Esperada").
        {140202: 'npl_provision',   # Perda Esperada (e2), under Operacoes de Credito
         140199: 'aplic_interfin',  # Aplicacoes Interfinanceiras de Liquidez
         # 78190 bundled TVM + derivatives; 2025 splits them. Both map to 'tvm' so the
         # groupby-sum re-assembles the pre-2025 aggregate (no definition change).
         140200: 'tvm',             # Titulos e Valores Mobiliarios
         141612: 'tvm',             # Instrumentos Derivativos
         140205: 'credit_net',      # Operacoes de Credito (e) — already net of Perda Esperada
         140210: 'leasing_net'},    # Operacoes de Arrendamento Financeiro (f) — net
        # NOT mapped (genuinely NEW 2025 lines with no pre-2025 counterpart, so including
        # them would break comparability): 140216 Outras Operacoes com Caracteristicas de
        # Concessao de Credito, 145833 Valores a Receber de Transacoes de Pagamentos.
    ))

    # 3. Wholesale (Passivo)
    p3 = load_and_pivot(3, _accounts(
        {78288: 'repos',                  # Obrigacoes por Operacoes Compromissadas
         78289: 'lci',                    # Letras de Credito Imobiliario
         78290: 'lca',                    # Letras de Credito do Agronegocio
         78291: 'letras_financeiras',     # Letras Financeiras
         78295: 'emprestimos_repasses'},  # Obrigacoes por Emprestimos e Repasses
        {140230: 'repos',                 # 2025: same five lines, renumbered 1:1
         140231: 'lci',
         140232: 'lca',
         140233: 'letras_financeiras',
         140238: 'emprestimos_repasses'},
    ))

    # 4. Costs + financial income (DRE)
    p4 = load_and_pivot(4, _accounts(
        {78218: 'personnel_expenses',  # Despesas de Pessoal
         78219: 'admin_expenses',      # Despesas Administrativas
         78220: 'tax_expenses',        # Despesas Tributarias
         78208: 'fin_income'},         # Receitas de Intermediacao Financeira (= credit + TVM + deriv + ...)
        # 2025 rebuilt the DRE. The three expense lines survive 1:1 by name. The single
        # gross-revenue total 78208 does NOT: it was decomposed into per-asset-class
        # "Rendas de ..." lines (the 2025 total 141851 is the NET result of intermediation
        # — the analogue of the old 78215, not of 78208). We therefore re-assemble 78208
        # from exactly the components it used to contain, so 'fin_income' keeps its
        # pre-2025 meaning and stays the income earned on `earning_assets` below.
        {141858: 'personnel_expenses',  # Despesas de Pessoal (o)
         141859: 'admin_expenses',      # Despesas Administrativas (p)
         141862: 'tax_expenses',        # Despesas Tributarias (s)
         141835: 'fin_income',          # Rendas de Operacoes de Credito      (old a1)
         141836: 'fin_income',          # Rendas de Arrendamento Financeiro   (old a2)
         141830: 'fin_income',          # Rendas de Titulos e Valores Mobiliarios (old a3)
         141849: 'fin_income',          # Resultado com Derivativos           (old a4)
         141825: 'fin_income'},         # Rendas de Aplicacoes Interfinanceiras de Liquidez
                                        #   (carries the old a6 Rendas de Aplicacoes Compulsorias;
                                        #    old a5 Resultado de Cambio has no 2025 line — FX is now
                                        #    folded into the "Ajuste de Variacao Cambial" subcomponents
                                        #    already inside the totals above).
        # NOT mapped: 141837 Rendas de Outras Operacoes com Caracteristicas de Concessao de
        # Credito and 141850 Outros Resultados de Intermediacao Financeira — new buckets whose
        # assets are likewise excluded from `earning_assets`, so leaving them out keeps the
        # income/asset numerator and denominator consistent.
        #
        # CAVEAT (fin_income only): unlike every other series here, 78208 has NO exact 2025
        # counterpart, so this is a best-faith reconstruction, not an identity. It does NOT
        # splice perfectly: system-wide 2025Q1 = R$336bn vs 2024Q1 = R$451bn. The shortfall is
        # concentrated in TVM income (R$103bn vs R$159bn) and in derivatives, which flipped from
        # a GROSS revenue line (old a4, +R$10bn) to a NET result (new (i), -R$24bn) — the 2025
        # DRE simply does not decompose the old way. Alternative groupings land at R$347-370bn;
        # none recover R$451bn. Consequence: `asset_return_qoq` (r^j) is biased DOWN in 2025
        # relative to earlier years. Revisit before leaning on 2025 asset returns.
    ))

    # 5. Capital (Informacoes de Capital) — report 5 was NOT renumbered in 2025.
    p5 = load_and_pivot(5, {79664: 'indice_basileia_raw'})

    # Merge everything
    panel = p1
    if not p2.empty: panel = panel.merge(p2, on=['CodConglomeradoPrudencial', 'Year', 'Quarter'], how='left')
    if not p3.empty: panel = panel.merge(p3, on=['CodConglomeradoPrudencial', 'Year', 'Quarter'], how='left')
    if not p4.empty: panel = panel.merge(p4, on=['CodConglomeradoPrudencial', 'Year', 'Quarter'], how='left')
    if not p5.empty: panel = panel.merge(p5, on=['CodConglomeradoPrudencial', 'Year', 'Quarter'], how='left')

    panel.sort_values(['CodConglomeradoPrudencial', 'Year', 'Quarter'], inplace=True)

    # The IF-Data DRE is cumulative WITHIN EACH SEMESTER, not within the calendar year: the
    # series RESETS in July.  A large bank's 2022 admin expenses run
    #     Q1 -5.79bn   Q2 -12.10bn   Q3 -6.77bn   Q4 -14.06bn
    # i.e. Q3 (H2 to date) is *smaller* than Q2 (H1 total).  Differencing by year therefore gets
    # Q1/Q2/Q4 right but computes Q3 = M09 - M06, which is the wrong sign for every expense line
    # (and turns fin_income negative, so the `>0` guard below silently drops it and the asset
    # return gets median-imputed for a quarter of the panel).  The cost ratios feed the BLP's
    # cost-shifter instruments, so a sign flip in 25% of quarters is not cosmetic.
    #
    # Correct disaccumulation is therefore by (conglomerate, year, SEMESTER):
    #     Q1 = M03            Q2 = M06 - M03            Q3 = M09            Q4 = M12 - M09
    # The first quarter of each semester keeps its reported (already single-quarter) value; the
    # second is a within-semester difference.  Same defect class as the COSIF semester-cumulative
    # bug fixed in panel_2/panel_3.
    if not p4.empty:
        semester = np.where(panel['Quarter'] <= 2, 1, 2)
        for cost_col in ['personnel_expenses', 'admin_expenses', 'tax_expenses', 'fin_income']:
            if cost_col in panel.columns:
                panel[cost_col] = panel[cost_col].fillna(0)
                val_diff = panel.groupby(
                    ['CodConglomeradoPrudencial', 'Year', semester])[cost_col].diff()
                # NaN = first quarter present in that semester (normally Q1/Q3) -> keep as reported.
                panel[cost_col] = val_diff.fillna(panel[cost_col])

    # Core characteristics
    total_assets_no0 = panel['total_assets'].replace(0, np.nan)
    panel['equity_ratio'] = panel['equity'] / total_assets_no0
    panel['log_total_assets'] = np.log(panel['total_assets'].clip(lower=1))

    # NPL provision ratio (provision is reported as negative; take abs)
    if 'npl_provision' in panel.columns:
        panel['npl_provision_ratio'] = panel['npl_provision'].abs() / total_assets_no0
    else:
        panel['npl_provision_ratio'] = np.nan

    # Segment defaults
    for s in ['S2', 'S3', 'S4', 'S5']:
        panel[f'seg_{s}'] = (panel.get('segment') == s).astype(int)

    # Wholesale ratios
    if 'lci' in panel.columns and 'lca' in panel.columns:
        panel['lci_lca_ratio'] = panel[['lci', 'lca']].sum(axis=1) / total_assets_no0
    else:
        panel['lci_lca_ratio'] = np.nan

    wholesale_cols = [c for c in ['repos', 'lci', 'lca', 'letras_financeiras', 'emprestimos_repasses'] if c in panel.columns]
    panel['wholesale_ratio'] = panel[wholesale_cols].sum(axis=1) / total_assets_no0 if wholesale_cols else np.nan

    # Cost ratios (take absolute value since DRE sums are negative expenses)
    if not p4.empty:
        for c, out_n in [('personnel_expenses', 'personnel_cost_ratio'), 
                         ('admin_expenses', 'admin_cost_ratio'), 
                         ('tax_expenses', 'tax_cost_ratio')]:
            if c in panel.columns:
                panel[out_n] = panel[c].abs() / total_assets_no0
            else:
                panel[out_n] = np.nan
    else:
        for c in ['personnel_cost_ratio', 'admin_cost_ratio', 'tax_cost_ratio']:
            panel[c] = np.nan

    if 'indice_basileia_raw' in panel.columns:
        panel['indice_basileia'] = panel['indice_basileia_raw']
    else:
        panel['indice_basileia'] = np.nan

    # ── ASSET RETURN r^j (V_Main eq 16, ψ1 row) ─────────────────────────────────────────────
    # The return the bank earns on the assets its deposits fund. CF2 needs (r^j − r^f); with r^j
    # left at 0 the deposit franchise is worth only (ρ − c) and eq-18 can only rationalise the
    # observed spreads with a NEGATIVE marginal cost (see counterfactuals_plan.md §9.4).
    #
    # Realized portfolio yield = quarterly financial income ÷ LAGGED earning assets. Dividing by
    # the lagged stock mirrors the deposit implicit-rate convention (expense ÷ lagged stock) and
    # keeps the flow/stock timing honest. Because compulsórios sit inside the asset base earning
    # little, this realized yield is already reserve-drag ("compulsório") adjusted.
    ea_cols = [c for c in ['aplic_interfin', 'tvm', 'credit_net', 'leasing_net'] if c in panel.columns]
    if ea_cols and 'fin_income' in panel.columns:
        panel['earning_assets'] = panel[ea_cols].sum(axis=1, min_count=1)
        ea_lag = panel.groupby('CodConglomeradoPrudencial')['earning_assets'].shift(1)
        ea_lag = ea_lag.where(ea_lag > 0)                      # guard: no yield off a zero/neg base
        # The DRE differencing above fillna(0)s missing flows. A bank with NO reported income would
        # then get r^j = 0 and hence an asset margin of −r^f (≈ −10.5pp/yr) — reintroducing exactly
        # the pathology this column exists to remove. So only trust a strictly positive income, and
        # impute the rest from the cross-sectional median for that quarter (r^j is a bank
        # characteristic; the quarter median is the natural fallback and keeps the sign right).
        raw = panel['fin_income'].where(panel['fin_income'] > 0) / ea_lag
        raw = raw.replace([np.inf, -np.inf], np.nan)
        raw = raw.where((raw > 0) & (raw < 0.5))               # drop absurd yields (>50%/quarter)
        med_by_q = raw.groupby([panel['Year'], panel['Quarter']]).transform('median')
        panel['asset_return_qoq'] = raw.fillna(med_by_q).fillna(raw.median())
        panel['asset_return_imputed'] = raw.isna().astype(int)
    else:
        panel['earning_assets'] = np.nan
        panel['asset_return_qoq'] = np.nan
        panel['asset_return_imputed'] = 1

    # Override has_ip using IF-Data List files, which include Payment Institutions.
    # The Prudential Conglomerate report (Report 1) never lists IPs, so has_ip from
    # the pivot above is always 0. List files are the authoritative source.
    ip_lookup = _has_ip_from_list_files(IF_LIST_DIR)
    if ip_lookup:
        panel['has_ip'] = (
            panel['CodConglomeradoPrudencial'].astype(str)
            .str.replace(r'\.0$', '', regex=True)
            .map(ip_lookup)
            .fillna(panel.get('has_ip', 0))
            .astype(int)
        )

    # FINALLY, Lag all variables to be used cleanly in regressions
    lag_cols = ['total_assets', 'equity', 'equity_ratio', 'log_total_assets',
                'lci_lca_ratio', 'wholesale_ratio', 'indice_basileia',
                'personnel_cost_ratio', 'admin_cost_ratio', 'tax_cost_ratio',
                'npl_provision_ratio', 'asset_return_qoq']

    present_lag_cols = [c for c in lag_cols if c in panel.columns]
    panel[[c + '_lag' for c in present_lag_cols]] = panel.groupby('CodConglomeradoPrudencial')[present_lag_cols].shift(1)

    # Gross factor for the Julia consumer (cost_2_fwd_sim.jl --asset-return-col), which does
    #     gr = col − 1 ;  asset_ret = gr − risk_free_qoq_lag
    # so this must be 1 + a LAGGED quarterly DECIMAL rate — the same vintage and units as
    # gross_return_lag = 1 + deposit_rate_lag (≈1.0126) and risk_free_qoq_lag (≈0.0253).
    if 'asset_return_qoq_lag' in panel.columns:
        panel['asset_gross_return_lag'] = 1.0 + panel['asset_return_qoq_lag']

    panel.rename(columns={'Year': 'year', 'Quarter': 'quarter'}, inplace=True)
    return panel

def main() -> None:
    panel = build_panel()
    if not panel.empty:
        import pyarrow as pa
        import pyarrow.csv as pa_csv
        pa_csv.write_csv(pa.Table.from_pandas(panel, preserve_index=False), OUT_CSV)
        logging.info(f'Saved bank characteristics panel to {OUT_CSV}')
    else:
        logging.warning('Build returned an empty panel.')

if __name__ == '__main__':
    main()
