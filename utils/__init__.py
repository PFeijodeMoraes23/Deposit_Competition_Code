"""Utility modules for the Egan_et_al_2025_Rep pipeline."""

def load_panel_cached(panel_csv, **read_csv_kwargs):
    """Read market_panel.csv, caching a Parquet copy for faster subsequent loads.

    On first call the CSV is read (passing any extra kwargs to read_csv) and a
    .parquet sidecar is written next to the CSV.  On subsequent calls the
    Parquet is returned instead – typically 3-5× faster and lower RAM than CSV
    parsing.  If the CSV is newer than the Parquet (i.e. the panel was
    rebuilt) the Parquet cache is refreshed automatically.

    The mca_code column is always returned as str, consistent with the
    dtype={'mca_code': str} convention used across the pipeline.
    """
    from pathlib import Path
    import pandas as pd

    panel_csv = Path(panel_csv)
    parquet_path = panel_csv.with_suffix('.parquet')

    cache_valid = (
        parquet_path.exists()
        and parquet_path.stat().st_mtime >= panel_csv.stat().st_mtime
    )

    if cache_valid:
        df = pd.read_parquet(parquet_path)
        if 'mca_code' in df.columns:
            df['mca_code'] = df['mca_code'].astype(str)
        return df

    # Read CSV – honour caller's kwargs (e.g. dtype, low_memory)
    read_csv_kwargs.setdefault('low_memory', False)
    read_csv_kwargs.setdefault('dtype', {})
    read_csv_kwargs['dtype']['mca_code'] = str  # ensure consistent str type
    df = pd.read_csv(panel_csv, **read_csv_kwargs)

    try:
        df.to_parquet(parquet_path, index=False)
    except Exception:
        pass  # non-fatal; fall back to CSV next time

    return df
