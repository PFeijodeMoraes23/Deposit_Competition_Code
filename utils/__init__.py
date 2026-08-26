"""Utility modules for the Egan_et_al_2025_Rep pipeline."""

def load_panel_cached(panel_csv, **read_csv_kwargs):
    """Read the market panel, from whichever of the CSV/Parquet pair is authoritative.

    TWO MODES, decided by whether the CSV is on disk.

    CSV PRESENT -- the sidecar is a CACHE.  The CSV is the source of truth; the
    .parquet next to it is a faster copy of it.  On first call the CSV is read
    (passing any extra kwargs to read_csv) and the sidecar is written.  On later
    calls the Parquet is returned instead -- typically 3-5x faster and lower RAM
    than CSV parsing -- but only while `_cache_is_current` can still vouch for it
    against the CSV; otherwise the CSV is re-read and the sidecar refreshed.

    CSV ABSENT -- the sidecar IS the source.  The cluster upload ships
    market_panel.parquet (39 MB) instead of market_panel.csv (724 MB), so on the
    cluster the CSV never exists at all and the Parquet is the only copy of the
    panel there is.  There is then nothing to validate the sidecar against and
    nothing to refresh, so it is read directly.  Its integrity is established
    before it leaves the local machine (stage_cluster_upload.py checks its
    `source_csv_bytes` stamp against the local CSV and records its sha256) and on
    arrival (gate G0, env_job.sh ENV_STEP=preflight, opens it and counts rows).

    Neither on disk is an error naming both paths, since a panel that resolved to
    the wrong directory otherwise surfaces as an opaque parser failure.

    The mca_code column is always returned as str, consistent with the
    dtype={'mca_code': str} convention used across the pipeline.
    """
    from pathlib import Path
    import pandas as pd

    import logging

    panel_csv = Path(panel_csv)
    parquet_path = sidecar_path(panel_csv)
    csv_exists = panel_csv.exists()

    if not csv_exists and not parquet_path.exists():
        raise FileNotFoundError(
            f"no market panel to read: neither {panel_csv} nor {parquet_path} exists "
            f"(the CSV is the local source of truth; the Parquet is what the cluster "
            f"upload ships)")

    if _cache_is_current(panel_csv, parquet_path):
        if not csv_exists:
            logging.info(f"panel source is the Parquet {parquet_path}: no "
                         f"{panel_csv.name} on disk, so the sidecar is the panel "
                         f"(nothing to validate against, nothing to refresh)")
        df = pd.read_parquet(parquet_path)
        if 'mca_code' in df.columns:
            df['mca_code'] = df['mca_code'].astype(str)
        return df

    if parquet_path.exists():
        logging.info(f"panel cache stale for {panel_csv.name}; re-reading the CSV and "
                     f"refreshing {parquet_path.name}")

    # Read CSV – honour caller's kwargs (e.g. dtype, low_memory)
    read_csv_kwargs.setdefault('low_memory', False)
    read_csv_kwargs.setdefault('dtype', {})
    read_csv_kwargs['dtype']['mca_code'] = str  # ensure consistent str type
    df = pd.read_csv(panel_csv, **read_csv_kwargs)

    refresh_panel_cache(panel_csv, df)
    return df


def sidecar_path(panel_csv):
    """Path of the Parquet sidecar belonging to a panel CSV."""
    from pathlib import Path
    return Path(panel_csv).with_suffix('.parquet')


def _cache_is_current(panel_csv, parquet_path) -> bool:
    """True when the sidecar is the frame to read.

    With the CSV PRESENT this asks whether the sidecar can be trusted to hold that
    CSV's contents, on two conditions, because mtime alone is not safe here. The panel
    lives inside a OneDrive-synced tree, where a sync can touch mtimes without the
    content changing, and a restored-from-backup CSV can be OLDER than a sidecar built
    from different data. So the sidecar also records the size of the CSV it was built
    from, and that must still match. A rebuilt panel essentially always changes size, so
    a matching size plus a non-regressed mtime is a much stronger signal than mtime alone.

    With the CSV ABSENT there is no such question to ask: the sidecar is the only copy
    of the panel on this machine (the cluster ships the Parquet, not the 18x larger
    CSV), so it is the frame to read and the two staleness conditions have no operand.
    Answering False there would only send the caller to a CSV that does not exist.
    """
    import logging
    if not parquet_path.exists():
        return False
    if not panel_csv.exists():
        return True
    try:
        csv_stat = panel_csv.stat()
        if parquet_path.stat().st_mtime < csv_stat.st_mtime:
            return False
        recorded = _recorded_source_size(parquet_path)
        if recorded is not None and recorded != csv_stat.st_size:
            logging.warning(
                f"panel cache {parquet_path.name} was built from a {recorded:,}-byte CSV "
                f"but {panel_csv.name} is now {csv_stat.st_size:,} bytes -- rebuilding")
            return False
        return True
    except OSError:
        return False


def _recorded_source_size(parquet_path):
    """Size of the CSV the sidecar was built from, or None if it predates the stamp."""
    try:
        import pyarrow.parquet as pq
        md = pq.read_schema(parquet_path).metadata or {}
        raw = md.get(b'source_csv_bytes')
        return int(raw) if raw is not None else None
    except Exception:  # noqa: BLE001 - an unreadable stamp just means "cannot verify"
        return None


def refresh_panel_cache(panel_csv, df=None):
    """Rewrite a panel CSV's Parquet sidecar so it is never left stale on disk.

    `load_panel_cached` serves the sidecar whenever it looks current, and
    `estimation_1_sleep` / `estimation_2_sleep` / `step_entry_dynamics` read through it.
    A script that rewrites the panel CSV without calling this leaves a stale sidecar
    behind until some later read happens to notice and heal it -- so every writer of
    market_panel.csv calls this immediately after writing.

    Pass `df` when the frame is already in memory to avoid re-reading a large CSV.
    Failure is logged and swallowed: a missing sidecar only costs speed, since
    `load_panel_cached` falls back to the CSV.

    With no CSV on disk this is a no-op returning False. The stamp it writes is the
    size of the CSV the sidecar was built from, so without a CSV there is nothing to
    stamp -- and where that happens (the cluster, which is shipped the Parquet alone)
    the sidecar is the panel rather than a cache of one, so overwriting it from a
    caller's frame would replace the input instead of refreshing a copy of it.
    """
    import logging
    from pathlib import Path
    import pandas as pd

    panel_csv = Path(panel_csv)
    parquet_path = sidecar_path(panel_csv)
    if not panel_csv.exists():
        logging.info(f"panel cache refresh skipped: {panel_csv} is not on disk, so "
                     f"{parquet_path.name} is the panel itself, not a cache of it")
        return False
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        if df is None:
            df = pd.read_csv(panel_csv, low_memory=False, dtype={'mca_code': str})
        table = pa.Table.from_pandas(df, preserve_index=False)
        meta = dict(table.schema.metadata or {})
        meta[b'source_csv_bytes'] = str(panel_csv.stat().st_size).encode()
        meta[b'source_csv_name'] = panel_csv.name.encode()
        pq.write_table(table.replace_schema_metadata(meta), parquet_path)
        logging.info(f"panel cache refreshed: {parquet_path.name} "
                     f"({len(df):,} rows, from a {panel_csv.stat().st_size:,}-byte CSV)")
        return True
    except Exception as e:  # noqa: BLE001 - cache is an optimisation, never load-bearing
        logging.warning(f"could not refresh panel cache {parquet_path.name}: {e}")
        return False
