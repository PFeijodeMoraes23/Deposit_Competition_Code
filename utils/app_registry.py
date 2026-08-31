"""app_registry.py
# Author: Pedro Feijó de Moraes
#
# Which mobile app belongs to which firm, for the service-quality panel
# (scrape_app_ratings.py, scrape_app_ratings_wayback.py, panel_quality.py).
#
# Seeded from utils/firm_registry.FIRMS so the two registries name firms the same way:
# `firm_key` and `cnpj_root` are copied from there, and the conglomerate code is
# resolved the same way every other source is -- through
# panel_deposits.build_cnpj_conglomerate_map.
#
# Why store ids rather than search at scrape time
# -----------------------------------------------
# A store search for "Inter" returns dozens of apps and the ranking is not stable
# between runs, so resolving by search would silently repoint a firm's time series at a
# different app. The store ids below are the identity of the series; they are looked up
# ONCE (see `propose_candidates`) and then frozen.
#
# One firm, several apps
# ----------------------
# Banks ship more than one app and the split matters:
#   Caixa       CAIXA (full bank) vs Caixa Tem (the mass-market benefit account)
#   Itau        Itau (retail) vs iti (the payments-first app)
#   Mercado     Mercado Pago (wallet) vs Mercado Livre (the marketplace, NOT a bank app)
#   Nubank      one retail app; Nu's business app is separate
# `app_role` marks which app carries the firm's headline rating; every other app is
# kept as 'secondary' so the choice is visible and reversible rather than implicit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

PLAY = "play"
IOS = "ios"


@dataclass(frozen=True)
class AppEntry:
    firm_key: str            # matches utils.firm_registry.FIRMS[*]['firm_key']
    cnpj_root: str           # 8-digit CNPJ root of the lead institution
    label: str               # the app's store name, for logs and figures
    app_role: str            # 'primary' (headline series) or 'secondary'
    android_package: str | None = None   # e.g. 'com.nu.production'
    ios_app_id: str | None = None        # numeric App Store id, as a string
    reclameaqui_slug: str | None = None  # path segment in /empresa/<slug>/
    verified: bool = False   # True once the ids have been confirmed against the stores
    notes: str = ""


# Ids start empty on purpose: a guessed package name scrapes a real but WRONG app and
# nothing downstream can tell. Fill them from `propose_candidates` output (one-time),
# set verified=True, and they are frozen from then on.
APPS: list[AppEntry] = [
    AppEntry("nubank", "18236120", "Nubank", "primary"),
    AppEntry("inter", "00416968", "Inter", "primary"),
    AppEntry("c6", "31872495", "C6 Bank", "primary"),
    AppEntry("pagseguro", "08561701", "PagBank", "primary"),
    AppEntry("mercadopago", "10573521", "Mercado Pago", "primary"),
    AppEntry("picpay", "22896431", "PicPay", "primary"),
    AppEntry("stone", "16501555", "Stone", "primary"),
    AppEntry("xp", "33264668", "XP", "primary"),
    AppEntry("pan", "59285411", "Banco Pan", "primary"),
    AppEntry("bmg", "61186680", "Banco BMG", "primary"),
    AppEntry("bs2", "71027866", "Banco BS2", "primary"),
    AppEntry("sofisa", "60889128", "Sofisa Direto", "primary"),
    AppEntry("itau", "60701190", "Itau", "primary"),
    AppEntry("itau", "60701190", "iti", "secondary",
             notes="payments-first app; separate store listing from the retail bank app"),
    AppEntry("bradesco", "60746948", "Bradesco", "primary"),
    AppEntry("santander_br", "90400888", "Santander", "primary"),
    AppEntry("bb", "00000000", "Banco do Brasil", "primary"),
    AppEntry("caixa", "00360305", "CAIXA", "primary"),
    AppEntry("caixa", "00360305", "Caixa Tem", "secondary",
             notes="mass-market benefit account; far more installs than the main app"),
    AppEntry("banrisul", "92702067", "Banrisul", "primary"),
    AppEntry("brb", "00000208", "BRB", "primary"),
]


def validate(apps: list[AppEntry] | None = None) -> list[str]:
    """Return the registry's problems as a list of strings; empty means it is usable."""
    apps = APPS if apps is None else apps
    problems: list[str] = []

    for store, attr in ((PLAY, "android_package"), (IOS, "ios_app_id")):
        seen: dict[str, str] = {}
        for a in apps:
            val = getattr(a, attr)
            if val is None:
                continue
            if val in seen:
                problems.append(f"{store} id {val!r} is claimed by both "
                                f"{seen[val]} and {a.firm_key}/{a.label}")
            seen[val] = f"{a.firm_key}/{a.label}"

    by_firm: dict[str, list[AppEntry]] = {}
    for a in apps:
        by_firm.setdefault(a.firm_key, []).append(a)
    for firm, entries in by_firm.items():
        n_primary = sum(e.app_role == "primary" for e in entries)
        if n_primary != 1:
            problems.append(f"{firm} has {n_primary} primary apps (expected exactly 1)")
    for a in apps:
        if a.app_role not in ("primary", "secondary"):
            problems.append(f"{a.firm_key}/{a.label}: app_role {a.app_role!r} is neither "
                            f"'primary' nor 'secondary'")
    return problems


def resolved(store: str = PLAY, apps: list[AppEntry] | None = None) -> list[AppEntry]:
    """Entries carrying a verified id for `store` -- the ones a scraper can actually use."""
    apps = APPS if apps is None else apps
    attr = "android_package" if store == PLAY else "ios_app_id"
    return [a for a in apps if getattr(a, attr) and a.verified]


def unresolved(store: str = PLAY, apps: list[AppEntry] | None = None) -> list[AppEntry]:
    apps = APPS if apps is None else apps
    attr = "android_package" if store == PLAY else "ios_app_id"
    return [a for a in apps if not getattr(a, attr) or not a.verified]


def propose_candidates(entry: AppEntry, n: int = 5) -> list[dict]:
    """
    Search Google Play for `entry.label` and return the top candidates, so the ids above
    can be filled in from data rather than from memory.

    Returns [] when google-play-scraper is not installed -- this is a curation aid, not
    something any pipeline step depends on.
    """
    try:
        from google_play_scraper import search
    except ImportError:
        log.warning("google-play-scraper not installed; cannot propose candidates "
                    "(pip install google-play-scraper)")
        return []
    try:
        hits = search(entry.label, lang="pt", country="br", n_hits=n)
    except Exception as exc:  # noqa: BLE001 - a failed search must not break curation
        log.warning("Play search failed for %s: %s", entry.label, exc)
        return []
    return [{"appId": h.get("appId"), "title": h.get("title"),
             "developer": h.get("developer"), "score": h.get("score")} for h in hits]


def conglomerate_map() -> dict[str, str]:
    """
    {firm_key: CodConglomeradoPrudencial}, resolved through the same canonical map the
    market panel and the fee sources use, so the quality panel joins their key space.
    Firms the IF Data List files do not place in a conglomerate get the panel's
    'CNPJ_<int>' fallback (see panel_fee_merge.cnpj_fallback_key).
    """
    from panel_deposits import build_cnpj_conglomerate_map

    cmap = build_cnpj_conglomerate_map()
    out: dict[str, str] = {}
    for a in APPS:
        hit = cmap.get(int(a.cnpj_root))
        out[a.firm_key] = (str(hit[0]) if hit and hit[0] is not None
                           else f"CNPJ_{int(a.cnpj_root)}")
    return out


def _cli() -> int:
    """Report the registry's state and, where possible, propose ids for what is missing."""
    problems = validate()
    print(f"registry: {len(APPS)} app entries, "
          f"{len({a.firm_key for a in APPS})} firms")
    print(f"  with a verified Play id: {len(resolved(PLAY))}")
    print(f"  with a verified iOS id : {len(resolved(IOS))}")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
    else:
        print("\nvalidate(): no problems")

    todo = unresolved(PLAY)
    if not todo:
        return 1 if problems else 0
    try:
        import google_play_scraper  # noqa: F401
    except ImportError:
        print(f"\n{len(todo)} entries still need a Play id. Install google-play-scraper "
              f"to have this command propose store candidates for each of them.")
        return 1 if problems else 0

    print(f"\n{len(todo)} entries still need a Play id. Candidates from the store:")
    for a in todo:
        print(f"\n  {a.firm_key}/{a.label} ({a.app_role})")
        for c in propose_candidates(a):
            print(f"     {str(c['appId']):<38s} {str(c['title'])[:30]:<32s} "
                  f"{str(c['developer'])[:24]:<26s} {c['score']}")
    return 1 if problems else 0


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    sys.exit(_cli())
