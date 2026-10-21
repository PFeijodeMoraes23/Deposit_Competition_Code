# Handoff: open scraping tasks (merged, 2026-10-21)

**To:** scraping agent. **From:** main thread, for Pedro.

**Replaces:**
- `breakfast_followups_scraping_handoff_2026-10-21.md` (S6 to S10)
- your reply `breakfast_followups_scraping_reply_2026-10-21.md`

Both rounds (S1 to S5 and S6 to S10) are delivered. Their status files stay the record:
- `DATA_RAW/devo_followups_2026-10-20/STATUS.md`
- `DATA_RAW/breakfast_followups_2026-10-21/STATUS.md`

**Rule:** V_Full.tex, V_Friction.tex and the decks are edited by the author only. Deliver raw files, documentation and source-log rows, with no analysis.

## Rules (unchanged)

- Official sources first. No guessed endpoints.
- Raw files untouched, one folder per task, each with `NOTES.md` and its source-log rows.
- If a source does not exist or is not public, say so.
- **Bot walls and terms of use are respected.**
  - Tesouro Direto is collected by hand, by Pedro.
  - B3 forbids automated collection.
  - Latinobarómetro files stay internal.

## Delivered

```
delivered[6]{task,where}:
  S1-S5,DATA_RAW/devo_followups_2026-10-20/
  S6-S10,DATA_RAW/breakfast_followups_2026-10-21/
  ESTBAN per branch,DATA_RAW/estban_agencia/ (2013-01..2026-06)
  national series,DATA_RAW/national_series/ (Ibovespa; Selic SGS 11/4390/4189)
  scrapers,"scrape_national_series.py, scrape_estban_branches.py, scrape_bcb_ccs_relationships.py, scrape_consumidor_gov.py"
  Tesouro Direto,S8_outside_options/tesouro/live_manual/ (Pedro's manual download)
```

## Tasks

```
tasks[2]{id,agent,target,when}:
  P1,scraping,"Pix by municipality: re-download 2026-10 once the month closes",early November 2026
  P2,scraping,"re-run scrape_national_series.py for final monthly values",after each month end
```

**P1.** 2026-10 is partial in the current files, and the last full month is 2026-09. Replace only that month's file, and log the new capture.

**P2.** The current month is partial in the monthly series. Re-run after the month closes, and log it.

## On hold: Pedro decides

None of these runs until Pedro says yes.

```
hold[4]{id,question}:
  H1,"S6: are dated rules enough, or is a monthly series per firm needed? Chase Nubank RDB rates by maturity, Inter CDB (JS-rendered, needs a browser) and the big banks' digital accounts?"
  H2,"S7: chase Itau's own regulamento and the June 2021 BCB norm that ended automatic fund sweeps? Draft a LAI request for BB Rende Facil terms?"
  H3,"S9: fill the WVS download form, and download Edelman through a browser? Clarify whether the Findex 2024 trust value (5.672) is a share of all adults or of the unbanked."
  H4,"From 10-20: a Findex World Bank account; IBGE codes and full history for consumidor.gov.br; SGS series 25105-25119; monthly snapshots of the branch and correspondent registers."
```

## Waiting on outside answers (track only)

```
pending[6]{request,protocol,subject}:
  B3,B3PF0031927,"individual investors by state x month or quarter from 2016, municipality if available, by product, with custody value"
  LAI,18810.025400/2026-04,"CCS relationships by institution x municipality (still open)"
  LAI appeal,18810.025656/2026-11,appeal filed
  Ouvidoria,18810026133202684,request about the open LAI 18810.025400/2026-04
  LAI,not recorded,"CDB registry, tariffs, CDB by investor state (per the 10-21 reply)"
  LAI,NUP not recorded,Pix request
```

When an answer arrives, save it under the matching task folder and add a source-log row.

## Open questions for Pedro

1. Record the missing protocol numbers (the Pix LAI, and the CDB registry, tariff and CDB-by-state requests) so they can be tracked here.

## Response from the scraping agent (2026-10-21 17:50)

- **P1 and P2 are DROPPED, per Pedro (2026-10-21):** "No need for 2026, recall I need only until 2024." The analysis window ends 2024Q4. October 2026 and later months come along with downloads but are never re-run or gap-filled. Nothing is scheduled.
- **Three of the "not recorded" protocols are recorded** (the 10-21 reply omitted them by mistake):

  ```
  pending_resolved[3]{request,protocol}:
    LAI CDB/RDB registry (B3/Cetip) by institution,18810.025399/2026-18
    LAI tariff history per institution,18810.025439/2026-13
    LAI CDB/RDB by issuer x investor state + ESTBAN verbete 907,18810.025440/2026-48
  ```

  Only the Pix LAI NUP is still unrecorded, so question 1 reduces to that one number.
- **Recorded:** the LAI appeal 18810.025656/2026-11 and the Ouvidoria request 18810026133202684 (on LAI 18810.025400/2026-04).
- Holds H1–H4 are untouched until Pedro says yes.
