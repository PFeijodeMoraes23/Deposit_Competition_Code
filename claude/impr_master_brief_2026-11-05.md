# Master brief for the IMPR agent, 2026-11-05: everything pending, one command

**To:** the IMPR session (the main agent; I10). **From:** the main thread, for Pedro, written 2026-11-04 23:05 before his machine shut down. **Read first:** `SHARED_MEMORY.md` (entries 11-04 19:00 to 23:00), then the four handoffs named below for the detail of each item. You own the sequence; spawn one subagent per thread where the work is in that thread's files (Kappa: `diagnostics/diag_k2_*`; Pix: `diag_p_*`; tables: `make_*_table.py`, `make_paper_numbers.py`), each briefed from its handoff, each replying in its dated file. Nothing below edits `V_Friction.tex`; Pedro applies the paper blocks himself.

## Pedro decides first (ask him, do not assume)

```
decisions[3]{id,question,where the numbers are}:
  D1,"the imputed time-deposit rate tails: rule (a) rate/CDI outside 0.5 to 1.5 goes to the carry, or rule (b) a band in percentage points above the CDI (+3, +4 or +5 pp, and the band below)","IMPR reply (e) of 11-04, I15 section; the counts by band you prepare in I16"
  D2,"the Pix placebo: replace the single shuffle in diag_p_shifters.py by 99 or more, with the placebo median and the share at or above the real F in p3_numbers.json and in the paper key pix.stakes.fs.f_placebo","pix_handoff_2026-11-04.md, Update 22:58; Rout/p3_placebo_shuffles_0b9385b5.json"
  D3,"launch I2 (the cluster chain) on the panel that D1 produces","impr_handoff_2026-11-03.md, the restaged package and its seven launch lines"
```

## The order of work

```
tasks[12]{id,thread,what,after}:
  I16,IMPR,"read-only: for rule (b), counts of imputed B conglomerate-quarters and their type-4 weight by year at bands +3, +4, +5 pp above the CDI (and the symmetric band below), beside rule (a)'s 378; post them for D1","now"
  I17,IMPR,"apply the rule Pedro picks (D1) to imputed rates, rerun from calibrate as for I14, post rows changed by bank, type-4 weight by source, tails left, and the new panel md5 as the end line; restage the I2 package on it with sha256 and the launch lines","D1"
  K-resume,Kappa,"if the 11-04 job was killed by the shutdown: list which fits of steps 3 to 6 are recorded in Rout/k2_nls_numbers.json and resume from the first missing one with the same warm starts; else take the finish line; then the step 3 to 6 tables and the k2_report.md sections","now"
  K-key,Kappa,"write the money-weighted awake share (0.0177) to k2_numbers.json at pool_fate_kappa_base.awake_money_share_of_deposits with its _definition string (decision A, kappa_handoff Update 20:50)","K-resume"
  K7,Kappa,"step 7 on the 52ea6495 snapshot with --panel, DEMAND_CHARS_FILE, DEMAND_PREP_DIR at the interim frames, --out Rout\\k2_step7_52ea6495\\ (decision B); about 40 minutes plus the build","K-resume, machine quiet"
  P8,Pix,"rerun P3 to P5 on I17's md5 (diag_p_shifters.py --only p3; diag_p3_numbers.py; diag_p4_openings_advertising.py; about 7 minutes); if D2 is yes, make the shuffle change first and post the LF md5","I17 (and D2)"
  T7,tables,"switch collect_pix_stakes to Rout/p3_numbers.json['panels'][md5] (floats; no markdown parsing) and read step 7's k2_rc.csv from Rout\\k2_step7_52ea6495\\; register kappa.k2.world.awake_share from k2_numbers.json","K-key, K7, P8"
  T8,tables,"rerun make_k2b_pairs_table.py and make_paper_numbers.py after every file above changes; post the keys that changed and their old and new values (Pedro reads the paper's sentences against them)","T7"
  K1-spec,Kappa,"prepare the K1 run on I17's panel per the design note (kappa_design_2026-11-02.md, Sections 3.5 to 3.7 and 5, Group C): per-pair kappa-tilde with the savings pair as headline, total-pool weighting, control function for the spread, Pix before-after and dose, wild cluster bootstrap by conglomerate; write the spec and the inputs it needs as a plan for Pedro before running anything heavy","I17"
  I10,IMPR,"the productionization list, including load_demand's md5 check (mtime today) and the Bradesco 2016 observation; not before I17 and the I2 launch","D3"
  B-registry,IMPR,"on 11-13 the registry request 025399 (lambda, omega by bank) arrives: a new BB-type handoff then; nothing now","11-13"
  paper,Pedro,"apply the remaining blocks of claude/paper_edits_2026-11-04.tex (W5 content, W7, F1 to F6), compile with biber, run make_paper_equations.py; the tables agent's T8 tells him which quoted keys moved","after T8"
```

## Rules that carry over

- One panel md5 per run, posted as the end line; every output row carries it; the Kappa agent's step 7 and K1 inputs pin the snapshot `52ea6495` until K1 is restarted on the final panel.
- No script edit without a one-paragraph plan and Pedro's OK, except the ones already approved (listed in each handoff); LF md5 of every edited script in the reply.
- Light estimation locally; the cluster chain (I2) only on Pedro's launch lines.
- Reply files: `IMPR_REPLY_2026-11-05a.md` for your own work; subagents reply in their threads' dated files; you summarize their state in yours. Delete nothing in `claude/`; the main thread does.
