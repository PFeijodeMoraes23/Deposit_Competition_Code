# Switch-cost copies: source manifest

Each `switch_cost_*` file is a copy of the production file on the left, taken from the
working tree on 2026-10-13 (the stage C versions of the demand preps). The md5 identifies the
exact source. The originals were never edited. "Rewired" counts the lines whose references
to a copied module now point at its copy.

| Source | Copy | Source md5 | Rewired |
|---|---|---|---|
| `sleep_est_e2.py` | `switch_cost_sleep_est_e2.py` | `3a2b630b4573470316eb85ed6cad7bf6` | 4 |
| `sleep_est_single.py` | `switch_cost_sleep_est_single.py` | `5814aa975311e57a0bc94dcc5ccbfbae` | 12 |
| `sleep_demand_prep_link.py` | `switch_cost_sleep_demand_prep_link.py` | `ab6975a96d497813ad72161198e181d6` | 7 |
| `utils/sleep_links.py` | `utils/switch_cost_sleep_links.py` | `761d09a4eaabde699241ef66ea7abb68` | 5 |
| `cluster_lib.sh` | `switch_cost_cluster_lib.sh` | `8bcd02f43cf3337c040fe0db3ba807a2` | 346 |
| `sleep_job.sh` | `switch_cost_sleep_job.sh` | `823267f9962a010a8df54dbb131b959e` | 235 |
| `sleep_run.sh` | `switch_cost_sleep_run.sh` | `bab5ac543908edda6a152e5c122de75a` | 195 |

For the three shell copies (copied 2026-10-13 17:47), "Rewired" is the number of lines of the copy
that differ from the source (`diff` lines marked `>`, header included). The shell copies were
isolated and cut down to the cluster scope (E3/E4, spec 12, two fit variants, optional demand
arms), so they also drop source lines: 177 (library), 493 (job), 370 (run script). Every edit is
listed in `CLUSTER_SCRIPTS.md`.
| `utils/se_national.py` | `utils/switch_cost_se_national.py` | `150109ce33e298a1fd1fbcf6f865fb15` | 2 |

Removed on 2026-10-13 at 21:12, unused: the copies of `sleep_est_e1.py`, `sleep_demand_prep.py`,
`sleep_demand_prep_e1.py`, `sleep_demand_prep_e2.py`, `sleep_ident_augmented.py` and
`sleep_ident_interaction.py`.
