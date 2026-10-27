# DC1: where C0080903's 2022Q3 type-5 balance went

**Answer.** No balance leaves the panel. In 2022Q3, PicPay's payment institution (22896431) lost R$7.66 bn of prepaid balances, and PicPay Bank (09516419) gained R$8.16 bn of time deposits (CDB). IF-Data puts 09516419 in C0080903 that quarter, but the panel's ESTBAN tier maps it to C0088022 in every quarter, so the CDBs appear as C0088022 `dep_a4` = 8.158.

**Evidence** (R$ bn; IF-Data report 3, `shared/IF_DATA/*/IF_DATA_Values_YYYY_M.csv`; ESTBAN `V432`):

| | 22Q2 | 22Q3 | 22Q4 | 23Q1 |
|---|---|---|---|---|
| 22896431 prepaid (110560) | 8.377 | 0.721 | 0.847 | 0.543 |
| 09516419 time deposits (78286 = V432) | 0 | 8.158 | 9.187 | 9.097 |
| 92894922 Original time deposits | 10.214 | 10.540 | 11.818 | 10.810 |
| C0080903 consolidated time deposits | 11.817 | 20.352 | 22.618 | 21.466 |

**Construction.**
- `dep_a5` is IF-Data type-1 (consolidated) account 110560 (`panel_deposits.py:82,316-352`), spread over municipalities by `dep_a1` (`:481-529`).
- ESTBAN CNPJs map through the time-invariant `build_cnpj_conglomerate_map`, where the last List snapshot wins (`utils/conglomerates.py:263-342`). An unmapped CNPJ becomes `CNPJ_<cnpj>` (`panel_deposits.py:222-238`). There is no quarter-specific fallback.
- C0080903's IF-Data row is dropped by the tier rule (`:566`), so nothing is counted twice.

**Panel.** Both codes come from ESTBAN, have `is_B`=0, and sit in the NATIONAL market.

**D3.** From 2020Q4 to 2024Q3, the payment institution accounts for at least 99% of C0080903's `dep_a5`, so PicPay's rates remain the right price after the drop. Points I could not settle:
- 2020Q2–Q3: the payment institution did not yet file on its own.
- It is unclear whether the 102% CDI rule from 2022-08 on prices the leftover wallet balance or the CDBs now in C0088022.
- Downstream, the move reads as C0080903 leaving type 5 and C0088022 entering type 4.
