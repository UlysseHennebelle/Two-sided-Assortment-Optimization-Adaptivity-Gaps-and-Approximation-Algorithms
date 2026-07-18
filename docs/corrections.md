# Correction ledger

Every correction is public, tested where executable, and linked to regenerated
outputs. Status `implemented` means the corrected source is present; the full
paper campaign has not yet been run.

| ID | Status | Correction and choice | Tests / affected outputs |
|---|---|---|---|
| C001 | implemented | `ALG(FS)` supplier-high edges are transposed once. The notebook transposed the temporary graph twice before evaluation. | `test_fully_static_supplier_high_case_uses_single_pure_transposition`; Tables 1-3, Figures 3-4 |
| C002 | implemented | `UB(OA)` constructs and solves both immutable orientations. The notebook omitted parentheses on the final `b.symmetric()` and left the input reversed. | `test_ub_oa_does_not_mutate_or_leave_instance_transposed`; Tables 2-4 |
| C003 | implemented | High-value FS greedy skips an initiator with no positive available edge. The notebook assigned supplier zero and could dilute later matches. | `test_high_value_greedy_skips_customer_with_no_available_edge`; all ALG(FS) outputs |
| C004 | implemented | Solver results separate incumbent, best bound, and gap. The notebook's `OPT_FS` wrapper returned only `ObjBoundC`. | Appendix F tests; Tables 1-2 |
| C005 | implemented | Adaptive state transformations are immutable; completed matches and dead choices are discarded canonically rather than through mutation, undo, and lossy string hashing. | DP parity/state tests; OPT(OA), OPT(FA) |
| C006 | implemented | Appendix G uses nonzero probability `sqrt(2)/2`, equivalently zero probability `1-sqrt(2)/2`, so a pair is mutually acceptable with probability 1/2. The current PDF reverses the zero probability. | generator correction test; Figures 3-4 |
| C007 | implemented, paper text pending | Appendix G retains the confirmed historical distribution orientation: customer `v` is sparse exponential and supplier `w` is sparse uniform. The current PDF states the reverse. | generator correction test; Figures 3-4 |
| C008 | implemented, paper text pending | OA uses 50 simulations per initiating side. The revised text should say 100 total, 50 per side, rather than 100 per side. | bounded algorithm tests; Tables 2-3, Figures 3-4 |
| C009 | implemented, paper text pending | The empirical FS threshold remains the confirmed `(sqrt(5)-1)/2`; it is not the approximately 0.7574 optimizer of the theoretical displayed bound. | alpha regression test; all ALG(FS) outputs |
| C010 | implemented, paper text pending | The empirical high-value FS method remains the confirmed marginal-demand greedy substitute. The paper must not call it continuous greedy without qualification. | algorithm documentation; all ALG(FS) outputs |
| C011 | implemented, paper text pending | Experimental `ALG(FA)` reuses `ALG(OA)`, as confirmed. The Theorem 4.3 fair-coin policy is implemented separately. | algorithm test; Table 3 |
| C012 | implemented | Outside-option scenarios store base matrices once and vary the outside weight. Per-agent normalization is mathematically equivalent to the notebook's repeated weight scaling. | model/algorithm tests; Figure 4 |
| C013 | chosen for new campaign | Use the paper's `100 + floor(100/q^2)` sample rule. The notebook used Python `round`, which explains the historical 5,775 rows. | Appendix G generation; Figure 3 |
| C014 | chosen for new campaign | Generate sizes 20, 50, 100, 200, 500 and configure the paper figure to display 50, 100, 200, 500. The current prose lists 20 instead of 100 while the figure displays 100. | Figure 3-4 configuration |
| C015 | chosen for new campaign | Reuse the same algorithm seed for every outside-option value of a base instance. This creates common random numbers across each Figure 4 curve; the legacy global RNG continued between scales. | Figure 4 |

## Known data boundary

The historical large matrices truncated by Excel cannot be reconstructed.
Their scalar outputs remain historical evidence, but the revised paper uses a
new seeded campaign whose complete matrices and results are stored in Parquet.
