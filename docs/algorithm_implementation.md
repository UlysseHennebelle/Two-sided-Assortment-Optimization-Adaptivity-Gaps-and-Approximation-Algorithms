# Algorithm implementation details

This document describes the corrected executable algorithms. The paper is the
mathematical specification; the legacy notebook is used to recover the
computational methods that actually generated the accepted paper's results.

## Common MNL oracle

`src/tsao/oracle.py` implements the optimization used throughout the notebook.
Each alternative has an MNL weight and a continuation revenue. Alternatives are
sorted by decreasing revenue and included while the next revenue is at least the
current expected value. The update is

\[
R' = \frac{(v_0+W)R+w r}{v_0+W+w}.
\]

This solves the unconstrained single-agent MNL assortment problem without
enumerating subsets. It is used in Algorithm 1 and in both optimized dynamic
programs.

## ALG(FS): Fully Static

File: `src/tsao/algorithms/fully_static.py`.

The empirical algorithm follows notebook cell 8, including the author-confirmed
choices that differ from the proof implementation.

1. Normalize each agent's MNL weights by its outside-option weight. This is
   exactly equivalent to the notebook's matrix scaling in Figure 4.
2. Use the confirmed empirical threshold
   \(\alpha=(\sqrt 5-1)/2\), not the approximately 0.7574 value that optimizes
   the displayed theoretical bound in Section 5.3.
3. Partition edges into:
   - low/low: \(v_{ij}<\alpha\) and \(w_{ji}<\alpha\);
   - customer-high: \(v_{ij}\ge\alpha\);
   - supplier-high remainder: \(v_{ij}<\alpha\) and \(w_{ji}\ge\alpha\).
4. For low/low edges, solve the factored form of the notebook LP, independently
   round every edge with its LP value, evaluate Problem (11), and report the
   Monte Carlo mean over the configured number of roundings. The best realized
   edge set is retained only as diagnostic metadata; it is not substituted for
   the mean.
5. For customer-high edges, process customers in index order. Assign each
   customer to the available supplier with largest marginal increase
   \[
   \frac{B_j+w_{ji}}{v_{0j}+B_j+w_{ji}}
   -\frac{B_j}{v_{0j}+B_j}.
   \]
   This is the legacy greedy computational substitute for the continuous-greedy
   method used in the proof of Lemma 5.4.
6. For supplier-high edges, transpose the subinstance once and run the same
   greedy method.
7. Report the largest of the three candidate values.

Corrections: the second, accidental transpose of the supplier-high subinstance
was removed. Customers with no positive available edge are skipped; the legacy
routine incorrectly assigned them to supplier zero and could dilute later
matches.

Experiment use: Table 1, the FS terms in Tables 2-3, and Figures 3-4.

## Algorithm 1: Arbitrary Order Greedy

File: `src/tsao/algorithms/arbitrary_order_greedy.py`.

For one initiating side:

1. Draw an arbitrary order as a random permutation, matching the experiments.
2. Maintain only the total responding-side MNL backlog weight \(B_j\).
3. For the next initiator \(i\), give each responder \(j\) the marginal-demand
   continuation revenue shown above and weight \(v_{ij}\).
4. Call the revenue-ordered MNL oracle to select the assortment.
5. Sample and observe the initiating agent's choice.
6. If responder \(j\) is selected, add \(w_{ji}\) to \(B_j\).
7. After all initiators are processed, return
   \(\sum_j B_j/(v_{0j}+B_j)\), the exact conditional expected number of final
   matches from the responding side.

The supplier-initiated version uses an immutable transposed instance.

## ALG(OS): One-sided Static

File: `src/tsao/algorithms/one_sided_static.py`.

This implements the Section 7 simulated-greedy benchmark.

1. Simulate Algorithm 1 to construct a full family of menus. Simulated choices
   update marginal-demand weights used for later menus, but the menus themselves
   are retained as the static policy.
2. Evaluate those fixed menus in independent simulations. No observed choice is
   allowed to change a menu during evaluation.
3. Repeat menu construction and evaluation with the configured counts.
4. Repeat from the opposite initiating side and report the larger mean.

This separation makes the paper's distinction between "simulate" for OS and
"observe" for OA explicit. The production configuration preserves 50 menu
constructions per side and 50 evaluations per construction.

Experiment use: Tables 2-3 and Figures 3-4.

## ALG(OA): One-sided Adaptive

File: `src/tsao/algorithms/one_sided_adaptive.py`.

Run Algorithm 1 independently 50 times with customers initiating and 50 times
with suppliers initiating. Report the larger sample mean and its side. All
individual values and child seeds are retained in Parquet. This is the confirmed
empirical implementation of Algorithm 2; the paper sentence should say 100
total runs, 50 per side.

Experiment use: Tables 2-3 and Figures 3-4.

## ALG(FA)

File: `src/tsao/algorithms/fully_adaptive.py`.

Two names are deliberately available:

- `fully_adaptive_theorem_algorithm` implements Theorem 4.3: choose the
  initiating side with a fair coin and run Algorithm 1.
- `fully_adaptive_experiment_algorithm` implements the author-confirmed
  numerical convention: reuse the already computed `ALG(OA)` value. It does not
  rerun a stochastic procedure and records zero additional runtime.

The revised paper must keep these theoretical and empirical conventions
distinct. Appendix G plots only ALG(OA).

## OPT(FS)

File: `src/tsao/benchmarks/opt_fully_static.py`.

Problem (38) is implemented in Gurobi with continuous `y` and `z`, packing
constraints, a bilinear objective, and `NonConvex=2`. The result object stores
the incumbent, best bound, status, runtime, and relative gap separately. The
legacy function returned only `ObjBoundC`; the reporting layer uses the bound
only when explicitly constructing the conservative historical denominator.
Tiny exact validation uses reciprocal-edge enumeration.

Experiment use: Table 1 and the first Table 2 ratio.

## OPT(OS)

File: `src/tsao/benchmarks/opt_one_sided_static.py`.

For each initiating orientation, enumerate all static menu families. For a menu
family, enumerate independent initiating choices, compute the probability of
each backlog realization, and evaluate the responding-side MNL demand exactly.
Return the better orientation. It is intentionally limited to the small sizes
for which Table 2 reports OPT(OS).

## Optimized OPT(OA) dynamic program

Files: `src/tsao/benchmarks/dynamic_programs.py` and `src/tsao/state.py`.

The implementation reproduces the optimized `solve_one_sided_adaptive_*_mnl`
logic rather than the earlier powerset recursion.

State:

- a bit mask of unprocessed initiators;
- the responding agents, which remain unprocessed;
- for each processed initiator, only a pending choice toward an unprocessed
  responder.

Completed/dead choices are discarded. At a state, every possible next
initiator is considered, preserving adaptive order optimization.

For candidate initiator \(i\):

1. Evaluate the continuation \(r_0\) if the outside option is selected.
2. For every responder \(j\), evaluate the continuation \(r_j\) if \(j\) is
   selected.
3. Create an MNL option with revenue \(r_j-r_0\) and weight \(v_{ij}\).
4. Solve the optimal assortment with the revenue-ordered oracle.
5. The candidate value is \(r_0\) plus the oracle value.

When no initiators remain, return the sum of responding-side backlog demands.
Memoization uses the canonical immutable state directly; it avoids the legacy
SHA-256 string construction and the mutate/undo operations. Both initiating
orientations are solved and the larger value is OPT(OA).

## Optimized OPT(FA) dynamic program

File: `src/tsao/benchmarks/dynamic_programs.py`.

The state contains:

- bit masks of unprocessed customers and suppliers;
- pending customer choices toward unprocessed suppliers;
- pending supplier choices toward unprocessed customers.

It deliberately excludes completed matches. A matching choice produces an
immediate transition reward of one, and cached values always represent future
matches. This is the explicit equivalent of the legacy cache's
`best - len(matches)` optimization.

For every possible next customer or supplier:

1. Evaluate the outside-option continuation once.
2. Aggregate all processed backlog alternatives into one pseudo-option. Every
   such alternative has continuation increment one, so their MNL weights can be
   summed exactly.
3. Evaluate one continuation for each unprocessed opposite-side agent.
4. Use continuation differences as MNL revenues and solve the assortment with
   the revenue-ordered oracle.
5. Maximize over the identity and side of the next processed agent.

When either side is exhausted, remaining agents cannot create new pending
choices. Their optimal expected matches are the closed-form sum of backlog MNL
demands. This terminal shortcut is retained from the legacy optimized DP.

The new implementation matches the optimized notebook OA and FA values to
machine precision on direct 2-by-2 comparisons.

## UB(OA)

File: `src/tsao/benchmarks/ub_one_sided_adaptive.py`.

Solve concave Problem (39) in CVXPY for both initiating orientations and return
the maximum. The input is never mutated. This fixes the missing parentheses on
the notebook's final `symmetric` call, which left subsequent algorithms with
reversed sides.

## UB(FA)

File: `src/tsao/benchmarks/ub_fully_adaptive.py`.

Solve linear Problem (40) directly using one match-probability matrix. The
notebook used auxiliary `x`, `y`, and `z` variables; the direct paper
formulation is smaller and equivalent.

## Cardinality-constrained validation

File: `src/tsao/algorithms/cardinality.py`.

The current paper experiments are unconstrained. This module provides exact
tiny-instance edge enumeration and capacity checks for Section 6. The
constrained MNL oracle in `oracle.py` also uses explicit enumeration and is
documented as a validation implementation, not a scalable hidden substitute
for Assumption 6.1.
