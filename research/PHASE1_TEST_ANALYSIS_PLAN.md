# Approved Phase 1 decision thresholds and MAESTRO test analysis

## Status

**Approved and frozen on November 12, 2026.**

This plan originated as a proposal written after inspecting MAESTRO validation
results. It is
therefore a prospective freeze for the one-time test analysis, not a claim that
the numerical thresholds below were preregistered before validation.

## Approved thresholds

Use absolute macro-F1 differences, reported as **D minus C**.

1. **Minimum worthwhile note+offset improvement:** `+0.0020` F1.
2. **Note-only non-inferiority margin:** `-0.0010` F1.

In percentage-point language, these are a required `+0.20` point improvement
for the primary offset-aware outcome and a maximum tolerated `0.10` point loss
for the note-only safeguard.

### Rationale

- The note+offset threshold requires more than a merely positive numerical
  difference while remaining realistic for a tightly controlled comparison in
  which C and D share architecture, training budget, data order, and attribute
  losses.
- The note-only margin is deliberately narrower than the required offset-aware
  gain. It prevents a claimed offset benefit from being purchased with a
  similarly sized deterioration in basic note detection.
- The values are simple absolute effect thresholds and do not depend on the
  number of notes or recordings in the test split.
- These thresholds require supervisor approval because validation was already
  observed. They must not be adjusted after any test result is inspected.

## Frozen systems and checkpoints

Evaluate only the three matched final seed pairs below. Do not select a single
best validation seed and do not average model weights.

| Seed | C checkpoint SHA-256 | D checkpoint SHA-256 |
|---|---|---|
| `20261108` | `1a47a13fd5f91d043bc7b9b831b2f7b0b79172f825e990cba22e6ad17a9bcca0` | `8d0f4d50b095e92b68fc1cd1524b9fa78a5cae20e788114f03776326c552fa34` |
| `20261109` | `62f3359bef39e767bf5e0474eea6898a5c976aacc8f21ffc730ad6f66c108848` | `af43f72b920af1a026fc150a7806425241c76fbbbc41c2a18c8aab0873f18407` |
| `20261110` | `937046318dbce19257f1b2772de197a6328fb77fbba0d754cdc428f61a535e93` | `b4499b1b31623ec2b5141cd74de3986342ea77df88ac5aaf77b44e0971a101ce` |

Frozen configs:

- C: `experiments/configs/posthoc_attributes.conf`, SHA-256
  `a5a7d04ed249ba00348dc3b8bea7fd4c0cf78c3ac9749886fddb4fce1f7f448a`;
- D: `experiments/configs/joint_attributes.conf`, SHA-256
  `08a5952bc906738c3ca150c64b0778f4e8d7c46cd223f0ee63affcf49859daad`.

## Frozen outcomes

### Primary outcome

Macro per-piece **note+offset F1**.

### Safeguard

Macro per-piece **note-only F1**.

### Key supportive outcome

Macro per-piece **note+velocity+offset F1**.

Precision and recall must accompany every F1 result. Note+velocity F1 may be
reported as descriptive support but is not part of the success rule.

## Frozen aggregation

For each metric:

1. Score every complete test recording independently for C and D under each of
   the three matched seeds.
2. Compute the per-recording F1 difference, D minus C, separately for each seed.
3. Average the three seed-specific differences within each recording.
4. Take the unweighted mean of those recording-level averages as the primary
   aggregate effect. Every recording receives equal weight.
5. Compute C and D macro precision, recall, and F1 by first taking the per-seed
   macro over recordings and then averaging those three seed macros.

This treats the three seeds as repeated realizations of the fixed training
procedure while retaining the recording as the independent evaluation unit. In
addition to the aggregate result, report every seed separately as a sensitivity
analysis.

## Frozen uncertainty procedure

Use a paired recording-level percentile bootstrap:

- bootstrap unit: complete recording;
- replicates: `10,000`;
- random seed: `20261112`;
- resample the test recording indices with replacement;
- preserve all six C/D predictions for a selected recording;
- within each replicate, average D-C across seeds within recording and then
  average across sampled recordings;
- interval: 2.5th and 97.5th percentiles.

Do not bootstrap individual notes. Do not pool the three seed-by-recording rows
as independent observations.

## Approved decision rule

Classify Phase 1 as supporting progression only if **both** conditions hold on
the one-time MAESTRO test analysis:

1. **Primary benefit:**
   - aggregate note+offset D-C is at least `+0.0020`; and
   - the paired 95% bootstrap interval has a lower bound greater than `0`.
2. **Note-only safeguard:**
   - the lower bound of the paired 95% bootstrap interval for note-only D-C is
     greater than `-0.0010`.

The effect threshold addresses practical importance; the interval condition
addresses uncertainty. Passing only one of the two primary-benefit clauses is
not sufficient.

### Outcome labels

- **Progression supported:** both conditions above pass.
- **Offset improvement detected but below the worthwhile threshold:** the
  note+offset interval excludes zero, but D-C is below `+0.0020`, while the
  safeguard passes.
- **Inconclusive:** the safeguard passes, but the note+offset interval includes
  zero.
- **Safeguard failed:** the note-only interval lower bound is at or below
  `-0.0010`, regardless of the offset result.
- **Negative result:** the note+offset effect is non-positive and its interval
  does not support improvement.

## Multiplicity and interpretation

The progression decision is a gate requiring the primary note+offset criterion
and note-only safeguard simultaneously. No additional multiplicity correction
is applied to that gate. Note+velocity+offset, note+velocity, per-seed results,
and all strata are supportive or exploratory and must not override a failed
primary gate.

## Pre-specified strata

Retain the strata already listed in `research/PROTOCOL.md`:

- reference duration: `<250 ms`, `250-500 ms`, `500 ms-1 s`, `>=1 s`;
- reference velocity: `<=40`, `41-80`, `>=81`;
- pitch: `<=48`, `49-72`, `>=73`;
- pedal active/inactive;
- local polyphony;
- repeated same-pitch notes;
- recording.

Stratum analyses are explanatory and cannot change the primary decision. Any
stratum implementation that is not complete before test access must be omitted
from the one-time confirmatory analysis rather than improvised afterward.

## Test-access controls

Before the test run:

1. Record supervisor approval and approval date below.
2. Confirm test metadata SHA-256 without examining metric results.
3. Finalize and test the aggregation script using validation artifacts only.
4. Commit the approved document and analysis code with a clean worktree.
5. Run the test evaluation once for all three seed pairs.
6. Do not alter thresholds, checkpoints, metrics, bootstrap settings, or strata
   after test access.

If an infrastructure failure occurs, resume only under an identical hashed
protocol. Code changes after partial test access require documenting the defect,
showing that no scientific choice was informed by the partial result, and
restarting all affected runs under one corrected revision.

## Approval

- Approver: **Project owner/user; personal name not supplied**
- Decision: **Approved as recommended**
- Approval date: **November 12, 2026**
- Approved minimum worthwhile note+offset effect: **`+0.0020` macro F1**
- Approved note-only non-inferiority margin: **`-0.0010` macro F1**
- Notes: **Approval recorded before accessing MAESTRO test metrics.**
