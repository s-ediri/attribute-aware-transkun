# Research protocol: attribute-aware structured decoding

## Working title

**Attribute-Aware Structured Decoding for Event-Based Piano Transcription:
Controlled Evaluation and Duration-Aware Offset Modeling**

## Scope

The project investigates whether note attributes should contribute directly to
semi-CRF event selection instead of being predicted only after an interval has
been decoded. Phase 1 isolates this effect from additional fine-tuning. A
duration-aware offset factor may be considered in Phase 2, but only after the
controlled comparison and the planned mechanism analysis have been completed.

## Primary research question

Does using trained velocity, sub-frame timing, and boundary-presence factors
inside semi-CRF decoding improve piano transcription over predicting the same
attributes without allowing them to affect interval selection?

## Confirmatory systems

| ID | System | Purpose |
|---|---|---|
| A | Packaged TransKun V2 checkpoint | Frozen reference |
| B | Original TransKun V2 architecture, equally fine-tuned | Controls for continued training |
| C | Structured heads with loss weight 1 and score weight 0 | Trained-attribute/post-hoc control |
| D | Structured heads with loss weight 1 and score weight 1 | Attribute-aware joint decoding |

The main comparison is **D minus C**. The **B minus A** comparison measures the
effect of continued fine-tuning. System B uses the historical architecture in
`transkun/ModelTransformerLegacy.py`, sourced from commit `03b23776`, and must
strictly load the packaged checkpoint.

## Fixed training conventions for the first comparison

- Dataset: MAESTRO v3 prepared at 44.1 kHz.
- Ground-truth sustain-pedal extension: disabled.
- Initialization: packaged `transkun/pretrained/2.0.pt` where compatible.
- Optimizer-step budget: identical for B, C, and D.
- Seeds: `20261108`, `20261109`, and `20261110` for final runs.
- Data order, augmentation policy, batch size, learning-rate schedule, and
  checkpoint selection rule must match across B, C, and D.
- Development runs must be labelled and cannot replace the final-seed runs.

## Data-use rules

- MAESTRO validation has already informed development and is used for model
  selection and exploratory analysis only.
- SMD has already been used for repeated comparisons. It is therefore treated
  as an exploratory external dataset rather than an independent confirmation
  set.
- Freeze the method, checkpoint rule, metrics, and strata before evaluating the
  MAESTRO test split. Evaluate that split once for the final analysis.
- Do not choose new weights or checkpoints from partial full-piece results.

## Primary outcomes

1. Macro per-piece note+offset F1.
2. Macro per-piece note-only F1 as a non-inferiority safeguard.
3. Macro per-piece note+velocity+offset F1.

Precision and recall are reported with each F1 result. Uncertainty is estimated
from paired recording-level differences using a recording bootstrap; individual
notes are not treated as independent observations.

## Mechanism outcomes

- Velocity: categorical accuracy, MIDI velocity MAE, tolerance accuracy, NLL,
  and calibration.
- Onset/offset timing: bin accuracy, adjacent-bin accuracy, signed error,
  absolute error, NLL, and calibration.
- Boundary presence: precision, recall, F1, average precision, and calibration.
- Compare factor confidence and correctness with interval additions, removals,
  and boundary changes between systems C and D.

## Pre-specified strata

- Reference duration: `<250 ms`, `250-500 ms`, `500 ms-1 s`, and `>=1 s`.
- Reference velocity: `<=40`, `41-80`, and `>=81`.
- Pitch: `<=48`, `49-72`, and `>=73`.
- Pedal active/inactive, local polyphony, repeated same-pitch notes, and piece.

Timing is scored only where the relevant boundary is present. Duration groups
require both onset and offset to be present, which avoids assigning misleading
durations to notes clipped by a window boundary. The legacy model uses a
continuous Bernoulli timing distribution, whereas the structured model uses
categorical bins. Their negative log-likelihood values are therefore reported
separately and are not compared directly.

## Decision rule for Phase 2

Proceed to duration-aware offset modeling only if either:

1. D improves note+offset F1 over C without a practically important reduction
   in note-only F1; or
2. mechanism analysis identifies a stable duration-conditioned failure mode
   attributable to the boundary-local offset factor.

The minimum worthwhile effect and non-inferiority margin must be agreed with the
supervisor before the final analysis. Until then, results should be described in
terms of effect size and uncertainty rather than pass or fail.

## Phase 2 candidate

Replace or augment the boundary-local end score with a candidate-interval term:

`S(i, j, p) = S_interval(i, j, p) + A_onset(i, p) + A_offset(j, p) + D(j-i, p)`

Only one duration-aware formulation may be selected after Phase 1. A broad
hyperparameter or architecture search is outside the scope of the project.

## Immediate milestones

1. Complete and validate the existing 137-piece full-recording robustness run.
2. Run matched short smoke tests for B, C, and D using identical declared seeds.
3. Verify checkpoint metadata and strict initialization before full runs.
4. Validate direct reference-interval attribute and calibration evaluation.
5. Freeze the final training/evaluation manifest before multi-seed runs.