# Attribute-Aware Structured Decoding Research Protocol

## Working title

**Attribute-Aware Structured Decoding for Event-Based Piano Transcription:
Controlled Evaluation and Duration-Aware Offset Modeling**

## Scope

This project tests whether note attributes should influence semi-CRF event
selection, rather than only being read out after an interval has been decoded.
The first phase isolates this effect from additional fine-tuning. The second
phase may introduce a duration-aware offset factor only after the controlled
comparison is complete.

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

The key method contrast is **D minus C**. The contrast **B minus A** measures
continued fine-tuning. System B requires a maintained legacy-model training
entry point and must be implemented before confirmatory training begins.

## Fixed training conventions for the first comparison

- Dataset: MAESTRO v3 prepared at 44.1 kHz.
- Ground-truth sustain-pedal extension: disabled.
- Initialization: packaged `transkun/pretrained/2.0.pt` where compatible.
- Optimizer-step budget: identical for B, C, and D.
- Seeds: `20261108`, `20261109`, and `20261110` for final runs.
- Data order, augmentation policy, batch size, learning-rate schedule, and
  checkpoint selection rule must match across B, C, and D.
- Development runs must be labeled and must not be substituted for final seeds.

## Data-use rules

- MAESTRO validation has already informed development and is used for model
  selection and exploratory analysis only.
- SMD has already informed repeated comparisons and is an exploratory external
  robustness set, not independent confirmation.
- Freeze the method, checkpoint rule, metrics, and strata before evaluating the
  MAESTRO test split. Evaluate that split once for the final analysis.
- Do not choose new weights or checkpoints from partial full-piece results.

## Primary outcomes

1. Macro per-piece note+offset F1.
2. Macro per-piece note-only F1 as a non-inferiority safeguard.
3. Macro per-piece note+velocity+offset F1.

Report precision and recall alongside every F1 result. Use paired piece-level
differences and a piece-cluster bootstrap; individual notes are not independent
statistical units.

## Mechanism outcomes

- Velocity: categorical accuracy, MIDI velocity MAE, tolerance accuracy, NLL,
  and calibration.
- Onset/offset timing: bin accuracy, adjacent-bin accuracy, signed error,
  absolute error, NLL, and calibration.
- Boundary presence: precision, recall, F1, average precision, and calibration.
- Relate factor confidence and correctness to interval additions, removals, and
  boundary changes between systems C and D.

## Pre-specified strata

- Reference duration: `<250 ms`, `250-500 ms`, `500 ms-1 s`, and `>=1 s`.
- Reference velocity: `<=40`, `41-80`, and `>=81`.
- Pitch: `<=48`, `49-72`, and `>=73`.
- Pedal active/inactive, local polyphony, repeated same-pitch notes, and piece.

## Decision rule for Phase 2

Proceed to duration-aware offset modeling only if either:

1. D improves note+offset F1 over C without a practically important reduction
   in note-only F1; or
2. mechanism analysis identifies a stable duration-conditioned failure mode
   attributable to the boundary-local offset factor.

The minimum worthwhile effect and non-inferiority margin must be agreed with
the supervisor before final training. Until then, report effect sizes and
uncertainty without classifying a result as successful.

## Phase 2 candidate

Replace or augment the boundary-local end score with a candidate-interval term:

`S(i, j, p) = S_interval(i, j, p) + A_onset(i, p) + A_offset(j, p) + D(j-i, p)`

Only one duration-aware formulation should be selected after Phase 1. A broad
hyperparameter or architecture sweep is outside the planned scope.

## Immediate milestones

1. Complete and validate the existing 137-piece full-recording robustness run.
2. Implement and test the equally fine-tunable legacy TransKun control (B).
3. Run short smoke tests for C and D using identical declared seeds.
4. Add direct attribute accuracy and calibration evaluation.
5. Freeze the final training/evaluation manifest before multi-seed runs.