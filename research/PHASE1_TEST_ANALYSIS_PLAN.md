# Phase 1 corrected-rerun analysis plan

## Status

**Pending corrected training and validation as of November 21, 2026.**

The earlier Phase 1 validation and MAESTRO test results were withdrawn because
the B/C/D systems were initialized from the wrong baseline checkpoint. Their
checkpoints, generated metrics, frozen hashes, and result reports are not valid
evidence and must not be cited.

The corrected baseline is:

- checkpoint: `transkun/pretrained/Transkun V2.pt`;
- checkpoint SHA-256:
  `76469be487911e110e0e661fadb583645e89c35364faa1ef8897d5600ba6ce20`;
- config: `transkun/pretrained/Transkun V2.conf`;
- config SHA-256:
  `b5f55fdd6fcca19871510bc68151eb1d09ea5e4db59aeeb7cdc59f5977ff950c`.

## Corrected-run controls

1. Train B, C, and D again for seeds `20261108`, `20261109`, and `20261110`.
2. Use the matched budget and data conventions in
   `experiments/manifests/phase1.json`.
   Ground-truth note offsets include sustain-pedal extension.
3. Verify that every checkpoint records the corrected baseline path and SHA-256.
4. Complete the full-recording validation comparison before freezing any test
   checkpoint hashes.
5. Do not run `scripts/run_phase1_c_vs_d_test.ps1` until
   `experiments/manifests/phase1-test-analysis.json` is repopulated and frozen.

## Outcomes and decision settings

To avoid adapting the analysis to previously inspected MAESTRO test results, the
corrected rerun retains the earlier outcomes and numerical decision settings:

- primary: macro per-piece note+offset F1, D minus C;
- safeguard: macro per-piece note-only F1, D minus C;
- supportive: macro per-piece note+velocity+offset F1;
- minimum worthwhile note+offset difference: `+0.0020`;
- note-only non-inferiority margin: `-0.0010`;
- paired recording bootstrap: 10,000 replicates;
- aggregate seed differences within recording before resampling recordings.

Precision and recall accompany every F1 result. Individual notes are not treated
as independent bootstrap units.

## Test-split disclosure

The MAESTRO test split was already evaluated during the withdrawn run. Therefore,
the corrected test rerun is a disclosed protocol-correction analysis, not
pristine first access. No threshold, outcome, checkpoint-selection rule, metric,
or bootstrap setting may be changed in response to the withdrawn test values.

## Freeze requirements

After corrected validation is complete and before the corrected test rerun:

1. record all six corrected C/D checkpoint paths and SHA-256 values;
2. record corrected C/D config hashes;
3. record evaluator, aggregator, and launcher hashes;
4. verify test metadata identity without examining new metrics;
5. set the test manifest status to
   `approved_and_frozen_before_test_metrics` in a clean committed revision;
6. execute all three corrected test seed pairs under that revision.