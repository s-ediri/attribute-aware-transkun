# Research workspace

- `PROTOCOL.md` defines the current thesis question, controlled systems,
  outcomes, data-use rules, and progression criteria.
- `PHASE1_C_VS_D_VALIDATION.md` records the audited three-seed full-recording
  MAESTRO validation comparison and the frozen interpretation boundary before
  test-set analysis.
- `PHASE1_TEST_ANALYSIS_PLAN.md` freezes the approved practical-effect and
  non-inferiority thresholds plus the one-time MAESTRO test analysis.
- `../experiments/configs/` contains tracked model configurations.
- `../experiments/manifests/phase1.json` freezes the matched B/C/D training
  settings, seeds, dataset metadata hashes, and initialization checkpoint hash.
- `../experiments/manifests/phase1-test-analysis.json` freezes the approved
  one-time MAESTRO test inputs, hashes, outputs, and decision settings.
- `../scripts/` contains reproducible evaluation drivers.
- `../scripts/aggregate_phase1_c_vs_d.py` enforces the approved three-seed
  aggregation, bootstrap, and progression decision rule.
- `../scripts/run_phase1_c_vs_d_test.ps1` performs a fail-closed hash/Git
  preflight and launches the three frozen test runs plus aggregation.
- `../scripts/smoke_training_step.py` runs one forward/backward step without
  updating weights and should pass for B, C, and D before full training.
- `../scripts/optimizer_pilot.py` runs bounded real optimizer updates, saves a
  metadata-bearing checkpoint, and supports deterministic infrastructure resume.
- `../scripts/evaluate_reference_attributes.py` evaluates velocity, timing, and
  boundary-presence heads on reference intervals, independently of detection.
- `../checkpoint/` is intentionally local and ignored; it contains large model
  checkpoints, predictions, TensorBoard logs, and generated metric files.
- `../datasets/` is intentionally local and ignored.

Research conclusions should reference a tracked protocol/configuration plus
checkpoint and dataset hashes. Generated artifacts should not be committed
unless they are compact, final result summaries needed to support the thesis.