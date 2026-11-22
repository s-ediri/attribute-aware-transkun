# Research files

This directory contains the protocol and reports for the thesis experiments.

- `PROTOCOL.md` sets out the research question, systems, outcomes, data-use
  rules, and conditions for moving to Phase 2.
- The earlier Phase 1 validation and test reports were removed on November 21,
  2026 after discovering that training used the wrong baseline checkpoint.
- `PHASE1_TEST_ANALYSIS_PLAN.md` records the corrected-rerun controls and the
  disclosure required because the test split was accessed during the withdrawn
  run.
- `../experiments/configs/` contains tracked model configurations.
- `../experiments/manifests/phase1.json` freezes the matched B/C/D training
  settings, seeds, dataset metadata hashes, and initialization checkpoint hash.
- `../experiments/manifests/phase1-test-analysis.json` remains pending until the
  corrected C/D checkpoints have completed validation and can be frozen.
- `../scripts/` contains reproducible evaluation drivers.
- `../scripts/aggregate_phase1_c_vs_d.py` applies the approved three-seed
  aggregation, bootstrap, and decision rule.
- `../scripts/run_phase1_c_vs_d_test.ps1` checks the Git state and file hashes
  before launching the three test runs and their aggregation.
- `../scripts/smoke_training_step.py` runs one forward/backward step without
  updating weights and should pass for B, C, and D before full training.
- `../scripts/optimizer_pilot.py` runs bounded real optimizer updates, saves a
  metadata-bearing checkpoint, and supports deterministic infrastructure resume.
- `../scripts/evaluate_reference_attributes.py` evaluates velocity, timing, and
  boundary-presence heads on reference intervals, independently of detection.
- `../checkpoint/` is kept local and ignored by Git. It contains model
  checkpoints, predictions, TensorBoard logs, and generated metrics.
- `../datasets/` is also kept local and ignored by Git.

Research conclusions should cite the relevant tracked protocol or configuration
and the checkpoint and dataset hashes. Generated files should only be committed
when they are small final summaries needed for the thesis record.