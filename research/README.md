# Research workspace

- `PROTOCOL.md` defines the current thesis question, controlled systems,
  outcomes, data-use rules, and progression criteria.
- `../experiments/configs/` contains tracked model configurations.
- `../experiments/manifests/phase1.json` freezes the matched B/C/D training
  settings, seeds, dataset metadata hashes, and initialization checkpoint hash.
- `../scripts/` contains reproducible evaluation drivers.
- `../scripts/smoke_training_step.py` runs one forward/backward step without
  updating weights and should pass for B, C, and D before full training.
- `../scripts/evaluate_reference_attributes.py` evaluates velocity, timing, and
  boundary-presence heads on reference intervals, independently of detection.
- `../checkpoint/` is intentionally local and ignored; it contains large model
  checkpoints, predictions, TensorBoard logs, and generated metric files.
- `../datasets/` is intentionally local and ignored.

Research conclusions should reference a tracked protocol/configuration plus
checkpoint and dataset hashes. Generated artifacts should not be committed
unless they are compact, final result summaries needed to support the thesis.