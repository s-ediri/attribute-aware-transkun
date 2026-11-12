import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.evaluate_maestro_full_validation import (
    METRICS,
    resolve_sides,
    side_protocol,
    summarize,
    validate_saved_records,
)


class FullValidationEvaluationTest(unittest.TestCase):
    def arguments(self, **updates):
        values = {
            "config": None,
            "baseline_checkpoint": None,
            "candidate_checkpoint": None,
            "left_config": None,
            "left_checkpoint": None,
            "left_label": "left",
            "left_model": "structured",
            "right_config": None,
            "right_checkpoint": None,
            "right_label": "right",
            "right_model": "structured",
        }
        values.update(updates)
        return SimpleNamespace(**values)

    def test_default_arguments_preserve_historical_compatibility_mode(self):
        sides = resolve_sides(self.arguments())

        self.assertEqual(sides["left"]["model_kind"], "historical")
        self.assertEqual(sides["left"]["label"], "baseline")
        self.assertEqual(sides["right"]["model_kind"], "structured")
        self.assertEqual(sides["right"]["label"], "fine_tuned_10000")

    def test_explicit_sides_support_c_versus_d(self):
        sides = resolve_sides(self.arguments(
            left_config=Path("c.conf"),
            left_checkpoint=Path("c.pt"),
            left_label="C",
            right_config=Path("d.conf"),
            right_checkpoint=Path("d.pt"),
            right_label="D",
        ))

        self.assertEqual(sides["left"]["label"], "C")
        self.assertEqual(sides["right"]["label"], "D")

    def test_mixed_compatibility_and_explicit_flags_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Do not combine"):
            resolve_sides(self.arguments(
                config=Path("old.conf"),
                left_config=Path("c.conf"),
            ))

    def test_summary_reports_right_minus_left_piece_differences(self):
        records = []
        for left_f1, right_f1 in ((0.5, 0.7), (0.8, 0.7)):
            records.append({
                "metrics": {
                    metric: {
                        "C": [0.0, 0.0, left_f1, 0.0],
                        "D": [0.0, 0.0, right_f1, 0.0],
                    }
                    for metric in METRICS
                }
            })

        result = summarize(
            records, left_label="C", right_label="D",
            bootstrap_replicates=100, bootstrap_seed=1)

        note_offset = result["note+offset"]
        self.assertAlmostEqual(note_offset["C_macro_f1"], 0.65)
        self.assertAlmostEqual(note_offset["D_macro_f1"], 0.7)
        self.assertAlmostEqual(note_offset["paired_delta_macro_f1"], 0.05)
        self.assertEqual(note_offset["pieces_improved"], 1)
        self.assertEqual(note_offset["pieces_declined"], 1)

    def test_side_protocol_hashes_explicit_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "config.json"
            checkpoint = root / "checkpoint.pt"
            config.write_text("config", encoding="utf-8")
            checkpoint.write_bytes(b"checkpoint")

            result = side_protocol({
                "label": "C",
                "model_kind": "structured",
                "config": config,
                "checkpoint": checkpoint,
            })

        self.assertEqual(result["label"], "C")
        self.assertEqual(result["model_kind"], "structured")
        self.assertEqual(len(result["config_sha256"]), 64)
        self.assertEqual(len(result["checkpoint_sha256"]), 64)

    def test_resume_records_must_be_an_exact_piece_prefix(self):
        record = {
            "audio_filename": "piece-a.wav",
            "metrics": {
                metric: {"C": [0, 0, 0, 0], "D": [0, 0, 0, 0]}
                for metric in METRICS
            },
            "n_estimated": {"C": 1, "D": 1},
        }

        completed = validate_saved_records(
            [record], ["piece-a.wav", "piece-b.wav"], "C", "D")

        self.assertEqual(completed, {"piece-a.wav"})

    def test_resume_rejects_out_of_order_or_wrong_label_records(self):
        record = {
            "audio_filename": "piece-b.wav",
            "metrics": {
                metric: {"C": [0, 0, 0, 0], "D": [0, 0, 0, 0]}
                for metric in METRICS
            },
            "n_estimated": {"C": 1, "D": 1},
        }
        with self.assertRaisesRegex(ValueError, "exact prefix"):
            validate_saved_records(
                [record], ["piece-a.wav", "piece-b.wav"], "C", "D")

        record["audio_filename"] = "piece-a.wav"
        record["metrics"]["note"] = {"left": [0, 0, 0, 0]}
        with self.assertRaisesRegex(ValueError, "unexpected labels"):
            validate_saved_records(
                [record], ["piece-a.wav", "piece-b.wav"], "C", "D")


if __name__ == "__main__":
    unittest.main()