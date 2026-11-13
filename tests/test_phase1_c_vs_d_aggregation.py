import copy
import unittest

from scripts.aggregate_phase1_c_vs_d import (
    CHECKPOINT_HASHES,
    CONFIG_HASHES,
    METRICS,
    SEEDS,
    aggregate,
    classify,
    source_split,
)


def source(seed, deltas):
    records = []
    for index, delta in enumerate(deltas):
        metrics = {}
        for metric in METRICS:
            left = 0.5
            metric_delta = delta if metric == "note+offset" else 0.0005
            metrics[metric] = {
                "C": [0.6, 0.4, left, 0.0],
                "D": [0.61, 0.41, left + metric_delta, 0.0],
            }
        records.append({
            "audio_filename": f"piece-{index}.wav",
            "metrics": metrics,
            "n_estimated": {"C": 1, "D": 1},
        })
    summary = {}
    for metric in METRICS:
        metric_deltas = [
            record["metrics"][metric]["D"][2]
            - record["metrics"][metric]["C"][2]
            for record in records
        ]
        mean = sum(metric_deltas) / len(metric_deltas)
        summary[metric] = {
            "C_macro_f1": 0.5,
            "D_macro_f1": 0.5 + mean,
            "paired_delta_macro_f1": mean,
            "paired_piece_bootstrap_95_percentile_interval": [mean, mean],
        }
    return {
        "protocol": {
            "evaluation": "full_piece_paired_validation",
            "split": "validation",
            "piece_count": len(records),
            "limit": None,
            "metadata_sha256": "metadata",
            "evaluator": {"sha256": "evaluator"},
            "matching": {"aggregation": "macro mean of per-piece F1"},
            "transcription": {"scope": "complete audio file"},
            "left": {
                "label": "C",
                "model_kind": "structured",
                "config_sha256": CONFIG_HASHES["C"],
                "checkpoint_sha256": CHECKPOINT_HASHES[seed]["C"],
            },
            "right": {
                "label": "D",
                "model_kind": "structured",
                "config_sha256": CONFIG_HASHES["D"],
                "checkpoint_sha256": CHECKPOINT_HASHES[seed]["D"],
            },
        },
        "complete": True,
        "completed_pieces": len(records),
        "records": records,
        "summary": summary,
    }


class PhaseOneAggregationTest(unittest.TestCase):
    def test_aggregate_averages_seeds_within_piece(self):
        results = [source(seed, [0.001, 0.003]) for seed in SEEDS]

        result = aggregate(results, bootstrap_replicates=100, bootstrap_seed=1)

        primary = result["summary"]["note+offset"]
        self.assertAlmostEqual(primary["paired_delta_macro_f1"], 0.002)
        self.assertEqual(primary["pieces_improved"], 2)
        self.assertEqual(result["protocol"]["piece_count"], 2)
        self.assertEqual(len(result["protocol"]["piece_order_sha256"]), 64)
        self.assertEqual(result["decision"], "progression_supported")

    def test_rejects_duplicate_seed_or_different_piece_order(self):
        results = [source(seed, [0.002, 0.003]) for seed in SEEDS]
        duplicate = copy.deepcopy(results)
        duplicate[1]["protocol"]["left"]["checkpoint_sha256"] = (
            CHECKPOINT_HASHES[SEEDS[0]]["C"])
        duplicate[1]["protocol"]["right"]["checkpoint_sha256"] = (
            CHECKPOINT_HASHES[SEEDS[0]]["D"])
        with self.assertRaisesRegex(ValueError, "Duplicate source"):
            aggregate(duplicate, bootstrap_replicates=10)

        reordered = copy.deepcopy(results)
        reordered[1]["records"].reverse()
        with self.assertRaisesRegex(ValueError, "order differs"):
            aggregate(reordered, bootstrap_replicates=10)

    def test_decision_labels_follow_the_frozen_gate(self):
        base = {
            "note": {
                "paired_delta_macro_f1": 0.0,
                "paired_piece_bootstrap_95_percentile_interval": [-0.0005, 0.0005],
            },
            "note+offset": {
                "paired_delta_macro_f1": 0.0021,
                "paired_piece_bootstrap_95_percentile_interval": [0.0001, 0.004],
            },
        }
        self.assertEqual(classify(base), "progression_supported")

        base["note"]["paired_piece_bootstrap_95_percentile_interval"][0] = -0.001
        self.assertEqual(classify(base), "safeguard_failed")

        base["note"]["paired_piece_bootstrap_95_percentile_interval"][0] = -0.0005
        base["note+offset"]["paired_delta_macro_f1"] = 0.0019
        self.assertEqual(
            classify(base), "offset_improvement_below_worthwhile_threshold")

    def test_split_identity_is_required_and_consistent(self):
        self.assertEqual(source_split({
            "split": "test", "evaluation": "full_piece_paired_test"}), "test")
        self.assertEqual(source_split({
            "evaluation": "full_piece_paired_validation"}), "validation")
        with self.assertRaisesRegex(ValueError, "disagree"):
            source_split({
                "split": "test",
                "evaluation": "full_piece_paired_validation",
            })

        results = [source(seed, [0.002, 0.003]) for seed in SEEDS]
        results[1]["protocol"].update({
            "split": "test",
            "evaluation": "full_piece_paired_test",
        })
        with self.assertRaisesRegex(ValueError, "splits differ"):
            aggregate(results, bootstrap_replicates=10)


if __name__ == "__main__":
    unittest.main()
