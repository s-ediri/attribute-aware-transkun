import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np


METRICS = ("note", "note+offset", "note+velocity", "note+velocity+offset")
SEEDS = ("20261108", "20261109", "20261110")
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261112
MINIMUM_WORTHWHILE_NOTE_OFFSET_EFFECT = 0.0020
NOTE_ONLY_NONINFERIORITY_MARGIN = -0.0010
CONFIG_HASHES = {
    "C": "a5a7d04ed249ba00348dc3b8bea7fd4c0cf78c3ac9749886fddb4fce1f7f448a",
    "D": "08a5952bc906738c3ca150c64b0778f4e8d7c46cd223f0ee63affcf49859daad",
}
CHECKPOINT_HASHES = {
    "20261108": {
        "C": "1a47a13fd5f91d043bc7b9b831b2f7b0b79172f825e990cba22e6ad17a9bcca0",
        "D": "8d0f4d50b095e92b68fc1cd1524b9fa78a5cae20e788114f03776326c552fa34",
    },
    "20261109": {
        "C": "62f3359bef39e767bf5e0474eea6898a5c976aacc8f21ffc730ad6f66c108848",
        "D": "af43f72b920af1a026fc150a7806425241c76fbbbc41c2a18c8aab0873f18407",
    },
    "20261110": {
        "C": "937046318dbce19257f1b2772de197a6328fb77fbba0d754cdc428f61a535e93",
        "D": "b4499b1b31623ec2b5141cd74de3986342ea77df88ac5aaf77b44e0971a101ce",
    },
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def git_provenance():
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        check=True, capture_output=True, text=True).stdout.splitlines()
    return {"commit": commit, "dirty": bool(status), "status": status}


def seed_for_result(result):
    left_hash = result["protocol"]["left"]["checkpoint_sha256"]
    right_hash = result["protocol"]["right"]["checkpoint_sha256"]
    matches = [
        seed for seed in SEEDS
        if CHECKPOINT_HASHES[seed] == {"C": left_hash, "D": right_hash}
    ]
    if len(matches) != 1:
        raise ValueError("Source does not contain one frozen C/D checkpoint pair")
    return matches[0]


def source_split(protocol):
    split = protocol.get("split")
    if split is not None:
        if protocol.get("evaluation") != f"full_piece_paired_{split}":
            raise ValueError("Source split and evaluation name disagree")
        return split
    evaluation = protocol.get("evaluation")
    if evaluation == "full_piece_paired_validation":
        return "validation"
    if evaluation == "full_piece_paired_test":
        return "test"
    raise ValueError("Source protocol has no recognized split identity")


def validate_sources(results):
    if len(results) != len(SEEDS):
        raise ValueError(f"Expected exactly {len(SEEDS)} source results")

    by_seed = {}
    expected_names = None
    metadata_hash = None
    evaluator_hash = None
    matching_protocol = None
    transcription_protocol = None
    split = None
    for result in results:
        protocol = result.get("protocol", {})
        if not result.get("complete"):
            raise ValueError("Every source evaluation must be complete")
        records = result.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError("Every source evaluation must contain records")
        if result.get("completed_pieces") != len(records):
            raise ValueError("completed_pieces does not match the record count")
        if protocol.get("piece_count") != len(records):
            raise ValueError("piece_count does not match the record count")
        if protocol.get("limit") is not None:
            raise ValueError("Limited evaluations cannot be aggregated")
        if protocol.get("left", {}).get("label") != "C":
            raise ValueError("Left source label must be C")
        if protocol.get("right", {}).get("label") != "D":
            raise ValueError("Right source label must be D")
        if protocol["left"].get("model_kind") != "structured":
            raise ValueError("C model kind must be structured")
        if protocol["right"].get("model_kind") != "structured":
            raise ValueError("D model kind must be structured")
        if protocol["left"].get("config_sha256") != CONFIG_HASHES["C"]:
            raise ValueError("C config hash does not match the frozen protocol")
        if protocol["right"].get("config_sha256") != CONFIG_HASHES["D"]:
            raise ValueError("D config hash does not match the frozen protocol")

        current_split = source_split(protocol)
        if split is None:
            split = current_split
        elif current_split != split:
            raise ValueError("Source evaluation splits differ across seeds")

        seed = seed_for_result(result)
        if seed in by_seed:
            raise ValueError(f"Duplicate source for seed {seed}")

        names = [record.get("audio_filename") for record in records]
        if any(name is None for name in names) or len(set(names)) != len(names):
            raise ValueError("Source recordings must have unique filenames")
        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise ValueError("Source recording order differs across seeds")

        current_metadata_hash = protocol.get("metadata_sha256")
        if not current_metadata_hash:
            raise ValueError("Source protocol is missing metadata_sha256")
        if metadata_hash is None:
            metadata_hash = current_metadata_hash
        elif current_metadata_hash != metadata_hash:
            raise ValueError("Source metadata hashes differ across seeds")

        current_evaluator_hash = protocol.get("evaluator", {}).get("sha256")
        if not current_evaluator_hash:
            raise ValueError("Source protocol is missing the evaluator hash")
        if evaluator_hash is None:
            evaluator_hash = current_evaluator_hash
        elif current_evaluator_hash != evaluator_hash:
            raise ValueError("Source evaluator hashes differ across seeds")

        if matching_protocol is None:
            matching_protocol = protocol.get("matching")
            transcription_protocol = protocol.get("transcription")
        elif (
                protocol.get("matching") != matching_protocol
                or protocol.get("transcription") != transcription_protocol):
            raise ValueError("Source scoring protocols differ across seeds")

        for index, record in enumerate(records):
            if set(record.get("metrics", {})) != set(METRICS):
                raise ValueError(f"Record {index} has an unexpected metric set")
            for metric in METRICS:
                values = record["metrics"][metric]
                if set(values) != {"C", "D"}:
                    raise ValueError(
                        f"Record {index} has unexpected labels for {metric}")
                if any(len(values[label]) < 3 for label in ("C", "D")):
                    raise ValueError(
                        f"Record {index} has incomplete values for {metric}")
        by_seed[seed] = result

    if set(by_seed) != set(SEEDS):
        raise ValueError("The frozen seed set is incomplete")
    return {
        "by_seed": by_seed,
        "names": expected_names,
        "metadata_hash": metadata_hash,
        "evaluator_hash": evaluator_hash,
        "matching": matching_protocol,
        "transcription": transcription_protocol,
        "split": split,
    }


def classify(summary):
    primary = summary["note+offset"]
    safeguard = summary["note"]
    primary_delta = primary["paired_delta_macro_f1"]
    primary_low, primary_high = primary[
        "paired_piece_bootstrap_95_percentile_interval"]
    safeguard_low = safeguard[
        "paired_piece_bootstrap_95_percentile_interval"][0]

    if safeguard_low <= NOTE_ONLY_NONINFERIORITY_MARGIN:
        return "safeguard_failed"
    if (
            primary_delta >= MINIMUM_WORTHWHILE_NOTE_OFFSET_EFFECT
            and primary_low > 0):
        return "progression_supported"
    if primary_low > 0 and primary_delta < MINIMUM_WORTHWHILE_NOTE_OFFSET_EFFECT:
        return "offset_improvement_below_worthwhile_threshold"
    if primary_delta <= 0 and primary_high <= 0:
        return "negative_result"
    return "inconclusive"


def aggregate(results, bootstrap_replicates=BOOTSTRAP_REPLICATES,
              bootstrap_seed=BOOTSTRAP_SEED):
    validated = validate_sources(results)
    by_seed = validated["by_seed"]
    names = validated["names"]
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(
        0, len(names), size=(bootstrap_replicates, len(names)))

    per_seed = {}
    for seed in SEEDS:
        source_summary = by_seed[seed]["summary"]
        per_seed[seed] = {
            metric: {
                "C_macro_f1": float(source_summary[metric]["C_macro_f1"]),
                "D_macro_f1": float(source_summary[metric]["D_macro_f1"]),
                "paired_delta_macro_f1": float(
                    source_summary[metric]["paired_delta_macro_f1"]),
                "paired_piece_bootstrap_95_percentile_interval": [
                    float(value) for value in source_summary[metric][
                        "paired_piece_bootstrap_95_percentile_interval"]
                ],
            }
            for metric in METRICS
        }

    summary = {}
    for metric in METRICS:
        seed_values = []
        seed_deltas = []
        for seed in SEEDS:
            records = by_seed[seed]["records"]
            values = np.asarray([
                [
                    record["metrics"][metric]["C"][0],
                    record["metrics"][metric]["C"][1],
                    record["metrics"][metric]["C"][2],
                    record["metrics"][metric]["D"][0],
                    record["metrics"][metric]["D"][1],
                    record["metrics"][metric]["D"][2],
                ]
                for record in records
            ], dtype=np.float64)
            seed_values.append(values.mean(axis=0))
            seed_deltas.append(values[:, 5] - values[:, 2])

        macro = np.asarray(seed_values).mean(axis=0)
        piece_deltas = np.asarray(seed_deltas).mean(axis=0)
        bootstrap = piece_deltas[indices].mean(axis=1)
        summary[metric] = {
            "C_macro_precision": float(macro[0]),
            "C_macro_recall": float(macro[1]),
            "C_macro_f1": float(macro[2]),
            "D_macro_precision": float(macro[3]),
            "D_macro_recall": float(macro[4]),
            "D_macro_f1": float(macro[5]),
            "paired_delta_macro_f1": float(piece_deltas.mean()),
            "paired_piece_bootstrap_95_percentile_interval": [
                float(value) for value in np.percentile(bootstrap, [2.5, 97.5])
            ],
            "pieces_improved": int(np.count_nonzero(piece_deltas > 0)),
            "pieces_declined": int(np.count_nonzero(piece_deltas < 0)),
            "pieces_unchanged": int(np.count_nonzero(piece_deltas == 0)),
        }

    return {
        "protocol": {
            "comparison": "Phase 1 D minus C",
            "split": validated["split"],
            "seeds": list(SEEDS),
            "piece_count": len(names),
            "piece_order_sha256": hashlib.sha256(
                "\n".join(names).encode("utf-8")).hexdigest(),
            "metadata_sha256": validated["metadata_hash"],
            "source_evaluator_sha256": validated["evaluator_hash"],
            "matching": validated["matching"],
            "transcription": validated["transcription"],
            "aggregation": (
                "average D-C across seeds within recording, then macro mean "
                "across recordings"
            ),
            "bootstrap": {
                "unit": "recording",
                "replicates": bootstrap_replicates,
                "seed": bootstrap_seed,
                "interval": "percentile 95%",
            },
            "minimum_worthwhile_note_offset_effect": (
                MINIMUM_WORTHWHILE_NOTE_OFFSET_EFFECT),
            "note_only_noninferiority_margin": NOTE_ONLY_NONINFERIORITY_MARGIN,
        },
        "per_seed": per_seed,
        "summary": summary,
        "decision": classify(summary),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate the frozen three-seed Phase 1 C versus D results")
    parser.add_argument("inputs", nargs=3, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results = []
    sources = []
    for path in args.inputs:
        results.append(json.loads(path.read_text(encoding="utf-8")))
        sources.append({
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        })
    output = aggregate(results)
    output["sources"] = sources
    output["analysis"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": sha256_file(Path(__file__)),
        "git": git_provenance(),
    }
    write_json(args.output, output)
    print(json.dumps(output["summary"], indent=2))
    print(f"Decision: {output['decision']}")
    print(f"Saved aggregate result to {args.output}")


if __name__ == "__main__":
    main()