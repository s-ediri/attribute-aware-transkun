import argparse
import hashlib
import json
import sys
from pathlib import Path

import moduleconf
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from transkun import Data
from transkun.AttributeEvaluation import (
    reference_attribute_outputs,
    stratified_summaries,
    summarize_reference_attributes,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def concatenate(batches):
    return {
        "parameterization": batches[0]["parameterization"],
        "velocity": torch.cat([item["velocity"] for item in batches]),
        "timing": torch.cat([item["timing"] for item in batches]),
        "presence": torch.cat([item["presence"] for item in batches]),
        "pitch": torch.cat([item["pitch"] for item in batches]),
        "duration": torch.cat([item["duration"] for item in batches]),
        "outputs": {
            key: torch.cat([item["outputs"][key] for item in batches])
            for key in batches[0]["outputs"]
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate attribute heads on deterministic reference intervals.")
    parser.add_argument("--system-id", choices=("B", "C", "D"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--chunk-size", type=float, default=16.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be positive")
    device = torch.device(args.device)
    configuration = moduleconf.parseFromFile(str(args.config))
    model = configuration["Model"].module.TransKun(
        configuration["Model"].config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = checkpoint.get("best_state_dict", checkpoint.get("state_dict"))
    model.load_state_dict(state, strict=True)
    model.eval()

    dataset = Data.DatasetMaestro(str(args.dataset_path), str(args.metadata))
    iterator = Data.DatasetMaestroIterator(
        dataset,
        hopSizeInSecond=configuration["Model"].config.segmentHopSizeInSecond,
        chunkSizeInSecond=args.chunk_size,
        notesStrictlyContained=False,
        ditheringFrames=False,
        seed=args.seed,
    )
    batches = []
    with torch.inference_mode():
        for index in range(min(args.limit, len(iterator))):
            sample = iterator[index]
            audio = torch.from_numpy(sample["audioSlice"]).unsqueeze(0).to(device)
            batch = reference_attribute_outputs(model, audio, [sample["notes"]])
            batches.append({
                key: value.cpu() if isinstance(value, torch.Tensor) else
                ({name: tensor.cpu() for name, tensor in value.items()}
                 if key == "outputs" else value)
                for key, value in batch.items()
            })

    combined = concatenate(batches)
    hop_seconds = model.hopSize / model.fs
    result = {
        "protocol": {
            "evaluation": "reference_interval_attribute_quality",
            "system_id": args.system_id,
            "config": str(args.config.resolve()),
            "config_sha256": sha256_file(args.config),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "dataset_path": str(args.dataset_path.resolve()),
            "metadata": str(args.metadata.resolve()),
            "metadata_sha256": sha256_file(args.metadata),
            "window_count": len(batches),
            "chunk_size_seconds": args.chunk_size,
            "seed": args.seed,
            "pedal_events_excluded": True,
            "timing_scored_only_when_boundary_present": True,
            "duration_strata_require_complete_onset_and_offset": True,
            "calibration_bins": 10,
            "legacy_timing_nll": (
                "continuous density negative log-likelihood; values may be negative"),
        },
        "overall": summarize_reference_attributes(combined, hop_seconds),
        "strata": stratified_summaries(combined, hop_seconds),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["overall"], indent=2))
    print(f"Saved attribute evaluation to {args.output}")


if __name__ == "__main__":
    main()