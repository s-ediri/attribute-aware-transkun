import argparse
import json
import sys
from pathlib import Path

import torch
import moduleconf


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from transkun import Data
from transkun.TrainUtil import load_pretrained_weights
from transkun.train import seed_everything


def gradient_sum(model, prefix):
    return sum(
        parameter.grad.abs().sum().item()
        for name, parameter in model.named_parameters()
        if name.startswith(prefix) and parameter.grad is not None
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run one forward/backward training smoke step without updating weights.")
    parser.add_argument("--system-id", choices=("B", "C", "D"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--chunk-size", type=float, default=2.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--strict-pretrained", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    configuration = moduleconf.parseFromFile(str(args.config))
    model = configuration["Model"].module.TransKun(
        configuration["Model"].config).to(device)
    load_pretrained_weights(
        model, args.pretrained_checkpoint, device,
        strict=args.strict_pretrained)
    model.train()

    dataset = Data.DatasetMaestro(str(args.dataset_path), str(args.metadata))
    iterator = Data.DatasetMaestroIterator(
        dataset,
        hopSizeInSecond=configuration["Model"].config.segmentHopSizeInSecond,
        chunkSizeInSecond=args.chunk_size,
        notesStrictlyContained=False,
        ditheringFrames=False,
        seed=args.seed,
    )
    sample = iterator[0]
    audio = torch.from_numpy(sample["audioSlice"]).unsqueeze(0).to(device)

    model.zero_grad(set_to_none=True)
    log_probability = model.log_prob(audio, [sample["notes"]])
    loss = -log_probability.sum()
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite smoke-test loss: {loss.item()}")
    loss.backward()

    result = {
        "system_id": args.system_id,
        "model_module": configuration["Model"].module.__name__,
        "seed": args.seed,
        "audio_shape": list(audio.shape),
        "note_count": len(sample["notes"]),
        "log_probability_shape": list(log_probability.shape),
        "loss": loss.item(),
        "gradient_sums": {
            "backbone": gradient_sum(model, "backbone."),
            "scorer": gradient_sum(model, "scorer."),
        },
    }
    if hasattr(model, "velocityPredictor"):
        result["gradient_sums"]["velocity_predictor"] = gradient_sum(
            model, "velocityPredictor.")
        result["gradient_sums"]["refined_offset_predictor"] = gradient_sum(
            model, "refinedOFPredictor.")
    if hasattr(model, "structuredAttributePredictor"):
        result["gradient_sums"]["structured_attribute_predictor"] = gradient_sum(
            model, "structuredAttributePredictor.")

    print(json.dumps(result, indent=2))
    missing = [
        name for name, value in result["gradient_sums"].items()
        if not value > 0]
    if missing:
        raise RuntimeError(
            "The smoke step produced no gradient for: " + ", ".join(missing))


if __name__ == "__main__":
    main()