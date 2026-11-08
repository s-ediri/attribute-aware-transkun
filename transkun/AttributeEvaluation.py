import numpy as np
import torch
import torch.nn.functional as F

from .Data import prepareIntervals
from .Util import makeFrame


def average_precision(scores, targets):
    scores = np.asarray(scores, dtype=float)
    targets = np.asarray(targets, dtype=bool)
    positives = int(targets.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    ranked = targets[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].sum() / positives)


def expected_calibration_error(confidence, correct, n_bins=10):
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if confidence.size == 0:
        return None
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    result = 0.0
    for index in range(n_bins):
        if index == n_bins - 1:
            selected = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            selected = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if selected.any():
            result += selected.mean() * abs(
                correct[selected].mean() - confidence[selected].mean())
    return float(result)


def _safe_mean(values):
    values = np.asarray(values, dtype=float)
    return None if values.size == 0 else float(values.mean())


def _categorical_metrics(logits, targets, tolerance=None, values=None,
                         continuous_targets=None, n_calibration_bins=10):
    probabilities = torch.softmax(logits, dim=-1)
    targets = targets.long()
    prediction = probabilities.argmax(dim=-1)
    confidence = probabilities.max(dim=-1).values
    correct = prediction.eq(targets)
    result = {
        "count": int(targets.numel()),
        "accuracy": _safe_mean(correct.cpu().numpy()),
        "nll": _safe_mean(F.cross_entropy(logits, targets, reduction="none").cpu().numpy()),
        "ece": expected_calibration_error(
            confidence.cpu().numpy(), correct.cpu().numpy(),
            n_bins=n_calibration_bins),
    }
    if tolerance is not None:
        result["tolerance_accuracy"] = _safe_mean(
            prediction.sub(targets).abs().le(tolerance).cpu().numpy())
    if values is not None:
        expected = probabilities @ values
        target_values = continuous_targets
        if target_values is None:
            target_values = values.index_select(0, targets)
        error = expected - target_values
        result.update({
            "mean_error": _safe_mean(error.cpu().numpy()),
            "mean_absolute_error": _safe_mean(error.abs().cpu().numpy()),
        })
    return result


def _binary_metrics(logits, targets, n_calibration_bins=10):
    probabilities = torch.sigmoid(logits)
    targets = targets.bool()
    prediction = probabilities >= 0.5
    true_positive = int((prediction & targets).sum().item())
    false_positive = int((prediction & ~targets).sum().item())
    false_negative = int((~prediction & targets).sum().item())
    precision = true_positive / (true_positive + false_positive) \
        if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) \
        if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) \
        if precision + recall else 0.0
    return {
        "count": int(targets.numel()),
        "positive_count": int(targets.sum().item()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "average_precision": average_precision(
            probabilities.cpu().numpy(), targets.cpu().numpy()),
        "nll": _safe_mean(F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="none").cpu().numpy()),
        "ece": expected_calibration_error(
            probabilities.cpu().numpy(), targets.cpu().numpy(),
            n_bins=n_calibration_bins),
    }


def reference_attribute_outputs(model, audio_batch, notes_batch):
    """Return native attribute outputs on ground-truth intervals.

    This measures attribute heads independently of note-detection errors.
    Pedal pseudo-pitches are removed from the returned note-level arrays.
    """
    frames = makeFrame(
        audio_batch.transpose(-1, -2), model.hopSize, model.windowSize)
    prepared = [
        prepareIntervals(notes, model.hopSize / model.fs, model.targetMIDIPitch)
        for notes in notes_batch
    ]
    intervals_batch = [item["intervals"] for item in prepared]

    velocity = []
    timing = []
    presence = []
    pitch = []
    duration = []
    for item in prepared:
        for symbol_index, event_pitch in enumerate(model.targetMIDIPitch):
            for interval, cur_velocity, refine, cur_presence in zip(
                    item["intervals"][symbol_index],
                    item["velocity"][symbol_index],
                    item["endPointRefine"][symbol_index],
                    item["endPointPresence"][symbol_index]):
                velocity.append(cur_velocity)
                timing.append(refine)
                presence.append(cur_presence)
                pitch.append(event_pitch)
                duration.append(
                    ((interval[1] + refine[1]) -
                     (interval[0] + refine[0])) * model.hopSize / model.fs)

    if hasattr(model, "structuredAttributePredictor"):
        _, _, attribute_logits = model.processFramesBatch(frames)
        outputs, _ = model.gatherStructuredAttributeLogits(
            attribute_logits, intervals_batch)
        parameterization = "categorical"
    else:
        _, context = model.processFramesBatch(frames)
        begin, end, _, _ = model.fetchIntervalFeaturesBatch(
            context, intervals_batch)
        attribute_input = torch.cat([begin, end, begin * end], dim=-1)
        refined, boundary_presence = model.refinedOFPredictor(
            attribute_input).chunk(2, dim=-1)
        outputs = {
            "velocity": model.velocityPredictor(attribute_input),
            "timing": refined,
            "presence": boundary_presence,
        }
        parameterization = "continuous_bernoulli"

    device = audio_batch.device
    pitch = torch.tensor(pitch, dtype=torch.long, device=device)
    note_mask = pitch >= 0
    result = {
        "parameterization": parameterization,
        "velocity": torch.tensor(velocity, dtype=torch.long, device=device)[note_mask],
        "timing": torch.tensor(timing, dtype=torch.float, device=device)[note_mask],
        "presence": torch.tensor(presence, dtype=torch.bool, device=device)[note_mask],
        "pitch": pitch[note_mask],
        "duration": torch.tensor(duration, dtype=torch.float, device=device)[note_mask],
    }
    result["outputs"] = {
        key: value[note_mask] for key, value in outputs.items()
    }
    return result


def _subset(batch, mask):
    return {
        key: value[mask] if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
        if key != "outputs"
    } | {
        "outputs": {key: value[mask] for key, value in batch["outputs"].items()}
    }


def summarize_reference_attributes(batch, hop_seconds=None, n_calibration_bins=10):
    outputs = batch["outputs"]
    velocity_values = torch.arange(128, device=batch["velocity"].device).float()
    summary = {
        "note_count": int(batch["velocity"].numel()),
        "parameterization": batch["parameterization"],
        "velocity": _categorical_metrics(
            outputs["velocity"], batch["velocity"], tolerance=12,
            values=velocity_values,
            n_calibration_bins=n_calibration_bins),
    }

    if batch["parameterization"] == "categorical":
        n_bins = outputs["onsetTiming"].shape[-1]
        timing_values = torch.linspace(
            -0.5, 0.5, n_bins, device=batch["timing"].device)
        for boundary_index, name in enumerate(("onset", "offset")):
            present = batch["presence"][:, boundary_index]
            targets = torch.clamp(torch.round(
                (batch["timing"][:, boundary_index] + 0.5) * (n_bins - 1)),
                0, n_bins - 1).long()
            metrics = _categorical_metrics(
                outputs[f"{name}Timing"][present], targets[present],
                tolerance=1, values=timing_values,
                continuous_targets=batch["timing"][present, boundary_index],
                n_calibration_bins=n_calibration_bins)
            scale = hop_seconds if hop_seconds is not None else 1.0
            for key in ("mean_error", "mean_absolute_error"):
                if metrics[key] is not None:
                    metrics[f"{key}_seconds"] = metrics[key] * scale
            summary[f"{name}_timing"] = metrics
            presence_logits = outputs[f"{name}Presence"]
            binary_logits = presence_logits[:, 1] - presence_logits[:, 0]
            summary[f"{name}_presence"] = _binary_metrics(
                binary_logits, batch["presence"][:, boundary_index],
                n_calibration_bins=n_calibration_bins)
    else:
        timing_logits = outputs["timing"]
        distribution = torch.distributions.ContinuousBernoulli(
            logits=timing_logits)
        predicted = torch.clamp((distribution.mean - 0.5) / 0.99, -0.5, 0.5)
        shifted_target = batch["timing"] * 0.99 + 0.5
        for boundary_index, name in enumerate(("onset", "offset")):
            present = batch["presence"][:, boundary_index]
            error = predicted[present, boundary_index] - batch["timing"][present, boundary_index]
            nll = -distribution.log_prob(shifted_target)[present, boundary_index]
            scale = hop_seconds if hop_seconds is not None else 1.0
            summary[f"{name}_timing"] = {
                "count": int(present.sum().item()),
                "mean_error": _safe_mean(error.cpu().numpy()),
                "mean_absolute_error": _safe_mean(error.abs().cpu().numpy()),
                "mean_error_seconds": _safe_mean((error * scale).cpu().numpy()),
                "mean_absolute_error_seconds": _safe_mean(
                    (error.abs() * scale).cpu().numpy()),
                "nll": _safe_mean(nll.cpu().numpy()),
            }
            summary[f"{name}_presence"] = _binary_metrics(
                outputs["presence"][:, boundary_index],
                batch["presence"][:, boundary_index],
                n_calibration_bins=n_calibration_bins)
    return summary


def stratified_summaries(batch, hop_seconds):
    duration = batch["duration"]
    velocity = batch["velocity"]
    pitch = batch["pitch"]
    complete = batch["presence"][:, 0] & batch["presence"][:, 1]
    strata = {
        "duration_lt_250ms": complete & (duration < 0.25),
        "duration_250_500ms": complete & (duration >= 0.25) & (duration < 0.5),
        "duration_500ms_1s": complete & (duration >= 0.5) & (duration < 1.0),
        "duration_ge_1s": complete & (duration >= 1.0),
        "velocity_le_40": velocity <= 40,
        "velocity_41_80": (velocity >= 41) & (velocity <= 80),
        "velocity_ge_81": velocity >= 81,
        "pitch_le_48": pitch <= 48,
        "pitch_49_72": (pitch >= 49) & (pitch <= 72),
        "pitch_ge_73": pitch >= 73,
    }
    return {
        name: summarize_reference_attributes(_subset(batch, mask), hop_seconds)
        for name, mask in strata.items() if mask.any()
    }