import torch

from . import CRF


def add_attribute_scores(crf, attribute_logits, weight):
    begin_score = (
        torch.logsumexp(attribute_logits["velocity"], dim=-1)
        + torch.logsumexp(attribute_logits["onsetTiming"], dim=-1)
        + torch.logsumexp(attribute_logits["onsetPresence"], dim=-1)
    ) * weight
    end_score = (
        torch.logsumexp(attribute_logits["offsetTiming"], dim=-1)
        + torch.logsumexp(attribute_logits["offsetPresence"], dim=-1)
    ) * weight

    begin_score = begin_score.permute(2, 0, 1).flatten(1, 2)
    end_score = end_score.permute(2, 0, 1).flatten(1, 2)
    score = crf.score + begin_score.unsqueeze(0) + end_score.unsqueeze(1)
    return CRF.NeuralSemiCRFInterval(score, crf.noiseScore)


def timing_value_to_index(value, n_timing_bins):
    return torch.clamp(
        torch.round((value + 0.5) * (n_timing_bins-1)),
        0, n_timing_bins-1).long()


def timing_index_to_value(index, n_timing_bins):
    return index.float()/(n_timing_bins-1) - 0.5


def gather_attribute_logits(attribute_logits, intervals_batch, n_symbols):
    gathered = {key: [] for key in attribute_logits}
    scatter_indices = []
    device = attribute_logits["velocity"].device

    for batch_idx, tracks in enumerate(intervals_batch):
        intervals = sum(tracks, [])
        if len(intervals) == 0:
            continue
        symbol_idx = torch.tensor(
            [idx for idx, track in enumerate(tracks) for _ in track],
            dtype=torch.long, device=device)
        positions = torch.tensor(intervals, dtype=torch.long, device=device)
        begin = positions[:, 0]
        end = positions[:, 1]

        gathered["velocity"].append(attribute_logits["velocity"][batch_idx, symbol_idx, begin])
        gathered["onsetTiming"].append(attribute_logits["onsetTiming"][batch_idx, symbol_idx, begin])
        gathered["offsetTiming"].append(attribute_logits["offsetTiming"][batch_idx, symbol_idx, end])
        gathered["onsetPresence"].append(attribute_logits["onsetPresence"][batch_idx, symbol_idx, begin])
        gathered["offsetPresence"].append(attribute_logits["offsetPresence"][batch_idx, symbol_idx, end])
        scatter_indices.append(batch_idx*n_symbols + symbol_idx)

    if len(scatter_indices) == 0:
        return None, None

    return (
        {key: torch.cat(value, dim=0) for key, value in gathered.items()},
        torch.cat(scatter_indices, dim=0),
    )


def attribute_log_prob(gathered, velocity, timing, presence, n_timing_bins, weight):
    device = gathered["velocity"].device
    velocity = torch.tensor(velocity, dtype=torch.long, device=device)
    timing = torch.tensor(timing, dtype=torch.float, device=device)
    presence = torch.tensor(presence, dtype=torch.long, device=device)
    onset_timing = timing_value_to_index(timing[:, 0], n_timing_bins)
    offset_timing = timing_value_to_index(timing[:, 1], n_timing_bins)

    def categorical_log_prob(logits, target):
        selected = logits.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        return selected - torch.logsumexp(logits, dim=-1)

    result = categorical_log_prob(gathered["velocity"], velocity)
    result = result + categorical_log_prob(gathered["onsetTiming"], onset_timing)
    result = result + categorical_log_prob(gathered["offsetTiming"], offset_timing)
    result = result + categorical_log_prob(gathered["onsetPresence"], presence[:, 0])
    result = result + categorical_log_prob(gathered["offsetPresence"], presence[:, 1])
    return result * weight