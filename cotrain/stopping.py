"""Plateau detection for co-training."""


def should_stop(round_logs: list[dict], patience: int = 2, epsilon: float = 0.02) -> bool:
    """
    Stop when the optimizer's best mean slop has not decreased by more than epsilon
    for `patience` consecutive round-to-round comparisons.
    """
    if len(round_logs) < patience + 1:
        return False
    recent = [r["optimizer_best_mean_slop"] for r in round_logs[-(patience + 1) :]]
    improvements = [recent[i] - recent[i + 1] for i in range(len(recent) - 1)]
    return all(imp < epsilon for imp in improvements)
