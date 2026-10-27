"""Opt-in, first-answer-token mask intervention. No RNG or tensor work when disabled."""
import json
import os
from functools import lru_cache


@lru_cache(maxsize=1)
def _load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def intervene_mask(mask, sample_id, token_idx, target_stage, condition):
    path = os.environ.get("SECOND_INTERVENTION_PLAN")
    if not path:
        return mask
    if condition != "D":
        raise ValueError("Intervention requires condition D")
    if token_idx != 0:
        return mask
    import torch
    plan = _load(path)
    replacement = plan["masks"][str(sample_id)][str(target_stage)]
    result = torch.tensor(replacement, dtype=mask.dtype, device=mask.device)
    if result.shape != mask.shape or not torch.all((result == 0) | (result == 1)):
        raise ValueError("Invalid intervention mask")
    return result
