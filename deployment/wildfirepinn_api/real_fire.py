"""Real-fire evaluation hook. It does not invent scores."""

from __future__ import annotations


def evaluation_status() -> dict:
    return {
        "status": "not_executed",
        "reason": "No held-out real-fire forecast has been run through this service.",
        "required_inputs": [
            "Chronological perimeters in a projected meter CRS, or in the local Cartesian meters this API accepts.",
            "Observation times that all fall strictly before the held-out time.",
            "One later perimeter used only for scoring, never for fitting or for choosing the model.",
            "A stated cell size and the same grid for every perimeter.",
        ],
        "metrics": None,
    }
