"""
GAP aggregation sensitivity for edge-DP vs node-DP.
"""


def compute_gap_sensitivity(gap_privacy: str, max_degree: int, hops: int) -> float:
    """
    Conservative sensitivity for GAP aggregation (used by PMA).

    - edge: 1.0 (unchanged behavior)
    - node: max_degree * hops
    """
    gap_privacy = gap_privacy.lower()
    if gap_privacy == "edge":
        return 1.0
    if gap_privacy == "node":
        return float(max_degree * max(1, hops))
    raise ValueError(f"Unknown gap_privacy mode: {gap_privacy}")
