"""Musical key detection (Krumhansl-Schmuckler) and Camelot wheel helpers.

The Camelot mapping and compatibility rules mirror src/camelot.ts in the
soundcloud-dj-plugin repository.
"""
from __future__ import annotations

import numpy as np

PITCHES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Kessler key profiles.
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# pitch class -> Camelot number
CAMELOT_MAJOR = {0: 8, 7: 9, 2: 10, 9: 11, 4: 12, 11: 1, 6: 2, 1: 3, 8: 4, 3: 5, 10: 6, 5: 7}
CAMELOT_MINOR = {9: 8, 4: 9, 11: 10, 6: 11, 1: 12, 8: 1, 3: 2, 10: 3, 5: 4, 0: 5, 7: 6, 2: 7}


def estimate_key(chroma_mean: np.ndarray) -> tuple[str, str, float]:
    """Return (key name, Camelot code, confidence 0..1) from a 12-bin mean chroma."""
    c = np.asarray(chroma_mean, dtype=float)
    if c.shape != (12,) or not np.any(c):
        return ("unknown", "", 0.0)
    scores = []
    for pc in range(12):
        for minor, prof in ((False, MAJOR_PROFILE), (True, MINOR_PROFILE)):
            r = np.corrcoef(c, np.roll(prof, pc))[0, 1]
            scores.append((r, pc, minor))
    scores.sort(reverse=True)
    best, second = scores[0], scores[1]
    r, pc, minor = best
    name = f"{PITCHES[pc]} {'minor' if minor else 'major'}"
    code = f"{(CAMELOT_MINOR if minor else CAMELOT_MAJOR)[pc]}{'A' if minor else 'B'}"
    confidence = float(np.clip((r - second[0]) * 5 + max(r, 0) * 0.5, 0, 1))
    return (name, code, confidence)


def parse_camelot(code: str) -> tuple[int, str] | None:
    code = code.strip().upper()
    if len(code) < 2 or code[-1] not in "AB" or not code[:-1].isdigit():
        return None
    n = int(code[:-1])
    return (n, code[-1]) if 1 <= n <= 12 else None


def compatibility(a: str, b: str) -> str:
    """same | adjacent | relative | energy_boost | clash | unknown"""
    pa, pb = parse_camelot(a), parse_camelot(b)
    if not pa or not pb:
        return "unknown"
    (na, la), (nb, lb) = pa, pb
    if na == nb and la == lb:
        return "same"
    if na == nb:
        return "relative"
    d = abs(na - nb) % 12
    d = min(d, 12 - d)
    if la == lb and d == 1:
        return "adjacent"
    if la == lb and ((na + 1) % 12) + 1 == nb:  # +2 on the wheel, wrapping 11->1, 12->2
        return "energy_boost"
    return "clash"
