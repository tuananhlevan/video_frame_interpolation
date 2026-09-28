"""Cadence detection and slow-motion de-stuttering filter for sports replays."""

import logging
from typing import List
import numpy as np

logger = logging.getLogger(__name__)


def is_duplicate_frame(f1: np.ndarray, f2: np.ndarray, l1_threshold: float = 0.8) -> bool:
    """Checks whether two frames are identical/duplicate within video compression noise.
    
    Subsamples the frames 4x for instantaneous L1 difference evaluation.
    """
    if f1.shape != f2.shape:
        return False
    diff = float(np.mean(np.abs(f1[::4, ::4].astype(np.float32) - f2[::4, ::4].astype(np.float32))))
    return diff < l1_threshold


class CadenceHandler:
    """Tracks and manages frame repetition patterns in slow-motion replays."""

    def __init__(self, l1_threshold: float = 0.8) -> None:
        self.l1_threshold = l1_threshold
        self.consecutive_duplicates = 0

    def check_pair(self, f_curr: np.ndarray, f_next: np.ndarray) -> bool:
        """Returns True if f_next is a duplicate of f_curr."""
        is_dup = is_duplicate_frame(f_curr, f_next, self.l1_threshold)
        if is_dup:
            self.consecutive_duplicates += 1
        else:
            self.consecutive_duplicates = 0
        return is_dup
