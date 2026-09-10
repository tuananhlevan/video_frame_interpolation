"""State persistence and crash recovery storage for long-running upframing jobs."""

from dataclasses import asdict
import json
import logging
import os
from typing import Any, Dict, Optional
from upframe.core.types import ChunkResult

logger = logging.getLogger(__name__)


class StateStore:
    """Manages persistent job state and chunk resumption."""

    def __init__(self, state_file_path: str, enabled: bool = True) -> None:
        self.state_file_path = os.path.abspath(state_file_path)
        self.enabled = enabled
        self._state: Dict[str, Any] = {
            "chunks": {}
        }
        if self.enabled:
            self.load()

    def load(self) -> Dict[str, Any]:
        """Loads state from JSON file if available."""
        if os.path.exists(self.state_file_path):
            try:
                with open(self.state_file_path, "r") as f:
                    self._state = json.load(f)
                logger.info(f"Loaded existing pipeline state from {self.state_file_path}")
            except Exception as e:
                logger.warning(f"Could not load state from {self.state_file_path}: {e}")
                self._state = {"chunks": {}}
        return self._state

    def save(self) -> None:
        """Writes current state to JSON file."""
        if not self.enabled:
            return
        os.makedirs(os.path.dirname(self.state_file_path), exist_ok=True)
        try:
            with open(self.state_file_path, "w") as f:
                json.dump(self._state, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write state file {self.state_file_path}: {e}")

    def record_chunk(self, result: ChunkResult) -> None:
        """Records the result of a completed or failed chunk."""
        self._state.setdefault("chunks", {})
        self._state["chunks"][str(result.chunk_id)] = asdict(result)
        self.save()

    def get_chunk_result(self, chunk_id: int) -> Optional[ChunkResult]:
        """Retrieves recorded ChunkResult if completed and valid on disk."""
        c_str = str(chunk_id)
        chunks = self._state.get("chunks", {})
        if c_str in chunks:
            info = chunks[c_str]
            if info.get("status") == "COMPLETED" and info.get("output_file"):
                if os.path.exists(info["output_file"]):
                    return ChunkResult(
                        chunk_id=chunk_id,
                        status="COMPLETED",
                        output_file=info["output_file"],
                        source_frames_count=info.get("source_frames_count", 0),
                        output_frames_count=info.get("output_frames_count", 0),
                        scene_cuts_count=info.get("scene_cuts_count", 0),
                        elapsed_seconds=info.get("elapsed_seconds", 0.0)
                    )
        return None

    def is_chunk_completed(self, chunk_id: int) -> bool:
        """Checks whether a chunk has already been completed and verified."""
        return self.get_chunk_result(chunk_id) is not None
