"""Audio integrity and A/V synchronization validation."""

from typing import Dict, List, Tuple
from upframe.core.types import VideoMetadata


def validate_audio_integrity(
    source_meta: VideoMetadata,
    output_meta: VideoMetadata,
    max_duration_diff_sec: float = 0.25
) -> Tuple[bool, Dict[str, any], List[str]]:
    """Validates audio presence, duration consistency, channel count, and sync.
    
    Returns:
        (passed, details, warnings)
    """
    warnings: List[str] = []
    passed = True

    # 1. Audio stream existence
    if source_meta.has_audio and not output_meta.has_audio:
        warnings.append("Original video contained audio, but output file has no audio stream.")
        return False, {"has_audio": False}, warnings

    if not source_meta.has_audio and not output_meta.has_audio:
        return True, {"has_audio": False, "note": "Source and output have no audio"}, []

    src_audio = source_meta.audio_streams[0]
    out_audio = output_meta.audio_streams[0]

    # 2. Duration check
    src_a_dur = src_audio.duration or source_meta.duration
    out_a_dur = out_audio.duration or output_meta.duration
    dur_diff = abs(out_a_dur - src_a_dur)

    if dur_diff > max_duration_diff_sec:
        warnings.append(
            f"Audio duration mismatch: source audio {src_a_dur:.2f}s vs output audio {out_a_dur:.2f}s "
            f"(diff: {dur_diff:.2f}s > tolerance {max_duration_diff_sec:.2f}s)"
        )
        passed = False

    # 3. Audio vs Video duration drift in output
    av_dur_diff = abs(out_a_dur - output_meta.duration)
    if av_dur_diff > max_duration_diff_sec:
        warnings.append(
            f"Potential A/V drift: output audio ({out_a_dur:.2f}s) vs video ({output_meta.duration:.2f}s) "
            f"delta is {av_dur_diff:.2f}s"
        )
        # Note: Container padding can cause minor differences, warn if significant

    # 4. Channel and sample rate checks
    if src_audio.channels != out_audio.channels:
        warnings.append(
            f"Audio channel count changed from {src_audio.channels} to {out_audio.channels}"
        )
    if src_audio.sample_rate != out_audio.sample_rate:
        warnings.append(
            f"Audio sample rate changed from {src_audio.sample_rate} Hz to {out_audio.sample_rate} Hz"
        )

    details = {
        "has_audio": True,
        "source_audio_codec": src_audio.codec_name,
        "output_audio_codec": out_audio.codec_name,
        "source_sample_rate": src_audio.sample_rate,
        "output_sample_rate": out_audio.sample_rate,
        "source_channels": src_audio.channels,
        "output_channels": out_audio.channels,
        "audio_duration_diff_sec": dur_diff,
        "av_duration_diff_sec": av_dur_diff,
        "passed": passed
    }

    return passed, details, warnings
