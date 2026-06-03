import librosa
import numpy as np

# Hop length used only for the rhythmic-salience analysis below. It is
# independent from the mel hop (we don't need fine resolution to locate a
# 12 s window), so we keep librosa's default for speed.
ANALYSIS_HOP = 512


def estimate_bpm(wav_np, sr):
    """Estimate the global tempo (BPM) of a track via librosa's beat tracker."""
    tempo, _ = librosa.beat.beat_track(
        y=wav_np, sr=sr, hop_length=ANALYSIS_HOP)
    return float(np.atleast_1d(tempo)[0])


def select_segments(wav_np, sr, segment_seconds, n_segments):
    """Select up to ``n_segments`` rhythmically-salient windows from a track.

    Each candidate window is scored by (onset strength x RMS energy): we want
    sections that are both energetic and rhythmically active (the groove / drop),
    not intros, breakdowns or outros. Picks are non-overlapping so the segments
    cover the track's signature rather than collapsing onto a single spot.

    Returns a list of (start_sample, end_sample) tuples, sorted by position.
    """
    n_samples = len(wav_np)
    seg_len = int(segment_seconds * sr)

    # Track shorter than one segment -> use it whole (padded downstream).
    if n_samples <= seg_len:
        return [(0, n_samples)]

    # Frame-level rhythmic salience.
    onset_env = librosa.onset.onset_strength(
        y=wav_np, sr=sr, hop_length=ANALYSIS_HOP)
    rms = librosa.feature.rms(y=wav_np, hop_length=ANALYSIS_HOP)[0]
    n_frames = min(len(onset_env), len(rms))  # they can differ by one frame
    salience = onset_env[:n_frames] * rms[:n_frames]

    seg_frames = seg_len // ANALYSIS_HOP
    if seg_frames >= n_frames:
        return [(0, n_samples)]

    # Mean salience of every contiguous window, computed in O(n) via a prefix sum.
    cumsum = np.concatenate([[0.0], np.cumsum(salience)])
    window_scores = (cumsum[seg_frames:] - cumsum[:-seg_frames]) / seg_frames

    # Greedy, non-overlapping selection of the top-scoring windows.
    order = np.argsort(window_scores)[::-1]
    selected_frames = []
    for frame_start in order:
        if len(selected_frames) >= n_segments:
            break
        if all(abs(frame_start - s) >= seg_frames for s in selected_frames):
            selected_frames.append(int(frame_start))

    selected_frames.sort()
    segments = []
    for f in selected_frames:
        start = min(f * ANALYSIS_HOP, n_samples -
                    seg_len)  # clamp inside track
        segments.append((start, start + seg_len))
    return segments
