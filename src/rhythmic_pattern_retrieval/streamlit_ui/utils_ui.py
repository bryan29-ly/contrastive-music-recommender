from pathlib import Path

from rhythmic_pattern_retrieval.config import RAW_DATA_DIR

def get_mp3_path(track_id):
    demo_path = Path("demo_audio") / f"{track_id}.mp3"
    if demo_path.exists():
        return str(demo_path)
        
    candidates = list(RAW_DATA_DIR.rglob(f"{track_id}.mp3"))
    if candidates:
        return str(candidates[0])
        
    return None