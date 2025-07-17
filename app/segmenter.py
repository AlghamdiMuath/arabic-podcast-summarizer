import json
from collections import defaultdict
from pathlib import Path

def structure_transcript(transcript: dict, diarization: dict, output_id: str = "default") -> dict:
    """
    Combine Whisper transcript segments with diarization to create per-speaker blocks.

    Args:
        transcript (dict): Whisper output with 'segments', each containing 'start', 'end', and 'text'.
        diarization (dict): Pyannote output as list of segments with 'speaker', 'start', 'end'.
        output_id (str): Unique identifier to save output file.

    Returns:
        dict: {
            "speakers": {
                "SPEAKER_0": "combined text...",
                "SPEAKER_1": "combined text..."
            },
            "host_guess": "SPEAKER_0"
        }
    """
    speaker_blocks = defaultdict(str)

    # Build list of diarization segments
    diarized_segments = diarization.get("segments", [])

    # Go through Whisper transcript segments
    for segment in transcript.get("segments", []):
        seg_start = segment["start"]
        seg_end = segment["end"]
        seg_text = segment["text"].strip()

        # Find matching diarization speaker (based on time overlap)
        for dia in diarized_segments:
            dia_start = dia["start"]
            dia_end = dia["end"]

            # If there's sufficient overlap, assign
            if seg_start < dia_end and seg_end > dia_start:
                speaker = dia["speaker"]
                speaker_blocks[speaker] += " " + seg_text
                break  # Move to next transcript segment

    # Heuristically guess the host: speaker with most words
    word_counts = {spk: len(txt.split()) for spk, txt in speaker_blocks.items()}
    host_guess = max(word_counts, key=word_counts.get, default=None)

    structured = {
        "speakers": dict(speaker_blocks),
        "host_guess": host_guess
    }

    # Save output
    output_path = Path(f"data/transcripts/structured_{output_id}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(structured, f, ensure_ascii=False, indent=2)

    return structured
