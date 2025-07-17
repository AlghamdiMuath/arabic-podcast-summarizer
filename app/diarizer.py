import os
import json
from datetime import timedelta
from pyannote.audio import Pipeline


def diarize_audio(audio_path: str, hf_token: str, output_dir: str = "data/diarization") -> list:
    """
    Perform speaker diarization on a .wav audio file using pyannote-audio.

    Args:
        audio_path (str): Path to the input .wav audio file.
        hf_token (str): Hugging Face access token to load the diarization pipeline.
        output_dir (str): Directory to save diarization results (.json and .rttm).

    Returns:
        List[Dict]: List of diarized segments with start, end, and speaker.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load diarization pipeline
    print("Loading pyannote speaker diarization pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=hf_token
    )

    # Run diarization
    print(f"Processing audio file: {audio_path}")
    diarization = pipeline(audio_path)

    # Collect segments
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": str(timedelta(seconds=turn.start)),
            "end": str(timedelta(seconds=turn.end)),
            "speaker": speaker
        })

    # Define output filenames
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    json_path = os.path.join(output_dir, f"{base_name}.json")
    rttm_path = os.path.join(output_dir, f"{base_name}.rttm")

    # Save as JSON
    with open(json_path, "w", encoding="utf-8") as f_json:
        json.dump(segments, f_json, ensure_ascii=False, indent=2)
    print(f"Saved diarization JSON: {json_path}")

    # Save as RTTM
    with open(rttm_path, "w", encoding="utf-8") as f_rttm:
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            f_rttm.write(
                f"SPEAKER {base_name} 1 {turn.start:.3f} {(turn.end - turn.start):.3f} <NA> <NA> {speaker} <NA> <NA>\n"
            )
    print(f"Saved diarization RTTM: {rttm_path}")

    return segments


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_TOKEN")

    if len(sys.argv) != 2:
        print("Usage: python app/diarizer.py path/to/audio.wav")
    else:
        diarize_audio(sys.argv[1], hf_token)
