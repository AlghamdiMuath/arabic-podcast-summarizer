import os
import json
import torch
from faster_whisper import WhisperModel

def transcribe(audio_path: str, output_dir: str = "data/transcripts", model_size: str = "large-v3") -> dict:
    """
    Transcribe Arabic audio using faster-whisper and return the transcript dictionary.
    
    Args:
        audio_path (str): Path to the input .wav or .mp3 file.
        output_dir (str): Where to save .json and .txt transcripts.
        model_size (str): Model size: base | small | medium | large-v3
    
    Returns:
        dict: Transcript data with segments, language, and duration.
    """
    # Determine compute device
    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float32"
    else:
        device = "cpu"
        compute_type = "int8"

    print(f"Loading Whisper model: {model_size} on {device} ({compute_type})")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments, info = model.transcribe(audio_path, language="ar")

    transcript = {
        "language": info.language,
        "duration": info.duration,
        "segments": []
    }

    for seg in segments:
        transcript["segments"].append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip()
        })

    # Prepare filenames
    os.makedirs(output_dir, exist_ok=True)
    audio_id = os.path.splitext(os.path.basename(audio_path))[0]
    json_path = os.path.join(output_dir, f"{audio_id}.json")
    txt_path = os.path.join(output_dir, f"{audio_id}.txt")

    # Save .json
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)

    # Save flat .txt
    with open(txt_path, "w", encoding="utf-8") as f:
        for seg in transcript["segments"]:
            f.write(f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['text']}\n")

    print(f" Saved transcript: {json_path}\n Flat text version: {txt_path}")
    return transcript


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python app/transcriber.py path/to/audio.wav")
        exit(1)

    transcribe(sys.argv[1])
