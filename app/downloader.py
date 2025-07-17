import sys
import os
import json
import subprocess
from typing import Optional, Dict, List
import re

def slugify(text: str) -> str:
    """Clean title to create a filesystem-safe filename."""
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '_', text.strip())  
    return text[:50]  

def download_audio(youtube_url: str, output_dir: str = "data/audio") -> Optional[Dict]:
    """Download audio from YouTube and save metadata."""
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Get metadata only (no download yet)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "yt_dlp", "--skip-download", "--print-json", "--no-playlist", youtube_url],
            capture_output=True, text=True, check=True
        )
        info = json.loads(result.stdout)
        title_slug = slugify(info["title"])
    except Exception as e:
        print(f"Failed to retrieve metadata: {e}")
        return None

    audio_path = os.path.join(output_dir, f"{title_slug}.wav")
    info_path = os.path.join(output_dir, f"{title_slug}.info.json")
    metadata_path = os.path.join(output_dir, f"{title_slug}.metadata.json")

    # Step 2: Skip if already downloaded
    if os.path.exists(audio_path):
        print(f"Skipping (already exists): {title_slug}.wav")
        return None

    # Step 3: Run yt-dlp to download
    try:
        command = [
            sys.executable, "-m", "yt_dlp",
            "-f", "bestaudio[ext=m4a]/bestaudio",
            "-x", "--audio-format", "wav",
            "--output", os.path.join(output_dir, f"{title_slug}.%(ext)s"),
            "--write-info-json",
            "--no-playlist",
            youtube_url
        ]

        subprocess.run(command, check=True)

        metadata = {
            "id": info.get("id"),
            "title": info.get("title"),
            "channel": info.get("channel"),
            "duration": info.get("duration"),
            "upload_date": info.get("upload_date"),
            "webpage_url": info.get("webpage_url")
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"Downloaded: {title_slug}.wav")
        return metadata

    except subprocess.CalledProcessError:
        print(f"yt-dlp failed for: {youtube_url}")
    except Exception as e:
        print(f"Error downloading: {e}")

    return None

def batch_download(urls: List[str], output_dir: str = "data/audio"):
    """Download multiple podcast audio files from a list of URLs."""
    for url in urls:
        url = url.strip()
        if url:
            download_audio(url, output_dir)

# CLI interface
if __name__ == "__main__":
    if len(sys.argv) == 2 and os.path.isfile(sys.argv[1]):
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            urls = f.readlines()
        batch_download(urls)
    else:
        # Single input mode
        url = sys.argv[1] if len(sys.argv) > 1 else input("Enter a YouTube podcast URL: ").strip()
        download_audio(url)
