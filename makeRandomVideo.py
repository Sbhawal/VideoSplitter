import os
import random
import subprocess
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ====== CONFIG ======
INPUT_DIR = r"Z:\test\1080p"
OUTPUT_FILE = r"Z:\test\generated\testOutput.mp4"
TEMP_DIR = r"Z:\test\temp"
TARGET_RESOLUTION = "1920x1080"
TARGET_FPS = 30
AUDIO_CODEC = "aac"
AUDIO_BITRATE = "192k"
MAX_WORKERS = 3  # Parallel encoding processes (2–3 is best for RTX 4060)
# ====================

def get_video_files(folder):
    exts = (".mp4", ".mov", ".mkv", ".avi", ".flv")
    return [str(Path(folder) / f) for f in os.listdir(folder) if f.lower().endswith(exts)]

def normalize_video(input_file, output_file):
    """Re-encode to uniform video/audio specs with GPU."""
    cmd = [
        "ffmpeg", "-y", "-i", input_file,
        "-vf", f"scale={TARGET_RESOLUTION},fps={TARGET_FPS},format=yuv420p",
        "-c:v", "h264_nvenc", "-preset", "fast", "-cq", "19",
        "-c:a", AUDIO_CODEC, "-b:a", AUDIO_BITRATE, "-ar", "48000", "-ac", "2",
        "-map", "0:v:0", "-map", "0:a:0?",
        output_file
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def concat_videos(file_list_path, output_file):
    """Concat pre-encoded videos into one."""
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", file_list_path, "-c", "copy",
        output_file
    ]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    videos = get_video_files(INPUT_DIR)
    if not videos:
        raise FileNotFoundError("No video files found in input folder.")

    random.shuffle(videos)

    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    futures = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for idx, vf in enumerate(videos, start=1):
            out_path = os.path.join(TEMP_DIR, f"part_{idx:05d}.mp4")
            futures.append(executor.submit(normalize_video, vf, out_path))

        for i, future in enumerate(as_completed(futures), start=1):
            print(f"[{i}/{len(futures)}] Done")

    # Create concat list
    list_file_path = os.path.join(TEMP_DIR, "file_list.txt")
    with open(list_file_path, "w", encoding="utf-8") as list_file:
        for idx in range(1, len(videos) + 1):
            out_path = os.path.join(TEMP_DIR, f"part_{idx:05d}.mp4")
            list_file.write(f"file '{out_path}'\n")

    print("✅ All parts normalized. Concatenating...")
    concat_videos(list_file_path, OUTPUT_FILE)

    shutil.rmtree(TEMP_DIR, ignore_errors=True)

    print(f"🎬 Final video saved to: {OUTPUT_FILE}")
