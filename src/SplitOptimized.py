import os
import time
import shutil
import json
import subprocess
from imohash import hashfile
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from pathlib import Path

# ===== CONFIG =====
TARGET_FPS = 30
AUDIO_CODEC = "aac"
AUDIO_BITRATE = "192k"
MAX_WORKERS = 3

TOTAL_SIZE_BEFORE_ENCODING = 0
TOTAL_SIZE_AFTER_ENCODING = 0
DELETED_FILE_SIZE = 0

# ===== FAST VIDEO DURATION USING FFMPEG =====
def get_video_duration(path):
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'json', str(path)],
            capture_output=True, text=True
        )
        return float(json.loads(result.stdout)["format"]["duration"])
    except:
        return None

# ===== FAST HASH FUNCTION =====
def get_video_hash(file_path):
    try:
        return hashfile(file_path, hexdigest=True, sample_size=8*1024)  # reduced sample size for speed
    except Exception as e:
        print(f"Error hashing {file_path}: {e}")
        return None

# ===== SPLIT VIDEO (ENCODE FIRST, HASH LATER) =====
def split_video_with_gpu(input_path, output_dir):
    global TOTAL_SIZE_BEFORE_ENCODING

    filename = input_path.name
    name = input_path.stem

    duration = get_video_duration(input_path)
    if duration is None:
        shutil.move(input_path, delete_dir / filename)
        return []

    total_parts = int(duration // 10) + (1 if duration % 10 else 0)
    chunk_paths = []

    for i in range(total_parts):
        start_time = i * 10
        output_path = output_dir / f"{name}_part{i+1}.mp4"

        ffmpeg_command = [
            'ffmpeg',
            '-y',
            '-ss', str(start_time),
            '-i', str(input_path),
            '-t', '10',
            "-c:v", "h264_nvenc", "-preset", "medium", "-cq", "19",
            "-c:a", AUDIO_CODEC, "-b:a", AUDIO_BITRATE, "-ar", "48000", "-ac", "2",
            "-map", "0:v:0", "-map", "0:a:0?",
            str(output_path)
        ]

        try:
            subprocess.run(ffmpeg_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            chunk_paths.append(output_path)
        except subprocess.CalledProcessError:
            shutil.move(input_path, error_dir / filename)
            return []

    # Track size before encoding
    TOTAL_SIZE_BEFORE_ENCODING += input_path.stat().st_size
    shutil.move(input_path, delete_dir / filename)

    return chunk_paths

# ===== PROCESS A SINGLE VIDEO =====
def process_video(file_path):
    global TOTAL_SIZE_AFTER_ENCODING, DELETED_FILE_SIZE

    duration = get_video_duration(file_path)
    if duration is None:
        shutil.move(file_path, delete_dir / file_path.name)
        return []

    if duration < 3:
        DELETED_FILE_SIZE += file_path.stat().st_size
        shutil.move(file_path, delete_dir / file_path.name)
        return []

    elif duration > 10:
        return split_video_with_gpu(file_path, file_path.parent)

    else:
        hash_string = get_video_hash(file_path)
        if hash_string:
            target_path = file_path.parent / f"{hash_string}.mp4"
            if not target_path.exists():
                shutil.move(file_path, target_path)
                TOTAL_SIZE_AFTER_ENCODING += target_path.stat().st_size
            else:
                DELETED_FILE_SIZE += file_path.stat().st_size
                file_path.unlink()
        else:
            DELETED_FILE_SIZE += file_path.stat().st_size
            shutil.move(file_path, hash_error_dir / file_path.name)
        return []

# ===== HASH & DEDUPLICATE CHUNKS AFTER ENCODING =====
def finalize_chunks(chunk_paths):
    global TOTAL_SIZE_AFTER_ENCODING, DELETED_FILE_SIZE
    for chunk in chunk_paths:
        hash_string = get_video_hash(chunk)
        if hash_string:
            target_path = chunk.parent / f"{hash_string}.mp4"
            if not target_path.exists():
                chunk.rename(target_path)
                TOTAL_SIZE_AFTER_ENCODING += target_path.stat().st_size
            else:
                DELETED_FILE_SIZE += chunk.stat().st_size
                chunk.unlink()
        else:
            DELETED_FILE_SIZE += chunk.stat().st_size
            shutil.move(chunk, hash_error_dir / chunk.name)

# ===== MAIN VIDEO SPLITTER =====
def split_large_videos(source_dir, max_workers=MAX_WORKERS):
    files = [
        f for f in Path(source_dir).iterdir()
        if f.suffix.lower() == '.mp4'
        and 'DELETE' not in f.name
        and 'ERROR' not in f.name
        and len(f.name) != 36
        and len(f.name) != 37
    ]

    all_chunks = []

    # Encode in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for chunks in tqdm(executor.map(process_video, files), total=len(files), desc="Processing videos"):
            if chunks:
                all_chunks.extend(chunks)

    # Hash in parallel after encoding
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(finalize_chunks, [ [c] for c in all_chunks ]), total=len(all_chunks), desc="Hashing chunks"))

# ===== ENTRY POINT =====
if __name__ == "__main__":
    input_dir = Path(r"Z:\Projects\VideoSplitter\HDD\Scenes\1080p")
    delete_dir = input_dir / 'DELETE'
    error_dir = input_dir / 'ERROR'
    hash_error_dir = input_dir / 'HASH_ERROR'

    for d in (delete_dir, error_dir, hash_error_dir):
        d.mkdir(exist_ok=True)

    split_large_videos(input_dir, max_workers=MAX_WORKERS)

    print("\nProcessing complete.")
    print("\n\n ===================================================================================== \n\n")
    print(f"Total size before encoding: {TOTAL_SIZE_BEFORE_ENCODING / (1024 * 1024):.2f} MB")
    print(f"Total size after encoding: {TOTAL_SIZE_AFTER_ENCODING / (1024 * 1024):.2f} MB")
    print(f"Total size of deleted files: {DELETED_FILE_SIZE / (1024 * 1024):.2f} MB")
