import os,time
import shutil
import subprocess
from moviepy.editor import VideoFileClip
from imohash import hashfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

TARGET_FPS = 30
AUDIO_CODEC = "aac"
AUDIO_BITRATE = "192k"
MAX_WORKERS = 3  # Parallel encoding processes (2–3 is best for RTX 4060)

TOTAL_SIZE_BEFORE_ENCODING = 0
TOTAL_SIZE_AFTER_ENCODING = 0
DELETED_FILE_SIZE = 0

def get_video_duration(path):
    try:
        with VideoFileClip(path) as clip:
            return clip.duration
    except Exception as e:
        return None


def get_video_hash(file_path):
    try:
        return hashfile(file_path, hexdigest=True, sample_size=25*1024)
    except Exception as e:
        print(f"Error hashing {file_path}: {e}")
        return None


def split_video_with_gpu(input_path, output_dir):
    global TOTAL_SIZE_BEFORE_ENCODING, TOTAL_SIZE_AFTER_ENCODING,DELETED_FILE_SIZE
    filename = os.path.basename(input_path)
    name, _ = os.path.splitext(filename)

    duration = get_video_duration(input_path)
    if duration is None:
        os.rename(input_path, os.path.join(delete_dir, filename))  # Move to DELETE directory
        os.remove(os.path.join(delete_dir, filename))
        return

    total_parts = int(duration // 10) + (1 if duration % 10 else 0)

    for i in range(total_parts):
        start_time = i * 10
        output_filename = f"{name}_part{i+1}.mp4"
        output_path = os.path.join(output_dir, output_filename)

        ffmpeg_command = [
            'ffmpeg',
            '-y',
            '-ss', str(start_time),
            '-i', input_path,
            '-t', '10',
            "-c:v", "h264_nvenc", "-preset", "medium", "-cq", "21",
            "-c:a", AUDIO_CODEC, "-b:a", AUDIO_BITRATE, "-ar", "48000", "-ac", "2",
            "-map", "0:v:0", "-map", "0:a:0?",
            output_path
        ]

        try:
            subprocess.run(ffmpeg_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            hash_string = get_video_hash(output_path)
            if hash_string:
                try:
                    FILE_SIZE = os.path.getsize(output_path)
                    os.rename(output_path, os.path.join(output_dir, f"{hash_string}.mp4"))
                    TOTAL_SIZE_AFTER_ENCODING += FILE_SIZE
                except FileExistsError:
                    # print(f"Hash {hash_string} already exists, removing {output_path}")
                    FILE_SIZE = os.path.getsize(output_path)
                    os.remove(output_path)  # Remove the file if hash already exists
                    DELETED_FILE_SIZE += FILE_SIZE
                    continue
            else:
                FILE_SIZE = os.path.getsize(output_path)
                shutil.move(input_path, os.path.join(hash_error_dir, filename))  # Move to HASH ERROR directory on failure
                DELETED_FILE_SIZE += FILE_SIZE
        except subprocess.CalledProcessError as e:
            try:
                FILE_SIZE = os.path.getsize(input_path)
                shutil.move(input_path, os.path.join(error_dir, filename))  # Move to ERROR directory on failure
                DELETED_FILE_SIZE += FILE_SIZE
            except:
                time.sleep(2)  # Wait a bit before retrying
                try:
                    if os.path.exists(input_path):
                        FILE_SIZE = os.path.getsize(input_path)
                        shutil.move(input_path, os.path.join(error_dir, filename))
                        DELETED_FILE_SIZE += FILE_SIZE
                except Exception as e:
                    print(f"Failed to move {input_path} to ERROR directory: {e}")
                    pass

    # Move original file to DELETE directory after all chunks are processed
    if os.path.exists(input_path):
        FILE_SIZE = os.path.getsize(input_path)
        os.rename(input_path, os.path.join(delete_dir, filename))
        os.remove(os.path.join(delete_dir, filename))
        TOTAL_SIZE_BEFORE_ENCODING += FILE_SIZE


def process_video(file, source_dir, delete_dir, chunks_dir):
    global TOTAL_SIZE_BEFORE_ENCODING, TOTAL_SIZE_AFTER_ENCODING,DELETED_FILE_SIZE
    chunks_dir = source_dir
    full_path = os.path.join(source_dir, file)
   
    duration = get_video_duration(full_path)
    if duration is None:
        try:
            os.rename(full_path, os.path.join(delete_dir, file))  # Move to DELETE directory
            os.remove(os.path.join(delete_dir, file))
        except FileExistsError:
            shutil.move(full_path, os.path.join(delete_dir, file))
            os.remove(os.path.join(delete_dir, file))
        return

    if duration < 3:
        FILE_SIZE = os.path.getsize(full_path)
        DELETED_FILE_SIZE += FILE_SIZE
        shutil.move(full_path, os.path.join(delete_dir, file))
        os.remove(os.path.join(delete_dir, file))
    elif duration > 10:
        FILE_SIZE = os.path.getsize(full_path)
        TOTAL_SIZE_BEFORE_ENCODING += FILE_SIZE
        split_video_with_gpu(full_path, chunks_dir)
    else:
        hash_string = get_video_hash(full_path)
        if hash_string:
            try:
                os.rename(full_path, os.path.join(chunks_dir, f"{hash_string}.mp4"))
                FILE_SIZE = os.path.getsize(os.path.join(chunks_dir, f"{hash_string}.mp4"))
                TOTAL_SIZE_AFTER_ENCODING += FILE_SIZE
            except FileExistsError:
                # print(f"Hash {hash_string} already exists, removing {full_path}")
                FILE_SIZE = os.path.getsize(full_path)
                DELETED_FILE_SIZE += FILE_SIZE
                os.remove(full_path)  # Remove the file if hash already exists
        else:
            FILE_SIZE = os.path.getsize(full_path)
            DELETED_FILE_SIZE += FILE_SIZE
            shutil.move(full_path, os.path.join(hash_error_dir, file))


def split_large_videos(source_dir, max_workers=4):
    mp4_files = [
        f for f in os.listdir(source_dir)
        if f.lower().endswith('.mp4') 
        and 'DELETE' not in f 
        and 'ERROR' not in f 
        and len(f) != 36
        and len(f) != 37
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_video, file, source_dir, delete_dir, source_dir)
            for file in mp4_files
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
            try:
                future.result()
            except Exception as e:
                pass
                print(f"Worker failed: {e}")


# if __name__ == "__main__":
input_dir = r"Z:\Projects\VideoSplitter\HDD\Scenes\1080p"
delete_dir = os.path.join(input_dir, 'DELETE')
error_dir = os.path.join(input_dir, 'ERROR')
hash_error_dir = os.path.join(input_dir, 'HASH_ERROR')
os.makedirs(delete_dir, exist_ok=True)
os.makedirs(error_dir, exist_ok=True)
os.makedirs(hash_error_dir, exist_ok=True)

split_large_videos(input_dir, max_workers=4)

print("\nProcessing complete.")

print("\n\n ===================================================================================== \n\n")

print(f"Total size before encoding: {TOTAL_SIZE_BEFORE_ENCODING / (1024 * 1024 * 1024):.2f} GB")
print(f"Total size after encoding: {TOTAL_SIZE_AFTER_ENCODING / (1024 * 1024 * 1024):.2f} GB")
print(f"Total size of deleted files: {DELETED_FILE_SIZE / (1024 * 1024 * 1024):.2f} GB")

try:
    shutil.rmtree(delete_dir)
    shutil.rmtree(error_dir)
    shutil.rmtree(hash_error_dir)
    print("Temporary directories removed.")
except Exception as e:
    print(f"Error removing temporary directories: {e}")
    