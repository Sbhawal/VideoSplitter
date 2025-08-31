import multiprocessing
import psutil
import os, time
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import pandas as pd
import shutil
import subprocess
from tqdm import tqdm
from multiprocessing import Process

ffmpeg_path = r"C:\binaries\ffmpeg\bin\ffmpeg.exe" 
TARGET_FPS = 30
AUDIO_CODEC = "aac"
AUDIO_BITRATE = "192k"
MAX_WORKERS = 3  # Parallel encoding processes (2–3 is best for RTX 4060)

count = 10

def get_video_resolution(video_path):
    """Use ffprobe to get the resolution of a video."""
    cmd = [
        ffmpeg_path.replace("ffmpeg.exe", "ffprobe.exe"),
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'csv=s=x:p=0',
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    resolution = result.stdout.strip()
    return resolution

def set_cpu_affinity(use_cores):
    """Set CPU affinity for current process."""
    p = psutil.Process(os.getpid())
    p.cpu_affinity(use_cores)

def detect_scene_cuts(video_path, csv_output_path):
    total_cores = multiprocessing.cpu_count()
    use_cores = list(range(total_cores - 2))  # Use all but 2 cores
    set_cpu_affinity(use_cores)

    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=20.0))  # Lower threshold for more cuts if needed

    video_manager.set_downscale_factor(1)
    video_manager.start()

    scene_manager.detect_scenes(video_manager, show_progress=True)
    scene_list = scene_manager.get_scene_list()

    # Store both start & end times
    scene_ranges = []
    for i, (start, end) in enumerate(scene_list, start=1):
        scene_ranges.append({
            "Scene_Number": i,
            "Start_Timecode": start.get_timecode(),
            "End_Timecode": end.get_timecode(),
        })

    df = pd.DataFrame(scene_ranges)
    df.to_csv(csv_output_path, index=False)

    video_manager.release()
    print(f"Scene cuts saved to {csv_output_path} ({len(scene_ranges)} scenes)")



import random
import string

def random_string(length=32):
    """Generate a random lowercase alphanumeric string."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

def split_video_with_gpu(video_path, csv_path, _unused_output_dir=None):
    # Read scene cut ranges
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        print(f"No scenes found in {csv_path}, skipping {video_path}.")
        return

    def timecode_to_seconds(tc):
        h, m, s = tc.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)

    start_seconds = [timecode_to_seconds(tc) for tc in df["Start_Timecode"]]
    end_seconds = [timecode_to_seconds(tc) for tc in df["End_Timecode"]]

    # Detect resolution
    resolution = get_video_resolution(video_path)
    if not resolution:
        print(f"Could not detect resolution for {video_path}. Skipping.")
        return
    
    if resolution != "1920x1080":
        print(f"Warning: Detected resolution {resolution} for {video_path}, expected 1920x1080... Skipping !!")
        return

    resolution_map = {
        "1920x1080": "1080p",
        "1280x720": "720p",
        "3840x2160": "4k",
        "640x480": "480p"
    }
    folder_name = resolution_map.get(resolution, resolution)
    if "x" in folder_name:
        folder_name = folder_name.split('x')[-1] + 'p'

    resolution_dir = os.path.join(r"Z:\Projects\VideoSplitter\HDD\Scenes", folder_name)
    os.makedirs(resolution_dir, exist_ok=True)

    base_filename = os.path.basename(video_path).replace('.mp4', '')

    for i, (start, end) in enumerate(zip(start_seconds, end_seconds), start=1):
        duration = end - start
        
        # Generate random 32-char prefix for this split
        prefix = random_string(32)
        output_name = f"{base_filename}_split_{i:03d}_{prefix}.mp4".replace(" ","_")
        output_file = os.path.join(resolution_dir, output_name)

        cmd = [
            ffmpeg_path,
            '-y',
            '-hwaccel', 'cuda',
            '-ss', str(start),
            '-i', video_path,
            '-t', str(duration),
            "-c:v", "h264_nvenc", "-preset", "medium", "-cq", "21",
            "-c:a", AUDIO_CODEC, "-b:a", AUDIO_BITRATE, "-ar", "48000", "-ac", "2",
            "-map", "0:v:0", "-map", "0:a:0?",
            output_file
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[ERROR] FFmpeg failed for {output_file}")
            print(result.stderr)
        else:
            # print(f"[OK] Created {output_file}")
            pass

    print(f"All splits saved for {video_path} in {resolution_dir}")




def print_seperator():
    print("\n" + "="*50 + "\n")



def startDetection():
    print("\nStarting scene cut detection...")
    print_seperator()
    video_files = [x for x in os.listdir() if x.endswith('.mp4')]
    print(f"Found {len(video_files)} video files to process.")

    for file in video_files:
        print(f"\nProcessing file: {file}")
        try:
            video_file = file
            os.rename(video_file, video_file.replace(' ', '_'))  # Replace spaces with underscores
            video_file = video_file.replace(' ', '_')  # Update the variable to the new
            csv_output = video_file.replace('.mp4', '_cuts.csv')
            detect_scene_cuts(video_file, csv_output)
            os.rename(video_file, os.path.join("COMPLETED_DETECTION", video_file))  # Move processed file to COMPLETED_DETECTION
        except Exception as e:
            print(f"Error processing {file}: {e}")
            os.rename(file, os.path.join("ERRORS", file))  # Move errored file to ERRORS
            continue

    print("\nAll scene cuts detected and saved to CSV files.\n")

def splitSingleVideo():
    csv_files = [x for x in os.listdir() if x.endswith('_cuts.csv')]
    if len(csv_files) == 0:
        return
    global count
    count = 100
    for csv in csv_files:
        print(f"\nProcessing CSV: {csv}\n")
        try:
            video_file = os.path.join("COMPLETED_DETECTION", csv.replace('_cuts.csv', '.mp4'))
            output_folder = "SPLITS" + video_file.replace('.mp4', '_splits').replace('COMPLETED_DETECTION', '')
            df = pd.read_csv(csv)
            if len(df) == 0:
                # No splits, move video to NO_SPLITS
                dest_path = os.path.join("NO_SPLITS", os.path.basename(video_file))
                os.rename(video_file, dest_path)
                print(f"No splits found for {video_file}. Moved to NO_SPLITS.")
                os.remove(csv)  # Remove CSV after processing
                continue
            # os.makedirs(output_folder, exist_ok=True)  # Create output folder for splits
            split_video_with_gpu(video_file, csv, output_folder)
            os.remove(csv)  # Remove CSV after processing
            os.remove(video_file)  # Remove original video after splitting
        except Exception as e:
            dest_path = os.path.join("NO_SPLITS", os.path.basename(video_file))
            os.rename(video_file, dest_path)
            print(f"No splits found for {video_file}. Moved to NO_SPLITS.")
            os.remove(csv)  # Remove CSV after processing
            print(f"Error processing {csv}: {e}")
            continue
        print_seperator()


def startSplits():
    global count
    print_seperator()
    while(count):
        splitSingleVideo()
        time.sleep(10)
        count = count - 1

    




if __name__ == '__main__':
    INPUT_DIR = r"Z:\Projects\VideoSplitter\HDD\Scenes\temp"
    if INPUT_DIR:
        os.chdir(INPUT_DIR)
    else:
        exit()

    os.makedirs("ERRORS", exist_ok=True)
    os.makedirs("COMPLETED_DETECTION", exist_ok=True)
    os.makedirs("SPLITS", exist_ok=True)
    os.makedirs("NO_SPLITS", exist_ok=True)

    p1 = Process(target=startDetection)
    p2 = Process(target=startSplits)

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    print("\nDetection and Splitting finished in parallel.\n")