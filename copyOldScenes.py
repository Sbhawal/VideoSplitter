import os, subprocess,shutil
from tqdm import tqdm

DIR_TREE = input("Enter the directory tree to search for videos: ").strip()
videos = []

_720= r"Z:\Projects\VideoSplitter\HDD\Scenes\720p"
_1080= r"Z:\Projects\VideoSplitter\HDD\Scenes\1080p"
_2160= r"Z:\Projects\VideoSplitter\HDD\Scenes\2160p"

for root, dirs, files in os.walk(DIR_TREE):
    for file in files:
        if file.lower().endswith(('.pkl')):
            os.remove(os.path.join(root, file))
        if len(files) < 2:
            continue
        if file.lower().endswith(('.mp4')):
            videos.append(os.path.join(root, file))
    for dir in dirs:
        dirpath = os.path.join(root,dir)
        if len(os.listdir(dirpath)) == 0:
            shutil.rmtree(dirpath)
            print(f"Deleting {dirpath}.")

        

def get_vid_res(video_path):
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=p=0",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    height = int(result.stdout.split(',')[1])
    return height

from concurrent.futures import ThreadPoolExecutor, as_completed

def process_video(video):
    try:
        height = get_vid_res(video)
        if height < 720:
            os.remove(video)
        elif height < 1080:
            try:
                shutil.move(video, os.path.join(_720, os.path.basename(video)))
            except FileExistsError:
                os.remove(video)
        elif height < 1900:
            try:
                shutil.move(video, os.path.join(_1080, os.path.basename(video)))
            except FileExistsError:
                os.remove(video)
        elif height > 1900:
            try:
                shutil.move(video, os.path.join(_2160, os.path.basename(video)))
            except FileExistsError:
                os.remove(video)
        else:
            return
    except Exception as e:
        print(f"Error processing {video}: {e}")

with ThreadPoolExecutor(max_workers=8) as executor:
    list(tqdm(executor.map(process_video, videos), total=len(videos)))

    