import os
import json
import cv2
import torch
import warnings
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import torchvision.transforms.functional as TVF
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from src.Models import VisionModel

warnings.filterwarnings(
    "ignore",
    message=".*torch.backends.cuda.sdp_kernel.*is deprecated.*",
    category=FutureWarning
)

# ==== Config ====
SCENE_DETAILS_TSV = r"Z:\Projects\VideoSplitter\HDD\Scenes\scene_details.tsv"
SCENE_FOLDER = r"Z:\Projects\VideoSplitter\HDD\Scenes\1080p"
TEMP_DIR = r"R:\Temp"
path = 'model'
THRESHOLD = 0.3
SAVE_INTERVAL = 100  # Save TSV every N videos
MAX_WORKERS = 6      # Number of threads to use

# ==== Globals ====
ProcessedVideos = set()
df_lock = Lock()
progress_lock = Lock()

# ==== Load model ====
model = VisionModel.load_model(path)
model.eval().to('cuda')

# Load processed list if exists
if os.path.exists(SCENE_DETAILS_TSV):
    df = pd.read_csv(SCENE_DETAILS_TSV, sep='\t')
    ProcessedVideos = set(df['name'].tolist())
else:
    df = pd.DataFrame(columns=['name', 'scores'])

# Load tag list
with open(Path(path) / 'top_tags.txt', 'r') as f:
    top_tags = [line.strip() for line in f if line.strip()]

# ==== Helper functions ====
def prepare_image(image: Image.Image, target_size: int) -> torch.Tensor:
    max_dim = max(image.size)
    pad_left = (max_dim - image.size[0]) // 2
    pad_top = (max_dim - image.size[1]) // 2
    padded = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
    padded.paste(image, (pad_left, pad_top))
    if max_dim != target_size:
        padded = padded.resize((target_size, target_size), Image.BICUBIC)
    tensor = TVF.pil_to_tensor(padded) / 255.0
    tensor = TVF.normalize(tensor, mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711])
    return tensor

@torch.no_grad()
def predict(image: Image.Image):
    tensor = prepare_image(image, model.image_size).unsqueeze(0).to('cuda')
    with torch.amp.autocast_mode.autocast('cuda', enabled=True):
        preds = model({'image': tensor})
        tag_preds = preds['tags'].sigmoid().cpu()
    scores = {top_tags[i]: tag_preds[0][i].item() for i in range(len(top_tags))}
    return scores

EPS = 1e-8
def noisy_or(ps): return 1.0 - np.prod(1.0 - np.clip(ps, EPS, 1-EPS))
def freq_times_conf(ps, threshold=0.3):
    ps = np.asarray(ps, dtype=float)
    voted = ps >= threshold
    return 0.0 if not voted.any() else float(voted.mean() * ps[voted].mean())
def hybrid_score(ps, threshold=0.3, lam=0.8):
    return lam * noisy_or(ps) + (1 - lam) * freq_times_conf(ps, threshold)

def aggregate_tags(tag_score_list, method="hybrid", threshold=0.3, lam=0.8):
    tag_dict = {}
    for tag, score in tag_score_list:
        tag_dict.setdefault(tag, []).append(score)
    final_scores = []
    for tag, scores in tag_dict.items():
        if method == "noisy_or":
            score = noisy_or(scores)
        elif method == "freq_conf":
            score = freq_times_conf(scores, threshold)
        else:
            score = hybrid_score(scores, threshold, lam)
        final_scores.append((tag, score))
    final_scores.sort(key=lambda x: x[1], reverse=True)
    return final_scores

def getScoreForAllFrames(filepath, fps=1):
    cap = cv2.VideoCapture(filepath)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(orig_fps / fps))
    scores_all = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            scores = predict(image)
            for tag, score in scores.items():
                if score > THRESHOLD:
                    scores_all.append((tag, score))
        idx += 1
    cap.release()
    return aggregate_tags(scores_all)

# ==== Worker ====
def process_video(video):
    with df_lock:
        if video in ProcessedVideos:
            return None  # Already processed
        ProcessedVideos.add(video)

    filepath = os.path.join(SCENE_FOLDER, video)
    try:
        scores = getScoreForAllFrames(filepath)
    except Exception as e:
        return f"Error processing {video}: {e}"

    with df_lock:
        df.loc[len(df)] = {'name': video, 'scores': json.dumps(scores)}
        if len(ProcessedVideos) % SAVE_INTERVAL == 0:
            df.to_csv(SCENE_DETAILS_TSV, sep='\t', index=False)
    return None

# ==== Main ====
def generateVideoScoreDataset():
    video_files = sorted(
        [f for f in os.listdir(SCENE_FOLDER)
         if f.endswith('.mp4') and f not in ProcessedVideos and (len(f) in (36, 37))],
        key=str.lower
    )
    if not video_files:
        print("No unprocessed videos found.")
        return

    print(f"Processing {len(video_files)} videos using {MAX_WORKERS} threads...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_video, v): v for v in video_files}
        with tqdm(total=len(video_files), desc="Processing videos") as pbar:
            for future in as_completed(futures):
                err = future.result()
                if err:
                    print(err)
                pbar.update(1)

    # Final save
    with df_lock:
        df.to_csv(SCENE_DETAILS_TSV, sep='\t', index=False)

if __name__ == "__main__":
    generateVideoScoreDataset()
