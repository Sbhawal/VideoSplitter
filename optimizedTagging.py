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
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from src.processData import *
from src.Models import VisionModel

warnings.filterwarnings(
    "ignore",
    message=".*torch.backends.cuda.sdp_kernel.*is deprecated.*",
    category=FutureWarning
)

# ==== Config ====
SCENE_FOLDER = r"Z:\Projects\VideoSplitter\HDD\Scenes\1080p"
TEMP_DIR = r"R:\Temp"
MODEL_PATH = 'model'
THRESHOLD = 0.3
SAVE_INTERVAL = 100        # Save every N videos
MAX_WORKERS = 4        # Number of threads to use for per-video processing (IO + preprocessing)
SAMPLING_FPS = 2           # frames/second to sample from each video (videos max 10s so small)
BATCH_SIZE = 32            # GPU batch for model forward

# ==== Globals ====
ProcessedVideos = set()
df_lock = Lock()

# ==== Device & model load ====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = VisionModel.load_model(MODEL_PATH)
model.eval()
model.to(device)

# Keep image size in local variable
IMAGE_SIZE = getattr(model, 'image_size', 224)

df = read_complete_df()
ProcessedVideos = set(df['name'].tolist())
df = pd.DataFrame(columns=['name', 'scores'])

# Load tag list
with open(Path(MODEL_PATH) / 'top_tags.txt', 'r') as f:
    top_tags = [line.strip() for line in f if line.strip()]

# ==== Helper functions ====
def prepare_image_pil(image: Image.Image, target_size: int) -> torch.Tensor:
    """
    Prepare PIL image -> normalized tensor on CPU (C,H,W), not moved to device here.
    Keeps same algorithm as original (pad to square, resize, normalize).
    """
    max_dim = max(image.size)
    pad_left = (max_dim - image.size[0]) // 2
    pad_top = (max_dim - image.size[1]) // 2
    padded = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
    padded.paste(image, (pad_left, pad_top))
    if max_dim != target_size:
        padded = padded.resize((target_size, target_size), Image.BICUBIC)
    arr = np.asarray(padded).astype(np.float32) / 255.0  # H,W,C
    # HWC -> CHW and normalize
    arr = arr.transpose(2, 0, 1)
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32).reshape(3,1,1)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32).reshape(3,1,1)
    arr = (arr - mean) / std
    return torch.from_numpy(arr)  # float32 tensor on CPU

@torch.no_grad()
def batch_predict(tensor_batch: torch.Tensor):
    """
    tensor_batch: float32 tensor (B, C, H, W) already moved to device
    returns: numpy array (B, num_tags) with sigmoid outputs on CPU
    """
    with torch.amp.autocast(device.type, enabled=(device.type == 'cuda')):
        preds = model({'image': tensor_batch})
        tag_logits = preds['tags']
        tag_probs = torch.sigmoid(tag_logits)  # keep on device
    return tag_probs.cpu().numpy()  # move to CPU numpy

EPS = 1e-8
def noisy_or(ps): return 1.0 - np.prod(1.0 - np.clip(ps, EPS, 1-EPS))
def freq_times_conf(ps, threshold=0.3):
    ps = np.asarray(ps, dtype=float)
    voted = ps >= threshold
    return 0.0 if not voted.any() else float(voted.mean() * ps[voted].mean())
def hybrid_score(ps, threshold=0.3, lam=0.8):
    return lam * noisy_or(ps) + (1 - lam) * freq_times_conf(ps, threshold,)

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
        final_scores.append((tag, float(score)))
    final_scores.sort(key=lambda x: x[1], reverse=True)
    return final_scores

def getScoreForAllFrames(filepath, sampling_fps=SAMPLING_FPS, batch_size=BATCH_SIZE):
    """
    Reads video, samples frames at sampling_fps, converts frames to tensors on CPU,
    runs batched GPU inference, and aggregates tag scores.
    """
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {filepath}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # compute interval in frames (integer)
    frame_interval = max(1, int(round(orig_fps / float(sampling_fps))))
    # Collect tensors (CPU) for batched forward
    tensors_cpu = []
    frame_indices = []

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            # convert BGR (cv2) to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            try:
                t = prepare_image_pil(pil, IMAGE_SIZE)  # CPU tensor
                tensors_cpu.append(t)
                frame_indices.append(idx)
            except Exception as e:
                # skip problematic frames but keep processing
                print(f"Warning: failed prepare frame {idx} in {os.path.basename(filepath)}: {e}")
        idx += 1
    cap.release()

    if not tensors_cpu:
        return []  # no tags

    # Run batched prediction
    scores_all = []
    num_tags = len(top_tags)
    for start in range(0, len(tensors_cpu), batch_size):
        batch_tensors = torch.stack(tensors_cpu[start:start + batch_size], dim=0)  # B,C,H,W on CPU
        batch_tensors = batch_tensors.to(device, non_blocking=True)
        try:
            probs = batch_predict(batch_tensors)  # numpy (B, num_tags)
        except Exception as e:
            print("1")
            # if model forward fails, free GPU memory and raise
            torch.cuda.empty_cache()
            raise
        # iterate batch rows
        B = probs.shape[0]
        for b in range(B):
            row = probs[b]
            # only keep tags above threshold to reduce memory
            for i, p in enumerate(row):
                if p > THRESHOLD:
                    scores_all.append((top_tags[i], float(p)))
        # free GPU allocation of batch
        del batch_tensors
        torch.cuda.empty_cache()

    # aggregate
    return aggregate_tags(scores_all)

# ==== Worker ====
def process_video(video):
    # quick check + reserve
    # print(video)
    with df_lock:
        if video in ProcessedVideos:
            return None
        ProcessedVideos.add(video)

    filepath = os.path.join(SCENE_FOLDER, video)
    try:
        scores = getScoreForAllFrames(filepath, sampling_fps=SAMPLING_FPS, batch_size=BATCH_SIZE)
    except Exception as e:
        print("2")
        # remove from processed so it can be retried later
        with df_lock:
            ProcessedVideos.discard(video)
        return f"Error processing {video}: {repr(e)}"

    with df_lock:
        df.loc[len(df)] = {'name': video, 'scores': json.dumps(scores)}
        if len(ProcessedVideos) % SAVE_INTERVAL == 0:
            # df.to_csv(, sep='\t', index=False)
            # Save last 10 appended rows to a separate file
            last_INTERVAL_df = df.tail(SAVE_INTERVAL)
            save_new_temp_df(last_INTERVAL_df)
    return None

# ==== Main ====
def generateVideoScoreDataset():
    # List videos (faster with scandir)
    entries = [entry.name for entry in os.scandir(SCENE_FOLDER)
               if entry.is_file() and entry.name.endswith('.mp4') and entry.name not in ProcessedVideos and (len(entry.name) in (36, 37))]
    video_files = sorted(entries, key=str.lower)

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
        last_INTERVAL_df = df.tail(SAVE_INTERVAL)
        save_new_temp_df(last_INTERVAL_df)

if __name__ == "__main__":
    import time
    split_and_save_df_chunks()
    time.sleep(5)  # wait for file system to settle
    generateVideoScoreDataset()
    split_and_save_df_chunks()
