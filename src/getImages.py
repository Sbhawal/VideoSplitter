import os
from tqdm import tqdm
import random
from PIL import Image
import imageio

def get_random_video(input_dir):
    video_exts = ('.mp4', '.avi', '.mov', '.mkv')
    videos = [f for f in os.listdir(input_dir) if f.lower().endswith(video_exts)]
    if not videos:
        raise FileNotFoundError("No video files found in the input directory.")
    return os.path.join(input_dir, random.choice(videos))

def save_random_frame(video_path, output_path):
    reader = imageio.get_reader(video_path)
    frame_count = reader.count_frames()
    if frame_count == 0:
        reader.close()
        raise ValueError("No frames found in the video.")
    rand_frame = random.randint(0, frame_count - 1)
    frame = reader.get_data(rand_frame)
    img = Image.fromarray(frame)
    img.save(output_path)
    reader.close()

if __name__ == "__main__":
    input_dir = input("Enter the path to the video directory: ").strip()
    os.chdir(input_dir)
    for x in tqdm(os.listdir(input_dir)):
        if x.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            save_random_frame(x, x.replace('.mp4', '.jpg').replace('.avi', '.jpg').replace('.mov', '.jpg').replace('.mkv', '.jpg'))