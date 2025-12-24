import os, tensorflow_datasets as tfds, numpy as np, imageio

data_dir = ""
split = "train[:32]"
out_dir = "./fd_videos"
os.makedirs(out_dir, exist_ok=True)

ds = tfds.load("droid_100", data_dir=data_dir, split=split, try_gcs=False, shuffle_files=False)

for epi_idx, episode in enumerate(ds):
    frames = []
    for step in tfds.as_numpy(episode["steps"]):
        obs = step["observation"]
        img = np.concatenate([
            # obs["exterior_image_1_left"],
            obs["exterior_image_2_left"],
            # obs["wrist_image_left"],
        ], axis=1)
        frames.append(img)
    if not frames:
        continue
    video_path = os.path.join(out_dir, f"droid_100_{epi_idx:03d}.mp4")
    imageio.mimwrite(video_path, frames, fps=15, quality=8)
    print("saved", video_path, "frames:", len(frames))
