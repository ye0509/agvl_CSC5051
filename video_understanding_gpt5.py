# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
from typing import Optional
from enum import Enum
import os
import base64
import shutil
import requests
from openai import OpenAI
from datetime import datetime
import re
import cv2
import numpy as np
import random

repo_root = os.path.dirname(__file__)

seed_vl_version = os.getenv("OPENAI_MODEL", "gpt-5-2025-08-07")
_base = os.getenv("OPENAI_API_BASE", "https://jeniya.top/v1")
_key = os.getenv("OPENAI_API_KEY", "sk-lSIZWYQ8n3dxWzsT4mGkENIOxEpDAHVXIAYaNDL7VXUixem4")
client = OpenAI(
    base_url=_base,
    api_key=_key,
    timeout=300,
)

class Strategy(Enum):
    # sampling stragegies
    # constant interval: sampling at a constant interval, fps sampling
    CONSTANT_INTERVAL = "constant_interval"
    # even interval: sampling at an even interval, uniform sampling
    EVEN_INTERVAL = "even_interval"

def resize(image):
    height, width = image.shape[:2]
    if height < width:
        target_height, target_width = 480, 640
    else:
        target_height, target_width = 640, 480
    if height <= target_height and width <= target_width:
        return image
    if height / target_height < width / target_width:
        new_width = target_width
        new_height = int(height * (new_width / width))
    else:
        new_height = target_height
        new_width = int(width * (new_height / height))
    return cv2.resize(image, (new_width, new_height))

def encode_image(image_path):
    image = cv2.imread(image_path)
    image_resized = resize(image)
    _, encoded_image = cv2.imencode(".jpg", image_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 39])
    return base64.b64encode(encoded_image).decode("utf-8")

def construct_messages(image_paths, timestamps, prompt):
    """
    construct messages for the video understanding
    """
    shuffle_index = list(range(1, len(image_paths)))
    random.shuffle(shuffle_index)
    # successfully
    overall_prompt = (
        """You are an expert roboticist analyzing single image frames from a robot manipulation task involving grasping and relocating an object.

The specific task description is: <TASK_DESCRIPTION>.

For each frame, you must estimate the task completion percentage for this specific task.
Use detailed physical reasoning about robot manipulation (e.g., object poses, grasps, contact states, relative positions, motion direction), but keep all reasoning internal and hidden.
First, silently think step by step to decide how far along the task is in this frame; then ONLY output the final required line, without showing your reasoning.

Guidelines for the percentage:
- Percentages must be integers between 0 and 100 inclusive.
  - 0%: the task has clearly not started (e.g., the target object is still at its initial/source location, not being approached or manipulated).
  - 100%: the target object has been successfully placed at the intended target location/region, is no longer grasped by the robot, and appears stable; the task is clearly finished.
- Assume there is a main target object defined in <TASK_DESCRIPTION>. If multiple objects are present, focus on the object that is most clearly identified as the target in the task description.
- When estimating progress, consider factors such as:
  - Whether the robot is approaching the target object or still far away.
  - Whether a grasp has been formed (or is about to be formed), including partial/unstable grasps.
  - Whether the object has been lifted or at least partially detached from its support surface.
  - How far the object has been transported from the source region toward the target region.
  - Whether the object is close to, touching, or already resting on the target region.
  - Whether the robot has released the object and whether the final configuration looks stable.
- Frames are in random order; do NOT assume any temporal order or continuity.
- Frames that show visually similar progress toward the goal should receive similar percentages.
- If the task has clearly not begun in the frame, use percentages close to 0.
- If the task is clearly finished in the frame, use percentages close to or equal to 100.
- If some details are occluded or ambiguous, make your best estimate based only on the visible evidence.

Output constraints (VERY IMPORTANT):
- Output STRICTLY one line per frame in the following format:
  'Frame N: Frame Description: <short description>. Task Completion Percentage: X%.'
- Replace N with the frame index, <short description> with a concise, neutral description of the frame
  (e.g., 'robot hand approaching object near source', 'object held above table near target', 'object resting on target area, robot hand retracted'), and X with the integer percentage.
- Do NOT add any other text, explanations, reasoning steps, summaries, notes, headers, or final confirmations.
- Do NOT output anything before the first frame line or after the last frame line.


 """
    ).format(task_description=prompt)
    
    content = []

    content.append(
        {
            "type": "text",
            "text": overall_prompt,
    })

    content.append({
        "type": "text",
        "text": "Initial robot scene:"
    })

    content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image(image_paths[0])}",
                    "detail":"low"
                },
            }
        )
    
    content.append({
        "type": "text",
        "text": "In the initial robot scene, the task completion percentage is 0."
    })

    content.append(
        {
            "type": "text",
            "text": "Now, for the task of {task_description}, output the task completion percentage for the following frames that are presented in random order.".format(task_description=prompt)+
              "For each frame, format your response as follow: Frame {i}: Frame Description: {}, Task Completion Percentages:{}%\n",
    })

    for i, idx in enumerate(shuffle_index):
        content.append({
            "type": "text",
            "text": "Frame {i}:".format(i=i+1)
        })
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image(image_paths[idx])}",
                    "detail":"low"
                },
            }
        )
  
  
    return [
        {
            "role": "user",
            "content": content,
        }
    ], shuffle_index

def api_complete(messages):
    """
    调用 Gemini
    """
    resp = client.chat.completions.create(
        model=seed_vl_version,
        messages=messages,
    )
    return resp.choices[0]

if __name__ == "__main__":

    import os

    image_base_path = "/home/ubt/NLP/fd_videos/droid_100_017"

    instruction = ""

    image_path_list = []

    for filename in os.listdir(image_base_path):
        image_path_list.append(os.path.join(image_base_path, filename))

    image_path_list.sort()
    max_frame = int(os.getenv("MAX_FRAMES", 50))
    print("Original_frames:", len(image_path_list))
    if len(image_path_list)>max_frame:
        seq_len = len(image_path_list)
        image_path_list = image_path_list[seq_len%max_frame:] 

        new_image_path_list = []
        interval = int(len(image_path_list)/max_frame)
        for i in range(max_frame):
            new_image_path_list.append(image_path_list[i*interval])
        
        image_path_list = new_image_path_list
    
    print("Sampling_frames:", len(image_path_list))
    print("Instruction:", instruction)

    message, shuffle_index = construct_messages(image_paths=image_path_list, timestamps=None, prompt=instruction)
    result = api_complete(message)
    result_content = result.message.content if hasattr(result, "message") else str(result)
    print("Seed1.5-VL:\n", result_content)
    print("shuffle_index:", shuffle_index)

    result_text = str(result_content)
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*%", result_text)
    print(f"[Debug] extracted {len(matches)} percentage tokens")
    pred_percent = []
    for m in matches:
        try:
            pred_percent.append(float(m) / 100)
        except ValueError:
            continue
    if not pred_percent:
        print("[Warn]")

    model_order_percent = pred_percent[:]
    print("model_order_percent (response order):", model_order_percent)

    frame_count = len(image_path_list)
    if len(pred_percent) != len(shuffle_index):
        print(
            f"[Warn] predicted {len(pred_percent)} values, but shuffle_index has {len(shuffle_index)} entries; filling missing with 0."
        )
    mapping = {0: 0.0}
    usable = min(len(pred_percent), len(shuffle_index))
    for idx in range(usable):
        frame_id = shuffle_index[idx]
        mapping[frame_id] = pred_percent[idx]

    sort_pred_percent = [mapping.get(fid, 0.0) for fid in range(frame_count)]
    
    order = range(len(sort_pred_percent))
    print("order:", order)
    print("sort_pred_percent:", sort_pred_percent)

    from metric import voc_calculation

    array1 = np.array(order)
    array2 = np.array(sort_pred_percent)

    spearman_corr, kendall_corr = voc_calculation(array1, array2)

    print(f"Kendall rank cof: {kendall_corr}, spearman rank cof: {spearman_corr}")

    video_name = os.path.basename(image_base_path.rstrip("/"))
    log_path = os.path.join(repo_root, "run_new.log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"{datetime.now().isoformat()} | video={video_name} | "
            f"kendall={kendall_corr:.6f} | spearman={spearman_corr:.6f}\n"
        )
    print(f"Logged metrics to {log_path}")

    import matplotlib.pyplot as plt

    plt.plot(array1, array2, '-o')

    plt.xlabel('frame number')
    plt.ylabel('percent')
    plt.savefig('frame_percent.png')
    plt.show()
