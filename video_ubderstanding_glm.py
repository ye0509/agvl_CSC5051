# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
from typing import Optional
from enum import Enum
import os
import base64
import shutil
import requests

import cv2
import numpy as np
import random

seed_vl_version = os.getenv("QWEN_MODEL", "zai-org/GLM-4.5V")
api_key = os.getenv("SILICONFLOW_TOKEN")
if not api_key:
    raise RuntimeError("SILICONFLOW_TOKEN（Bearer token）。")
base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1/chat/completions")

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
    overall_prompt = "You are an expert roboticist tasked to predict task completion percentages for frames of a robot for the task of {task_description}." \
            "The task completion percentages are between 0 and 100, where 100 corresponds to full task completion. " \
            "Calfully distinguish the difference among images ,Images that are closer together have more similar completion percentages."\
            "We provide several examples of the robot performing the task at various stages and their corresponding task completion percentages." \
            "Note that these frames are in random order, so please pay attention to the individual frames when reasoning about task completion percentage.".format(task_description=prompt)
    
    content = []

    content.append(
        {
            "type": "text",
            "text": overall_prompt,
    })

    # content.append({
    #     "type": "text",
    #     "text": "For example:"
    # })

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
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": seed_vl_version,
        "messages": messages,
    }
    resp = requests.post(base_url, json=payload, headers=headers, timeout=120)
    if not resp.ok:
        print(f"[Error] status: {resp.status_code}, body: {resp.text}")
        raise RuntimeError(f"request failed: {resp.status_code}")
    data = resp.json()
    return data["choices"][0]

if __name__ == "__main__":

    import os
    image_base_path = "/home/ubt/NLP/fd_videos/cmu_stretch_001"
    instruction = ""

    image_path_list = []

    for filename in os.listdir(image_base_path):
        image_path_list.append(os.path.join(image_base_path, filename))

    image_path_list.sort()
    max_frame = int(os.getenv("MAX_FRAMES", 65))
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
    print("Seed1.5-VL:\n", result["message"]["content"])
    print("shuffle_index:", shuffle_index)

    result_text = result["message"]["content"]
    result_text_list = result_text.split("%\n") 

    pred_percent = []

    for i in range(len(result_text_list)):
        s_index = result_text_list[i].find("Percentages:")
        if s_index == -1:
            continue
        if i != len(result_text_list) - 1:
            tmp_str = result_text_list[i][s_index + 12 :]
        else:
            tmp_str = result_text_list[i][s_index + 12 : -1]
        try:
            pred_percent.append(float(tmp_str) / 100)
        except ValueError:
            continue

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

    import matplotlib.pyplot as plt

    plt.plot(array1, array2, '-o')

    plt.xlabel('frame number')
    plt.ylabel('percent')
    plt.savefig('frame_percent.png')

    plt.show()
