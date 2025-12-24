import cv2
import json
import os

def read_sample(data_name, video_name, save_flag, frame_interval):

    video_path = os.path.join("./fd_videos",video_name+".mp4")

    cap = cv2.VideoCapture(video_path)
    if not os.path.exists(os.path.join("./", data_name)):
        os.makedirs(os.path.join("./", data_name))

    save_frame_path = os.path.join("./", data_name, video_name)

    if not os.path.exists(save_frame_path):
        os.makedirs(save_frame_path)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            image_path = os.path.join(
                save_frame_path, f'frame_'+"{:04d}".format(frame_count)+".png")
            if save_flag:
                cv2.imwrite(
                    image_path,
                    frame,
                )
        frame_count += 1

    print(video_path+" original frame number:", frame_count)
    print(video_path+" sample frame number:", int(frame_count / frame_interval))

    return save_frame_path

if __name__ == "__main__":
    data_name = "fd_videos"
    video_name = "droid_100_031"
    anno_path = ""
    read_sample(data_name, video_name, True, 4)
