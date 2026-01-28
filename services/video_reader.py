import cv2

def open_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError("Cannot open video")
    return cap

def get_video_info(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = max(1, int(1000 / fps))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_seconds = int(total_frames / fps)
    return fps, delay, total_seconds

def get_current_second(cap):
    return int(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)