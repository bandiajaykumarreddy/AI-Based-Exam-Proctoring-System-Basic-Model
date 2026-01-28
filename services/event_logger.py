import csv
import os
import cv2

class EventLogger:
    def __init__(self, log_file, screenshot_dir):
        self.log_file = log_file
        self.screenshot_dir = screenshot_dir
        os.makedirs(screenshot_dir, exist_ok=True)

        self.fp = open(log_file, "w", newline="")
        self.writer = csv.writer(self.fp)
        self.writer.writerow(["Time(s)", "Event", "Evidence"])

    def log(self, sec, event, frame):
        name = f"{event.replace(' ', '_').lower()}_{sec}.jpg"
        path = os.path.join(self.screenshot_dir, name)
        cv2.imwrite(path, frame)
        self.writer.writerow([sec, event, name])

    def close(self):
        self.fp.close()