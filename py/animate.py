import cv2
import os
import re

OUTFILE = "video.mp4"

images = [f for f in os.listdir(".") if f.endswith(".png")]
images = sorted(
    images,
    key=lambda f: int(re.match(r"out(?P<i>\d+).png", f).group("i"))
)

frame = cv2.imread(images[0])
height, width, channels = frame.shape
video = cv2.VideoWriter(
    OUTFILE, cv2.VideoWriter_fourcc(*"MP4V"), 5, (width, height)
)

for image in images:
    video.write(cv2.imread(image))

video.release()
