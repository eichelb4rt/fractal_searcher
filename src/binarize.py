import cv2
from pathlib import Path

source_file = "images/penis_2_balls.png"
out_file = f"images/{Path(source_file).stem}_binarized.png"
img = cv2.imread(source_file, 0)
# binarized = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 6)
_, binarized = cv2.threshold(img, 40, 255, cv2.THRESH_BINARY)
cv2.imwrite(out_file, binarized)
