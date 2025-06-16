import os 
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


img_dir = 'E:/license_detection_proj/license_detection/card_images/data/real_world_test_v2'
img_path = [os.path.join(img_dir, path) for path in os.listdir(img_dir)]

img = cv2.imread(img_path[5])
# print(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

# plt.imshow(img[1500:2200, 900:2000, :])
# cv2.imwrite(f'{img_dir}/cropped_1.png', cv2.cvtColor(img[1500:2200, 900:2000, :], cv2.COLOR_BGR2RGB))
cv2.imwrite(f'{img_dir}/cropped_2.png', cv2.cvtColor(img[1500:2800, 189:2600, :], cv2.COLOR_BGR2RGB))
# cv2.imwrite(f'{img_dir}/cropped_1.png', img[1500:2200, 900:2000, :])

