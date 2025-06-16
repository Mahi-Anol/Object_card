import os
from pathlib import Path
from ultralytics import YOLO
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
# path to test image directory


SHOW_VISUAL_RESULT=True 

### DIRS
test_path = r'D:\card_detection_proj_tashfiq\data\images\test'
test_images_path = [os.path.join(test_path, path) for path in os.listdir(test_path)]
model_path = r'D:\card_detection_proj_tashfiq\yolov8n_seg_twostage_card_detection\twostage_run_stage1\weights\best.pt'


model = YOLO(model_path)


st=time.time()
results = model(
    # 'dl_41.jpg'
    r'D:\card_detection_proj_tashfiq\0bc30109-15dabe18-IMG_6247_aug_1.jpg'
    # test_images_path[4] ### GIVE COMPLETE TEST IMAGES PATH or 1 single image , if multiple image inference used we might need to add some more modification in the crooped image saving code.....
)
end=time.time()

print("Model Inference Time: %f seconds" %(end-st))


if SHOW_VISUAL_RESULT:
    results[0].show()

st=time.time()
for r in results:
    img = np.copy(r.orig_img)
    img_name = Path(r.path).stem  # source image base-name

    # Iterate each object contour (multiple detections)
    for ci, c in enumerate(r):
        label = c.names[c.boxes.cls.tolist().pop()]
        # Create binary mask
        b_mask = np.zeros(img.shape[:2], np.uint8)

        #  Extract contour result
        contour = c.masks.xy.pop()
        # contour = c.obb.xyxy.pop()
        # Changing the type
        contour = contour.astype(np.int32)
        # Reshaping
        contour = contour.reshape(-1, 1, 2)


        # Draw contour onto mask
        _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
        # Create 3-channel mask
        # mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)

        # # Isolate object with binary mask
        # isolated = cv2.bitwise_and(mask3ch, img)
        # Isolate object with transparent background (when saved as PNG)
        isolated = np.dstack([img, b_mask])
        x1, y1, x2, y2 = c.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
        # Crop image to object region
        iso_crop = isolated[y1:y2, x1:x2]

end=time.time()
print("Model output post processing Time: %f seconds" %(end-st))

lower_cased_label=label.lower()
if 'front' in lower_cased_label:
    infered_label_type='Front'
else:
    infered_label_type='Back'

### CARD TYPE EXTRACTION
label_to_card_types= {
  "Driving License Back":"DRIVING LICENSE",
  "Driving License Front":"DRIVING LICENSE",
  "OLD_NID_BACK":"OLD NID",
  "OLD_NID_FRONT":"OLD NID",
  "SMART_NID_BACK":"SMART NID",
  "SMART_NID_FRONT":"SMART NID",
  "Vehicle Registration Card Back":"VEHICLE REGISTRATION",
  "Vehicle Registration Card Front":"VEHICLE REGISTRATION",
}
card_type=label_to_card_types[label]

# print(f"\nLabel: {label}") ###uncomment if you need to see original label name.
print("Infered Card Type: ",card_type)
print("Infered Card Side: ",infered_label_type)


### CODE TO SAVE CROPEED IMAGE
os.makedirs('cropped_image',exist_ok=True)
cv2.imwrite('cropped_image/ouput{i}.png', isolated)
print('Cropped images saved at : ','./cropped_image')



### Code to see cropped image
# if SHOW_VISUAL_RESULT:
#     plt.imshow(isolated)
#     plt.show()