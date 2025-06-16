import os
import time
from pathlib import Path
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import cv2



SHOW_RESULT=False ### SET TO False if you don't want to see the visual result.

class DetectCrop:
    def __init__(self, model_path, output_dir):
        """
        DetectCrop class constructor

        Args: 
            model_path [str]: Path to the YOLO model
            output_dir [str]: Path to the output_directory where the output images will be saved
        """

        self.model_path = model_path
        self.model = YOLO(model_path)
        self.out_dir = output_dir

    
    def make_dirs(self, out_dir, card_type=None, side=None, dir_name='segmented_crops'):
        if side and card_type:
            output_dir = Path(os.path.join(out_dir, dir_name, f"nid_{card_type}", side))
        else:
            output_dir = Path(os.path.join(out_dir, dir_name))
        
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def detect(self, image=None, image_path=None, side=None, seg=False, crop=True, mask=False):
        with open('logger.txt', 'w') as f:
            f.write("Logger\n")
            f.write("Image_id, conf")
        f.close()

        if image_path:
            if isinstance(image_path, list):
                test_images = [os.path.join(image_path, path) for path in os.listdir(image_path)]
                print(f"Total Image count: {len(test_images)}\n")
            else:
                test_images = image_path
        if isinstance(image, np.ndarray):
            test_images = image
        results = self.model(test_images)
        
        if SHOW_RESULT:
            results[0].show() # only for test purpose, commnet it when done

        counter = 0

        for r in results:
            
            for ci, c in enumerate(r):
                conf = c.boxes.conf[0] # get the confidence score of the prediction
                if conf >= 0.8:
                    img = np.copy(c.orig_img)
                    img_name = Path(c.path).stem

                    label = c.names[c.boxes.cls.tolist().pop()]
                    card_type = label.split("_")[0].lower() # get the type of card OLD or SMART
                    # side = label.split("_")[2].lower()
                    # create binary mask
                    b_mask = np.zeros(img.shape[:2], np.uint8)

                    # create contour mask
                    contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                    _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

                    isolated = np.dstack([img, b_mask])
                    x1, y1, x2, y2 = c.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
                    if seg:
                        if side:
                            output_dir = self.make_dirs(out_dir=self.out_dir, card_type=card_type, side=side, dir_name='segmented')
                            output_path_seg = str(output_dir / f"output.png")
                            cv2.imwrite(output_path_seg)
                        else:
                            output_dir = self.make_dirs(out_dir=self.out_dir, dir_name='segmented')
                            output_path_seg = str(output_dir / f"{img_name}_corrected_seg{ci}.png")
                            cv2.imwrite(output_path_seg, isolated)
                    
                    if mask:
                            if side:
                                output_dir = self.make_dirs(out_dir=self.out_dir, card_type=card_type, side=side, dir_name='segmented_masks')
                                output_path_mask = str(output_dir / f"output.png")
                                cv2.imwrite(output_path_mask)
                            else:
                                output_dir = self.make_dirs(out_dir=self.out_dir, side=side, dir_name='segmented_masks')
                                output_path_mask = str(output_dir / f"{img_name}_corrected_mask{ci}.png")
                                c.plot(save=True, filename=output_path_mask)

                    if crop:
                        if side:
                            output_dir = self.make_dirs(out_dir=self.out_dir, card_type=card_type, side=side)
                            # output_path = str(output_dir / f"{img_name}_corrected_seg_{ci}.png")
                            output_path = str(output_dir / f"output.png")
                            # self.crop(isolated, b_mask, contour, output_path)
                            counter += 1
                        else:
                            output_dir = self.make_dirs(out_dir=self.out_dir)
                            output_path = str(output_dir / f"{img_name}_corrected_crop_{ci}.png")
                            counter += 1
                        
                        cropped_image = self.crop(isolated, b_mask, contour, output_path)

                    # else:
                    #     return img_name, img, b_mask, contour
                else:
                    with open("logger.txt", 'a') as f:
                         f.write(f"\n{c.path.split('/')[-1]}, {conf}")
        
        print(f"Total Segementation saved: {counter}\n")

        ## Return the ndaryy image 

        ############
        return cropped_image, side, card_type
        ############

    def crop(self, image, mask, contour, output_path,  min_area=100):
        """
        Crop and rotate the segmented region to correct the contour's angle, and save it.
        Args:
            image: Original image (numpy array, HxWx3, RGB).
            mask: Binary mask (numpy array, HxW).
            contour: Contour of the mask (numpy array, Nx1x2).
            output_path: Path to save the cropped and rotated image.
            min_area: Minimum area of the mask to consider (to filter noise).
        Returns:
            bool: True if saved successfully, False otherwise.
        """

        # Ensure mask is binary (0 or 255)

        mask = mask.astype(np.uint8)
        if np.sum(mask > 0) < min_area:
            print(f"Mask too small (area={np.sum(mask > 0)}). Skipping.")
            return False

        print(f"Image shape: {image.shape}")
        # Create RGBA image for transparency
        rgba_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        rgba_image[:, :, :3] = image[:, :, :3]  # Copy RGB channels
        rgba_image[:, :, 3] = mask  # Alpha channel: 255 where mask is 255, 0 elsewhere

        # Find the bounding rectangle of non-zero mask pixels (initial crop)
        coords = np.column_stack(np.where(mask > 0))
        if coords.size == 0:
            print("Empty mask. Skipping.")
            return False
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # Crop the region
        cropped_image = rgba_image[y_min:y_max+1, x_min:x_max+1]


        # Get the contour's angle
        rect = cv2.minAreaRect(contour)
        mask_angle = rect[2]
        print(f"Mask angle: {mask_angle}")

        if mask_angle > 45:
            mask_angle = -1 * (90 - mask_angle)

        print(f"Contour angle: {mask_angle}, Rotation applied: {mask_angle}")

        # Get the center of the cropped image
        (h, w) = cropped_image.shape[:2]
        # (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, mask_angle, scale=1.0)

        # Rotate the cropped image with a larger canvas to avoid clipping
        rotated_image = cv2.warpAffine(
            cropped_image,
            M,
            (w * 2, h * 2),  # Larger canvas
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)  # Transparent border
        )

        # Recrop to remove extra padding
        alpha_channel = rotated_image[:, :, 3]
        coords = np.column_stack(np.where(alpha_channel > 0))
        if coords.size == 0:
            print("Empty rotated mask. Skipping.")
            return False
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        final_image = rotated_image[y_min:y_max+1, x_min:x_max+1]

        # Save as PNG to preserve transparency
        cv2.imwrite(output_path, final_image)
        print(f"Saved cropped and rotated segmented region to {output_path}")

        return final_image[:, :, :3]

if __name__ == "__main__":
    model_path = r"D:\card_detection_proj_tashfiq\yolov8n_seg_twostage_card_detection\twostage_run_stage1\weights\best.pt"
    out_dir = Path().resolve()
    dc = DetectCrop(model_path=model_path, output_dir=out_dir)
    image_base_dir = r"D:\card_detection_proj_tashfiq\data\images\test"
    image_paths = os.listdir(image_base_dir)
    start_time = time.time()
    # for fname in image_paths:
    #     img_path = os.path.join(image_base_dir, fname)
    #     dc.detect(image_path=img_path, seg=True, mask=True, crop=True)
    img = cv2.imread(os.path.join(image_base_dir, image_paths[0]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cropped_img, side, card_type = dc.detect(image=img, side='front')
    print(f"Side: {side}")
    print(f"Card type: {card_type}")
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nElapsed Time: {elapsed}")

    if SHOW_RESULT:
        plt.imshow(cropped_img)
        plt.show()
